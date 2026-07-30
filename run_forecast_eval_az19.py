#!/usr/bin/env python3
"""Run forecast evaluation from az19.ipynb using forecast_eval.py."""

from __future__ import annotations

import os

# Limit native BLAS/OpenMP threads; oversubscription can segfault on some Linux hosts.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import gc
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

_T0 = time.perf_counter()


def _log_phase(message: str) -> None:
    elapsed = time.perf_counter() - _T0
    print(f"[{elapsed:6.1f}s] {message}", flush=True)


def _warn_numpy_pandas_compat() -> None:
    """Pandas 2.3+ against numpy 1.24 is a known segfault combo in read_csv/merge."""
    np_parts = [int(x) for x in np.__version__.split(".")[:2]]
    pd_parts = [int(x) for x in pd.__version__.split(".")[:2]]
    if tuple(pd_parts) >= (2, 3) and tuple(np_parts) < (1, 26):
        print(
            f"WARNING: pandas {pd.__version__} with numpy {np.__version__} is "
            "unsupported and often segfaults in pd.read_csv. Preload defaults "
            "to stdlib csv; upgrade numpy to >=1.26 on the test machine if you "
            "want the pandas preload engine.",
            flush=True,
        )


_warn_numpy_pandas_compat()

_log_phase("importing forecast_eval")
from forecast_eval import (
    build_minute_bin_function_metadata,
    default_holdout_forecasters,
    evaluate_series,
    sample_representative_functions,
    summarize_aggregate_results,
    summarize_results,
    summarize_results_weighted,
    temporal_split,
)

DEFAULT_DATA = Path("data/azure2019")
TRACE_START = pd.Timestamp("2019-07-15", tz="UTC")
TRACE_DAYS = 14
MINUTE_COLS = [str(m) for m in range(1, 1441)]
DAY_CSV_USECOLS = ["HashApp", "HashFunction", *MINUTE_COLS]
MINUTE_DTYPES = {col: np.int32 for col in MINUTE_COLS}


def _validate_freq(freq: str) -> str:
    """Pandas offset for eval buckets; source data is per-minute."""
    td = pd.Timedelta(freq)
    if td <= pd.Timedelta(0):
        raise ValueError(f"freq must be a positive pandas offset, got {freq!r}")
    if td < pd.Timedelta("1min"):
        raise ValueError(
            f"freq must be >= 1min (Azure 2019 CSVs are minute bins), got {freq!r}"
        )
    return freq


def trace_minute_count() -> int:
    return TRACE_DAYS * 1440


def minute_index() -> pd.DatetimeIndex:
    return pd.date_range(
        TRACE_START,
        periods=trace_minute_count(),
        freq="1min",
        tz="UTC",
    )


def trace_index(freq: str = "1min") -> pd.DatetimeIndex:
    """Datetime index spanning the trace at the evaluation frequency."""
    freq = _validate_freq(freq)
    periods = max(1, int(pd.Timedelta(days=TRACE_DAYS) / pd.Timedelta(freq)))
    return pd.date_range(TRACE_START, periods=periods, freq=freq, tz="UTC")


def trace_minutes(freq: str = "1min") -> pd.DatetimeIndex:
    """Alias for trace_index (minute-native CSV layout resampled to freq)."""
    return trace_index(freq)


def series_at_freq(minute_values: np.ndarray, freq: str) -> pd.Series:
    """Build an eval-frequency count series from per-minute counts."""
    freq = _validate_freq(freq)
    minute_series = pd.Series(minute_values, index=minute_index(), dtype=np.int64)
    if pd.Timedelta(freq) <= pd.Timedelta("1min"):
        return minute_series.astype(int)
    return minute_series.resample(freq).sum().astype(int)


def resample_series(series: pd.Series, freq: str) -> pd.Series:
    """Resample a minute-indexed (or reindexable) series to eval frequency."""
    freq = _validate_freq(freq)
    minute_series = series.reindex(minute_index(), fill_value=0).astype(np.int64)
    if pd.Timedelta(freq) <= pd.Timedelta("1min"):
        return minute_series.astype(int)
    return minute_series.resample(freq).sum().astype(int)


def _day_from_path(path: Path) -> int:
    return int(re.search(r"\.d(\d+)\.csv$", path.name).group(1))


def _preload_init_buffers(
    eval_keys: list[tuple[str, str]],
) -> tuple[dict[tuple[str, str], int], np.ndarray]:
    n_minutes = trace_minute_count()
    return (
        {key: idx for idx, key in enumerate(eval_keys)},
        np.zeros((len(eval_keys), n_minutes), dtype=np.int32),
    )


def _accumulate_day_row(
    row: dict[str, str],
    *,
    key_to_idx: dict[tuple[str, str], int],
    accum: np.ndarray,
    day_start: int,
) -> None:
    idx = key_to_idx.get((row["HashApp"], row["HashFunction"]))
    if idx is None:
        return
    day_counts = np.fromiter(
        (int(row[col] or 0) for col in MINUTE_COLS),
        dtype=np.int32,
        count=len(MINUTE_COLS),
    )
    accum[idx, day_start : day_start + 1440] += day_counts


def _preload_function_minute_series_csv(
    eval_keys: list[tuple[str, str]],
    data_dir: Path,
    *,
    freq: str = "1min",
) -> dict[tuple[str, str], pd.Series]:
    """Stream daily CSVs with stdlib csv (no pd.read_csv)."""
    if not eval_keys:
        return {}

    key_to_idx, accum = _preload_init_buffers(eval_keys)

    for path in sorted(data_dir.glob("invocations_per_function*.csv")):
        day_start = (_day_from_path(path) - 1) * 1440
        _log_phase(f"preload {path.name} ({len(eval_keys)} eval functions, csv)")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                _accumulate_day_row(
                    row,
                    key_to_idx=key_to_idx,
                    accum=accum,
                    day_start=day_start,
                )

    return {
        key: series_at_freq(accum[idx], freq)
        for key, idx in key_to_idx.items()
    }


def _preload_function_minute_series_pandas(
    eval_keys: list[tuple[str, str]],
    data_dir: Path,
    *,
    freq: str = "1min",
    chunk_size: int = 2000,
) -> dict[tuple[str, str], pd.Series]:
    """Chunked pandas preload (fast locally; can segfault on old numpy builds)."""
    if not eval_keys:
        return {}

    key_to_idx, accum = _preload_init_buffers(eval_keys)
    keys_df = pd.DataFrame(eval_keys, columns=["HashApp", "HashFunction"])

    for path in sorted(data_dir.glob("invocations_per_function*.csv")):
        day_start = (_day_from_path(path) - 1) * 1440
        _log_phase(f"preload {path.name} ({len(eval_keys)} eval functions, pandas)")
        for chunk in pd.read_csv(
            path,
            usecols=DAY_CSV_USECOLS,
            dtype=MINUTE_DTYPES,
            chunksize=chunk_size,
        ):
            filtered = chunk.merge(
                keys_df, on=["HashApp", "HashFunction"], how="inner"
            )
            del chunk
            if filtered.empty:
                continue
            for (app, func), group in filtered.groupby(
                ["HashApp", "HashFunction"], sort=False
            ):
                idx = key_to_idx.get((app, func))
                if idx is None:
                    continue
                day_counts = group[MINUTE_COLS].sum(axis=0, numeric_only=True)
                accum[idx, day_start : day_start + 1440] += day_counts.to_numpy(
                    dtype=np.int32
                )
            del filtered
        gc.collect()

    return {
        key: series_at_freq(accum[idx], freq)
        for key, idx in key_to_idx.items()
    }


def preload_function_minute_series(
    eval_keys: list[tuple[str, str]],
    data_dir: Path,
    *,
    freq: str = "1min",
    chunk_size: int = 2000,
    engine: str = "csv",
) -> dict[tuple[str, str], pd.Series]:
    if engine == "pandas":
        return _preload_function_minute_series_pandas(
            eval_keys, data_dir, freq=freq, chunk_size=chunk_size
        )
    if engine != "csv":
        raise ValueError(f"unknown preload engine {engine!r}; use 'csv' or 'pandas'")
    return _preload_function_minute_series_csv(eval_keys, data_dir, freq=freq)


def load_function_invocations(
    app: str,
    func: str,
    *,
    data_dir: Path = DEFAULT_DATA,
) -> pd.DataFrame:
    """Per-minute invocation counts for one function across all days."""
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("invocations_per_function*.csv")):
        day = _day_from_path(path)
        day_df = pd.read_csv(path, usecols=DAY_CSV_USECOLS)
        mask = (day_df["HashApp"] == app) & (day_df["HashFunction"] == func)
        rows = day_df.loc[mask, MINUTE_COLS]
        if rows.empty:
            continue
        long = rows.melt(var_name="minute", value_name="count")
        long["minute"] = long["minute"].astype(int)
        long["timestamp"] = (
            TRACE_START
            + pd.Timedelta(days=day - 1)
            + pd.to_timedelta(long["minute"] - 1, unit="m")
        )
        frames.append(long[["timestamp", "count"]])

    if not frames:
        raise ValueError(f"No rows found for app={app!r}, func={func!r}")

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def function_minute_series(
    app: str,
    func: str,
    *,
    data_dir: Path = DEFAULT_DATA,
    freq: str = "1min",
) -> pd.Series:
    """Invocation counts for one function across the trace at eval frequency."""
    long = load_function_invocations(app, func, data_dir=data_dir)
    series = long.groupby("timestamp")["count"].sum()
    series.index = pd.DatetimeIndex(series.index, tz="UTC")
    minute_series = series.reindex(minute_index(), fill_value=0).astype(int)
    return resample_series(minute_series, freq)


def load_total_invocations_per_minute(data_dir: Path = DEFAULT_DATA) -> pd.Series:
    """Platform-wide invocation counts per minute."""
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("invocations_per_function*.csv")):
        day = int(re.search(r"\.d(\d+)\.csv$", path.name).group(1))
        minute_counts = pd.read_csv(path, usecols=lambda c: c.isdigit()).sum(axis=0)
        minute_counts.index = minute_counts.index.astype(int)
        timestamps = (
            TRACE_START
            + pd.Timedelta(days=day - 1)
            + pd.to_timedelta(minute_counts.index - 1, unit="m")
        )
        frames.append(
            pd.DataFrame({"timestamp": timestamps, "count": minute_counts.values})
        )
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return df.set_index("timestamp")["count"].astype(int)


def load_or_build_metadata(
    data_dir: Path,
    *,
    metadata_path: Path,
    min_invocations: int,
    rebuild: bool,
) -> pd.DataFrame:
    if metadata_path.exists() and not rebuild:
        print(f"Loading metadata from {metadata_path}")
        return pd.read_csv(metadata_path)

    print("Building function metadata (one CSV scan)...", flush=True)
    t0 = time.perf_counter()
    metadata = build_minute_bin_function_metadata(
        data_dir,
        minute_cols=MINUTE_COLS,
        min_invocations=min_invocations,
        trace_minutes=TRACE_DAYS * 1440,
        log=_log_phase,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {len(metadata):,} rows to {metadata_path} ({elapsed:.1f}s)")
    return metadata


def run_sample_holdout(
    eval_keys: list[tuple[str, str]],
    *,
    data_dir: Path,
    freq: str,
    season_days: int,
    include_m7: bool,
    include_ma1: bool,
    include_prophet: bool,
    include_auto_arima: bool,
    verbose: bool,
    preload_chunk_size: int,
    preload_engine: str,
) -> pd.DataFrame:
    if not eval_keys:
        return pd.DataFrame()

    print(f"Evaluating {len(eval_keys)} functions")
    forecasters = default_holdout_forecasters(
        freq=freq,
        season_days=season_days,
        include_seasonal_arima=False,
        include_ma1=include_ma1,
        include_prophet=include_prophet,
        include_auto_arima=include_auto_arima,
    )
    print(f"  models: {[f.name for f in forecasters]}", flush=True)

    _log_phase(f"preloading series at freq={freq!r} for {len(eval_keys)} functions")
    series_by_key = preload_function_minute_series(
        eval_keys,
        data_dir,
        freq=freq,
        chunk_size=preload_chunk_size,
        engine=preload_engine,
    )
    _log_phase(f"preloaded {len(series_by_key)} series")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, (app, func) in enumerate(eval_keys, start=1):
        series = series_by_key.get((app, func))
        if series is None:
            continue
        split = temporal_split(series)
        if split is None:
            continue

        for forecaster in forecasters:
            if verbose:
                print(
                    f"  {i}/{len(eval_keys)} {app[:8]}…/{func[:8]}… {forecaster.name}",
                    flush=True,
                )
            try:
                result = evaluate_series(
                    split.train,
                    split.test,
                    forecaster,
                    freq=freq,
                    season_days=season_days,
                    include_m7=include_m7,
                )
            except Exception as exc:
                print(f"skip {app[:8]}…/{func[:8]}… {forecaster.name}: {exc}")
                continue
            rows.append(
                {
                    "app": app,
                    "func": func,
                    "model": result["model"],
                    "freq": freq,
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "mase_m1": result["mase_m1"],
                    "rmsse_m1": result["rmsse_m1"],
                    **(
                        {
                            "mase_m7": result["mase_m7"],
                            "rmsse_m7": result["rmsse_m7"],
                        }
                        if include_m7
                        else {}
                    ),
                }
            )

        if i % 10 == 0 or i == len(eval_keys):
            elapsed = time.perf_counter() - t0
            print(f"  {i}/{len(eval_keys)} functions ({elapsed:.1f}s)")

    return pd.DataFrame(rows)


def run_aggregate(
    total_series: pd.Series,
    *,
    freq: str,
    season_days: int,
    include_m7: bool,
    include_ma1: bool,
    include_prophet: bool,
    include_auto_arima: bool,
    verbose: bool,
) -> pd.DataFrame:
    forecasters = default_holdout_forecasters(
        freq=freq,
        season_days=season_days,
        include_seasonal_arima=False,
        include_ma1=include_ma1,
        include_prophet=include_prophet,
        include_auto_arima=include_auto_arima,
    )
    split = temporal_split(total_series)
    if split is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for forecaster in forecasters:
        if verbose:
            print(f"  aggregate {forecaster.name}", flush=True)
        try:
            result = evaluate_series(
                split.train,
                split.test,
                forecaster,
                freq=freq,
                season_days=season_days,
                include_m7=include_m7,
            )
        except Exception as exc:
            print(f"skip {forecaster.name}: {exc}")
            continue
        rows.append(
            {
                "app": "all",
                "func": "all",
                "model": result["model"],
                "freq": freq,
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "mae": result["mae"],
                "rmse": result["rmse"],
                "mase_m1": result["mase_m1"],
                "rmsse_m1": result["rmsse_m1"],
                **(
                    {
                        "mase_m7": result["mase_m7"],
                        "rmsse_m7": result["rmsse_m7"],
                    }
                    if include_m7
                    else {}
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Holdout forecast evaluation for Azure Functions 2019 (az19.ipynb workflow).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Directory with invocations_per_function*.csv (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Cached function metadata CSV (default: <data-dir>/function_metadata.csv)",
    )
    parser.add_argument(
        "--freq",
        default="1min",
        help=(
            "Evaluation bucket size (pandas offset). Source CSVs are per-minute; "
            "coarser bins sum minute counts. Examples: 1min, 5min, 15min, 1h"
        ),
    )
    parser.add_argument(
        "--season-days",
        type=int,
        default=7,
        help="Season length in days for seasonal naive / MASE scaling",
    )
    parser.add_argument(
        "--min-invocations",
        type=int,
        default=50,
        help="Minimum total invocations for eligibility",
    )
    parser.add_argument(
        "--census-n",
        type=int,
        default=30,
        help="Always include top N functions by invocation volume",
    )
    parser.add_argument(
        "--per-cell-k",
        type=int,
        default=5,
        help="Sample up to K functions per (volume quintile, trigger) cell",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling",
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=None,
        help="Optional cap on evaluation sample size (after sampling)",
    )
    parser.add_argument(
        "--rebuild-metadata",
        action="store_true",
        help="Rebuild function metadata even if cache exists",
    )
    parser.add_argument(
        "--no-ma1",
        action="store_true",
        help="Exclude MA(1) / ARIMA(0,0,1) forecaster from evaluation",
    )
    parser.add_argument(
        "--no-m7",
        action="store_true",
        help="Skip MASE/RMSSE with 7-day seasonal-naive scaling (mase_m7, rmsse_m7)",
    )
    parser.add_argument(
        "--no-prophet",
        action="store_true",
        help="Exclude Prophet (CmdStan); often needed on headless Linux hosts",
    )
    parser.add_argument(
        "--no-auto-arima",
        action="store_true",
        help="Exclude pmdarima auto-ARIMA search",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each (function, model) fit (helps locate native crashes)",
    )
    parser.add_argument(
        "--preload-engine",
        choices=("csv", "pandas"),
        default="csv",
        help="How to read daily CSVs during preload (csv avoids pd.read_csv segfaults)",
    )
    parser.add_argument(
        "--preload-chunk-size",
        type=int,
        default=2000,
        help="Rows per chunk when preloading daily CSVs (lower = less RAM)",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip platform-wide aggregate holdout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write result CSVs",
    )
    return parser.parse_args()


def main() -> None:
    global _T0
    _T0 = time.perf_counter()
    _log_phase("starting")

    args = parse_args()
    _validate_freq(args.freq)
    metadata_path = args.metadata_path or (args.data_dir / "function_metadata.csv")
    include_m7 = not args.no_m7
    include_ma1 = not args.no_ma1
    include_prophet = not args.no_prophet
    include_auto_arima = not args.no_auto_arima

    _log_phase("loading metadata")
    function_metadata = load_or_build_metadata(
        args.data_dir,
        metadata_path=metadata_path,
        min_invocations=args.min_invocations,
        rebuild=args.rebuild_metadata,
    )
    eval_keys, eval_manifest = sample_representative_functions(
        function_metadata,
        census_n=args.census_n,
        per_cell_k=args.per_cell_k,
        seed=args.seed,
    )
    if args.max_functions is not None:
        eval_keys = eval_keys[: args.max_functions]
        eval_manifest = eval_manifest.head(args.max_functions)

    print(f"Eligible functions: {len(function_metadata):,}")
    print(f"Evaluation sample: {len(eval_keys):,}")
    print(f"Eval frequency: {args.freq} ({len(trace_index(args.freq)):,} buckets / trace)")
    print(eval_manifest["sample_tier"].value_counts().to_string())

    _log_phase("starting per-function holdout")
    holdout_results = run_sample_holdout(
        eval_keys,
        data_dir=args.data_dir,
        freq=args.freq,
        season_days=args.season_days,
        include_m7=include_m7,
        include_ma1=include_ma1,
        include_prophet=include_prophet,
        include_auto_arima=include_auto_arima,
        verbose=args.verbose,
        preload_chunk_size=args.preload_chunk_size,
        preload_engine=args.preload_engine,
    )

    print("\n## Per-function holdout — macro (each function counts equally)")
    macro_sort = "mase_m7_mean" if include_m7 else "mase_m1_mean"
    macro_summary = summarize_results(holdout_results, sort_by=macro_sort)
    print(macro_summary.to_string(index=False))

    print("\n## Per-function holdout — micro (invocation-weighted)")
    micro_sort = "mase_m7_weighted" if include_m7 else "mase_m1_weighted"
    micro_summary = summarize_results_weighted(
        holdout_results,
        function_metadata,
        sort_by=micro_sort,
    )
    print(micro_summary.to_string(index=False))

    aggregate_results = pd.DataFrame()
    if not args.skip_aggregate:
        _log_phase(f"loading platform-wide series at freq={args.freq!r}")
        print(f"\nLoading platform-wide series at {args.freq}...")
        total_series = resample_series(
            load_total_invocations_per_minute(args.data_dir),
            args.freq,
        )
        aggregate_results = run_aggregate(
            total_series,
            freq=args.freq,
            season_days=args.season_days,
            include_m7=include_m7,
            include_ma1=include_ma1,
            include_prophet=include_prophet,
            include_auto_arima=include_auto_arima,
            verbose=args.verbose,
        )
        print("\n## All-invocation aggregate holdout")
        agg_sort = "mase_m7" if include_m7 else "mase_m1"
        print(
            summarize_aggregate_results(aggregate_results, sort_by=agg_sort).to_string(
                index=False
            )
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        function_metadata.to_csv(args.output_dir / "function_metadata.csv", index=False)
        eval_manifest.to_csv(args.output_dir / "eval_manifest.csv", index=False)
        holdout_results.to_csv(args.output_dir / "holdout_per_function.csv", index=False)
        macro_summary.to_csv(args.output_dir / "holdout_macro_summary.csv", index=False)
        micro_summary.to_csv(args.output_dir / "holdout_micro_summary.csv", index=False)
        if not aggregate_results.empty:
            aggregate_results.to_csv(
                args.output_dir / "holdout_aggregate.csv", index=False
            )
        print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
