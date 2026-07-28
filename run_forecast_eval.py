#!/usr/bin/env python3
"""Run forecast evaluation from az21.ipynb using forecast_eval.py."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from forecast_eval import (
    SeasonalNaiveForecaster,
    _default_season_length,
    build_invocation_series,
    default_forecasters,
    default_holdout_forecasters,
    evaluate_series,
    evaluate_total_invocations,
    evaluate_trace,
    evaluate_trace_cv,
    plot_holdout,
    summarize_aggregate_results,
    summarize_results,
    temporal_split,
    top_function_keys,
)

DEFAULT_DATA = Path("data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")


def load_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_timestamp"] = df["end_timestamp"] - df["duration"]
    return df


def run_holdout(
    df: pd.DataFrame,
    *,
    freq: str,
    season_days: int,
    top_n: int,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    top_keys = top_function_keys(df, n=top_n)
    print(f"Evaluating {len(top_keys)} functions")

    forecasters = default_holdout_forecasters(
        freq=freq, season_days=season_days, include_seasonal_arima=False
    )
    results = evaluate_trace(
        df,
        forecasters,
        freq=freq,
        keys=top_keys,
        season_days=season_days,
    )
    print("\n## Per-function holdout (aggregated by model)")
    print(summarize_results(results).to_string(index=False))
    return results, top_keys


def run_aggregate(
    df: pd.DataFrame,
    *,
    freq: str,
    season_days: int,
) -> pd.DataFrame:
    forecasters = default_holdout_forecasters(
        freq=freq, season_days=season_days, include_seasonal_arima=False
    )
    results = evaluate_total_invocations(
        df,
        forecasters,
        freq=freq,
        season_days=season_days,
    )
    print("\n## All-invocation aggregate holdout")
    print(summarize_aggregate_results(results).to_string(index=False))
    return results


def run_cv(
    df: pd.DataFrame,
    *,
    freq: str,
    top_keys: list[tuple[str, str]],
    h: int,
    max_folds: int,
) -> pd.DataFrame:
    forecasters = default_forecasters(freq=freq, include_prophet=False)
    t0 = time.perf_counter()
    results = evaluate_trace_cv(
        df,
        forecasters,
        h=h,
        freq=freq,
        keys=top_keys,
        max_folds=max_folds,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n## h-step-ahead cross-validation (h={h}, max_folds={max_folds})")
    print(f"CV finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(summarize_results(results).to_string(index=False))
    return results


def plot_busiest_holdout(
    df: pd.DataFrame,
    top_keys: list[tuple[str, str]],
    *,
    freq: str,
    season_days: int,
) -> None:
    if not top_keys:
        return

    app, func = top_keys[0]
    series = build_invocation_series(df, freq=freq)[app, func]
    split = temporal_split(series)
    if split is None:
        print("Skipping holdout plot: series too short for temporal split")
        return

    season_length = _default_season_length(freq, days=season_days)
    model = SeasonalNaiveForecaster(freq=freq, season_length=season_length)
    ev = evaluate_series(
        split.train, split.test, model, freq=freq, season_days=season_days
    )
    plot_holdout(split, ev["y_pred"], title=f"{app[:8]}… / {func[:8]}… ({ev['model']})")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Holdout and CV forecast evaluation (az21.ipynb workflow).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Azure trace CSV (default: {DEFAULT_DATA})",
    )
    parser.add_argument("--freq", default="1min", help="Resample frequency")
    parser.add_argument(
        "--season-days",
        type=int,
        default=7,
        help="Season length in days for seasonal naive / MASE scaling",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top functions by invocation volume",
    )
    parser.add_argument("--cv-h", type=int, default=10, help="CV forecast horizon")
    parser.add_argument(
        "--cv-max-folds",
        type=int,
        default=10,
        help="Max evenly spaced CV folds",
    )
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot holdout for the busiest function",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write result CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.style.use("ggplot")

    print(f"Loading {args.data}")
    df = load_trace(args.data)

    holdout_results, top_keys = run_holdout(
        df,
        freq=args.freq,
        season_days=args.season_days,
        top_n=args.top_n,
    )
    aggregate_results = run_aggregate(
        df,
        freq=args.freq,
        season_days=args.season_days,
    )

    cv_results: pd.DataFrame | None = None
    if not args.skip_cv:
        cv_results = run_cv(
            df,
            freq=args.freq,
            top_keys=top_keys,
            h=args.cv_h,
            max_folds=args.cv_max_folds,
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        holdout_results.to_csv(args.output_dir / "holdout_per_function.csv", index=False)
        aggregate_results.to_csv(args.output_dir / "holdout_aggregate.csv", index=False)
        if cv_results is not None:
            cv_results.to_csv(args.output_dir / "cv_results.csv", index=False)
        print(f"\nWrote results to {args.output_dir}")

    if args.plot:
        plot_busiest_holdout(
            df,
            top_keys,
            freq=args.freq,
            season_days=args.season_days,
        )


if __name__ == "__main__":
    main()
