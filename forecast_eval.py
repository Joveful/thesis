"""FaaS invocation-rate forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ForecasterT = TypeVar("ForecasterT", bound="Forecaster")

_FORECASTER_REGISTRY: dict[str, type] = {}


def register_forecaster(cls: type[ForecasterT]) -> type[ForecasterT]:
    """Register a forecaster class for discovery by name."""
    _FORECASTER_REGISTRY[cls.__name__] = cls
    return cls


def list_forecasters() -> list[str]:
    return sorted(_FORECASTER_REGISTRY.keys())


@dataclass
class EvalSplit:
    train: pd.Series
    test: pd.Series


def build_invocation_series(
    df: pd.DataFrame,
    freq: str = "5min",
    origin: str = "2021-01-31",
    min_invocations: int = 50,
    timestamp_col: str = "start_timestamp",
) -> dict[tuple[str, str], pd.Series]:
    """
    Convert raw invocation records into per-(app, func) count time series.

    Values are invocation counts per resample bucket (zero-filled gaps).
    """
    work = df.copy()
    if timestamp_col not in work.columns:
        if "end_timestamp" in work.columns and "duration" in work.columns:
            work[timestamp_col] = work["end_timestamp"] - work["duration"]
        else:
            raise ValueError(
                f"Missing {timestamp_col!r}; need start_timestamp or end_timestamp+duration"
            )

    work["datetime"] = pd.to_datetime(work[timestamp_col], origin=origin, unit="s")
    work = work.sort_values("datetime")

    series_by_key: dict[tuple[str, str], pd.Series] = {}
    for (app, func), group in work.groupby(["app", "func"], sort=False):
        if len(group) < min_invocations:
            continue
        ts = (
            group.set_index("datetime")
            .resample(freq)
            .size()
            .asfreq(freq, fill_value=0)
            .sort_index()
        )
        ts.name = "count"
        series_by_key[(app, func)] = ts

    return series_by_key


def temporal_split(
    series: pd.Series,
    test_fraction: float = 0.2,
    min_train_points: int = 48,
    min_test_points: int = 12,
) -> EvalSplit | None:
    """Hold out the trailing fraction of points as test (no shuffle)."""
    n = len(series)
    if n < min_train_points + min_test_points:
        return None

    test_size = max(min_test_points, int(np.ceil(n * test_fraction)))
    train_size = n - test_size
    if train_size < min_train_points or test_size < min_test_points:
        return None

    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    return EvalSplit(train=train, test=test)


class Forecaster(Protocol):
    @property
    def name(self) -> str: ...

    def fit(self, y: pd.Series) -> None: ...

    def predict(self, horizon: int) -> np.ndarray: ...


def _default_season_length(freq: str) -> int:
    """Buckets per day for common pandas offset strings."""
    td = pd.Timedelta(freq)
    if td <= pd.Timedelta(0):
        return 24
    return max(1, int(pd.Timedelta("1D") / td))


@register_forecaster
class NaiveForecaster:
    def __init__(self) -> None:
        self._last: float = 0.0

    @property
    def name(self) -> str:
        return "naive"

    def fit(self, y: pd.Series) -> None:
        self._last = float(y.iloc[-1]) if len(y) else 0.0

    def predict(self, horizon: int) -> np.ndarray:
        return np.clip(np.full(horizon, self._last), 0, None)


@register_forecaster
class SeasonalNaiveForecaster:
    def __init__(self, season_length: int | None = None, freq: str = "5min") -> None:
        self.season_length = season_length or _default_season_length(freq)
        self._season: np.ndarray = np.array([])

    @property
    def name(self) -> str:
        return "seasonal_naive"

    def fit(self, y: pd.Series) -> None:
        sl = min(self.season_length, len(y))
        self._season = y.iloc[-sl:].to_numpy(dtype=float) if sl else np.array([0.0])

    def predict(self, horizon: int) -> np.ndarray:
        if len(self._season) == 0:
            return np.zeros(horizon)
        reps = int(np.ceil(horizon / len(self._season)))
        tiled = np.tile(self._season, reps)[:horizon]
        return np.clip(tiled, 0, None)


@register_forecaster
class MovingAverageForecaster:
    def __init__(self, window: int = 12) -> None:
        self.window = window
        self._mean: float = 0.0

    @property
    def name(self) -> str:
        return f"moving_average_w{self.window}"

    def fit(self, y: pd.Series) -> None:
        tail = y.iloc[-self.window :] if len(y) else y
        self._mean = float(tail.mean()) if len(tail) else 0.0

    def predict(self, horizon: int) -> np.ndarray:
        return np.clip(np.full(horizon, self._mean), 0, None)


@register_forecaster
class ARIMAForecaster:
    """ARIMA on invocation counts; requires statsmodels."""

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)) -> None:
        self.order = order
        self._fitted = None

    @property
    def name(self) -> str:
        p, d, q = self.order
        return f"arima_{p}_{d}_{q}"

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        if len(y) < max(20, sum(self.order) + 5):
            raise ValueError(f"train series too short for ARIMA{self.order}")
        self._fitted = ARIMA(y.astype(float), order=self.order).fit()

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._fitted.forecast(steps=horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


@register_forecaster
class ProphetForecaster:
    """Prophet on invocation counts; requires prophet."""

    def __init__(
        self,
        freq: str = "5min",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        **prophet_kwargs,
    ) -> None:
        self.freq = freq
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.prophet_kwargs = prophet_kwargs
        self._model = None

    @property
    def name(self) -> str:
        return "prophet"

    def fit(self, y: pd.Series) -> None:
        from prophet import Prophet

        if len(y) < 20:
            raise ValueError("train series too short for Prophet")

        dfp = pd.DataFrame({"ds": y.index, "y": y.astype(float).values})
        self._model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            **self.prophet_kwargs,
        )
        self._model.fit(dfp)

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")

        future = self._model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self._model.predict(future)
        yhat = forecast["yhat"].iloc[-horizon:].to_numpy(dtype=float)
        return np.clip(yhat, 0, None)


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def evaluate_series(
    y_train: pd.Series,
    y_test: pd.Series,
    forecaster: Forecaster,
) -> dict:
    model = forecaster
    model.fit(y_train)
    y_pred = model.predict(len(y_test))
    return {
        "model": model.name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "y_pred": y_pred,
    }


def evaluate_trace(
    df: pd.DataFrame,
    forecasters: list[Forecaster],
    freq: str = "5min",
    origin: str = "2021-01-31",
    min_invocations: int = 50,
    test_fraction: float = 0.2,
    min_train_points: int = 48,
    min_test_points: int = 12,
    keys: list[tuple[str, str]] | None = None,
    max_functions: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate forecasters on trace data; one result row per (app, func, model).
    """
    all_series = build_invocation_series(
        df,
        freq=freq,
        origin=origin,
        min_invocations=min_invocations,
    )

    if keys is not None:
        selected = {k: all_series[k] for k in keys if k in all_series}
    else:
        selected = all_series

    if max_functions is not None and len(selected) > max_functions:
        top = sorted(
            selected.items(),
            key=lambda kv: kv[1].sum(),
            reverse=True,
        )[:max_functions]
        selected = dict(top)

    rows: list[dict] = []
    for (app, func), series in selected.items():
        split = temporal_split(
            series,
            test_fraction=test_fraction,
            min_train_points=min_train_points,
            min_test_points=min_test_points,
        )
        if split is None:
            continue

        for forecaster in forecasters:
            try:
                result = evaluate_series(split.train, split.test, forecaster)
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
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                }
            )

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and median MAE/RMSE per model."""
    if results.empty:
        return results
    agg = results.groupby("model", as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_median=("mae", "median"),
        rmse_mean=("rmse", "mean"),
        rmse_median=("rmse", "median"),
        n_functions=("func", "count"),
    )
    return agg.sort_values("mae_mean")


def rank_models_per_function(results: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Rank models within each function by metric (1 = best)."""
    if results.empty:
        return results
    out = results.copy()
    out["rank"] = out.groupby(["app", "func"])[metric].rank(method="min")
    return out.sort_values(["app", "func", "rank"])


def plot_holdout(
    split: EvalSplit,
    y_pred: np.ndarray,
    title: str = "Invocation forecast holdout",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot train history, test actuals, and holdout predictions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(split.train.index, split.train.values, label="train", color="C0")
    ax.plot(split.test.index, split.test.values, label="test (actual)", color="C1")
    ax.plot(split.test.index, y_pred, label="forecast", color="C2", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("invocations per bucket")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def default_baselines(freq: str = "5min") -> list[Forecaster]:
    """Standard baseline forecasters for experiments."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(freq=freq),
        MovingAverageForecaster(window=12),
    ]


def default_forecasters(
    freq: str = "5min",
    *,
    include_arima: bool = True,
    include_prophet: bool = True,
    arima_order: tuple[int, int, int] = (2, 1, 2),
) -> list[Forecaster]:
    """Baselines plus ARIMA and Prophet (new instance per model type)."""
    models: list[Forecaster] = default_baselines(freq)
    if include_arima:
        models.append(ARIMAForecaster(order=arima_order))
    if include_prophet:
        models.append(ProphetForecaster(freq=freq))
    return models


def summarize_invocation_count_bins(
    df: pd.DataFrame,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split (app, func) pairs into equal-frequency bins by total invocation count.

    Returns (per_function, bin_summary). bin_summary includes counts, invocation
    ranges, duration stats, and each bin's share of trace invocations.
    """
    per_function = (
        df.groupby(["app", "func"], as_index=False)
        .agg(
            invocation_count=("func", "size"),
            mean_duration=("duration", "mean"),
            median_duration=("duration", "median"),
            total_duration=("duration", "sum"),
        )
        .sort_values("invocation_count")
    )

    per_function["bin"] = pd.qcut(
        per_function["invocation_count"],
        q=n_bins,
        labels=[f"Q{i}" for i in range(1, n_bins + 1)],
        duplicates="drop",
    )

    total_invocations = int(per_function["invocation_count"].sum())
    bin_summary = (
        per_function.groupby("bin", observed=True)
        .agg(
            n_functions=("func", "count"),
            inv_count_min=("invocation_count", "min"),
            inv_count_max=("invocation_count", "max"),
            inv_count_mean=("invocation_count", "mean"),
            inv_count_median=("invocation_count", "median"),
            total_invocations=("invocation_count", "sum"),
            mean_duration=("mean_duration", "mean"),
            median_duration=("median_duration", "median"),
            mean_total_duration=("total_duration", "mean"),
        )
        .reset_index()
    )
    bin_summary["pct_invocations"] = (
        100.0 * bin_summary["total_invocations"] / total_invocations
    )
    bin_summary["pct_functions"] = (
        100.0 * bin_summary["n_functions"] / len(per_function)
    )

    return per_function, bin_summary


def top_function_keys(
    df: pd.DataFrame,
    n: int = 5,
    min_invocations: int = 50,
) -> list[tuple[str, str]]:
    """Return (app, func) keys with the most invocations."""
    counts = df.groupby(["app", "func"]).size()
    counts = counts[counts >= min_invocations]
    top = counts.nlargest(n)
    return list(top.index)
