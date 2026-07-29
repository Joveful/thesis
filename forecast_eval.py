"""FaaS invocation-rate forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def build_total_invocation_series(
    df: pd.DataFrame,
    freq: str = "5min",
    origin: str = "2021-01-31",
    timestamp_col: str = "start_timestamp",
) -> pd.Series:
    """Aggregate every invocation into one platform-wide count time series."""
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
    ts = (
        work.set_index("datetime")
        .resample(freq)
        .size()
        .asfreq(freq, fill_value=0)
        .sort_index()
    )
    ts.name = "count"
    return ts


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


def _default_season_length(freq: str, days: int = 1) -> int:
    """Buckets per `days`-day season for common pandas offset strings."""
    td = pd.Timedelta(freq)
    if td <= pd.Timedelta(0):
        return 24 * days
    return max(1, int(pd.Timedelta(f"{days}D") / td))


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
class MA1Forecaster:
    """Moving-average model of order 1 (MA(1) / ARIMA(0,0,1)); requires statsmodels."""

    def __init__(self) -> None:
        self._fitted = None

    @property
    def name(self) -> str:
        return "ma1"

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        if len(y) < 10:
            raise ValueError("train series too short for MA(1)")
        self._fitted = ARIMA(y.astype(float), order=(0, 0, 1)).fit()

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._fitted.forecast(steps=horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


@register_forecaster
class AR1Forecaster:
    """Autoregressive model of order 1 (AR(1)); requires statsmodels."""

    def __init__(self) -> None:
        self._fitted = None

    @property
    def name(self) -> str:
        return "ar1"

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.ar_model import AutoReg

        if len(y) < 10:
            raise ValueError("train series too short for AR(1)")
        self._fitted = AutoReg(y.astype(float), lags=1, old_names=False).fit()

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._fitted.forecast(steps=horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


@register_forecaster
class SESForecaster:
    """Simple exponential smoothing (level only); requires statsmodels."""

    def __init__(self) -> None:
        self._fitted = None

    @property
    def name(self) -> str:
        return "ses"

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if len(y) < 10:
            raise ValueError("train series too short for SES")
        self._fitted = ExponentialSmoothing(
            y.astype(float),
            trend=None,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._fitted.forecast(horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


@register_forecaster
class DESForecaster:
    """Double exponential smoothing (Holt linear trend); requires statsmodels."""

    def __init__(self) -> None:
        self._fitted = None

    @property
    def name(self) -> str:
        return "des"

    def fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if len(y) < 10:
            raise ValueError("train series too short for DES")
        self._fitted = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._fitted.forecast(horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


@register_forecaster
class ARIMAForecaster:
    """Fixed-order ARIMA on invocation counts; requires statsmodels."""

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
class AutoARIMAForecaster:
    """Auto-selected ARIMA (p,d,q) via pmdarima; for holdout evaluation."""

    def __init__(
        self,
        max_p: int = 4,
        max_d: int = 2,
        max_q: int = 4,
        seasonal: bool = False,
        **auto_arima_kwargs,
    ) -> None:
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.auto_arima_kwargs = auto_arima_kwargs
        self._model = None
        self._order: tuple[int, int, int] | None = None

    @property
    def name(self) -> str:
        if self._order is None:
            return "auto_arima"
        p, d, q = self._order
        return f"auto_arima_{p}_{d}_{q}"

    def fit(self, y: pd.Series) -> None:
        from pmdarima import auto_arima

        if len(y) < 20:
            raise ValueError("train series too short for auto_arima")

        self._model = auto_arima(
            y.astype(float).values,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            seasonal=self.seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            **self.auto_arima_kwargs,
        )
        self._order = tuple(self._model.order)

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        fc = self._model.predict(n_periods=horizon)
        return np.clip(np.asarray(fc, dtype=float), 0, None)


def _steps_per_resample(orig_freq: str, resample_freq: str) -> int:
    """Buckets of orig_freq in one resample_freq bucket (e.g. 60 for 1min -> 1h)."""
    return max(1, int(pd.Timedelta(resample_freq) / pd.Timedelta(orig_freq)))


def _expand_resampled_forecast(
    fc_resampled: np.ndarray, steps_per_bucket: int, horizon: int
) -> np.ndarray:
    """Split each coarse-bucket forecast evenly across fine-bucket steps."""
    per_step = np.repeat(fc_resampled / steps_per_bucket, steps_per_bucket)
    return per_step[:horizon]


@register_forecaster
class AutoSARIMAForecaster:
    """
    Auto-selected seasonal ARIMA via pmdarima (holdout evaluation).

    Fitted on hourly counts with seasonal period m = season_days * 24 (default
    one day); hourly forecasts are expanded back to the original bucket frequency.
    """

    def __init__(
        self,
        freq: str = "5min",
        season_days: int = 1,
        fit_freq: str = "1h",
        max_p: int = 2,
        max_d: int = 1,
        max_q: int = 2,
        max_P: int = 1,
        max_D: int = 1,
        max_Q: int = 1,
        min_hourly_points: int | None = None,
        **auto_arima_kwargs,
    ) -> None:
        self.freq = freq
        self.season_days = season_days
        self.fit_freq = fit_freq
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.auto_arima_kwargs = auto_arima_kwargs
        self._seasonal_period = season_days * 24
        self._steps_per_fit_bucket = _steps_per_resample(freq, fit_freq)
        self._min_hourly_points = min_hourly_points or (self._seasonal_period + 20)
        self._model = None
        self._order: tuple[int, int, int] | None = None
        self._seasonal_order: tuple[int, int, int, int] | None = None

    @property
    def name(self) -> str:
        if self._order is None:
            return "sarima"
        p, d, q = self._order
        P, D, Q, m = self._seasonal_order or (0, 0, 0, self._seasonal_period)
        return f"sarima_{p}_{d}_{q}_{P}_{D}_{Q}_{m}"

    def _hourly_train(self, y: pd.Series) -> pd.Series:
        return y.resample(self.fit_freq).sum()

    def fit(self, y: pd.Series) -> None:
        from pmdarima import auto_arima

        y_h = self._hourly_train(y.astype(float))
        if len(y_h) < self._min_hourly_points:
            raise ValueError(
                f"train series too short for sarima "
                f"(need {self._min_hourly_points} hourly points, got {len(y_h)})"
            )

        self._model = auto_arima(
            y_h.values,
            seasonal=True,
            m=self._seasonal_period,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            max_P=self.max_P,
            max_D=self.max_D,
            max_Q=self.max_Q,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            approximation=True,
            **self.auto_arima_kwargs,
        )
        self._order = tuple(self._model.order)
        self._seasonal_order = tuple(self._model.seasonal_order)

    def predict(self, horizon: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")

        steps = int(np.ceil(horizon / self._steps_per_fit_bucket))
        fc_h = self._model.predict(n_periods=steps)
        fc = _expand_resampled_forecast(
            np.asarray(fc_h, dtype=float),
            self._steps_per_fit_bucket,
            horizon,
        )
        return np.clip(fc, 0, None)


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

        ds = y.index
        if isinstance(ds, pd.DatetimeIndex) and ds.tz is not None:
            ds = ds.tz_localize(None)
        dfp = pd.DataFrame({"ds": ds, "y": y.astype(float).values})
        self._model = Prophet(
            growth="flat",
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


def _mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def _rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _effective_season_length(y: pd.Series, season_length: int) -> int:
    """Use seasonal scaling only when the train window is long enough."""
    n = len(y)
    if n <= 1:
        return 1
    if n > season_length:
        return season_length
    return 1


def _safe_nanmean(values: list[float]) -> float:
    """Mean ignoring NaNs; no warning when every value is NaN."""
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))


def _in_sample_naive_scale_abs(y: pd.Series, season_length: int = 1) -> float:
    """Mean absolute in-sample error of seasonal naive (MASE denominator)."""
    m = _effective_season_length(y, season_length)
    yt = np.asarray(y, dtype=float)
    if len(yt) <= m:
        return float("nan")
    diffs = np.abs(yt[m:] - yt[:-m])
    scale = float(np.mean(diffs))
    return scale if scale > 0 else float("nan")


def _in_sample_naive_scale_sq(y: pd.Series, season_length: int = 1) -> float:
    """Mean squared in-sample error of seasonal naive (RMSSE denominator)."""
    m = _effective_season_length(y, season_length)
    yt = np.asarray(y, dtype=float)
    if len(yt) <= m:
        return float("nan")
    diffs = yt[m:] - yt[:-m]
    scale = float(np.mean(diffs**2))
    return scale if scale > 0 else float("nan")


def mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: pd.Series,
    season_length: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scaled by in-sample MAE of a seasonal naive forecast on y_train
    (Hyndman & Koehler, 2006). Values < 1 beat the naive benchmark.
    """
    scale = _in_sample_naive_scale_abs(y_train, season_length)
    if np.isnan(scale):
        return float("nan")
    return _mae(y_true, y_pred) / scale


def rmsse(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: pd.Series,
    season_length: int = 1,
) -> float:
    """
    Root Mean Squared Scaled Error.

    Scaled by in-sample RMSE of a seasonal naive forecast on y_train
    (M4/M5 convention). Values < 1 beat the naive benchmark.
    """
    scale = _in_sample_naive_scale_sq(y_train, season_length)
    if np.isnan(scale):
        return float("nan")
    return _rmse(y_true, y_pred) / np.sqrt(scale)


def _season_length_m7(freq: str, season_days: int = 7) -> int:
    """Season length in buckets for m=7-day seasonal naive scaling."""
    return _default_season_length(freq, days=season_days)


def evaluate_series(
    y_train: pd.Series,
    y_test: pd.Series,
    forecaster: Forecaster,
    *,
    freq: str = "5min",
    season_days: int = 7,
    include_m7: bool = True,
) -> dict:
    model = forecaster
    model.fit(y_train)
    y_pred = model.predict(len(y_test))
    result = {
        "model": model.name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mae": _mae(y_test, y_pred),
        "rmse": _rmse(y_test, y_pred),
        "mase_m1": mase(y_test, y_pred, y_train, 1),
        "rmsse_m1": rmsse(y_test, y_pred, y_train, 1),
        "y_pred": y_pred,
    }
    if include_m7:
        m7 = _season_length_m7(freq, season_days)
        result["mase_m7"] = mase(y_test, y_pred, y_train, m7)
        result["rmsse_m7"] = rmsse(y_test, y_pred, y_train, m7)
    return result


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
    season_days: int = 1,
    include_m7: bool = True,
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

    return pd.DataFrame(rows)


def evaluate_total_invocations(
    df: pd.DataFrame,
    forecasters: list[Forecaster],
    freq: str = "5min",
    origin: str = "2021-01-31",
    test_fraction: float = 0.2,
    min_train_points: int = 48,
    min_test_points: int = 12,
    season_days: int = 1,
    include_m7: bool = True,
) -> pd.DataFrame:
    """
    Holdout evaluation on one series aggregating all function invocations.
    """
    series = build_total_invocation_series(df, freq=freq, origin=origin)
    split = temporal_split(
        series,
        test_fraction=test_fraction,
        min_train_points=min_train_points,
        min_test_points=min_test_points,
    )
    if split is None:
        return pd.DataFrame()

    rows: list[dict] = []
    for forecaster in forecasters:
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


def summarize_aggregate_results(
    results: pd.DataFrame,
    *,
    group_auto_arima: bool = True,
    sort_by: str = "mase_m7",
) -> pd.DataFrame:
    """Summary for all-invocation aggregate eval (MAE/RMSE + MASE/RMSSE at m=1 and m=7)."""
    if results.empty:
        return results
    work = results.copy()
    if group_auto_arima:
        work["model"] = work["model"].map(_model_group)
    metrics = [
        c
        for c in (
            "mae",
            "rmse",
            "mase_m1",
            "rmsse_m1",
            "mase_m7",
            "rmsse_m7",
            "mase",
            "rmsse",
        )
        if c in work.columns
    ]
    out = work.groupby("model", as_index=False)[metrics].mean()
    if sort_by in out.columns:
        return out.sort_values(sort_by)
    return out


def cv_origins(
    n: int,
    h: int,
    min_train: int,
    step: int | None = None,
    max_folds: int | None = None,
) -> list[int]:
    """
    Rolling-origin cut points for h-step-ahead CV.

    Each origin t trains on y[:t] and evaluates on y[t:t+h].
    When max_folds is set, origins are subsampled evenly across the series.
    """
    if n < min_train + h:
        return []
    step = h if step is None else step
    origins = list(range(min_train, n - h + 1, step))
    if max_folds is not None and len(origins) > max_folds:
        idx = np.linspace(0, len(origins) - 1, max_folds, dtype=int)
        origins = [origins[i] for i in np.unique(idx)]
    return origins


def evaluate_series_cv(
    y: pd.Series,
    forecaster: Forecaster,
    h: int,
    min_train: int,
    step: int | None = None,
    season_length: int = 1,
    max_folds: int | None = None,
) -> dict:
    """h-step-ahead rolling-origin CV for one series and forecaster."""
    origins = cv_origins(len(y), h, min_train, step, max_folds)
    if not origins:
        raise ValueError("series too short for CV")

    mases: list[float] = []
    rmsses: list[float] = []
    for origin in origins:
        train = y.iloc[:origin]
        test = y.iloc[origin : origin + h]
        forecaster.fit(train)
        y_pred = forecaster.predict(h)
        mases.append(mase(test, y_pred, train, season_length))
        rmsses.append(rmsse(test, y_pred, train, season_length))

    return {
        "model": forecaster.name,
        "h": h,
        "n_folds": len(origins),
        "n_valid_folds": int(np.sum(np.isfinite(mases))),
        "mase": _safe_nanmean(mases),
        "rmsse": _safe_nanmean(rmsses),
    }


def evaluate_trace_cv(
    df: pd.DataFrame,
    forecasters: list[Forecaster],
    h: int = 10,
    freq: str = "5min",
    origin: str = "2021-01-31",
    min_invocations: int = 50,
    min_train: int = 48,
    step: int | None = None,
    max_folds: int = 10,
    keys: list[tuple[str, str]] | None = None,
    max_functions: int | None = None,
) -> pd.DataFrame:
    """
    h-step-ahead rolling-origin CV on trace data.

    One result row per (app, func, model) with mean MASE/RMSSE across folds.
    Default max_folds=10 subsamples origins evenly across the trace.
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

    step = h if step is None else step
    season_length = _default_season_length(freq)
    rows: list[dict] = []

    for (app, func), series in selected.items():
        if len(series) < min_train + h:
            continue

        for forecaster in forecasters:
            try:
                result = evaluate_series_cv(
                    series,
                    forecaster,
                    h=h,
                    min_train=min_train,
                    step=step,
                    season_length=season_length,
                    max_folds=max_folds,
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
                    "h": h,
                    "step": step,
                    "n_folds": result["n_folds"],
                    "n_valid_folds": result["n_valid_folds"],
                    "mase": result["mase"],
                    "rmsse": result["rmsse"],
                }
            )

    return pd.DataFrame(rows)


def _model_group(model: str) -> str:
    """Collapse auto-selected ARIMA/SARIMA parameter variants to group labels."""
    if model == "auto_arima" or model.startswith("auto_arima_"):
        return "auto_arima"
    if model == "sarima" or model.startswith("sarima_"):
        return "sarima"
    if model == "seasonal_arima" or model.startswith("seasonal_arima_"):
        return "sarima"
    return model


def _scaled_metric_agg_kwargs(col: str) -> dict:
    """Aggregation kwargs for one scaled metric column (mean, median, std)."""
    return {
        f"{col}_mean": (col, "mean"),
        f"{col}_median": (col, "median"),
        f"{col}_std": (col, "std"),
    }


def summarize_results(
    results: pd.DataFrame,
    *,
    group_auto_arima: bool = True,
    sort_by: str = "mase_m7_mean",
) -> pd.DataFrame:
    """Mean, median, and std of MASE/RMSSE per model (m=1 and m=7-day scaling)."""
    if results.empty:
        return results
    work = results.copy()
    if group_auto_arima:
        work["model"] = work["model"].map(_model_group)
    agg_kwargs: dict = {}
    if "mase_m1" in work.columns:
        for col in ("mase_m1", "rmsse_m1", "mase_m7", "rmsse_m7"):
            if col in work.columns:
                agg_kwargs.update(_scaled_metric_agg_kwargs(col))
    elif "mase" in work.columns:
        agg_kwargs.update(_scaled_metric_agg_kwargs("mase"))
        agg_kwargs.update(_scaled_metric_agg_kwargs("rmsse"))
    agg_kwargs["n_functions"] = ("func", "count")
    if "n_folds" in work.columns:
        agg_kwargs["n_folds_mean"] = ("n_folds", "mean")
    if "n_valid_folds" in work.columns:
        agg_kwargs["n_valid_folds_mean"] = ("n_valid_folds", "mean")
    agg = work.groupby("model", as_index=False).agg(**agg_kwargs)
    if sort_by in agg.columns:
        return agg.sort_values(sort_by)
    elif "mase_m7_mean" in agg.columns:
        return agg.sort_values("mase_m7_mean")
    elif "mase_mean" in agg.columns:
        return agg.sort_values("mase_mean")
    return agg


def rank_models_per_function(results: pd.DataFrame, metric: str = "mase_m7") -> pd.DataFrame:
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


def default_baselines(
    freq: str = "5min",
    season_days: int = 1,
    *,
    include_ma1: bool = True,
) -> list[Forecaster]:
    """Standard baseline forecasters for experiments."""
    models: list[Forecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(
            freq=freq,
            season_length=_default_season_length(freq, days=season_days),
        ),
    ]
    if include_ma1:
        models.append(MA1Forecaster())
    return models


def default_forecasters(
    freq: str = "5min",
    *,
    include_ma1: bool = True,
    include_arima: bool = True,
    include_prophet: bool = True,
    arima_order: tuple[int, int, int] = (2, 1, 2),
) -> list[Forecaster]:
    """Baselines plus fixed-order ARIMA and Prophet (used by CV)."""
    models: list[Forecaster] = default_baselines(freq, include_ma1=include_ma1)
    if include_arima:
        models.append(ARIMAForecaster(order=arima_order))
    if include_prophet:
        models.append(ProphetForecaster(freq=freq))
    return models


def default_holdout_forecasters(
    freq: str = "5min",
    *,
    season_days: int = 7,
    sarima_season_days: int = 1,
    include_ma1: bool = True,
    include_auto_arima: bool = True,
    include_seasonal_arima: bool = True,
    include_prophet: bool = True,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
) -> list[Forecaster]:
    """Baselines plus holdout-only models (AR1, SES, DES, auto-ARIMA, SARIMA, Prophet)."""
    models: list[Forecaster] = default_baselines(
        freq, season_days=season_days, include_ma1=include_ma1
    )
    models.extend(
        [
            AR1Forecaster(),
            SESForecaster(),
            DESForecaster(),
        ]
    )
    if include_auto_arima:
        models.append(
            AutoARIMAForecaster(max_p=max_p, max_d=max_d, max_q=max_q)
        )
    if include_seasonal_arima:
        models.append(
            AutoSARIMAForecaster(freq=freq, season_days=sarima_season_days)
        )
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


def build_minute_bin_function_metadata(
    data_dir: Path | str,
    *,
    minute_cols: list[str],
    app_col: str = "HashApp",
    func_col: str = "HashFunction",
    trigger_col: str = "Trigger",
    min_invocations: int = 50,
    trace_minutes: int = 14 * 1440,
    n_volume_bins: int = 5,
    n_activity_bins: int = 3,
) -> pd.DataFrame:
    """
    Build per-function metadata from Azure 2019-style minute-count CSVs.

    Returns one row per eligible function with volume/activity bins for
    stratified sampling.
    """
    data_dir = Path(data_dir)
    usecols = [app_col, func_col, trigger_col, *minute_cols]
    totals: dict[tuple[str, str], int] = {}
    active: dict[tuple[str, str], int] = {}
    triggers: dict[tuple[str, str], str] = {}

    for path in sorted(data_dir.glob("invocations_per_function*.csv")):
        day_df = pd.read_csv(path, usecols=usecols)
        minute_totals = day_df[minute_cols].sum(axis=1).astype(np.int64)
        active_minutes = (day_df[minute_cols].to_numpy(dtype=np.int32) > 0).sum(
            axis=1
        )
        for app, func, trigger, total, active_count in zip(
            day_df[app_col],
            day_df[func_col],
            day_df[trigger_col],
            minute_totals,
            active_minutes,
        ):
            key = (app, func)
            if key not in triggers:
                triggers[key] = trigger
            totals[key] = totals.get(key, 0) + int(total)
            active[key] = active.get(key, 0) + int(active_count)

    rows: list[dict] = []
    for (app, func), invocations in totals.items():
        if invocations < min_invocations:
            continue
        active_minutes = active[(app, func)]
        rows.append(
            {
                "app": app,
                "func": func,
                "trigger": triggers[(app, func)],
                "invocations": invocations,
                "active_minutes": active_minutes,
                "active_frac": active_minutes / trace_minutes,
            }
        )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata

    metadata["inv_bin"] = pd.qcut(
        metadata["invocations"],
        q=n_volume_bins,
        labels=[f"Q{i}" for i in range(1, n_volume_bins + 1)],
        duplicates="drop",
    )
    metadata["activity_bin"] = pd.qcut(
        metadata["active_frac"],
        q=n_activity_bins,
        labels=[f"A{i}" for i in range(1, n_activity_bins + 1)],
        duplicates="drop",
    )
    return metadata.sort_values("invocations", ascending=False).reset_index(drop=True)


def sample_representative_functions(
    metadata: pd.DataFrame,
    *,
    census_n: int = 30,
    per_cell_k: int = 5,
    seed: int = 42,
    stratify_cols: tuple[str, ...] = ("inv_bin", "trigger"),
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """
    Hybrid census + stratified sample for representative forecast evaluation.

    Always includes the top ``census_n`` functions by invocation volume, then
    draws up to ``per_cell_k`` functions from each remaining
    (volume_bin, trigger) cell.
    """
    if metadata.empty:
        return [], metadata.iloc[0:0].copy()

    work = metadata.copy()
    work["key"] = list(zip(work["app"], work["func"]))
    ranked = work.sort_values("invocations", ascending=False)

    census = ranked.head(census_n).copy()
    census["sample_tier"] = "census"

    remaining = ranked.iloc[census_n:].copy()
    sampled_parts = [census]
    rng = np.random.default_rng(seed)

    for _, group in remaining.groupby(list(stratify_cols), observed=True):
        n_take = min(per_cell_k, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n_take, replace=False)
        part = group.loc[idx].copy()
        part["sample_tier"] = "stratified"
        sampled_parts.append(part)

    manifest = (
        pd.concat(sampled_parts, ignore_index=True)
        .drop_duplicates(subset=["app", "func"], keep="first")
        .sort_values("invocations", ascending=False)
        .reset_index(drop=True)
    )
    eval_keys = list(zip(manifest["app"], manifest["func"]))
    return eval_keys, manifest


def summarize_results_weighted(
    results: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    weight_col: str = "invocations",
    group_auto_arima: bool = True,
    sort_by: str = "mase_m7_weighted",
) -> pd.DataFrame:
    """Invocation-weighted mean MASE/RMSSE per model (micro average)."""
    if results.empty:
        return results

    weights = metadata[["app", "func", weight_col]]
    work = results.merge(weights, on=["app", "func"], how="left")
    if group_auto_arima:
        work["model"] = work["model"].map(_model_group)

    metric_cols = [
        c
        for c in ("mase_m1", "rmsse_m1", "mase_m7", "rmsse_m7", "mase", "rmsse")
        if c in work.columns
    ]
    rows: list[dict] = []
    for model, group in work.groupby("model", sort=False):
        w = group[weight_col].to_numpy(dtype=float)
        if np.sum(w) <= 0:
            continue
        row: dict = {"model": model, "n_functions": group["func"].nunique()}
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=float)
            row[f"{col}_weighted"] = float(np.average(vals, weights=w))
        rows.append(row)

    out = pd.DataFrame(rows)
    if sort_by in out.columns:
        return out.sort_values(sort_by)
    return out
