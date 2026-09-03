"""Explainable, point-in-time FX indicators.

Every indicator receives only an already truncated panel. It must derive all
normalisation statistics from that panel rather than fitting them globally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class IndicatorOut:
    """One explainable indicator evaluation at the final panel observation."""

    fired: bool
    strength: float
    details: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("Indicator strength must be finite and in [0, 1]")


IndicatorFn = Callable[..., IndicatorOut]
_REGISTRY: dict[str, IndicatorFn] = {}


def indicator(name: str) -> Callable[[IndicatorFn], IndicatorFn]:
    """Register an indicator under a stable config name."""

    def decorate(function: IndicatorFn) -> IndicatorFn:
        if name in _REGISTRY:
            raise ValueError(f"Indicator is already registered: {name}")
        _REGISTRY[name] = function
        return function

    return decorate


def get_indicator(name: str) -> IndicatorFn:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown indicator: {name}") from exc


def evaluate(name: str, panel: pd.DataFrame, **params: Any) -> IndicatorOut:
    """Evaluate an indicator using only the observations present in `panel`."""

    return get_indicator(name)(panel, **params)


def _ordered_prices(panel: pd.DataFrame) -> pd.Series:
    if "price" not in panel.columns:
        raise ValueError("Panel must contain a price column")
    order_columns = [column for column in ("known_at", "value_date") if column in panel.columns]
    ordered = (
        panel
        if panel.attrs.get("fxpulse_sorted_by_known_at")
        else panel.sort_values(order_columns, kind="mergesort")
        if order_columns
        else panel
    )
    prices = pd.to_numeric(ordered["price"], errors="raise")
    if prices.isna().any() or prices.le(0).any():
        raise ValueError("Indicators require strictly positive, non-missing prices")
    return prices.reset_index(drop=True)


def _returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _strength(value: float, scale: float) -> float:
    if scale <= 0 or not math.isfinite(scale):
        return 0.0
    return float(min(max(value / scale, 0.0), 1.0))


@indicator("level_percentile")
def level_percentile(panel: pd.DataFrame, *, window: int, pct: float) -> IndicatorOut:
    """Current price is in the lower `pct` percentile of its trailing window."""

    _require_positive_int("window", window)
    if not 0 < pct <= 100:
        raise ValueError("pct must be in (0, 100]")
    prices = _ordered_prices(panel)
    if len(prices) < window:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    trailing = prices.iloc[-window:]
    percentile = float(trailing.rank(method="max", pct=True).iloc[-1] * 100)
    fired = percentile <= pct
    return IndicatorOut(
        fired,
        _strength(pct - percentile, pct) if fired else 0.0,
        {"percentile": percentile, "window": window},
    )


@indicator("momentum_streak")
def momentum_streak(panel: pd.DataFrame, *, n: int) -> IndicatorOut:
    """Price has decreased in each of the latest `n` observed intervals."""

    _require_positive_int("n", n)
    prices = _ordered_prices(panel)
    if len(prices) < n + 1:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    returns = _returns(prices)
    streak = returns.iloc[-n:]
    fired = bool(streak.lt(0).all())
    sigma = float(returns.iloc[-max(20, n):].std(ddof=1))
    mean_decline = float(-streak.mean())
    return IndicatorOut(
        fired,
        _strength(mean_decline, sigma) if fired else 0.0,
        {"n": n, "mean_decline": mean_decline, "sigma": sigma},
    )


@indicator("filter_rule")
def filter_rule(panel: pd.DataFrame, *, x: float) -> IndicatorOut:
    """Latest price decrease is at least `x` percent in one observation interval."""

    if x <= 0:
        raise ValueError("x must be positive")
    prices = _ordered_prices(panel)
    if len(prices) < 2:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    latest_return = float(prices.iloc[-1] / prices.iloc[-2] - 1)
    threshold = x / 100
    fired = latest_return <= -threshold
    return IndicatorOut(
        fired,
        _strength(-latest_return - threshold, threshold) if fired else 0.0,
        {"change_pct": latest_return * 100, "threshold_pct": x},
    )


@indicator("reversal_from_low")
def reversal_from_low(panel: pd.DataFrame, *, lookback: int, bounce: float) -> IndicatorOut:
    """Price has rebounded from its earlier `lookback` low by `bounce` daily σ."""

    _require_positive_int("lookback", lookback)
    if bounce <= 0:
        raise ValueError("bounce must be positive")
    prices = _ordered_prices(panel)
    if len(prices) < lookback + 1:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    trailing = prices.iloc[-(lookback + 1) :]
    prior_low = float(trailing.iloc[:-1].min())
    current = float(trailing.iloc[-1])
    returns = _returns(prices).iloc[-lookback:]
    sigma = float(returns.std(ddof=1))
    rebound = current / prior_low - 1
    threshold = bounce * sigma
    fired = bool(prior_low < current and sigma > 0 and rebound >= threshold)
    return IndicatorOut(
        fired,
        _strength(rebound - threshold, threshold) if fired else 0.0,
        {"lookback": lookback, "rebound": rebound, "threshold": threshold, "sigma": sigma},
    )


@indicator("volatility_regime")
def volatility_regime(panel: pd.DataFrame, *, window: int = 20, regime: str = "high") -> IndicatorOut:
    """Realized volatility belongs to a trailing historical tercile."""

    _require_positive_int("window", window)
    if regime not in {"low", "mid", "high"}:
        raise ValueError("regime must be low, mid, or high")
    prices = _ordered_prices(panel)
    returns = _returns(prices)
    if len(returns) < 2 * window + 1:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    current = float(returns.iloc[-window:].std(ddof=1))
    historical = returns.iloc[:-1].rolling(window).std().dropna()
    low, high = (float(historical.quantile(1 / 3)), float(historical.quantile(2 / 3)))
    if regime == "low":
        fired = current <= low
        strength = _strength(low - current, low) if fired else 0.0
    elif regime == "high":
        fired = current >= high
        strength = _strength(current - high, high) if fired else 0.0
    else:
        fired = low < current < high
        strength = _strength(min(current - low, high - current), (high - low) / 2) if fired else 0.0
    return IndicatorOut(fired, strength, {"regime": regime, "volatility": current, "low_tercile": low, "high_tercile": high})


@indicator("seasonality")
def seasonality(
    panel: pd.DataFrame,
    *,
    calendar: str = "month",
    min_observations: int = 20,
    min_signal_to_noise: float = 0.25,
) -> IndicatorOut:
    """Past returns in the current calendar bucket form an explainable seasonal fact."""

    _require_positive_int("min_observations", min_observations)
    if calendar not in {"month", "day_of_month"}:
        raise ValueError("calendar must be month or day_of_month; holiday data are not loaded yet")
    if "value_date" not in panel.columns:
        raise ValueError("Seasonality requires value_date in the panel")
    ordered = panel.sort_values(["known_at", "value_date"], kind="mergesort").reset_index(drop=True)
    prices = _ordered_prices(ordered)
    dates = pd.to_datetime(ordered["value_date"], errors="raise")
    history = pd.DataFrame({"date": dates.iloc[:-1], "return": prices.pct_change().iloc[:-1]}).dropna()
    current_date = dates.iloc[-1]
    matching = history.loc[
        history["date"].dt.month.eq(current_date.month)
        if calendar == "month"
        else history["date"].dt.day.eq(current_date.day),
        "return",
    ]
    if len(matching) < min_observations:
        return IndicatorOut(False, 0.0, {"reason": "insufficient_history"})
    expected = float(matching.mean())
    sigma = float(matching.std(ddof=1))
    ratio = abs(expected) / sigma if sigma > 0 else 0.0
    fired = ratio >= min_signal_to_noise
    direction = "window_closing" if expected > 0 else "favorable"
    return IndicatorOut(
        fired,
        _strength(ratio - min_signal_to_noise, min_signal_to_noise) if fired else 0.0,
        {"calendar": calendar, "observations": len(matching), "mean_return": expected, "signal_to_noise": ratio, "direction": direction},
    )


__all__ = ["IndicatorOut", "evaluate", "get_indicator", "indicator"]
