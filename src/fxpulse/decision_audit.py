"""Reproducible audits for indicator timing and the pilot benefit metric."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _validated_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "corridor", "price"}
    if missing := required - set(frame):
        raise ValueError(f"price panel lacks {sorted(missing)}")
    data = frame[["timestamp", "corridor", "price"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data["price"] = pd.to_numeric(data["price"], errors="raise")
    if not np.isfinite(data["price"]).all() or data["price"].le(0).any():
        raise ValueError("prices must be finite and positive")
    if data.duplicated(["timestamp", "corridor"]).any():
        raise ValueError("price panel contains duplicate timestamp/corridor rows")
    return data.sort_values(["corridor", "timestamp"], kind="mergesort").reset_index(drop=True)


def _rolling_percentile_state(prices: pd.Series, *, window: int, maximum: float) -> pd.Series:
    if window <= 1 or not 0 < maximum <= 1:
        raise ValueError("slow window must exceed one and percentile must be in (0, 1]")
    return prices.rolling(window, min_periods=window).apply(
        lambda values: float((values <= values[-1]).sum() / len(values) <= maximum),
        raw=True,
    ).eq(1)


def fast_slow_pairs(
    frame: pd.DataFrame,
    *,
    start_year: int = 2022,
    end_year: int = 2026,
    fast_streak: int = 3,
    slow_window: int = 60,
    slow_percentile: float = 0.20,
    confirmation_horizon: int = 5,
    outcome_horizon: int = 5,
    tolerance_bps: float = 25,
) -> pd.DataFrame:
    """Pair a fast falling-streak onset with the first slow value confirmation.

    All windows contain trading observations, not calendar days. A positive
    ``waiting_cost_bps`` means waiting made the RUB-per-recipient-unit price
    worse; a negative value means the client obtained a lower price by waiting.
    """

    if fast_streak <= 0 or confirmation_horizon < 0 or outcome_horizon <= 0:
        raise ValueError("streak and outcome horizon must be positive")
    if tolerance_bps < 0 or not math.isfinite(float(tolerance_bps)):
        raise ValueError("tolerance_bps must be finite and non-negative")
    data = _validated_prices(frame)
    rows: list[dict[str, object]] = []
    for corridor, raw in data.groupby("corridor", sort=True):
        current = raw.reset_index(drop=True)
        prices = current["price"]
        falling = prices.diff().lt(0)
        fast_state = falling.rolling(fast_streak, min_periods=fast_streak).sum().eq(fast_streak)
        fast_event = fast_state & ~fast_state.shift(1, fill_value=False)
        slow_state = _rolling_percentile_state(
            prices, window=slow_window, maximum=slow_percentile
        )
        future = pd.concat(
            [prices.shift(-step) for step in range(1, outcome_horizon + 1)], axis=1
        )
        complete = future.notna().all(axis=1)
        regret = ((prices / future.min(axis=1)) - 1).clip(lower=0) * 10_000
        for position in np.flatnonzero(fast_event.to_numpy()):
            timestamp = pd.Timestamp(current.loc[position, "timestamp"])
            if not start_year <= timestamp.year <= end_year:
                continue
            search_end = min(position + confirmation_horizon, len(current) - 1)
            confirmations = [
                candidate
                for candidate in range(position, search_end + 1)
                if bool(slow_state.iloc[candidate])
            ]
            confirmed_at = confirmations[0] if confirmations else None
            confirmation_complete = confirmed_at is not None and bool(complete.iloc[confirmed_at])
            rows.append(
                {
                    "corridor": corridor,
                    "fast_date": timestamp.date().isoformat(),
                    "fast_price": float(prices.iloc[position]),
                    "confirmed": confirmed_at is not None,
                    "confirmation_date": (
                        pd.Timestamp(current.loc[confirmed_at, "timestamp"]).date().isoformat()
                        if confirmed_at is not None
                        else None
                    ),
                    "delay_trading_days": confirmed_at - position if confirmed_at is not None else np.nan,
                    "confirmation_price": (
                        float(prices.iloc[confirmed_at]) if confirmed_at is not None else np.nan
                    ),
                    "waiting_cost_bps": (
                        (float(prices.iloc[confirmed_at]) / float(prices.iloc[position]) - 1) * 10_000
                        if confirmed_at is not None
                        else np.nan
                    ),
                    "fast_regret_bps": float(regret.iloc[position]) if bool(complete.iloc[position]) else np.nan,
                    "fast_hit": (
                        bool(regret.iloc[position] <= tolerance_bps)
                        if bool(complete.iloc[position])
                        else np.nan
                    ),
                    "confirmation_regret_bps": (
                        float(regret.iloc[confirmed_at]) if confirmation_complete else np.nan
                    ),
                    "confirmation_hit": (
                        bool(regret.iloc[confirmed_at] <= tolerance_bps)
                        if confirmation_complete
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_fast_slow(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = list(pairs.groupby("corridor", sort=True)) + [("ALL", pairs)]
    for corridor, current in groups:
        confirmed = current.loc[current["confirmed"]].copy()
        delayed = confirmed.loc[confirmed["delay_trading_days"].gt(0)].copy()
        rows.append(
            {
                "corridor": corridor,
                "fast_events": int(len(current)),
                "confirmed_within_5d": int(len(confirmed)),
                "confirmation_share": float(len(confirmed) / len(current)) if len(current) else np.nan,
                "same_day_confirmation_share": (
                    float(confirmed["delay_trading_days"].eq(0).mean()) if len(confirmed) else np.nan
                ),
                "delayed_confirmations": int(len(delayed)),
                "median_delay_trading_days_delayed": (
                    float(delayed["delay_trading_days"].median()) if len(delayed) else np.nan
                ),
                "mean_waiting_cost_bps_delayed": (
                    float(delayed["waiting_cost_bps"].mean()) if len(delayed) else np.nan
                ),
                "median_waiting_cost_bps_delayed": (
                    float(delayed["waiting_cost_bps"].median()) if len(delayed) else np.nan
                ),
                "fast_hit_rate": float(current["fast_hit"].dropna().mean()),
                "confirmation_hit_rate": (
                    float(confirmed["confirmation_hit"].dropna().mean()) if len(confirmed) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def symmetric_benefit_bps(
    frame: pd.DataFrame,
    *,
    corridor: str = "UZS",
    horizon: int = 5,
    start_year: int = 2022,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Return the case-defined price benefit versus the surrounding ±h window."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    data = _validated_prices(frame)
    current = data.loc[data["corridor"].eq(corridor)].reset_index(drop=True)
    if current.empty:
        raise ValueError(f"corridor {corridor!r} is absent")
    prices = current["price"]
    surrounding = sum(
        (prices.shift(offset) for offset in range(-horizon, horizon + 1) if offset != 0),
        start=pd.Series(0.0, index=prices.index),
    ) / (2 * horizon)
    current["benefit_sym_bps"] = (surrounding / prices - 1) * 10_000
    years = current["timestamp"].dt.year
    return current.loc[
        years.between(start_year, end_year) & current["benefit_sym_bps"].notna(),
        ["timestamp", "corridor", "price", "benefit_sym_bps"],
    ].reset_index(drop=True)


def summarize_benefit(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = list(values.groupby(values["timestamp"].dt.year, sort=True)) + [("ALL", values)]
    for period, current in groups:
        benefit = current["benefit_sym_bps"]
        rows.append(
            {
                "period": str(period),
                "observations": int(len(current)),
                "mean_bps": float(benefit.mean()),
                "std_bps": float(benefit.std(ddof=1)),
                "p10_bps": float(benefit.quantile(0.10)),
                "median_bps": float(benefit.median()),
                "p90_bps": float(benefit.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "fast_slow_pairs",
    "summarize_fast_slow",
    "symmetric_benefit_bps",
    "summarize_benefit",
]
