"""Future-only labels and fixed signal-quality metrics.

Labels intentionally live outside the feature and signal modules: a signal is
constructed from the historical information set, then evaluated against these
future observations in a backtest only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


HORIZONS = (1, 3, 5, 10, 20)


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"price", "known_at", "value_date", "is_carried"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing columns: {', '.join(sorted(missing))}")
    ordered = panel.sort_values(["known_at", "value_date"], kind="mergesort").reset_index(drop=True).copy()
    ordered["price"] = pd.to_numeric(ordered["price"], errors="raise")
    if ordered["price"].isna().any() or ordered["price"].le(0).any():
        raise ValueError("Labels require strictly positive, non-missing prices")
    ordered["is_carried"] = ordered["is_carried"].fillna(False).astype(bool)
    return ordered


def label_observations(panel: pd.DataFrame, horizon: int, *, tolerance_bps: float = 0.0) -> pd.DataFrame:
    """Label each eligible observation using exactly `horizon` future observations.

    `benefit_sym_bps` follows the supplied specification; `benefit_fwd_bps` is
    the product-honest forward-only counterpart. No label is ever supplied to an
    indicator or `signals_as_of`.
    """

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if not isinstance(tolerance_bps, int | float) or not math.isfinite(float(tolerance_bps)) or tolerance_bps < 0:
        raise ValueError("tolerance_bps must be a finite non-negative number")
    ordered = _ordered(panel)
    rows: list[dict[str, Any]] = []
    prices = ordered["price"].tolist()
    carried = ordered["is_carried"].tolist()
    tolerance = 1 + float(tolerance_bps) / 10_000
    for position in range(len(ordered) - horizon):
        if carried[position] or any(carried[position + 1 : position + horizon + 1]):
            continue
        price = prices[position]
        future = prices[position + 1 : position + horizon + 1]
        has_full_past_window = position >= horizon - 1
        past = prices[position - horizon + 1 : position + 1] if has_full_past_window else []
        past_has_carried = has_full_past_window and any(carried[position - horizon + 1 : position + 1])
        symmetric = None
        if position >= horizon and not any(carried[position - horizon : position]):
            surrounding = prices[position - horizon : position] + future
            symmetric = (sum(surrounding) / len(surrounding) / price - 1) * 10_000
        rows.append(
            {
                "position": position,
                "known_at": ordered.loc[position, "known_at"],
                "value_date": ordered.loc[position, "value_date"],
                # A positive label means that buying at T is within the configured
                # tolerance of the best executable close in the next h observations.
                "hit_favorable": min(future) >= price / tolerance,
                # This is observable at T and is intentionally separate from the
                # future-only outcome. Product policies may only send on this gate.
                "past_min": bool(has_full_past_window and not past_has_carried and price <= min(past) * tolerance),
                "hit_closing": future[-1] > price,
                "benefit_sym_bps": symmetric,
                "benefit_fwd_bps": (sum(future) / len(future) / price - 1) * 10_000,
            }
        )
    return pd.DataFrame(
        rows,
        columns=(
            "position",
            "known_at",
            "value_date",
            "hit_favorable",
            "past_min",
            "hit_closing",
            "benefit_sym_bps",
            "benefit_fwd_bps",
        ),
    )


def newey_west_t(values: Iterable[float]) -> float | None:
    """HAC t-statistic for a mean with an automatic Bartlett lag selection."""

    series = pd.Series(list(values), dtype="float64").dropna()
    count = len(series)
    if count < 2:
        return None
    mean = float(series.mean())
    demeaned = [float(value - mean) for value in series]
    max_lag = max(1, min(count - 1, math.floor(4 * (count / 100) ** (2 / 9))))
    gamma_zero = sum(value * value for value in demeaned) / count
    long_run_variance = gamma_zero
    for lag in range(1, max_lag + 1):
        autocovariance = sum(demeaned[index] * demeaned[index - lag] for index in range(lag, count)) / count
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * autocovariance
    if long_run_variance <= 0:
        return 0.0 if mean == 0 else None
    return mean / math.sqrt(long_run_variance / count)


def _frequency_and_clustering(
    panel: pd.DataFrame, positions: list[int], base_positions: Iterable[int] | None = None
) -> dict[str, float | None]:
    period_positions = list(base_positions) if base_positions is not None else positions
    if not period_positions:
        return {"signals_per_week": 0.0, "signals_per_month": 0.0, "cluster_share": None, "interval_cv": None}
    dates = pd.to_datetime(panel.loc[period_positions, "known_at"], errors="raise").dt.tz_localize(None)
    weeks = max(1, len(dates.dt.to_period("W").drop_duplicates()))
    months = max(1, len(dates.dt.to_period("M").drop_duplicates()))
    if not positions:
        return {"signals_per_week": 0.0, "signals_per_month": 0.0, "cluster_share": None, "interval_cv": None}
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    if not gaps:
        cluster_share = None
        interval_cv = None
    else:
        cluster_share = sum(gap <= 5 for gap in gaps) / len(gaps)
        mean_gap = sum(gaps) / len(gaps)
        interval_cv = math.sqrt(sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)) / mean_gap if mean_gap else None
    return {
        "signals_per_week": len(positions) / weeks,
        "signals_per_month": len(positions) / months,
        "cluster_share": cluster_share,
        "interval_cv": interval_cv,
    }


def evaluate_positions(
    panel: pd.DataFrame,
    positions: Iterable[int],
    *,
    direction: str,
    horizon: int,
    base_positions: Iterable[int] | None = None,
    labels: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    """Evaluate fired positions against an eligible period of the same series.

    `base_positions` narrows the baseline to one out-of-time block. Supplying
    precomputed `labels` lets a grid run reuse identical future-only labels for
    every configuration without changing their definition.
    """

    if direction not in {"favorable", "window_closing"}:
        raise ValueError("direction must be favorable or window_closing")
    ordered = _ordered(panel)
    labels = labels.copy() if labels is not None else label_observations(ordered, horizon)
    selected_base_positions = set(base_positions) if base_positions is not None else None
    if selected_base_positions is not None:
        labels = labels.loc[labels["position"].isin(selected_base_positions)].copy()
    position_set = set(positions)
    signal = labels.loc[labels["position"].isin(position_set)].copy()
    hit_column = "hit_favorable" if direction == "favorable" else "hit_closing"
    hit_rate = float(signal[hit_column].mean()) if not signal.empty else None
    baseline_hit_rate = float(labels[hit_column].mean()) if not labels.empty else None
    lift = hit_rate / baseline_hit_rate if hit_rate is not None and baseline_hit_rate not in {None, 0} else None
    frequency = _frequency_and_clustering(
        ordered, sorted(signal["position"].tolist()), selected_base_positions
    )
    return {
        "signal_count": int(len(signal)),
        "eligible_base_count": int(len(labels)),
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline_hit_rate,
        "lift": lift,
        "benefit_sym_bps": float(signal["benefit_sym_bps"].mean()) if not signal.empty else None,
        "benefit_fwd_bps": float(signal["benefit_fwd_bps"].mean()) if not signal.empty else None,
        "benefit_fwd_newey_west_t": newey_west_t(signal["benefit_fwd_bps"]) if not signal.empty else None,
        **frequency,
    }


__all__ = ["HORIZONS", "evaluate_positions", "label_observations", "newey_west_t"]
