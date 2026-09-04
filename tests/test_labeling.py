from __future__ import annotations

import math

import pandas as pd
import pytest

from fxpulse.labeling import (
    evaluate_positions,
    label_dual_regret_observations,
    label_hybrid_observations,
    label_observations,
    newey_west_t,
)


def _panel(prices: list[float], carried: list[bool] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "known_at": dates.tz_localize("Europe/Moscow") + pd.DateOffset(hours=15, minutes=30),
            "value_date": dates.date,
            "price": prices,
            "is_carried": carried or [False] * len(prices),
        }
    )


def test_labels_use_only_future_prices_and_keep_symmetric_label_separate() -> None:
    labels = label_observations(_panel([10.0, 9.0, 11.0, 12.0, 13.0]), horizon=1)

    assert labels["position"].tolist() == [0, 1, 2, 3]
    assert not bool(labels.loc[0, "hit_favorable"])
    assert bool(labels.loc[1, "hit_favorable"])
    assert bool(labels.loc[1, "hit_closing"])
    assert pd.isna(labels.loc[0, "benefit_sym_bps"])
    assert labels.loc[1, "benefit_sym_bps"] == 1_666.6666666666674
    assert labels.loc[1, "benefit_fwd_bps"] == 2_222.222222222223


def test_labels_support_intraday_horizons() -> None:
    labels = label_observations(_panel([10.0, 8.0, 9.0, 10.0, 11.0, 12.0]), horizon=4)

    assert labels["position"].tolist() == [0, 1]
    assert not bool(labels.loc[0, "hit_favorable"])
    assert bool(labels.loc[1, "hit_favorable"])


def test_labels_reject_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        label_observations(_panel([10.0, 11.0]), horizon=0)


def test_carried_observations_are_excluded_instead_of_filled() -> None:
    labels = label_observations(
        _panel([10.0, 9.0, 11.0, 12.0, 13.0], [False, False, True, False, False]),
        horizon=1,
    )

    assert labels["position"].tolist() == [0, 3]


def test_regret_target_distinguishes_negligible_and_material_future_improvements() -> None:
    # A 5 bps better future price should be acceptable under a 10 bps budget;
    # a 100 bps improvement should not.  Exact-min labels would call both zero.
    panel = _panel([100.0, 99.95, 100.0, 100.0, 100.0, 99.0, 100.0])
    labels = label_observations(panel, horizon=1, tolerance_bps=10.0)

    assert bool(labels.loc[0, "hit_favorable"])
    assert math.isclose(float(labels.loc[0, "future_regret_bps"]), 5.0)
    assert not bool(labels.loc[4, "hit_favorable"])
    assert math.isclose(float(labels.loc[4, "future_regret_bps"]), 100.0)
    assert labels.loc[4, "future_best_price"] == 99.0


def test_hybrid_target_requires_short_strict_and_long_regret_conditions() -> None:
    # Position 0 is the strict 1-day minimum but loses 50 bps on day 2, so it
    # fails a 2 bps long-horizon regret budget. Position 2 passes both.
    labels = label_hybrid_observations(
        _panel([100.0, 101.0, 99.5, 101.0, 102.0, 103.0]),
        short_horizon=1,
        long_horizon=3,
        long_tolerance_bps=2.0,
    )

    assert bool(labels.loc[0, "short_strict_favorable"])
    assert not bool(labels.loc[0, "long_regret_favorable"])
    assert not bool(labels.loc[0, "hit_favorable"])
    assert bool(labels.loc[2, "short_strict_favorable"])
    assert bool(labels.loc[2, "long_regret_favorable"])
    assert bool(labels.loc[2, "hit_favorable"])


def test_dual_regret_target_accepts_small_short_improvement_and_rejects_material_one() -> None:
    # At position 0 the next close is only 5 bps lower. Exact minimum would
    # reject it, but both short and long product budgets accept it. Position 3
    # loses 100 bps on the next close and must fail the short condition.
    labels = label_dual_regret_observations(
        _panel([100.0, 99.95, 100.0, 100.0, 99.0, 100.0, 101.0]),
        short_horizon=1,
        short_tolerance_bps=10.0,
        long_horizon=3,
        long_tolerance_bps=50.0,
    )

    assert bool(labels.loc[0, "short_regret_favorable"])
    assert bool(labels.loc[0, "long_regret_favorable"])
    assert bool(labels.loc[0, "hit_favorable"])
    assert not bool(labels.loc[3, "short_regret_favorable"])
    assert not bool(labels.loc[3, "hit_favorable"])


def test_metrics_compare_to_full_eligible_period_and_expose_stability() -> None:
    panel = _panel([10.0, 9.0, 11.0, 12.0, 13.0])
    metrics = evaluate_positions(panel, [1, 3], direction="favorable", horizon=1)

    assert metrics["signal_count"] == 2
    assert metrics["eligible_base_count"] == 4
    assert metrics["hit_rate"] == 1.0
    assert metrics["baseline_hit_rate"] == 0.75
    assert math.isclose(float(metrics["lift"]), 4 / 3)
    assert metrics["signals_per_week"] == 1.0
    assert metrics["cluster_share"] == 1.0
    assert metrics["benefit_fwd_newey_west_t"] is not None
    assert metrics["regret_mean_bps"] is not None
    assert metrics["baseline_regret_p90_bps"] is not None
    assert newey_west_t([1.0, 2.0, 3.0]) is not None
