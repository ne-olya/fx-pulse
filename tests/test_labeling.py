from __future__ import annotations

import math

import pandas as pd

from fxpulse.labeling import evaluate_positions, label_observations, newey_west_t


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


def test_carried_observations_are_excluded_instead_of_filled() -> None:
    labels = label_observations(
        _panel([10.0, 9.0, 11.0, 12.0, 13.0], [False, False, True, False, False]),
        horizon=1,
    )

    assert labels["position"].tolist() == [0, 3]


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
    assert newey_west_t([1.0, 2.0, 3.0]) is not None
