from __future__ import annotations

import pandas as pd

from fxpulse.indicator_matrix import _metrics, _week_key


def test_same_week_lift_uses_only_the_signals_matched_weeks() -> None:
    labels = pd.DataFrame(
        {
            "position": [0, 1, 2, 3],
            "value_date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]),
            "hit_favorable": [1, 0, 0, 0],
            "future_regret_bps": [0.0, 50.0, 50.0, 50.0],
        }
    )
    labels["week"] = _week_key(labels["value_date"])
    rates = labels.groupby("week")["hit_favorable"].mean()
    labels["matched_week_hit_rate"] = labels["week"].map(rates)

    result = _metrics(labels, {0})

    assert result["hit_rate"] == 1.0
    assert result["raw_lift"] == 4.0
    assert result["same_week_lift"] == 2.0


def test_frequency_denominator_includes_years_without_signals() -> None:
    labels = pd.DataFrame(
        {
            "position": [0, 1, 2, 3],
            "value_date": pd.to_datetime(["2025-01-01", "2025-12-31", "2026-01-01", "2026-12-31"]),
            "hit_favorable": [1, 1, 1, 1],
            "future_regret_bps": [0.0, 0.0, 0.0, 0.0],
            "matched_week_hit_rate": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = _metrics(labels, {0})

    assert 0 < result["signals_per_week"] < 0.02
