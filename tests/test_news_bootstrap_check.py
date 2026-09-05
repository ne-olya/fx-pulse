from __future__ import annotations

import pandas as pd

from fxpulse.news_bootstrap_check import weekly_policy_table


def test_weekly_policy_table_keeps_weeks_without_signals() -> None:
    scores = pd.DataFrame(
        {
            "feature_set": ["market_only", "market_only"],
            "horizon": [5, 5],
            "timestamp": ["2026-01-01", "2026-01-08"],
            "target": [1, 0],
        }
    )
    signals = pd.DataFrame(
        {
            "feature_set": ["market_only", "candidate"],
            "horizon": [5, 5],
            "timestamp": ["2026-01-01", "2026-01-01"],
            "target": [1, 1],
            "matched_week_hit_rate": [0.5, 0.5],
        }
    )

    result = weekly_policy_table(
        scores, signals, feature_sets=["market_only", "candidate"], horizon=5
    )

    assert len(result) == 2
    assert result.loc[1, "candidate__count"] == 0
