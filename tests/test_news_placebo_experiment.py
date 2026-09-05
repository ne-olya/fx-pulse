from __future__ import annotations

import pandas as pd

from fxpulse.news_placebo_experiment import add_lagged_placebo, feature_sets


def test_placebo_uses_news_from_exactly_one_year_earlier() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2026-01-01"]),
            "corridor": ["AMD", "AMD"],
            "news__russia_count": [10.0, 99.0],
        }
    )

    result = add_lagged_placebo(frame, lag_days=365)

    assert len(result) == 1
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-01")
    assert result.loc[0, "placebo365__news__russia_count"] == 10.0


def test_recipient_placebo_keeps_only_recipient_levels() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2026-01-01"]),
            "corridor": ["TJS", "TJS"],
            "base__return_1": [0.1, 0.2],
            "news__russia_count": [10.0, 20.0],
            "news__recipient_count": [3.0, 8.0],
            "news__recipient_count_zscore_30d": [0.1, 0.2],
        }
    )

    result = add_lagged_placebo(
        frame,
        lag_days=365,
        news_scope="recipient_levels",
    )
    sets = feature_sets(result, news_scope="recipient_levels")

    assert "news__recipient_count" in sets["plus_recipient_levels_real"]
    assert "news__russia_count" not in sets["plus_recipient_levels_real"]
    assert "news__recipient_count_zscore_30d" not in sets["plus_recipient_levels_real"]
    assert result.loc[0, "placebo365__news__recipient_count"] == 3.0
