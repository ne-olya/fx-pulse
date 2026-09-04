from __future__ import annotations

import pandas as pd

from fxpulse.news_features import build_daily_features, build_hourly_features


def _config() -> dict[str, object]:
    return {
        "corridors": ["AMD"],
        "decision_timezone": "Europe/Moscow",
        "daily_decision_time": "09:00",
        "availability_delay_minutes": 15,
        "hourly_windows": [2],
        "daily_windows": [1, 2],
        "shock_baselines_hours": [4],
        "shock_baselines_days": [2],
    }


def _row(timestamp: str, articles: int, tone_sum: float = 0.0) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_utc": timestamp,
        "corridor": "AMD",
        "article_count": articles,
        "russia_article_count": articles,
        "recipient_article_count": 0,
        "bilateral_article_count": 0,
        "source_count": articles,
        "tone_sum": tone_sum,
        "tone_count": articles,
        "russia_tone_sum": tone_sum,
        "russia_tone_count": articles,
        "recipient_tone_sum": 0,
        "recipient_tone_count": 0,
        "negative_count": int(tone_sum < 0),
        "conflict_count": 0,
        "material_conflict_count": 0,
        "cooperation_count": articles,
        "coercion_count": 0,
        "protest_count": 0,
        "goldstein_sum": 2.0 * articles,
        "goldstein_count": articles,
    }
    return row


def test_hourly_features_become_available_after_completed_hour_and_delay() -> None:
    raw = pd.DataFrame([_row("2026-01-01 06:00:00+00:00", 2, -4.0)])

    result = build_hourly_features(raw, _config())

    assert result.loc[0, "available_at_utc"] == pd.Timestamp("2026-01-01 07:15:00+00:00")
    assert result.loc[0, "news__avg_tone"] == -2.0
    assert result.loc[0, "news__negative_share"] == 0.5


def test_daily_cutoff_moves_late_batch_to_next_decision_date() -> None:
    # 05:00 UTC is 08:00 Moscow; the hour becomes complete at 09:15 Moscow,
    # therefore it is not legal for the 09:00 decision and moves to tomorrow.
    raw = pd.DataFrame(
        [
            _row("2026-01-01 04:00:00+00:00", 2),
            _row("2026-01-01 05:00:00+00:00", 3),
        ]
    )

    result = build_daily_features(raw, _config())

    first = result.loc[result["feature_date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]
    second = result.loc[result["feature_date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    assert first["news__article_count"] == 2
    assert second["news__article_count"] == 3


def test_news_shock_uses_only_prior_values() -> None:
    raw = pd.DataFrame(
        [
            _row("2026-01-01 00:00:00+00:00", 1),
            _row("2026-01-01 01:00:00+00:00", 1),
            _row("2026-01-01 02:00:00+00:00", 1),
            _row("2026-01-01 03:00:00+00:00", 10),
        ]
    )

    original = build_hourly_features(raw, _config())
    changed = raw.copy()
    changed.loc[3, "article_count"] = 1000
    changed_result = build_hourly_features(changed, _config())

    prior_columns = ["news__volume_zscore_4h", "news__volume_ratio_4h"]
    pd.testing.assert_frame_equal(original.loc[:2, prior_columns], changed_result.loc[:2, prior_columns])
