from __future__ import annotations

import pandas as pd

from fxpulse.data.gdelt_doc import parse_timeline
from fxpulse.gdelt_doc_features import build_features


def test_parse_timeline_keeps_raw_count_and_normalization() -> None:
    payload = {
        "timeline": [
            {
                "series": "Article Count",
                "data": [{"date": "20260101T000000Z", "value": 20, "norm": 1000}],
            }
        ]
    }

    result = parse_timeline(payload, series="russia")

    assert result.loc[0, "article_count"] == 20
    assert result.loc[0, "article_share"] == 0.02


def _config() -> dict[str, object]:
    return {
        "queries": {
            "russia": "Russia",
            "sanctions": "sanctions",
            "currency": "ruble",
            "energy": "oil",
            "amd": "Armenia",
        },
        "corridor_series": {"AMD": "amd"},
        "rolling_windows_days": [1, 3],
        "shock_window_days": 10,
        "availability_lag_days": 1,
    }


def _raw(periods: int = 20) -> pd.DataFrame:
    rows = []
    for series_index, series in enumerate(_config()["queries"], start=1):
        for day, timestamp in enumerate(pd.date_range("2026-01-01", periods=periods, freq="D")):
            rows.append(
                {
                    "date": timestamp,
                    "series": series,
                    "article_count": series_index + day,
                    "all_article_count": 1000,
                    "article_share": (series_index + day) / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_daily_features_use_completed_previous_day() -> None:
    result = build_features(_raw(), _config())

    assert result.loc[0, "feature_date"] == pd.Timestamp("2026-01-02")
    assert result.loc[0, "news__russia_count"] == 1


def test_appending_future_news_does_not_change_past_features() -> None:
    raw = _raw()
    cut = pd.Timestamp("2026-01-12")
    before = build_features(raw.loc[raw["date"].le(cut)], _config())
    after = build_features(raw, _config())
    after = after.loc[
        after["feature_date"].le(cut + pd.Timedelta(1, unit="D"))
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)
