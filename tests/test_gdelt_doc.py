from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fxpulse.data.gdelt_doc import (
    _cache_path,
    _decode_response,
    _proxy_url,
    parse_timeline,
)
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


def test_cache_name_preserves_annual_checkpoints_and_multi_year_slices() -> None:
    assert (
        _cache_path(Path("cache"), "russia", date(2018, 1, 1), date(2018, 12, 31)).name
        == "russia_2018.json"
    )


def test_text_proxy_wrapper_is_decoded_without_changing_source_json() -> None:
    body = b'Title: x\n\nMarkdown Content:\n{"timeline": []}\n'

    assert _decode_response(body) == {"timeline": []}


def test_proxy_url_keeps_target_query_inside_the_proxied_url() -> None:
    target = "https://api.gdelt.test/doc?query=Tajikistan&mode=timelinevolraw"

    result = _proxy_url(target, "https://r.jina.ai/http://")

    assert result == (
        "https://r.jina.ai/http://api.gdelt.test/doc?"
        "query=Tajikistan%26mode=timelinevolraw"
    )
    assert (
        _cache_path(Path("cache"), "russia", date(2020, 1, 1), date(2026, 9, 2)).name
        == "russia_2020_2026.json"
    )


def _config() -> dict[str, object]:
    return {
        "queries": {
            "russia": "Russia",
            "sanctions": "sanctions",
            "currency": "ruble",
            "energy": "oil",
            "amd": "Armenia",
            "amd_macro": "Armenia dram",
        },
        "corridor_series": {"AMD": "amd"},
        "corridor_additional_series": {"AMD": {"recipient_macro": "amd_macro"}},
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
    assert result.loc[0, "news__recipient_macro_count"] == 6
    assert "news__cross_country_count_shock_gap_10d" in result
    assert "news__cross_macro_shock_gap_10d" in result


def test_appending_future_news_does_not_change_past_features() -> None:
    raw = _raw()
    cut = pd.Timestamp("2026-01-12")
    before = build_features(raw.loc[raw["date"].le(cut)], _config())
    after = build_features(raw, _config())
    after = after.loc[
        after["feature_date"].le(cut + pd.Timedelta(1, unit="D"))
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)


def test_global_source_gap_is_unknown_not_zero_news() -> None:
    raw = _raw(periods=20)
    missing_date = pd.Timestamp("2026-01-10")
    raw = raw.loc[pd.to_datetime(raw["date"]).ne(missing_date)]

    result = build_features(raw, _config())
    gap = result.loc[result["feature_date"].eq(pd.Timestamp("2026-01-11"))].iloc[0]

    assert pd.isna(gap["news__russia_count"])
    assert gap["news__source_missing"] == 1
