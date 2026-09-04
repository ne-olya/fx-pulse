from __future__ import annotations

import pandas as pd

from fxpulse.news_experiment import feature_sets, prepare_frame


def _config() -> dict[str, object]:
    return {
        "corridors": ["AMD"],
        "market_feature_prefixes": ["base__"],
        "news_feature_prefixes": ["news__"],
    }


def test_news_join_uses_exact_corridor_and_feature_date() -> None:
    market = pd.DataFrame(
        {
            "timestamp": ["2026-01-02", "2026-01-03"],
            "corridor": ["AMD", "AMD"],
            "base__return_1": [0.1, 0.2],
        }
    )
    news = pd.DataFrame(
        {
            "feature_date": ["2026-01-02", "2026-01-04"],
            "corridor": ["AMD", "AMD"],
            "news__article_count": [5, 999],
        }
    )

    result = prepare_frame(market, news, _config())

    assert len(result) == 1
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-02")
    assert result.loc[0, "news__article_count"] == 5


def test_feature_sets_keep_ablation_separate() -> None:
    frame = pd.DataFrame(
        {
            "base__return_1": [0.1],
            "news__article_count": [5],
            "outcome__regret_3": [0.0],
        }
    )

    result = feature_sets(frame, _config())

    assert result["market_only"] == ["base__return_1"]
    assert result["news_only"] == ["news__article_count"]
    assert result["market_plus_news"] == ["base__return_1", "news__article_count"]
