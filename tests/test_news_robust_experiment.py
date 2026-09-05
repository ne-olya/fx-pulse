from __future__ import annotations

import pandas as pd

from fxpulse.news_robust_experiment import feature_sets, nested_selection, prepare_frame


def _config() -> dict[str, object]:
    return {
        "corridors": ["AMD"],
        "horizons": [3],
        "test_years": [2022, 2023],
        "nested_selection": {
            "first_selection_year": 2023,
            "minimum_past_signals": 1,
            "metric": "lift_vs_matched_random",
            "fallback_feature_set": "market_only",
        },
    }


def test_prepare_frame_joins_exact_lagged_feature_date() -> None:
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
            "news__russia_count": [2, 99],
        }
    )

    result = prepare_frame(market, news, _config())

    assert len(result) == 1
    assert result.loc[0, "news__russia_count"] == 2


def test_feature_families_separate_russia_recipient_levels_and_shocks() -> None:
    frame = pd.DataFrame(
        {
            "base__return_1": [0.1],
            "liquidity__usd_num_trades": [1.0],
            "news__russia_count": [2.0],
            "news__russia_count_1d": [2.0],
            "news__russia_count_3d": [4.0],
            "news__russia_count_zscore_30d": [0.5],
            "news__recipient_share": [0.01],
            "news__recipient_share_zscore_30d": [-0.5],
            "news__recipient_macro_count": [1.0],
            "news__cross_country_count_shock_gap_30d": [1.0],
        }
    )

    result = feature_sets(frame)

    assert "news__russia_count_1d" not in result["plus_russia_levels"]
    assert "news__russia_count_3d" in result["plus_russia_levels"]
    assert "news__recipient_share" in result["plus_recipient_levels"]
    assert "news__recipient_share_zscore_30d" in result["plus_recipient_shocks"]
    assert "news__recipient_macro_count" in result["plus_macro_topics"]
    assert "news__cross_country_count_shock_gap_30d" in result["plus_cross_country"]
    assert "liquidity__usd_num_trades" in result["plus_liquidity_all_news"]


def test_nested_selection_uses_only_previous_year() -> None:
    rows = []
    signal_rows = []
    for feature_set, previous_hit in [("market_only", 0), ("news_only", 1)]:
        for year in [2022, 2023]:
            for day in [3, 4]:
                rows.append(
                    {
                        "timestamp": pd.Timestamp(year=year, month=1, day=day),
                        "corridor": "AMD",
                        "horizon": 3,
                        "test_year": year,
                        "feature_set": feature_set,
                        "target": 1 if day == 3 else 0,
                        "regret_bps": 0.0,
                        "benefit_bps": 0.0,
                        "score": 0.9,
                    }
                )
            signal_rows.append(
                {
                    **rows[-2],
                    "feature_set": feature_set,
                    "target": previous_hit if year == 2022 else 1 - previous_hit,
                    "matched_week_hit_rate": 0.5,
                }
            )
    scores = pd.DataFrame(rows)
    signals = pd.DataFrame(signal_rows)

    selections, selected_scores, selected_signals = nested_selection(scores, signals, _config())

    assert selections.loc[0, "selected_feature_set"] == "news_only"
    assert selected_scores["feature_set"].eq("nested_past_choice").all()
    assert selected_signals["feature_set"].eq("nested_past_choice").all()
