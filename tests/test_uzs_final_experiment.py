import pandas as pd

import fxpulse.uzs_final_experiment as final
from fxpulse.uzs_final_experiment import EXPECTED_FEATURE_SETS, feature_sets, load_config


def test_final_config_is_preregistered() -> None:
    config = load_config()
    assert config["development_years"] == [2022, 2023, 2024]
    assert config["pseudo_holdout_years"] == [2025, 2026]
    assert config["target_corridor"] == "UZS"


def test_feature_sets_keep_news_separate() -> None:
    frame = pd.DataFrame(
        {
            "base__return_1": [0.0],
            "extra__liquidity__fx__log_trades": [1.0],
            "extra__cross__rub_strength_1": [0.1],
            "extra__uzs__fx__usd_uzs_return_1": [0.2],
            "news__russia_count": [3.0],
            "news__sanctions_count": [1.0],
            "news__recipient_count": [2.0],
            "news__regional_other__recipient_count": [2.5],
            "aigpr__daily_gpr_ai_level": [4.0],
            "aigpr__regional_other__country_recipient_all_level": [4.5],
        }
    )
    sets = feature_sets(frame)
    assert list(sets) == EXPECTED_FEATURE_SETS
    assert "news__russia_count" not in sets["uzs_national"]
    assert "news__russia_count" in sets["uzs_national_plus_gdelt_russia"]
    assert "aigpr__daily_gpr_ai_level" in sets["uzs_national_plus_ai_gpr"]
    assert "news__regional_other__recipient_count" in sets["uzs_national_plus_regional_news"]
    assert "extra__cross__rub_strength_1" in sets["everything"]


def test_dual_lane_does_not_look_ahead_to_friday(monkeypatch) -> None:
    config = load_config()
    thursday = pd.Timestamp("2026-09-03")
    friday = pd.Timestamp("2026-09-04")
    current = pd.DataFrame(
        {
            "timestamp": [thursday, friday],
            "week": ["2026-36", "2026-36"],
            "score": [0.6, 0.9],
            "policy_score": [0.6, 0.9],
            "base__percentile_60": [0.2, 0.2],
            "policy": ["dual_lane", "dual_lane"],
        }
    )

    def candidates(frame, *, share, **_):
        # Thursday is only a fallback candidate; Friday is strong.
        return frame.loc[[1]].copy() if share == 0.2 else frame.copy()

    monkeypatch.setattr(final, "adaptive_candidates", candidates)
    selected = final._dual_lane(current, config=config)
    assert selected["timestamp"].tolist() == [thursday]
    assert selected["signal_tier"].tolist() == ["informational_value_fallback"]
