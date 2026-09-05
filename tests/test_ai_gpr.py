from __future__ import annotations

import pandas as pd

from fxpulse.ai_gpr_experiment import feature_sets
from fxpulse.ai_gpr_features import build_features


def _config() -> dict[str, object]:
    return {
        "date_from": "2026-01-01",
        "daily_availability_lag_days": 7,
        "monthly_availability_day_after_month_end": 7,
        "daily_columns": {"gpr_ai": "GPR_AI"},
        "corridor_countries": {"AMD": "Armenia"},
    }


def _monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = ["2026-01-01", "2026-02-01"]
    country = {"Date": dates}
    for nation in ("Russia", "Armenia"):
        for role in ("all", "initiator", "respondent", "spillover"):
            country[f"{nation}_{role}"] = [1.0, 2.0]
    bilateral = pd.DataFrame(
        {"Date": dates, "Russia|Armenia": [3.0, 4.0], "Armenia|Russia": [5.0, 6.0]}
    )
    return pd.DataFrame(country), bilateral


def test_daily_and_monthly_values_become_available_only_after_registered_delay() -> None:
    daily = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", "2026-02-10", freq="D"),
            "GPR_AI": range(41),
        }
    )
    country, bilateral = _monthly()

    result = build_features(daily, country, bilateral, _config())

    first = result.iloc[0]
    assert first["feature_date"] == pd.Timestamp("2026-01-08")
    assert first["aigpr__daily_gpr_ai_level"] == 0
    before_month = result.loc[result["feature_date"].eq(pd.Timestamp("2026-02-06"))].iloc[0]
    after_month = result.loc[result["feature_date"].eq(pd.Timestamp("2026-02-07"))].iloc[0]
    assert pd.isna(before_month["aigpr__country_russia_all_level"])
    assert after_month["aigpr__country_russia_all_level"] == 1
    assert after_month["aigpr__bilateral_russia_recipient_level"] == 8


def test_ai_gpr_feature_sets_keep_ablation_groups_separate() -> None:
    frame = pd.DataFrame(
        {
            "base__return_1": [0.0],
            "liquidity__usd": [1.0],
            "aigpr__daily_gpr_level": [2.0],
            "aigpr__country_russia_all_level": [3.0],
            "aigpr__cross_country_all_gap": [1.0],
            "aigpr__bilateral_russia_recipient_level": [4.0],
        }
    )

    result = feature_sets(frame)

    assert "aigpr__daily_gpr_level" in result["plus_daily_ai_gpr"]
    assert "aigpr__daily_gpr_level" not in result["plus_country_ai_gpr"]
    assert "aigpr__bilateral_russia_recipient_level" in result["plus_bilateral_ai_gpr"]
    assert "liquidity__usd" in result["plus_liquidity_all_ai_gpr"]
