import pandas as pd

from fxpulse.uzs_final_selection import matched_regret, reproduce_selection


def test_stability_rule_can_override_unstable_development_winner() -> None:
    rows = []
    candidates = [
        ("unstable", 1.4, 1.1, 1.4, 1.1, 1.05),
        ("stable", 1.25, 1.3, 1.4, 1.5, 1.2),
    ]
    for feature, dev, late, raw_dev, raw_late, worst in candidates:
        for period, same, raw in (("development", dev, raw_dev), ("pseudo_holdout", late, raw_late), ("all", min(dev, late), min(raw_dev, raw_late))):
            rows.append(
                {
                    "feature_set": feature,
                    "model_variant": "model",
                    "policy": "policy",
                    "period": period,
                    "same_week_lift": same,
                    "raw_lift": raw,
                    "signals": 100,
                    "signals_per_week": 0.7,
                    "weeks_covered_share": 0.5,
                    "worst_year_same_week_lift": worst,
                    "mean_regret_bps": 50.0,
                    "hit_rate": 0.6,
                    "feature_count": 10,
                }
            )
    _, selected = reproduce_selection(
        pd.DataFrame(rows),
        {"feature_set": "stable", "model_variant": "model", "policy": "policy"},
    )
    assert selected["feature_set"] == "stable"


def test_matched_regret_compares_only_with_same_calendar_week() -> None:
    identity = {"feature_set": "stable", "model_variant": "model", "policy": "policy"}
    scores = pd.DataFrame(
        {
            **{key: [value] * 4 for key, value in identity.items()},
            "timestamp": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-13", "2025-01-14"]),
            "test_year": [2025] * 4,
            "week": ["2025-2", "2025-2", "2025-3", "2025-3"],
            "regret_bps": [10.0, 30.0, 100.0, 200.0],
        }
    )
    signals = scores.iloc[[0, 2]].copy()
    result = matched_regret(
        scores,
        signals,
        identity,
        {"development_years": [2025], "pseudo_holdout_years": [2025], "test_years": [2025]},
    )
    row = result.loc[result["period"].eq("all")].iloc[0]
    assert row["matched_week_mean_regret_bps"] == 85.0
    assert row["selected_mean_regret_bps"] == 55.0
    assert row["reduction_vs_matched_week_bps"] == 30.0
