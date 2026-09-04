import pandas as pd

from fxpulse.temporal_sequence_experiment import add_sequence_features, comparison


def test_sequence_lags_do_not_cross_corridors() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "corridor": ["A", "A", "B", "B"],
            "base__return_1": [1.0, 2.0, 10.0, 20.0],
        }
    )

    result = add_sequence_features(frame, ["base__return_1"], [1])

    assert result.loc[result["corridor"].eq("A"), "sequence__base_return_1_lag_1"].tolist()[1] == 1.0
    assert result.loc[result["corridor"].eq("B"), "sequence__base_return_1_lag_1"].tolist()[1] == 10.0


def test_sequence_comparison_is_a_paired_delta() -> None:
    rows = [
        {"scope": "A", "model": "catboost", "horizon": 3, "feature_set": "interpretable", "lift": 1.1, "lift_vs_matched_random": 1.0, "benefit_mean_bps": 1.0, "signals_per_week": 1.0, "worst_fold_lift": 0.9},
        {"scope": "A", "model": "catboost", "horizon": 3, "feature_set": "plus_sequence", "lift": 1.3, "lift_vs_matched_random": 1.1, "benefit_mean_bps": 2.0, "signals_per_week": 1.1, "worst_fold_lift": 1.0},
    ]

    result = comparison(pd.DataFrame(rows))

    assert abs(float(result.loc[0, "delta_lift"]) - 0.2) < 1e-12
