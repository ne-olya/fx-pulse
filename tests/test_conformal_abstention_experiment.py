import pandas as pd

from fxpulse.conformal_abstention_experiment import conformal_positive_candidates


def test_conformal_calibration_delays_outcome_by_horizon() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "score": [0.9] * 8,
            "target": [1] * 8,
        }
    )
    result = conformal_positive_candidates(
        frame,
        horizon=3,
        alpha=0.2,
        lookback=10,
        minimum_history=3,
    )

    assert result["timestamp"].min() >= pd.Timestamp("2024-01-06")
