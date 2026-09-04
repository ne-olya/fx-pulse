import pandas as pd

from fxpulse.regime_policy_experiment import policy_candidates


def test_exclude_shock_never_returns_shock() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "score": [0.1, 0.2, 0.3, 0.9, 1.0],
            "regime": ["NORMAL", "NORMAL", "NORMAL", "SHOCK", "NORMAL"],
        }
    )
    config = {"top_score_share": 0.5, "lookback_observations": 3, "minimum_history_observations": 2}

    result = policy_candidates(frame, "exclude_shock", config)

    assert not result["regime"].eq("SHOCK").any()
