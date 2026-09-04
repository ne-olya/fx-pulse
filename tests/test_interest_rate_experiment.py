import pandas as pd

from fxpulse.interest_rate_experiment import add_interest_features


def test_interest_rate_is_never_available_same_day() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "corridor": ["TJS"] * 3,
        }
    )
    rates = pd.DataFrame({"rate_date": ["2024-01-02"], "key_rate_pct": [16.0]})

    result = add_interest_features(frame, rates, lag_days=1)

    assert pd.isna(result.loc[1, "interest__key_rate_pct"])
    assert result.loc[2, "interest__key_rate_pct"] == 16.0
