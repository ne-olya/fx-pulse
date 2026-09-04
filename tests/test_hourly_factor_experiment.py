import pandas as pd

from fxpulse.hourly_factor_experiment import _hourly_policy


def test_hourly_policy_is_causal_and_respects_cap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05 10:00", "2026-01-05 11:00", "2026-01-06 11:00", "2026-01-08 11:00"]),
            "score": [0.6, 0.9, 0.7, 0.8],
        }
    )

    selected = _hourly_policy(frame, cooldown_hours=24, weekly_cap=2)

    assert selected["timestamp"].tolist() == frame.loc[[0, 2], "timestamp"].tolist()
