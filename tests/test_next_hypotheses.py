import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _apply_policy, _future_outcomes


def test_future_outcomes_use_only_next_h_observations() -> None:
    price = pd.Series([100.0, 99.0, 98.0, 50.0])

    regret, benefit = _future_outcomes(price, 2)

    assert np.isclose(regret.iloc[0], (100 / 98 - 1) * 10_000)
    assert np.isclose(benefit.iloc[0], ((99 + 98) / 2 / 100 - 1) * 10_000)
    assert regret.iloc[1] > 0
    assert regret.iloc[2:].isna().all()


def test_selective_policy_applies_weekly_cap_and_cooldown() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-08", "2026-01-12"]),
            "score": [0.8, 0.9, 0.7, 0.6],
        }
    )

    selected = _apply_policy(candidates, cooldown_days=3, weekly_cap=2)

    assert selected["timestamp"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-05",
        "2026-01-08",
        "2026-01-12",
    ]
