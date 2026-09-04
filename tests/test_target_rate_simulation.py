import pandas as pd

from fxpulse.target_rate_simulation import simulate_orders


def test_target_rate_executes_at_first_observed_target_hit() -> None:
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "corridor": ["TJS"] * 10,
            "price": [100, 99.8, 99.0, 98.5, 101, 101, 101, 101, 101, 101],
        }
    )
    config = {
        "corridors": ["TJS"],
        "target_improvements_percent": [1.0],
        "deadlines_calendar_days": [7],
        "from_date": "2024-01-01",
        "to_date": "2024-01-10",
    }
    result = simulate_orders(prices, config)
    first = result.loc[result["start_date"].eq(pd.Timestamp("2024-01-01"))].iloc[0]

    assert bool(first["executed"])
    assert first["final_date"] == pd.Timestamp("2024-01-03")
    assert first["wait_calendar_days"] == 2
