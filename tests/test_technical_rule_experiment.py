import pandas as pd

from fxpulse.technical_rule_experiment import add_rules


def test_fixed_filter_and_ma_cross_use_current_and_past_prices() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D"),
            "corridor": ["TJS"] * 8,
            "price": [100, 100, 100, 100, 101, 102, 103, 104],
        }
    )
    config = {
        "return_windows": [1],
        "fixed_move_percent": [0.5],
        "streak_lengths": [3],
        "moving_average_pairs": [[2, 4]],
        "horizons": [3],
    }
    result = add_rules(frame, config)

    assert bool(result.loc[4, "rule__move_up_0.5pct_1d"])
    assert bool(result.loc[6, "rule__up_streak_3d"])
    assert not bool(result.loc[5, "rule__up_streak_3d"])
    assert result["rule__ma_cross_up_2_4"].sum() == 1
    assert pd.isna(result.loc[6, "regret_3"])
