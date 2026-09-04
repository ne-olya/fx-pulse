import pandas as pd

from fxpulse.optimal_stopping_simulation import choose_position


def test_relaxing_threshold_forces_a_decision_by_deadline() -> None:
    window = pd.DataFrame({"price": [100, 99, 98], "score_rank": [0.1, 0.2, 0.1]})
    config = {"static_top_share": 0.4, "relaxing_start_top_share": 0.2}

    assert choose_position(window, "static_score", config) == 2
    assert choose_position(window, "relaxing_score", config) == 2
    assert choose_position(window, "buy_first", config) == 0
    assert choose_position(window, "oracle", config) == 2
