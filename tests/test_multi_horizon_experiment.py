import pandas as pd

from fxpulse.multi_horizon_experiment import combine_scores


def test_multi_horizon_score_combinations() -> None:
    frame = pd.DataFrame({"score_3": [0.3], "score_5": [0.6], "score_10": [0.9]})

    assert combine_scores(frame, "single_h10").item() == 0.9
    assert combine_scores(frame, "mean_scores").item() == 0.6
    assert combine_scores(frame, "minimum_score").item() == 0.3
