import pandas as pd

from fxpulse.meta_labeling_experiment import VOTE_COLUMNS, candidate_mask


def test_candidate_rules_are_explainable() -> None:
    frame = pd.DataFrame({column: [0.0, 0.0, 0.0] for column in VOTE_COLUMNS})
    frame.loc[0, "indicator__cheap_60"] = 1.0
    frame.loc[1, "indicator__rub_strength_3"] = 1.0
    frame.loc[1, "indicator__volatility_falling"] = 1.0

    assert candidate_mask(frame, "all_days").tolist() == [True, True, True]
    assert candidate_mask(frame, "cheap_or_near_min").tolist() == [True, False, False]
    assert candidate_mask(frame, "indicator_vote_2").tolist() == [False, True, False]
