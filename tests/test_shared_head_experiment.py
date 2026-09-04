import pandas as pd

from fxpulse.shared_head_experiment import combine


def test_shared_and_individual_scores_can_require_agreement() -> None:
    individual = pd.Series([0.9, 0.2])
    pooled = pd.Series([0.5, 0.8])

    assert combine(individual, pooled, "balanced_mean").tolist() == [0.7, 0.5]
    assert combine(individual, pooled, "agreement_min").tolist() == [0.5, 0.2]
