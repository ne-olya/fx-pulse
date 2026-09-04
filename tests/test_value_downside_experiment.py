import math

import pandas as pd

from fxpulse.value_downside_experiment import causal_percentile, combine_score


def test_causal_percentile_uses_only_prior_values() -> None:
    values = pd.Series([1.0, 2.0, 0.0, 3.0])
    result = causal_percentile(values, lookback=2, minimum_history=2)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 1.0


def test_conjunction_requires_both_components() -> None:
    frame = pd.DataFrame({"model_score": [0.9], "model_rank": [0.8], "value_score": [0.2]})

    assert combine_score(frame, "model_only").item() == 0.9
    assert combine_score(frame, "conjunction_min").item() == 0.2
