import pandas as pd

from fxpulse.regret_formulation_experiment import regret_class


def test_regret_classes_include_boundaries_in_better_class() -> None:
    result = regret_class(pd.Series([0, 25, 25.1, 50, 75, 100, 101]), [25, 50, 100])

    assert result.tolist() == [0, 0, 1, 1, 2, 2, 3]
