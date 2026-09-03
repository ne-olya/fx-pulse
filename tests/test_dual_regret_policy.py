from __future__ import annotations

import numpy as np
import pandas as pd

from fxpulse.dual_regret_policy import load_dual_regret_policy_config
from fxpulse.local_minimum_models import FittedModel, refit_score_quantile_threshold


def test_dual_regret_config_is_one_fixed_two_budget_hypothesis() -> None:
    config = load_dual_regret_policy_config("configs/dual_regret_policy.json")
    evaluation = config["evaluation"]
    target = evaluation["dual_regret_target"]

    assert evaluation["models"] == ["xgboost"]
    assert target["short_horizon"] == 3
    assert target["short_tolerance_bps"] == 10
    assert target["long_horizon"] == 20
    assert target["long_tolerance_bps"] == 50
    assert evaluation["minimum_signals_per_week"] == 1
    assert evaluation["maximum_signals_per_week"] == 2


def test_refit_rank_threshold_uses_final_model_score_space() -> None:
    model = FittedModel(
        kind="xgboost",
        feature_columns=[],
        design_columns=[],
        median=pd.Series(dtype="float64"),
        mean=pd.Series(dtype="float64"),
        scale=pd.Series(dtype="float64"),
        model=None,
        train_scores=np.array([0.10, 0.20, 0.60, 0.95], dtype="float64"),
        train_positions=np.array([1, 2, 3, 4], dtype=int),
    )

    assert refit_score_quantile_threshold(model, 0.8) == np.quantile(model.train_scores, 0.8)
