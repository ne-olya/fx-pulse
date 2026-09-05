from __future__ import annotations

import pandas as pd

from fxpulse.news_risk_gate_experiment import gate_masks


def test_news_risk_gate_has_fixed_interpretable_rules() -> None:
    frame = pd.DataFrame(
        {
            "global": [0.0, 2.0, 0.0, 2.0],
            "oil": [0.0, 0.0, 2.0, 2.0],
        }
    )
    config = {
        "shock_zscore_threshold": 1.0,
        "global_feature": "global",
        "oil_feature": "oil",
    }

    masks = gate_masks(frame, config)

    assert masks["exclude_global_gpr_gt1"].tolist() == [True, False, True, False]
    assert masks["exclude_oil_gpr_gt1"].tolist() == [True, True, False, False]
    assert masks["exclude_global_or_oil_gt1"].tolist() == [True, False, False, False]
    assert masks["global_gpr_gt1_only"].tolist() == [False, True, False, True]
