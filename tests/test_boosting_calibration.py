from __future__ import annotations

import numpy as np

from fxpulse.boosting_calibration import _binary_metrics, _fit_calibrator, load_boosting_calibration_config
from fxpulse.hybrid_targets import load_hybrid_targets_config


def test_calibrators_return_finite_probabilities_and_none_keeps_scores() -> None:
    scores = np.array([0.05, 0.20, 0.70, 0.95], dtype="float64")
    target = np.array([0, 0, 1, 1], dtype=int)

    none = _fit_calibrator("none", scores, target).predict(scores)
    sigmoid = _fit_calibrator("sigmoid", scores, target).predict(scores)
    isotonic = _fit_calibrator("isotonic", scores, target).predict(scores)

    assert np.allclose(none, scores)
    for probability in (sigmoid, isotonic):
        assert np.isfinite(probability).all()
        assert ((probability > 0) & (probability < 1)).all()


def test_calibration_metrics_are_finite_and_config_is_multi_horizon() -> None:
    metrics = _binary_metrics(np.array([0, 1]), np.array([0.2, 0.8]), ece_bins=2)
    assert all(np.isfinite(value) for value in metrics.values())

    config = load_boosting_calibration_config("configs/boosting_calibration.json")
    evaluation = config["evaluation"]
    assert evaluation["horizons"] == [1, 3, 5, 10, 20]
    assert evaluation["calibration_methods"] == ["none", "sigmoid", "isotonic"]


def test_hybrid_config_registers_strict_short_and_regret_long_targets() -> None:
    config = load_hybrid_targets_config("configs/hybrid_targets.json")
    targets = config["evaluation"]["hybrid_targets"]

    assert len(targets) == 4
    assert all(target["short_horizon"] < target["long_horizon"] for target in targets)
    assert {target["long_tolerance_bps"] for target in targets} == {25.0, 50.0}
