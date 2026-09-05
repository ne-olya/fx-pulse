from __future__ import annotations

import numpy as np
import pandas as pd

from fxpulse.untested_hypotheses_experiment import (
    _first_improvement_steps,
    add_intraday_path_features,
    add_residual_factor_features,
    adwin_switch_scores,
    bootstrap_path_score,
)


def test_first_improvement_steps_uses_first_breach() -> None:
    prices = np.array([100.0, 99.9, 99.6, 100.0, 100.0])
    result = _first_improvement_steps(prices, horizon=2, tolerance_bps=25)
    assert result.tolist() == [2, 1, 3, 0, 0]


def test_intraday_features_are_exposed_one_session_late() -> None:
    raw = pd.DataFrame(
        {
            "dt_msk": ["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-02 10:00", "2026-01-02 11:00"],
            "secid": ["CNYRUB_TOM"] * 4,
            "open": [100.0, 99.0, 102.0, 103.0],
            "high": [101.0, 100.0, 103.0, 104.0],
            "low": [99.0, 98.0, 101.0, 102.0],
            "close": [100.0, 99.0, 103.0, 104.0],
        }
    )
    result = add_intraday_path_features(raw, secid="CNYRUB_TOM")
    assert result.loc[0, "intraday__observations"] != result.loc[0, "intraday__observations"]
    assert result.loc[1, "intraday__observations"] == 2
    assert result.loc[1, "intraday__realized_volatility"] > 0


def test_residual_beta_is_lagged() -> None:
    dates = pd.date_range("2026-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "corridor": ["AMD"] * 6,
            "market__cny_return_1": [1, 2, 3, 4, 5, 6],
            "market__gold_return_1": [2, 4, 6, 8, 100, 12],
        }
    )
    result = add_residual_factor_features(frame, assets=["gold"], window=4, minimum_observations=3)
    assert np.isclose(result.loc[4, "residual__gold_beta"], 2.0)
    assert result.loc[5, "residual__gold_beta"] > 2.0


def test_path_simulation_is_reproducible() -> None:
    history = np.tile(np.array([0.001, -0.001, 0.0005, 0.0002]), 50)
    kwargs = dict(
        horizon=3,
        tolerance_bps=25,
        paths=100,
        zero_drift=True,
        scale_bounds=(0.5, 2.0),
    )
    first = bootstrap_path_score(history, 0.001, rng=np.random.default_rng(7), **kwargs)
    second = bootstrap_path_score(history, 0.001, rng=np.random.default_rng(7), **kwargs)
    assert first == second


def test_adwin_switches_only_after_delayed_loss_change() -> None:
    target = np.array([1] * 100 + [0] * 100)
    baseline = np.full(200, 0.9)
    short = np.full(200, 0.2)
    switched, alarms = adwin_switch_scores(
        baseline,
        short,
        target,
        delay=5,
        delta=0.05,
        minimum_subwindow=20,
        maximum_window=100,
        alarm_hold=20,
    )
    assert alarms[:105].sum() == 0
    assert alarms.sum() >= 1
    assert np.any(switched == short)
