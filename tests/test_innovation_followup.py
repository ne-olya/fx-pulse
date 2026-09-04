import numpy as np
import pandas as pd

from fxpulse.innovation_followup import (
    EXPECTED_VARIANTS,
    _deoverlap_signals,
    causal_percentile,
    load_config,
    moving_block_bootstrap,
)


def test_followup_config_matches_implementation() -> None:
    config = load_config()
    assert config["variants"] == EXPECTED_VARIANTS
    assert config["bootstrap"]["moving_block_weeks"] == 4


def test_causal_percentile_ignores_future_values() -> None:
    first = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    changed_future = pd.Series([1.0, 2.0, 3.0, 400.0, 500.0])
    rank_first = causal_percentile(first, lookback=3, minimum_history=2)
    rank_changed = causal_percentile(changed_future, lookback=3, minimum_history=2)
    np.testing.assert_allclose(rank_first.iloc[:3], rank_changed.iloc[:3], equal_nan=True)


def test_deoverlap_uses_calendar_distance() -> None:
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-06"]),
            "target": [1, 0, 1],
        }
    )
    result = _deoverlap_signals(signals, cooldown_days=5)
    assert result["timestamp"].dt.day.tolist() == [1, 6]


def test_moving_block_bootstrap_is_deterministic() -> None:
    weekly = pd.DataFrame(
        {
            "signal_count": [1, 1, 1, 1, 1, 1],
            "signal_hits": [1, 1, 0, 1, 0, 1],
            "matched_expected_hits": [0.5] * 6,
            "base_count": [5] * 6,
            "base_hits": [2, 3, 2, 3, 2, 3],
        }
    )
    first = moving_block_bootstrap(weekly, samples=200, block_weeks=2, seed=11)
    second = moving_block_bootstrap(weekly, samples=200, block_weeks=2, seed=11)
    assert first == second
    assert first["block_raw_lift_low"] <= first["block_raw_lift_high"]
