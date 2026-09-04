from __future__ import annotations

import pandas as pd

from fxpulse.experiment import _metrics, _weekly_cap, best_by_horizon, build_features, build_labeled_dataset


SETTINGS = {
    "horizons": [1, 4],
    "return_windows": [1, 4],
    "volatility_windows": [2, 4],
    "level_windows": [2, 4],
}


def _candles(rows: int = 12) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-05 10:59:59", periods=rows, freq="h")
    close = pd.Series([100 + index * 0.1 for index in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        }
    )


def test_feature_at_t_does_not_change_when_future_changes() -> None:
    original = _candles()
    changed = original.copy()
    changed.loc[9:, ["open", "high", "low", "close"]] *= 2

    left = build_features(original, SETTINGS, granularity="hourly").iloc[:9]
    right = build_features(changed, SETTINGS, granularity="hourly").iloc[:9]

    pd.testing.assert_frame_equal(left, right)


def test_hourly_dataset_contains_scripted_targets() -> None:
    dataset = build_labeled_dataset(_candles(), SETTINGS, granularity="hourly", tolerance_bps=10)

    assert {"target_good_now_1", "target_good_now_4", "future_regret_bps_4"} <= set(dataset)
    assert dataset["target_good_now_4"].notna().sum() == len(dataset) - 4


def test_weekly_cap_keeps_first_visible_signals() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-07 12:00", "2026-01-05 12:00", "2026-01-06 12:00"]),
            "score": [0.99, 0.80, 0.90],
        }
    )

    capped = _weekly_cap(candidates, 2)

    assert capped["timestamp"].dt.day.tolist() == [5, 6]


def test_frequency_uses_elapsed_test_weeks() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-15"]),
            "target": [True, False],
            "benefit": [1.0, -1.0],
            "regret": [0.0, 1.0],
        }
    )

    metrics = _metrics(base, base, base)

    assert metrics["test_duration_weeks"] == 2
    assert metrics["signals_per_week"] == 1


def test_best_by_horizon_uses_lift_among_selected_models() -> None:
    summary = pd.DataFrame(
        {
            "granularity": ["hourly", "hourly", "hourly"],
            "horizon": [4, 4, 4],
            "model": ["logistic", "xgboost", "random_forest"],
            "selection_status": ["selected", "selected", "diagnostic_only"],
            "signals": [20, 10, 30],
            "lift": [1.1, 1.2, 2.0],
            "benefit_fwd_bps": [3.0, 2.0, 10.0],
            "test_duration_weeks": [20.0, 10.0, 30.0],
            "signals_per_week": [1.0, 1.0, 1.0],
        }
    )

    folds = pd.DataFrame(
        {
            "granularity": ["hourly"] * 6,
            "horizon": [4] * 6,
            "model": ["logistic", "logistic", "xgboost", "xgboost", "random_forest", "random_forest"],
            "test_duration_weeks": [10.0] * 6,
        }
    )

    best = best_by_horizon(summary, folds)

    assert best.loc[0, "best_model"] == "xgboost"
    assert best.loc[0, "lift"] == 1.2
    assert best.loc[0, "test_duration_weeks"] == 20
    assert best.loc[0, "signals_per_week"] == 0.5
