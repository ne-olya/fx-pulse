from __future__ import annotations

import json

import pandas as pd

from fxpulse.interpretable_models import cross_market_features
from fxpulse.rule_selection import (
    _target_panel,
    quarterly_folds,
    rule_candidates,
    run_rule_selection,
)


def _config(*, min_training_observations: int = 100) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation": {
            "target_instrument_id": "target",
            "horizon_observations": 5,
            "min_training_observations": min_training_observations,
            "test_period": "quarter",
            "factor_lag_observations": 1,
            "minimum_training_signals": 1,
            "minimum_signals_per_week": 0.1,
            "maximum_signals_per_week": 2.0,
            "selection_minimum_lift": 0.0,
            "promotion_minimum_lift": 0.0,
            "minimum_training_benefit_bps": -10_000.0,
            "no_signal_fallback": "do_not_send",
        },
        "rule_library": {
            "kind": "lagged_factor_return_quantile",
            "return_windows": [1],
            "comparison_lookback_observations": 20,
            "tail_quantiles": [0.2],
            "tails": ["lower", "upper"],
            "directions": ["favorable"],
        },
    }


def _prices(periods: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=periods, freq="B")
    target = [10.0]
    factor = [100.0]
    for position in range(1, periods):
        target.append(target[-1] * (1.001 if position % 7 else 1.0001))
        factor.append(factor[-1] * (0.99 if position % 11 == 0 else 1.001))
    return pd.DataFrame({"target": target, "factor": factor}, index=dates)


def test_rule_candidates_do_not_change_when_future_prices_are_appended() -> None:
    prices = _prices()
    config = _config()
    cut = 350

    before = rule_candidates(prices.iloc[:cut], config=config)
    after = rule_candidates(prices, config=config).query("position < @cut").reset_index(drop=True)

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)


def test_scorecard_features_do_not_change_when_future_prices_are_appended() -> None:
    prices = _prices()
    config = {"evaluation": {"target_instrument_id": "target", "factor_lag_observations": 1, "return_windows": [1, 3]}}
    cut = 350

    before = cross_market_features(prices.iloc[:cut], config=config)
    after = cross_market_features(prices, config=config).iloc[:cut]

    pd.testing.assert_frame_equal(before, after)


def test_quarterly_folds_purge_horizon_from_each_train_prefix() -> None:
    prices = _prices()
    panel = _target_panel(prices, "target")
    folds = quarterly_folds(panel, horizon=5, min_training_observations=100)

    assert folds
    assert all(fold.train_positions[-1] + fold.purged_tail_observations < fold.test_positions[0] for fold in folds)


def test_runner_uses_all_manifest_factors_and_never_exceeds_weekly_cap(tmp_path) -> None:
    prices = _prices()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    series = []
    for instrument_id in prices.columns:
        path = snapshot / f"{instrument_id}.csv"
        pd.DataFrame(
            {
                "trade_date": prices.index.date,
                "instrument_id": instrument_id,
                "close": prices[instrument_id],
            }
        ).to_csv(path, index=False)
        series.append({"instrument_id": instrument_id, "path": path.name})
    (snapshot / "manifest.json").write_text(
        json.dumps({"date_from": "2020-01-02", "date_to": "2021-12-01", "series": series}), encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    meta = run_rule_selection(snapshot_dir=snapshot, config_path=config_path, artifact_dir=artifacts)

    assert meta["factor_instruments"] == ["factor"]
    signals = pd.read_csv(artifacts / "signals.csv")
    if not signals.empty:
        weeks = pd.to_datetime(signals["timestamp"]).dt.to_period("W")
        assert signals.groupby(["fold", weeks]).size().max() <= 2
