from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from fxpulse.labeling import label_observations
from fxpulse.local_minimum_models import _fit_sklearn_model, _score_sklearn_model, local_minimum_features, run_local_minimum_models
from fxpulse.regret_benchmark import _product_policy_summary


def _panel(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "known_at": dates.tz_localize("Europe/Moscow") + pd.DateOffset(hours=23, minutes=59),
            "value_date": dates.date,
            "price": prices,
            "is_carried": False,
        }
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation": {
            "target_instrument_id": "target",
            "horizons": [1, 5],
            "candidate_policy": "all_observable_days",
            "tolerance_bps": 0.0,
            "min_training_observations": 100,
            "inner_selection_observations": 60,
            "minimum_candidate_training_observations": 20,
            "factor_lag_observations": 1,
            "target_return_windows": [1, 3],
            "factor_return_windows": [1],
            "volatility_windows": [5],
            "distance_to_min_windows": [5],
            "past_min_windows": [1, 5],
            "models": ["ridge_logistic"],
            "ridge_l2": 1.0,
            "max_iterations": 50,
            "boosted_stumps": {"iterations": 2, "learning_rate": 0.1, "min_leaf": 5, "quantiles": [0.5]},
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "min_child_weight": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
            },
            "sklearn": {
                "random_forest": {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 2, "max_features": 1.0},
                "extra_trees": {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 2, "max_features": 1.0},
                "adaboost": {"n_estimators": 2, "learning_rate": 0.1, "min_samples_leaf": 2},
                "gradient_boosting": {"n_estimators": 2, "learning_rate": 0.1, "max_depth": 2, "min_samples_leaf": 2, "subsample": 1.0},
                "svc_rbf": {"C": 0.5, "gamma": "scale"},
            },
            "score_quantiles": [0.5],
            "minimum_selection_signals": 1,
            "minimum_signals_per_week": 0.1,
            "maximum_signals_per_week": 2.0,
            "selection_minimum_lift": 0.0,
            "promotion_minimum_lift": 0.0,
            "minimum_selection_benefit_bps": -10_000.0,
            "no_signal_fallback": "do_not_send",
        },
    }


def _prices(periods: int = 420) -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=periods, freq="B")
    target = [10.0]
    factor = [100.0]
    for position in range(1, periods):
        target.append(target[-1] * (0.998 if position % 9 == 0 else 1.001))
        factor.append(factor[-1] * (1.004 if position % 11 else 0.993))
    return pd.DataFrame({"target": target, "factor": factor}, index=dates)


def test_future_min_marks_only_retrospective_interest_points() -> None:
    labels = label_observations(_panel([10.0, 9.0, 11.0, 8.0, 7.0, 6.0, 14.0, 15.0, 16.0, 17.0, 18.0]), horizon=5)

    assert labels["position"].tolist() == [0, 1, 2, 3, 4, 5]
    assert labels["hit_favorable"].tolist() == [False, False, False, False, False, True]
    # The observed trailing minimum is diagnostic only; it is not the target
    # population of the default all-days benchmark.
    assert labels["past_min"].tolist() == [False, False, False, False, True, True]


def test_local_minimum_features_do_not_change_when_future_prices_are_appended() -> None:
    prices = _prices()
    config = _config()
    cut = 300

    before = local_minimum_features(prices.iloc[:cut], config=config)
    after = local_minimum_features(prices, config=config).iloc[:cut]

    pd.testing.assert_frame_equal(before, after)
    assert pd.api.types.is_float_dtype(before["target__is_past_min_5"])


def test_all_days_runner_uses_all_labelled_days_as_its_baseline(tmp_path) -> None:
    prices = _prices()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    series = []
    for instrument_id in prices.columns:
        path = snapshot / f"{instrument_id}.csv"
        pd.DataFrame(
            {"trade_date": prices.index.date, "instrument_id": instrument_id, "close": prices[instrument_id]}
        ).to_csv(path, index=False)
        series.append({"instrument_id": instrument_id, "path": path.name})
    (snapshot / "manifest.json").write_text(
        json.dumps({"date_from": "2020-01-02", "date_to": "2021-08-12", "series": series}), encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    run_local_minimum_models(snapshot_dir=snapshot, config_path=config_path, artifact_dir=artifacts)

    folds = pd.read_csv(artifacts / "folds.csv")
    assert set(folds["candidate_policy"].dropna()) == {"all_observable_days"}
    selected = folds.dropna(subset=["test_candidate_count"])
    assert selected["test_candidate_count"].gt(0).all()


def test_rbf_svc_uses_margin_without_deprecated_probability_switch() -> None:
    features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype="float64")
    target = np.array([0, 0, 1, 1], dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = _fit_sklearn_model("svc_rbf", features, target, settings={"C": 0.5, "gamma": "scale"})
    scores = _score_sklearn_model("svc_rbf", model, features)

    assert np.isfinite(scores).all()
    assert ((scores > 0) & (scores < 1)).all()


def test_product_summary_pools_per_fold_winners_instead_of_selecting_outer_best_model() -> None:
    panel = _panel([10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 10.0])
    labels = label_observations(panel, horizon=1)
    dates = panel["value_date"].astype(str).tolist()
    folds = pd.DataFrame(
        [
            {
                "fold": "fold_a",
                "horizon_observations": 1,
                "candidate_policy": "all_observable_days",
                "selection_status": "promoted",
                "model": "ridge_logistic",
                "test_start": dates[1],
                "test_end": dates[2],
                "test_lift": 1.0,
                "test_dispatched_signal_count": 1,
                "test_regret_mean_bps": 0.0,
                "test_regret_p90_bps": 0.0,
            },
            {
                "fold": "fold_b",
                "horizon_observations": 1,
                "candidate_policy": "all_observable_days",
                "selection_status": "promoted",
                "model": "xgboost",
                "test_start": dates[4],
                "test_end": dates[5],
                "test_lift": 2.0,
                "test_dispatched_signal_count": 1,
                "test_regret_mean_bps": 0.0,
                "test_regret_p90_bps": 0.0,
            },
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "fold": "fold_a",
                "horizon_observations": 1,
                "candidate_policy": "all_observable_days",
                "selection_status": "promoted",
                "communication_allowed": True,
                "model": "ridge_logistic",
                "value_date": dates[1],
            },
            {
                "fold": "fold_b",
                "horizon_observations": 1,
                "candidate_policy": "all_observable_days",
                "selection_status": "promoted",
                "communication_allowed": True,
                "model": "xgboost",
                "value_date": dates[5],
            },
        ]
    )

    product, foldwise, bootstrap = _product_policy_summary(
        panel=panel,
        labels=labels,
        folds=folds,
        signals=signals,
        tolerance_bps=0.0,
        horizon=1,
        candidate_policy="all_observable_days",
        bootstrap_samples=20,
    )

    assert int(product.loc[0, "promoted_fold_count"]) == 2
    assert int(product.loc[0, "signal_count"]) == 2
    assert int(product.loc[0, "eligible_candidate_count"]) == 4
    assert float(product.loc[0, "lift"]) == 4 / 3
    assert int(foldwise.loc[0, "fold_count"]) == 2
    assert int(bootstrap.loc[0, "usable_samples"]) > 0
