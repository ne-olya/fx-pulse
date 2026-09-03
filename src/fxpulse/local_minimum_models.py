"""Leak-free benchmark for recommendations around future-minimum labels.

A future minimum is deliberately a *retrospective* outcome: today's CNY/RUB
close has no lower close in the next ``h`` observations.  The primary policy
scores every observable day.  A trailing minimum is available at decision time
and may be used as a diagnostic feature, but is not a mandatory product gate.
Models, thresholds, and the weekly cap are selected chronologically inside each
outer expanding fold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Literal

import numpy as np
import pandas as pd

from fxpulse.interpretable_models import _fit_ridge_logistic, _score, _transform
from fxpulse.labeling import HORIZONS, evaluate_positions, label_observations
from fxpulse.rule_selection import _target_panel, _weekly_cap, load_universe_prices, quarterly_folds, resolve_snapshot


ModelKind = Literal[
    "ridge_logistic",
    "gradient_boosted_stumps",
    "xgboost",
    "random_forest",
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "svc_rbf",
]

FOLD_COLUMNS = (
    "fold",
    "horizon_observations",
    "candidate_policy",
    "test_start",
    "test_end",
    "fit_observations",
    "selection_observations",
    "outer_training_observations",
    "purged_tail_observations",
    "selection_status",
    "model",
    "score_quantile",
    "score_threshold",
    "selection_candidate_count",
    "selection_signal_count",
    "selection_signals_per_week",
    "selection_hit_rate",
    "selection_candidate_baseline_hit_rate",
    "selection_lift",
    "selection_benefit_fwd_bps",
    "test_candidate_count",
    "test_candidate_signal_count",
    "test_dispatched_signal_count",
    "test_signals_per_week",
    "test_hit_rate",
    "test_candidate_baseline_hit_rate",
    "test_lift",
    "test_benefit_fwd_bps",
    "test_frequency_in_policy",
)
SIGNAL_COLUMNS = (
    "fold",
    "horizon_observations",
    "candidate_policy",
    "selection_status",
    "timestamp",
    "value_date",
    "model",
    "score",
    "score_threshold",
    "communication_allowed",
    "details",
)
ATTRIBUTION_COLUMNS = (
    "fold",
    "horizon_observations",
    "candidate_policy",
    "selection_status",
    "model",
    "feature",
    "feature_family",
    "importance",
    "signed_effect",
)
SUMMARY_COLUMNS = (
    "horizon_observations",
    "candidate_policy",
    "model",
    "selection_status",
    "communication_allowed",
    "folds",
    "signal_count",
    "eligible_candidate_count",
    "hit_rate",
    "candidate_baseline_hit_rate",
    "lift",
    "benefit_sym_bps",
    "benefit_fwd_bps",
    "benefit_fwd_newey_west_t",
    "signals_per_week",
    "signals_per_month",
    "cluster_share",
    "interval_cv",
    "weeks_with_signal",
    "max_signals_in_fold_week",
)


@dataclass(frozen=True)
class Stump:
    feature_index: int
    threshold: float
    left_update: float
    right_update: float
    gain: float


@dataclass(frozen=True)
class BoostedStumps:
    intercept: float
    learning_rate: float
    stumps: tuple[Stump, ...]


@dataclass
class FittedModel:
    kind: ModelKind
    feature_columns: list[str]
    design_columns: list[str]
    median: pd.Series
    mean: pd.Series
    scale: pd.Series
    model: Any
    train_scores: np.ndarray
    train_positions: np.ndarray


def local_minimum_config_sha256(path: Path | str = Path("configs/local_minimum_models.json")) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_local_minimum_config(path: Path | str = Path("configs/local_minimum_models.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("evaluation"), dict):
        raise ValueError("Only local-minimum schema_version 1 with evaluation is supported")
    evaluation = config["evaluation"]
    required = {
        "target_instrument_id",
        "horizons",
        "candidate_policy",
        "tolerance_bps",
        "min_training_observations",
        "inner_selection_observations",
        "minimum_candidate_training_observations",
        "factor_lag_observations",
        "target_return_windows",
        "factor_return_windows",
        "volatility_windows",
        "distance_to_min_windows",
        "past_min_windows",
        "models",
        "ridge_l2",
        "max_iterations",
        "boosted_stumps",
        "xgboost",
        "sklearn",
        "score_quantiles",
        "minimum_selection_signals",
        "minimum_signals_per_week",
        "maximum_signals_per_week",
        "selection_minimum_lift",
        "promotion_minimum_lift",
        "minimum_selection_benefit_bps",
        "no_signal_fallback",
    }
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks: {', '.join(sorted(missing))}")
    if not evaluation["horizons"] or not all(value in HORIZONS for value in evaluation["horizons"]):
        raise ValueError(f"horizons must be a non-empty subset of {HORIZONS}")
    if len(set(evaluation["horizons"])) != len(evaluation["horizons"]):
        raise ValueError("horizons must not contain duplicates")
    if evaluation["candidate_policy"] not in {"all_observable_days", "past_min_gate"}:
        raise ValueError("candidate_policy must be all_observable_days or past_min_gate")
    for name in ("min_training_observations", "inner_selection_observations", "minimum_candidate_training_observations", "factor_lag_observations", "minimum_selection_signals", "max_iterations"):
        if not isinstance(evaluation[name], int) or evaluation[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("target_return_windows", "factor_return_windows", "volatility_windows", "distance_to_min_windows", "past_min_windows"):
        if not evaluation[name] or not all(isinstance(value, int) and value > 0 for value in evaluation[name]):
            raise ValueError(f"{name} must contain positive integers")
    model_kinds = {
        "ridge_logistic",
        "gradient_boosted_stumps",
        "xgboost",
        "random_forest",
        "extra_trees",
        "adaboost",
        "gradient_boosting",
        "svc_rbf",
    }
    if set(evaluation["models"]) - model_kinds or not evaluation["models"]:
        raise ValueError("models contains an unsupported model kind")
    if not evaluation["score_quantiles"] or not all(
        isinstance(value, int | float) and 0 < value < 1 for value in evaluation["score_quantiles"]
    ):
        raise ValueError("score_quantiles must be in (0, 1)")
    if not isinstance(evaluation["tolerance_bps"], int | float) or evaluation["tolerance_bps"] < 0:
        raise ValueError("tolerance_bps must be non-negative")
    if evaluation["minimum_signals_per_week"] <= 0 or evaluation["maximum_signals_per_week"] < evaluation["minimum_signals_per_week"]:
        raise ValueError("invalid communication frequency range")
    boost = evaluation["boosted_stumps"]
    if not isinstance(boost, dict) or not all(key in boost for key in ("iterations", "learning_rate", "min_leaf", "quantiles")):
        raise ValueError("boosted_stumps needs iterations, learning_rate, min_leaf and quantiles")
    if not isinstance(boost["iterations"], int) or boost["iterations"] <= 0 or not isinstance(boost["min_leaf"], int) or boost["min_leaf"] <= 0:
        raise ValueError("boosted_stumps iterations and min_leaf must be positive integers")
    if not isinstance(boost["learning_rate"], int | float) or not 0 < boost["learning_rate"] <= 1:
        raise ValueError("boosted_stumps learning_rate must be in (0, 1]")
    if not boost["quantiles"] or not all(isinstance(value, int | float) and 0 < value < 1 for value in boost["quantiles"]):
        raise ValueError("boosted_stumps quantiles must be in (0, 1)")
    xgboost = evaluation["xgboost"]
    xgboost_fields = {"n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"}
    if not isinstance(xgboost, dict) or xgboost_fields - set(xgboost):
        missing_fields = xgboost_fields - set(xgboost) if isinstance(xgboost, dict) else xgboost_fields
        raise ValueError(f"xgboost lacks: {', '.join(sorted(missing_fields))}")
    for name in ("n_estimators", "max_depth", "min_child_weight"):
        if not isinstance(xgboost[name], int) or xgboost[name] <= 0:
            raise ValueError(f"xgboost {name} must be a positive integer")
    for name in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda"):
        if not isinstance(xgboost[name], int | float) or xgboost[name] <= 0:
            raise ValueError(f"xgboost {name} must be positive")
    sklearn = evaluation["sklearn"]
    sklearn_fields = {"random_forest", "extra_trees", "adaboost", "gradient_boosting", "svc_rbf"}
    if not isinstance(sklearn, dict) or sklearn_fields - set(sklearn):
        missing_fields = sklearn_fields - set(sklearn) if isinstance(sklearn, dict) else sklearn_fields
        raise ValueError(f"sklearn lacks: {', '.join(sorted(missing_fields))}")
    if not all(isinstance(sklearn[name], dict) for name in sklearn_fields):
        raise ValueError("each sklearn model needs a settings object")
    if evaluation["no_signal_fallback"] != "do_not_send":
        raise ValueError("Only no_signal_fallback=do_not_send is supported")
    return config


def local_minimum_features(prices: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    """Build current-target and strictly lagged external-market features."""

    evaluation = config["evaluation"]
    target = str(evaluation["target_instrument_id"])
    if target not in prices:
        raise ValueError(f"prices lacks target {target}")
    target_price = pd.to_numeric(prices[target], errors="coerce")
    target_return = target_price.pct_change(fill_method=None)
    features = pd.DataFrame(index=prices.index)
    for window in evaluation["target_return_windows"]:
        features[f"target__return_{window}"] = target_price.pct_change(int(window), fill_method=None)
    for window in evaluation["volatility_windows"]:
        features[f"target__volatility_{window}"] = target_return.rolling(int(window), min_periods=int(window)).std()
    for window in evaluation["distance_to_min_windows"]:
        rolling_min = target_price.rolling(int(window), min_periods=int(window)).min()
        features[f"target__distance_to_min_{window}"] = target_price / rolling_min - 1
    for window in evaluation["past_min_windows"]:
        rolling_min = target_price.rolling(int(window), min_periods=int(window)).min()
        features[f"target__is_past_min_{window}"] = (
            target_price.le(rolling_min).where(rolling_min.notna()).astype("float64")
        )
    lag = int(evaluation["factor_lag_observations"])
    for factor in sorted(column for column in prices.columns if column != target):
        factor_price = pd.to_numeric(prices[factor], errors="coerce")
        for window in evaluation["factor_return_windows"]:
            features[f"factor__{factor}__return_{window}"] = factor_price.pct_change(int(window), fill_method=None).shift(lag)
    return features.replace([np.inf, -np.inf], np.nan)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -35, 35)
    return 1 / (1 + np.exp(-clipped))


def _fit_boosted_stumps(features: np.ndarray, target: np.ndarray, *, settings: dict[str, Any]) -> BoostedStumps:
    """Fit deterministic logistic gradient boosting with shallow numeric stumps."""

    if len(features) == 0 or len(features) != len(target) or len(np.unique(target)) < 2:
        raise ValueError("boosted stumps require non-empty features and both target classes")
    prior = float(np.clip(np.mean(target), 1e-5, 1 - 1e-5))
    intercept = math.log(prior / (1 - prior))
    logits = np.full(len(target), intercept, dtype="float64")
    min_leaf = int(settings["min_leaf"])
    learning_rate = float(settings["learning_rate"])
    quantiles = np.asarray(settings["quantiles"], dtype="float64")
    stumps: list[Stump] = []
    for _ in range(int(settings["iterations"])):
        residual = target - _sigmoid(logits)
        total = float(np.square(residual).sum())
        best: Stump | None = None
        for feature_index in range(features.shape[1]):
            column = features[:, feature_index]
            thresholds = np.unique(np.quantile(column, quantiles))
            for threshold in thresholds:
                left = column <= threshold
                left_count = int(left.sum())
                right_count = len(column) - left_count
                if left_count < min_leaf or right_count < min_leaf:
                    continue
                left_sum = float(residual[left].sum())
                right_sum = float(residual[~left].sum())
                gain = left_sum * left_sum / left_count + right_sum * right_sum / right_count - total / len(column)
                candidate = Stump(
                    feature_index=feature_index,
                    threshold=float(threshold),
                    left_update=left_sum / left_count,
                    right_update=right_sum / right_count,
                    gain=float(gain),
                )
                if best is None or candidate.gain > best.gain + 1e-14:
                    best = candidate
        if best is None or best.gain <= 1e-14:
            break
        update = np.where(features[:, best.feature_index] <= best.threshold, best.left_update, best.right_update)
        logits += learning_rate * update
        stumps.append(best)
    return BoostedStumps(intercept=intercept, learning_rate=learning_rate, stumps=tuple(stumps))


def _score_boosted_stumps(features: np.ndarray, model: BoostedStumps) -> np.ndarray:
    logits = np.full(len(features), model.intercept, dtype="float64")
    for stump in model.stumps:
        logits += model.learning_rate * np.where(
            features[:, stump.feature_index] <= stump.threshold, stump.left_update, stump.right_update
        )
    return _sigmoid(logits)


def _fit_sklearn_model(kind: ModelKind, features: np.ndarray, target: np.ndarray, *, settings: dict[str, Any]) -> Any:
    """Fit fixed, compact classical ML baselines with one deterministic seed."""

    from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    settings = dict(settings)
    seed = 20260903
    if kind == "random_forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=1, **settings)
    elif kind == "extra_trees":
        model = ExtraTreesClassifier(random_state=seed, n_jobs=1, **settings)
    elif kind == "adaboost":
        stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=int(settings.pop("min_samples_leaf")), random_state=seed)
        model = AdaBoostClassifier(estimator=stump, random_state=seed, **settings)
    elif kind == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed, **settings)
    elif kind == "svc_rbf":
        # sklearn 1.9 deprecated the parameter itself.  Leaving it at the
        # default keeps probability calibration off; scoring uses the margin.
        model = SVC(kernel="rbf", **settings)
    else:
        raise ValueError(f"Unsupported sklearn model kind {kind}")
    model.fit(features, target)
    return model


def _score_sklearn_model(kind: ModelKind, model: Any, features: np.ndarray) -> np.ndarray:
    if kind == "svc_rbf":
        # RBF SVC has no time-aware probability calibration here. Its monotonic
        # sigmoid-transformed margin is used only for pre-registered ranking.
        return _sigmoid(np.asarray(model.decision_function(features), dtype="float64"))
    return np.asarray(model.predict_proba(features)[:, 1], dtype="float64")


def _feature_family(feature: str) -> str:
    base = feature.removesuffix("__missing")
    if base.startswith("target__return_"):
        return "динамика CNY/RUB"
    if base.startswith("target__volatility_"):
        return "волатильность CNY/RUB"
    if base.startswith("target__distance_to_min_"):
        return "положение CNY/RUB в диапазоне"
    if base.startswith("target__is_past_min_"):
        return "минимум CNY/RUB в прошлом окне"
    if base.startswith("factor__"):
        return "межрыночные факторы"
    return "прочее"


def _candidate_positions(
    labels: pd.DataFrame,
    positions: tuple[int, ...] | list[int],
    *,
    candidate_policy: str,
) -> tuple[int, ...]:
    label_positions = set(labels["position"])
    if candidate_policy == "all_observable_days":
        return tuple(position for position in positions if position in label_positions)
    if candidate_policy == "past_min_gate":
        eligible = labels.set_index("position")["past_min"]
        return tuple(position for position in positions if bool(eligible.get(position, False)))
    raise ValueError(f"Unknown candidate policy {candidate_policy}")


def _fit_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    positions: tuple[int, ...],
    kind: ModelKind,
    evaluation: dict[str, Any],
    candidate_policy: str,
) -> FittedModel:
    candidate_positions = _candidate_positions(labels, positions, candidate_policy=candidate_policy)
    if len(candidate_positions) < int(evaluation["minimum_candidate_training_observations"]):
        raise ValueError("not enough local-minimum candidates for model fit")
    feature_columns = list(features.columns)
    labels_by_position = labels.set_index("position")["hit_favorable"]
    train = features.loc[list(candidate_positions), feature_columns]
    target = labels_by_position.loc[list(candidate_positions)].astype(int).to_numpy()
    if len(np.unique(target)) < 2:
        raise ValueError("local-minimum train set has one target class")
    matrix, median, mean, scale, design_columns = _transform(train)
    if not np.isfinite(matrix).all():
        raise ValueError("local-minimum model received a non-finite feature")
    if kind == "ridge_logistic":
        model = _fit_ridge_logistic(
            matrix,
            target,
            l2=float(evaluation["ridge_l2"]),
            max_iterations=int(evaluation["max_iterations"]),
        )
        scores = _score(matrix, model)
    elif kind == "gradient_boosted_stumps":
        model = _fit_boosted_stumps(matrix, target, settings=evaluation["boosted_stumps"])
        scores = _score_boosted_stumps(matrix, model)
    elif kind == "xgboost":
        from xgboost import XGBClassifier

        settings = evaluation["xgboost"]
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(settings["n_estimators"]),
            max_depth=int(settings["max_depth"]),
            learning_rate=float(settings["learning_rate"]),
            min_child_weight=int(settings["min_child_weight"]),
            subsample=float(settings["subsample"]),
            colsample_bytree=float(settings["colsample_bytree"]),
            reg_lambda=float(settings["reg_lambda"]),
            random_state=20260903,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        model.fit(matrix, target)
        scores = model.predict_proba(matrix)[:, 1]
    else:
        model = _fit_sklearn_model(kind, matrix, target, settings=evaluation["sklearn"][kind])
        scores = _score_sklearn_model(kind, model, matrix)
    return FittedModel(
        kind=kind,
        feature_columns=feature_columns,
        design_columns=design_columns,
        median=median,
        mean=mean,
        scale=scale,
        model=model,
        train_scores=scores,
        train_positions=np.asarray(candidate_positions, dtype=int),
    )


def _predict(model: FittedModel, features: pd.DataFrame, positions: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    frame = features.loc[list(positions), model.feature_columns]
    matrix, _, _, _, _ = _transform(frame, median=model.median, mean=model.mean, scale=model.scale)
    if not np.isfinite(matrix).all():
        raise ValueError("local-minimum score received a non-finite feature")
    if model.kind == "ridge_logistic":
        return _score(matrix, model.model), matrix
    if model.kind == "gradient_boosted_stumps":
        return _score_boosted_stumps(matrix, model.model), matrix
    if model.kind == "xgboost":
        return model.model.predict_proba(matrix)[:, 1], matrix
    return _score_sklearn_model(model.kind, model.model, matrix), matrix


def _metrics(
    candidates: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    candidate_base_positions: tuple[int, ...],
    horizon: int,
    weekly_cap: float,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    capped = _weekly_cap(candidates, panel, maximum_signals_per_week=weekly_cap)
    metrics = evaluate_positions(
        panel,
        capped["position"].tolist(),
        direction="favorable",
        horizon=horizon,
        base_positions=candidate_base_positions,
        labels=labels,
    )
    return capped, metrics


def _eligible(metrics: dict[str, float | int | None], evaluation: dict[str, Any]) -> bool:
    frequency = metrics["signals_per_week"]
    return bool(
        metrics["signal_count"] >= int(evaluation["minimum_selection_signals"])
        and frequency is not None
        and float(evaluation["minimum_signals_per_week"]) <= float(frequency) <= float(evaluation["maximum_signals_per_week"])
        and metrics["lift"] is not None
        and float(metrics["lift"]) >= float(evaluation["selection_minimum_lift"])
        and metrics["benefit_fwd_bps"] is not None
        and float(metrics["benefit_fwd_bps"]) > float(evaluation["minimum_selection_benefit_bps"])
    )


def _inner_split(train_positions: tuple[int, ...], *, horizon: int, selection_observations: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if len(train_positions) <= selection_observations + horizon:
        return None
    selection = tuple(train_positions[-selection_observations:])
    fit = tuple(train_positions[: -selection_observations - horizon])
    return (fit, selection) if fit else None


def _local_contributions(model: FittedModel, matrix_row: np.ndarray) -> list[dict[str, float | str]]:
    values: dict[str, float] = {}
    if model.kind == "ridge_logistic":
        assert isinstance(model.model, np.ndarray)
        for feature, coefficient, value in zip(model.design_columns, model.model[1:], matrix_row, strict=True):
            family = _feature_family(feature)
            values[family] = values.get(family, 0.0) + float(coefficient * value)
    elif model.kind == "gradient_boosted_stumps":
        assert isinstance(model.model, BoostedStumps)
        for stump in model.model.stumps:
            family = _feature_family(model.design_columns[stump.feature_index])
            update = stump.left_update if matrix_row[stump.feature_index] <= stump.threshold else stump.right_update
            values[family] = values.get(family, 0.0) + float(model.model.learning_rate * update)
    elif model.kind == "xgboost":
        import xgboost as xgb

        contributions = model.model.get_booster().predict(
            xgb.DMatrix(matrix_row.reshape(1, -1)), pred_contribs=True
        )[0]
        for feature, value in zip(model.design_columns, contributions[:-1], strict=True):
            family = _feature_family(feature)
            values[family] = values.get(family, 0.0) + float(value)
    else:
        importances = getattr(model.model, "feature_importances_", None)
        if importances is None:
            return []
        for feature, importance in zip(model.design_columns, importances, strict=True):
            family = _feature_family(feature)
            values[family] = values.get(family, 0.0) + float(importance)
    return [
        {"feature_family": family, "logit_contribution": round(value, 6)}
        for family, value in sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))[:3]
    ]


def _global_attributions(
    model: FittedModel,
    *,
    fold: str,
    horizon: int,
    status: str,
    candidate_policy: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if model.kind == "ridge_logistic":
        assert isinstance(model.model, np.ndarray)
        for feature, coefficient in zip(model.design_columns, model.model[1:], strict=True):
            rows.append(
                {
                    "fold": fold,
                    "horizon_observations": horizon,
                    "candidate_policy": candidate_policy,
                    "selection_status": status,
                    "model": model.kind,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "importance": abs(float(coefficient)),
                    "signed_effect": float(coefficient),
                }
            )
    elif model.kind == "gradient_boosted_stumps":
        assert isinstance(model.model, BoostedStumps)
        for stump in model.model.stumps:
            feature = model.design_columns[stump.feature_index]
            rows.append(
                {
                    "fold": fold,
                    "horizon_observations": horizon,
                    "candidate_policy": candidate_policy,
                    "selection_status": status,
                    "model": model.kind,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "importance": float(max(stump.gain, 0.0)),
                    "signed_effect": float(model.model.learning_rate * (stump.right_update - stump.left_update)),
                }
            )
    elif model.kind == "xgboost":
        for feature, importance in zip(model.design_columns, model.model.feature_importances_, strict=True):
            rows.append(
                {
                    "fold": fold,
                    "horizon_observations": horizon,
                    "candidate_policy": candidate_policy,
                    "selection_status": status,
                    "model": model.kind,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "importance": float(importance),
                    "signed_effect": None,
                }
            )
    else:
        importances = getattr(model.model, "feature_importances_", None)
        if importances is None:
            return rows
        for feature, importance in zip(model.design_columns, importances, strict=True):
            rows.append(
                {
                    "fold": fold,
                    "horizon_observations": horizon,
                    "candidate_policy": candidate_policy,
                    "selection_status": status,
                    "model": model.kind,
                    "feature": feature,
                    "feature_family": _feature_family(feature),
                    "importance": float(importance),
                    "signed_effect": None,
                }
            )
    return rows


def _summary_rows(
    signals: pd.DataFrame,
    *,
    folds_by_name: dict[str, Any],
    candidate_positions: dict[tuple[str, int, str], tuple[int, ...]],
    panel: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    position_by_date = {str(value): position for position, value in enumerate(panel["value_date"].astype(str))}
    rows: list[dict[str, object]] = []
    groups = ["horizon_observations", "candidate_policy", "model", "selection_status", "communication_allowed"]
    for values, group in signals.groupby(groups, sort=True):
        horizon, candidate_policy, model, status, allowed = values
        group_folds = sorted(group["fold"].unique())
        base = tuple(position for fold in group_folds for position in candidate_positions[(fold, int(horizon), str(candidate_policy))])
        positions = [position_by_date[date] for date in group["value_date"] if date in position_by_date]
        metrics = evaluate_positions(
            panel,
            positions,
            direction="favorable",
            horizon=int(horizon),
            base_positions=base,
            labels=labels_by_horizon[int(horizon)],
        )
        timestamps = pd.to_datetime(group["timestamp"], errors="raise")
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_localize(None)
        weeks = timestamps.dt.to_period("W")
        per_fold_week = group.assign(_week=weeks).groupby(["fold", "_week"], sort=False).size()
        rows.append(
            {
                "horizon_observations": int(horizon),
                "candidate_policy": candidate_policy,
                "model": model,
                "selection_status": status,
                "communication_allowed": bool(allowed),
                "folds": len(group_folds),
                "eligible_candidate_count": len(base),
                "candidate_baseline_hit_rate": metrics.pop("baseline_hit_rate"),
                **metrics,
                "weeks_with_signal": int(weeks.nunique()),
                "max_signals_in_fold_week": int(per_fold_week.max()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def run_local_minimum_models(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/local_minimum_models.json"),
    artifact_dir: Path | str = Path("artifacts/local_minimum_models"),
    models: tuple[ModelKind, ...] | None = None,
) -> dict[str, Any]:
    """Select and OOT-test local-minimum models for every configured horizon."""

    config = load_local_minimum_config(config_path)
    evaluation = config["evaluation"]
    candidate_policy = str(evaluation["candidate_policy"])
    selected_models = tuple(models or tuple(evaluation["models"]))
    if not selected_models or set(selected_models) - set(evaluation["models"]):
        raise ValueError("models override must be a non-empty subset of config evaluation.models")
    snapshot = resolve_snapshot(snapshot_dir)
    target = str(evaluation["target_instrument_id"])
    prices = load_universe_prices(snapshot, target_instrument_id=target)
    panel = _target_panel(prices, target).reset_index(drop=True)
    features = local_minimum_features(prices, config=config).reset_index(drop=True)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    candidate_positions: dict[tuple[str, int], tuple[int, ...]] = {}
    labels_by_horizon: dict[int, pd.DataFrame] = {}
    folds_by_name: dict[str, Any] = {}
    for horizon_raw in evaluation["horizons"]:
        horizon = int(horizon_raw)
        labels = label_observations(panel, horizon, tolerance_bps=float(evaluation["tolerance_bps"]))
        labels_by_horizon[horizon] = labels
        folds = quarterly_folds(panel, horizon=horizon, min_training_observations=int(evaluation["min_training_observations"]))
        for fold in folds:
            folds_by_name[fold.name] = fold
            candidate_positions[(fold.name, horizon, candidate_policy)] = _candidate_positions(
                labels, fold.test_positions, candidate_policy=candidate_policy
            )
            split = _inner_split(
                fold.train_positions,
                horizon=horizon,
                selection_observations=int(evaluation["inner_selection_observations"]),
            )
            row: dict[str, object] = {
                "fold": fold.name,
                "horizon_observations": horizon,
                "candidate_policy": candidate_policy,
                "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
                "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
                "outer_training_observations": len(fold.train_positions),
                "purged_tail_observations": fold.purged_tail_observations,
                "test_candidate_count": len(candidate_positions[(fold.name, horizon, candidate_policy)]),
            }
            if split is None:
                row["selection_status"] = "not_enough_history_for_inner_selection"
                fold_rows.append(row)
                continue
            fit_positions, selection_positions = split
            row["fit_observations"] = len(fit_positions)
            row["selection_observations"] = len(selection_positions)
            selection_candidates = _candidate_positions(
                labels, selection_positions, candidate_policy=candidate_policy
            )
            row["selection_candidate_count"] = len(selection_candidates)
            options: list[dict[str, Any]] = []
            for kind_raw in selected_models:
                kind = kind_raw
                try:
                    fitted = _fit_model(
                        features,
                        labels,
                        positions=fit_positions,
                        kind=kind,
                        evaluation=evaluation,
                        candidate_policy=candidate_policy,
                    )
                except ValueError:
                    continue
                scores, _ = _predict(fitted, features, selection_candidates)
                for quantile_raw in evaluation["score_quantiles"]:
                    quantile = float(quantile_raw)
                    threshold = float(np.quantile(fitted.train_scores, quantile))
                    candidates = pd.DataFrame({"position": np.asarray(selection_candidates)[scores >= threshold], "score": scores[scores >= threshold]})
                    candidates["timestamp"] = panel.loc[candidates["position"], "known_at"].to_numpy()
                    capped, metrics = _metrics(
                        candidates,
                        panel=panel,
                        labels=labels,
                        candidate_base_positions=selection_candidates,
                        horizon=horizon,
                        weekly_cap=float(evaluation["maximum_signals_per_week"]),
                    )
                    options.append(
                        {
                            "kind": kind,
                            "quantile": quantile,
                            "selection_metrics": metrics,
                            "selection_candidates": capped,
                        }
                    )
            eligible = [option for option in options if _eligible(option["selection_metrics"], evaluation)]
            if not eligible:
                row["selection_status"] = "no_model_meets_inner_selection_gate"
                fold_rows.append(row)
                continue
            selected = max(
                eligible,
                key=lambda option: (
                    float(option["selection_metrics"]["lift"]),
                    float(option["selection_metrics"]["benefit_fwd_bps"]),
                    -float(option["quantile"]),
                    str(option["kind"]),
                ),
            )
            status = (
                "promoted"
                if float(selected["selection_metrics"]["lift"]) >= float(evaluation["promotion_minimum_lift"])
                else "research_only"
            )
            final_model = _fit_model(
                features,
                labels,
                positions=fold.train_positions,
                kind=selected["kind"],
                evaluation=evaluation,
                candidate_policy=candidate_policy,
            )
            threshold = float(np.quantile(final_model.train_scores, float(selected["quantile"])))
            test_candidates_base = candidate_positions[(fold.name, horizon, candidate_policy)]
            test_scores, test_matrix = _predict(final_model, features, test_candidates_base)
            test_candidates = pd.DataFrame(
                {"position": np.asarray(test_candidates_base)[test_scores >= threshold], "score": test_scores[test_scores >= threshold]}
            )
            test_candidates["timestamp"] = panel.loc[test_candidates["position"], "known_at"].to_numpy()
            capped_test, test_metrics = _metrics(
                test_candidates,
                panel=panel,
                labels=labels,
                candidate_base_positions=test_candidates_base,
                horizon=horizon,
                weekly_cap=float(evaluation["maximum_signals_per_week"]),
            )
            selection_metrics = selected["selection_metrics"]
            test_frequency = test_metrics["signals_per_week"]
            frequency_ok = (
                test_frequency is not None
                and float(evaluation["minimum_signals_per_week"])
                <= float(test_frequency)
                <= float(evaluation["maximum_signals_per_week"])
            )
            allowed = status == "promoted"
            row.update(
                {
                    "selection_status": status,
                    "model": selected["kind"],
                    "score_quantile": selected["quantile"],
                    "score_threshold": threshold,
                    "selection_signal_count": selection_metrics["signal_count"],
                    "selection_signals_per_week": selection_metrics["signals_per_week"],
                    "selection_hit_rate": selection_metrics["hit_rate"],
                    "selection_candidate_baseline_hit_rate": selection_metrics["baseline_hit_rate"],
                    "selection_lift": selection_metrics["lift"],
                    "selection_benefit_fwd_bps": selection_metrics["benefit_fwd_bps"],
                    "test_candidate_signal_count": test_metrics["signal_count"],
                    "test_dispatched_signal_count": len(capped_test) if allowed else 0,
                    "test_signals_per_week": test_metrics["signals_per_week"],
                    "test_hit_rate": test_metrics["hit_rate"],
                    "test_candidate_baseline_hit_rate": test_metrics["baseline_hit_rate"],
                    "test_lift": test_metrics["lift"],
                    "test_benefit_fwd_bps": test_metrics["benefit_fwd_bps"],
                    "test_frequency_in_policy": frequency_ok,
                }
            )
            attribution_rows.extend(
                _global_attributions(
                    final_model,
                    fold=fold.name,
                    horizon=horizon,
                    status=status,
                    candidate_policy=candidate_policy,
                )
            )
            matrix_by_position = {position: test_matrix[index] for index, position in enumerate(test_candidates_base)}
            for candidate in capped_test.itertuples(index=False):
                signal_rows.append(
                    {
                        "fold": fold.name,
                        "horizon_observations": horizon,
                        "candidate_policy": candidate_policy,
                        "selection_status": status,
                        "timestamp": candidate.timestamp,
                        "value_date": str(panel.loc[candidate.position, "value_date"]),
                        "model": selected["kind"],
                        "score": float(candidate.score),
                        "score_threshold": threshold,
                        "communication_allowed": allowed,
                        "details": json.dumps(
                            {
                                "candidate_policy": candidate_policy,
                                "past_minimum": bool(labels.set_index("position").loc[candidate.position, "past_min"]),
                                "future_target": f"no lower close in the following {horizon} observations within {evaluation['tolerance_bps']} bps",
                                "inner_selection_lift": selection_metrics["lift"],
                                "attribution_scope": (
                                    "local_additive"
                                    if selected["kind"] in {"ridge_logistic", "gradient_boosted_stumps", "xgboost"}
                                    else "global_feature_importance"
                                    if selected["kind"] != "svc_rbf"
                                    else "not_available_for_rbf_svc"
                                ),
                                "top_feature_attributions": _local_contributions(
                                    final_model, matrix_by_position[candidate.position]
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
            fold_rows.append(row)
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    signals_frame = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    attributions_frame = pd.DataFrame(attribution_rows, columns=ATTRIBUTION_COLUMNS)
    summary_frame = _summary_rows(
        signals_frame,
        folds_by_name=folds_by_name,
        candidate_positions=candidate_positions,
        panel=panel,
        labels_by_horizon=labels_by_horizon,
    )
    folds_frame.to_csv(output / "folds.csv", index=False)
    signals_frame.to_csv(output / "signals.csv", index=False)
    attributions_frame.to_csv(output / "feature_attributions.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "config_sha256": local_minimum_config_sha256(config_path),
        "target_instrument_id": target,
        "horizons": evaluation["horizons"],
        "candidate_policy": candidate_policy,
        "evaluated_models": selected_models,
        "tolerance_bps": evaluation["tolerance_bps"],
        "factor_instruments": sorted(column for column in prices if column != target),
        "feature_count": len(features.columns),
        "folds": len(folds_frame),
        "signals_written": len(signals_frame),
        "dispatched_signals": int(signals_frame["communication_allowed"].sum()) if not signals_frame.empty else 0,
        "method": "outer expanding quarterly walk-forward; h-observation outer and inner purges; inner chronological model/threshold selection; configurable all-days or observable-past-min candidate support; future-only local-min label; target same-close features and one-observation-lagged external features; chronological weekly cap; do-not-send fallback",
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/local_minimum_models.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/local_minimum_models"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            "ridge_logistic",
            "gradient_boosted_stumps",
            "xgboost",
            "random_forest",
            "extra_trees",
            "adaboost",
            "gradient_boosting",
            "svc_rbf",
        ),
        help="optional subset of preregistered model candidates",
    )
    args = parser.parse_args(argv)
    meta = run_local_minimum_models(
        snapshot_dir=args.snapshot,
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        models=tuple(args.models) if args.models else None,
    )
    print(
        f"Wrote {meta['folds']} fold-horizon decisions and {meta['signals_written']} candidate test signals "
        f"({meta['dispatched_signals']} communication-eligible) to {args.artifact_dir}"
    )


if __name__ == "__main__":
    main()


__all__ = ["load_local_minimum_config", "local_minimum_features", "run_local_minimum_models"]
