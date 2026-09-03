"""Leak-free calibration comparison for refitted boosting classifiers.

For each outer walk-forward quarter, calibration learns only from chronological
out-of-fold (OOF) scores inside the preceding, label-observable training
prefix.  The selected base booster is then *refit* on the full prefix before
scoring the outer test.  Thus an OOT label never enters feature fitting, base
model fitting, calibrator fitting, threshold construction, or model choice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from fxpulse.labeling import evaluate_positions, label_observations
from fxpulse.local_minimum_models import (
    ModelKind,
    _candidate_positions,
    _fit_model,
    _predict,
    load_local_minimum_config,
    local_minimum_features,
)
from fxpulse.rule_selection import _target_panel, _weekly_cap, load_universe_prices, quarterly_folds, resolve_snapshot


CalibrationMethod = Literal["none", "sigmoid", "isotonic"]
BOOSTING_KINDS: tuple[ModelKind, ...] = ("xgboost", "gradient_boosting")

FOLD_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "fold",
    "test_start",
    "test_end",
    "model",
    "calibration_method",
    "status",
    "outer_training_observations",
    "oof_observations",
    "oof_positive_observations",
    "final_refit_observations",
    "policy_score_quantile",
    "score_threshold",
    "test_candidate_count",
    "test_signal_count",
    "test_signals_per_week",
    "test_hit_rate",
    "test_candidate_baseline_hit_rate",
    "test_lift",
    "test_regret_mean_bps",
    "test_regret_p90_bps",
    "test_benefit_fwd_bps",
    "raw_brier",
    "calibrated_brier",
    "raw_log_loss",
    "calibrated_log_loss",
    "raw_ece",
    "calibrated_ece",
)

PREDICTION_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "fold",
    "model",
    "calibration_method",
    "value_date",
    "position",
    "hit_favorable",
    "future_regret_bps",
    "raw_score",
    "calibrated_probability",
    "dispatched",
)

SUMMARY_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "model",
    "calibration_method",
    "fold_count",
    "prediction_count",
    "positive_rate",
    "raw_brier",
    "calibrated_brier",
    "raw_log_loss",
    "calibrated_log_loss",
    "raw_ece",
    "calibrated_ece",
    "signal_count",
    "eligible_candidate_count",
    "hit_rate",
    "candidate_baseline_hit_rate",
    "lift",
    "regret_mean_bps",
    "regret_p90_bps",
    "benefit_fwd_bps",
    "signals_per_week",
    "fold_lift_median",
    "fold_lift_worst",
)


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: CalibrationMethod
    model: Any | None = None

    def predict(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype="float64")
        if self.method == "none":
            return np.clip(values, 1e-6, 1 - 1e-6)
        logit_scores = _score_logit(values)
        if self.method == "sigmoid":
            assert self.model is not None
            return np.clip(self.model.predict_proba(logit_scores.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            assert self.model is not None
            return np.clip(self.model.predict(logit_scores), 1e-6, 1 - 1e-6)
        raise ValueError(f"Unsupported calibration method {self.method}")


def _score_logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _binary_metrics(target: np.ndarray, probability: np.ndarray, *, ece_bins: int) -> dict[str, float]:
    labels = np.asarray(target, dtype="float64")
    scores = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1 - 1e-6)
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("calibration metrics need equally sized, non-empty arrays")
    brier = float(np.mean(np.square(scores - labels)))
    log_loss = float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)))
    ece = 0.0
    for bin_index in range(ece_bins):
        left = bin_index / ece_bins
        right = (bin_index + 1) / ece_bins
        mask = (scores >= left) & ((scores < right) if bin_index + 1 < ece_bins else (scores <= right))
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return {"brier": brier, "log_loss": log_loss, "ece": ece}


def _fit_calibrator(method: CalibrationMethod, scores: np.ndarray, target: np.ndarray) -> ProbabilityCalibrator:
    labels = np.asarray(target, dtype=int)
    if len(scores) != len(labels) or len(scores) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("calibrator needs chronological OOF scores with both target classes")
    if method == "none":
        return ProbabilityCalibrator(method="none")
    logit_scores = _score_logit(scores)
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1_000, random_state=20260903)
        model.fit(logit_scores.reshape(-1, 1), labels)
        return ProbabilityCalibrator(method="sigmoid", model=model)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(logit_scores, labels)
        return ProbabilityCalibrator(method="isotonic", model=model)
    raise ValueError(f"Unsupported calibration method {method}")


def load_boosting_calibration_config(path: Path | str = Path("configs/boosting_calibration.json")) -> dict[str, Any]:
    """Validate the base local-minimum contract and calibration extension."""

    config = load_local_minimum_config(path)
    evaluation = config["evaluation"]
    required = {
        "calibration_tolerance_bps",
        "calibration_methods",
        "calibration_oof_blocks",
        "calibration_minimum_observations",
        "calibration_minimum_positive_observations",
        "policy_score_quantile",
        "ece_bins",
    }
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks calibration fields: {', '.join(sorted(missing))}")
    if not evaluation["calibration_tolerance_bps"] or not all(
        isinstance(value, int | float) and math.isfinite(float(value)) and value >= 0
        for value in evaluation["calibration_tolerance_bps"]
    ):
        raise ValueError("calibration_tolerance_bps must be finite non-negative numbers")
    if len(set(evaluation["calibration_tolerance_bps"])) != len(evaluation["calibration_tolerance_bps"]):
        raise ValueError("calibration_tolerance_bps must not contain duplicates")
    if not evaluation["calibration_methods"] or set(evaluation["calibration_methods"]) - {"none", "sigmoid", "isotonic"}:
        raise ValueError("calibration_methods must be a non-empty subset of none, sigmoid, isotonic")
    if set(evaluation["models"]) - set(BOOSTING_KINDS) or not evaluation["models"]:
        raise ValueError("boosting calibration supports xgboost and gradient_boosting only")
    for field in ("calibration_oof_blocks", "calibration_minimum_observations", "calibration_minimum_positive_observations", "ece_bins"):
        if not isinstance(evaluation[field], int) or evaluation[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(evaluation["policy_score_quantile"], int | float) or not 0 < evaluation["policy_score_quantile"] < 1:
        raise ValueError("policy_score_quantile must be in (0, 1)")
    return config


def _chronological_oof_scores(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    positions: tuple[int, ...],
    horizon: int,
    kind: ModelKind,
    evaluation: dict[str, Any],
) -> pd.DataFrame:
    """Create calibration observations by repeated fit-past / score-future splits."""

    eligible = _candidate_positions(labels, positions, candidate_policy=str(evaluation["candidate_policy"]))
    minimum = int(evaluation["calibration_minimum_observations"])
    if len(eligible) < minimum + int(evaluation["minimum_candidate_training_observations"]) + horizon:
        return pd.DataFrame(columns=("position", "raw_score", "hit_favorable"))
    # Reserve the latter half of the train prefix for several chronological
    # score blocks. The h-gap keeps labels used for fitting observable at the
    # beginning of every OOF score block.
    first_score_index = max(
        len(eligible) // 2,
        int(evaluation["minimum_candidate_training_observations"]) + horizon,
    )
    score_indices = np.arange(first_score_index, len(eligible), dtype=int)
    blocks = [block for block in np.array_split(score_indices, int(evaluation["calibration_oof_blocks"])) if len(block)]
    records: list[pd.DataFrame] = []
    label_by_position = labels.set_index("position")["hit_favorable"]
    for block in blocks:
        first = int(block[0])
        fit_positions = tuple(eligible[: max(0, first - horizon)])
        score_positions = tuple(eligible[index] for index in block)
        try:
            fitted = _fit_model(
                features,
                labels,
                positions=fit_positions,
                kind=kind,
                evaluation=evaluation,
                candidate_policy=str(evaluation["candidate_policy"]),
            )
        except ValueError:
            continue
        scores, _ = _predict(fitted, features, score_positions)
        records.append(
            pd.DataFrame(
                {
                    "position": score_positions,
                    "raw_score": scores,
                    "hit_favorable": label_by_position.loc[list(score_positions)].astype(int).to_numpy(),
                }
            )
        )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=("position", "raw_score", "hit_favorable"))


def _summary(
    predictions: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels_by_task: dict[tuple[float, int], pd.DataFrame],
    ece_bins: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    groups = ["tolerance_bps", "horizon_observations", "model", "calibration_method"]
    position_by_date = dict(zip(panel["value_date"].astype(str), range(len(panel)), strict=True))
    for values, group in predictions.groupby(groups, sort=True):
        tolerance_bps, horizon, model, method = values
        labels = labels_by_task[(float(tolerance_bps), int(horizon))]
        base_positions = tuple(sorted(set(group["position"].astype(int))))
        signal_positions = group.loc[group["dispatched"], "position"].astype(int).tolist()
        policy_metrics = evaluate_positions(
            panel,
            signal_positions,
            direction="favorable",
            horizon=int(horizon),
            base_positions=base_positions,
            labels=labels,
        )
        outcome = group["hit_favorable"].astype(int).to_numpy()
        raw_metrics = _binary_metrics(outcome, group["raw_score"].to_numpy(), ece_bins=ece_bins)
        calibrated_metrics = _binary_metrics(outcome, group["calibrated_probability"].to_numpy(), ece_bins=ece_bins)
        lifts = []
        for _, fold in group.groupby("fold", sort=False):
            fold_positions = tuple(sorted(set(fold["position"].astype(int))))
            fold_signals = fold.loc[fold["dispatched"], "position"].astype(int).tolist()
            metric = evaluate_positions(
                panel,
                fold_signals,
                direction="favorable",
                horizon=int(horizon),
                base_positions=fold_positions,
                labels=labels,
            )
            if metric["lift"] is not None:
                lifts.append(float(metric["lift"]))
        rows.append(
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": horizon,
                "model": model,
                "calibration_method": method,
                "fold_count": int(group["fold"].nunique()),
                "prediction_count": int(len(group)),
                "positive_rate": float(np.mean(outcome)),
                "raw_brier": raw_metrics["brier"],
                "calibrated_brier": calibrated_metrics["brier"],
                "raw_log_loss": raw_metrics["log_loss"],
                "calibrated_log_loss": calibrated_metrics["log_loss"],
                "raw_ece": raw_metrics["ece"],
                "calibrated_ece": calibrated_metrics["ece"],
                "signal_count": policy_metrics["signal_count"],
                "eligible_candidate_count": policy_metrics["eligible_base_count"],
                "hit_rate": policy_metrics["hit_rate"],
                "candidate_baseline_hit_rate": policy_metrics["baseline_hit_rate"],
                "lift": policy_metrics["lift"],
                "regret_mean_bps": policy_metrics["regret_mean_bps"],
                "regret_p90_bps": policy_metrics["regret_p90_bps"],
                "benefit_fwd_bps": policy_metrics["benefit_fwd_bps"],
                "signals_per_week": policy_metrics["signals_per_week"],
                "fold_lift_median": float(np.median(lifts)) if lifts else None,
                "fold_lift_worst": float(np.min(lifts)) if lifts else None,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def run_boosting_calibration(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/boosting_calibration.json"),
    artifact_dir: Path | str = Path("artifacts/boosting_calibration"),
) -> dict[str, Any]:
    """Compare uncalibrated, sigmoid and isotonic boosters on OOT quarters."""

    config = load_boosting_calibration_config(config_path)
    evaluation = config["evaluation"]
    snapshot = resolve_snapshot(snapshot_dir)
    target = str(evaluation["target_instrument_id"])
    prices = load_universe_prices(snapshot, target_instrument_id=target)
    panel = _target_panel(prices, target).reset_index(drop=True)
    features = local_minimum_features(prices, config=config).reset_index(drop=True)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    labels_by_task: dict[tuple[float, int], pd.DataFrame] = {}
    candidate_policy = str(evaluation["candidate_policy"])
    methods = tuple(cast(CalibrationMethod, value) for value in evaluation["calibration_methods"])
    for tolerance_raw in evaluation["calibration_tolerance_bps"]:
        tolerance_bps = float(tolerance_raw)
        for horizon_raw in evaluation["horizons"]:
            horizon = int(horizon_raw)
            labels = label_observations(panel, horizon, tolerance_bps=tolerance_bps)
            labels_by_task[(tolerance_bps, horizon)] = labels
            labels_by_position = labels.set_index("position")
            folds = quarterly_folds(panel, horizon=horizon, min_training_observations=int(evaluation["min_training_observations"]))
            for fold in folds:
                test_positions = _candidate_positions(labels, fold.test_positions, candidate_policy=candidate_policy)
                base = {
                    "tolerance_bps": tolerance_bps,
                    "horizon_observations": horizon,
                    "fold": fold.name,
                    "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
                    "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
                    "outer_training_observations": len(fold.train_positions),
                    "test_candidate_count": len(test_positions),
                    "policy_score_quantile": float(evaluation["policy_score_quantile"]),
                }
                for kind_raw in evaluation["models"]:
                    kind = cast(ModelKind, kind_raw)
                    oof = _chronological_oof_scores(
                        features=features,
                        labels=labels,
                        positions=fold.train_positions,
                        horizon=horizon,
                        kind=kind,
                        evaluation=evaluation,
                    )
                    enough_oof = (
                        len(oof) >= int(evaluation["calibration_minimum_observations"])
                        and int(oof["hit_favorable"].sum()) >= int(evaluation["calibration_minimum_positive_observations"])
                        and int((1 - oof["hit_favorable"].astype(int)).sum()) >= int(evaluation["calibration_minimum_positive_observations"])
                    )
                    if not enough_oof:
                        for method in methods:
                            fold_rows.append(
                                {
                                    **base,
                                    "model": kind,
                                    "calibration_method": method,
                                    "status": "insufficient_chronological_oof_labels",
                                    "oof_observations": len(oof),
                                    "oof_positive_observations": int(oof["hit_favorable"].sum()) if not oof.empty else 0,
                                }
                            )
                        continue
                    # Critical refit: OOF models served solely to fit the
                    # calibrator. The deployable base learner sees every
                    # label-observable outer-train row exactly once here.
                    try:
                        final_model = _fit_model(
                            features,
                            labels,
                            positions=fold.train_positions,
                            kind=kind,
                            evaluation=evaluation,
                            candidate_policy=candidate_policy,
                        )
                    except ValueError:
                        for method in methods:
                            fold_rows.append({**base, "model": kind, "calibration_method": method, "status": "final_refit_failed"})
                        continue
                    raw_scores, _ = _predict(final_model, features, test_positions)
                    outcome = labels_by_position.loc[list(test_positions), "hit_favorable"].astype(int).to_numpy()
                    for method in methods:
                        try:
                            calibrator = _fit_calibrator(method, oof["raw_score"].to_numpy(), oof["hit_favorable"].to_numpy())
                        except ValueError:
                            fold_rows.append({**base, "model": kind, "calibration_method": method, "status": "calibrator_fit_failed"})
                            continue
                        calibrated = calibrator.predict(raw_scores)
                        # The final learner has a different score scale from
                        # the chronological OOF learners. The cutoff must be
                        # built from final-model scores, not reused as an OOF
                        # number. Calibration quality is assessed separately.
                        threshold = float(
                            np.quantile(
                                calibrator.predict(final_model.train_scores),
                                float(evaluation["policy_score_quantile"]),
                            )
                        )
                        candidates = pd.DataFrame(
                            {"position": np.asarray(test_positions)[calibrated >= threshold], "score": calibrated[calibrated >= threshold]}
                        )
                        candidates["timestamp"] = panel.loc[candidates["position"], "known_at"].to_numpy()
                        dispatched = _weekly_cap(candidates, panel, maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]))
                        dispatched_positions = set(dispatched["position"].astype(int))
                        policy_metrics = evaluate_positions(
                            panel,
                            dispatched_positions,
                            direction="favorable",
                            horizon=horizon,
                            base_positions=test_positions,
                            labels=labels,
                        )
                        raw_metrics = _binary_metrics(outcome, raw_scores, ece_bins=int(evaluation["ece_bins"]))
                        calibrated_metrics = _binary_metrics(outcome, calibrated, ece_bins=int(evaluation["ece_bins"]))
                        fold_rows.append(
                            {
                                **base,
                                "model": kind,
                                "calibration_method": method,
                                "status": "ok",
                                "oof_observations": len(oof),
                                "oof_positive_observations": int(oof["hit_favorable"].sum()),
                                "final_refit_observations": len(final_model.train_positions),
                                "score_threshold": threshold,
                                "test_signal_count": policy_metrics["signal_count"],
                                "test_signals_per_week": policy_metrics["signals_per_week"],
                                "test_hit_rate": policy_metrics["hit_rate"],
                                "test_candidate_baseline_hit_rate": policy_metrics["baseline_hit_rate"],
                                "test_lift": policy_metrics["lift"],
                                "test_regret_mean_bps": policy_metrics["regret_mean_bps"],
                                "test_regret_p90_bps": policy_metrics["regret_p90_bps"],
                                "test_benefit_fwd_bps": policy_metrics["benefit_fwd_bps"],
                                "raw_brier": raw_metrics["brier"],
                                "calibrated_brier": calibrated_metrics["brier"],
                                "raw_log_loss": raw_metrics["log_loss"],
                                "calibrated_log_loss": calibrated_metrics["log_loss"],
                                "raw_ece": raw_metrics["ece"],
                                "calibrated_ece": calibrated_metrics["ece"],
                            }
                        )
                        for position, raw_score, probability, hit, regret in zip(
                            test_positions,
                            raw_scores,
                            calibrated,
                            outcome,
                            labels_by_position.loc[list(test_positions), "future_regret_bps"].to_numpy(),
                            strict=True,
                        ):
                            prediction_rows.append(
                                {
                                    "tolerance_bps": tolerance_bps,
                                    "horizon_observations": horizon,
                                    "fold": fold.name,
                                    "model": kind,
                                    "calibration_method": method,
                                    "value_date": str(panel.loc[position, "value_date"]),
                                    "position": position,
                                    "hit_favorable": bool(hit),
                                    "future_regret_bps": float(regret),
                                    "raw_score": float(raw_score),
                                    "calibrated_probability": float(probability),
                                    "dispatched": position in dispatched_positions,
                                }
                            )
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    predictions_frame = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    summary_frame = _summary(
        predictions_frame,
        panel=panel,
        labels_by_task=labels_by_task,
        ece_bins=int(evaluation["ece_bins"]),
    )
    folds_frame.to_csv(output / "folds.csv", index=False)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "target_instrument_id": target,
        "horizons": evaluation["horizons"],
        "tolerances_bps": evaluation["calibration_tolerance_bps"],
        "models": evaluation["models"],
        "calibration_methods": evaluation["calibration_methods"],
        "final_base_model": "refit on every label-observable outer-train observation after OOF calibrator fitting",
        "method": "expanding quarterly walk-forward; h-observation outer purge; chronological OOF calibration blocks within outer train; fixed rank-quantile policy with chronological weekly cap; no OOT model or calibrator selection",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/boosting_calibration.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/boosting_calibration"))
    args = parser.parse_args(argv)
    meta = run_boosting_calibration(snapshot_dir=args.snapshot, config_path=args.config, artifact_dir=args.artifact_dir)
    print(f"Wrote calibrated boosting benchmark for h={meta['horizons']} and tau={meta['tolerances_bps']} to {args.artifact_dir}")


if __name__ == "__main__":
    main()


__all__ = ["load_boosting_calibration_config", "run_boosting_calibration"]
