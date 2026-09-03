"""Quarterly walk-forward search over interpretable cross-market scorecards.

The model class is deliberately small: a ridge-logistic scorecard with one or
three-day returns of every available MOEX factor and explicit missingness flags.
It is retrained for each test quarter, has no hidden feature engineering, and
writes its signed coefficients for every selected model card.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.labeling import HORIZONS, evaluate_positions, label_observations
from fxpulse.rule_selection import (
    _summary_rows,
    _target_panel,
    _weekly_cap,
    load_universe_prices,
    quarterly_folds,
    resolve_snapshot,
)


FOLD_COLUMNS = (
    "fold",
    "test_start",
    "test_end",
    "training_observations",
    "purged_tail_observations",
    "selection_status",
    "direction",
    "score_quantile",
    "score_threshold",
    "train_signal_count",
    "train_signals_per_week",
    "train_hit_rate",
    "train_baseline_hit_rate",
    "train_lift",
    "train_benefit_fwd_bps",
    "test_candidate_signal_count",
    "test_dispatched_signal_count",
    "test_signals_per_week",
    "test_hit_rate",
    "test_baseline_hit_rate",
    "test_lift",
    "test_benefit_fwd_bps",
    "test_frequency_in_policy",
)
SIGNAL_COLUMNS = (
    "fold",
    "selection_status",
    "timestamp",
    "value_date",
    "direction",
    "score",
    "communication_allowed",
    "details",
)
COEFFICIENT_COLUMNS = (
    "fold",
    "selection_status",
    "direction",
    "feature",
    "coefficient",
    "odds_ratio_per_standard_deviation",
    "is_missingness_indicator",
)


def model_config_sha256(path: Path | str = Path("configs/interpretable_models.json")) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_config(path: Path | str = Path("configs/interpretable_models.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("evaluation"), dict):
        raise ValueError("Only interpretable-model schema_version 1 with evaluation is supported")
    evaluation = config["evaluation"]
    required = {
        "target_instrument_id",
        "horizon_observations",
        "min_training_observations",
        "factor_lag_observations",
        "return_windows",
        "l2",
        "max_iterations",
        "score_quantiles",
        "minimum_training_signals",
        "minimum_signals_per_week",
        "maximum_signals_per_week",
        "selection_minimum_lift",
        "promotion_minimum_lift",
        "minimum_training_benefit_bps",
        "no_signal_fallback",
    }
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks: {', '.join(sorted(missing))}")
    if evaluation["horizon_observations"] not in HORIZONS:
        raise ValueError(f"horizon_observations must be one of {HORIZONS}")
    if not isinstance(evaluation["min_training_observations"], int) or evaluation["min_training_observations"] <= 0:
        raise ValueError("min_training_observations must be positive")
    if not isinstance(evaluation["factor_lag_observations"], int) or evaluation["factor_lag_observations"] <= 0:
        raise ValueError("factor_lag_observations must be positive")
    if not evaluation["return_windows"] or not all(
        isinstance(value, int) and value > 0 for value in evaluation["return_windows"]
    ):
        raise ValueError("return_windows must contain positive integers")
    if not evaluation["score_quantiles"] or not all(
        isinstance(value, int | float) and 0 < value < 1 for value in evaluation["score_quantiles"]
    ):
        raise ValueError("score_quantiles must be in (0, 1)")
    if evaluation["minimum_signals_per_week"] <= 0 or evaluation["maximum_signals_per_week"] < evaluation[
        "minimum_signals_per_week"
    ]:
        raise ValueError("invalid communication frequency range")
    if evaluation["no_signal_fallback"] != "do_not_send":
        raise ValueError("Only no_signal_fallback=do_not_send is supported")
    return config


def cross_market_features(prices: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    """Return strictly lagged factor features; gaps become explicit indicators."""

    evaluation = config["evaluation"]
    target = str(evaluation["target_instrument_id"])
    lag = int(evaluation["factor_lag_observations"])
    features = pd.DataFrame(index=prices.index)
    for factor in sorted(column for column in prices.columns if column != target):
        values = pd.to_numeric(prices[factor], errors="coerce")
        for window in evaluation["return_windows"]:
            name = f"{factor}__return_{int(window)}"
            features[name] = values.pct_change(int(window), fill_method=None).shift(lag)
    return features.replace([np.inf, -np.inf], np.nan)


def _fit_ridge_logistic(features: np.ndarray, target: np.ndarray, *, l2: float, max_iterations: int) -> np.ndarray:
    if len(features) == 0 or len(features) != len(target) or len(np.unique(target)) < 2:
        raise ValueError("ridge logistic requires non-empty features and both target classes")
    design = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros(design.shape[1])
    penalty = np.diag([0.0, *([l2] * (design.shape[1] - 1))])
    for _ in range(max_iterations):
        logits = np.clip(design @ weights, -35, 35)
        probability = 1 / (1 + np.exp(-logits))
        gradient = (design.T @ (probability - target)) / len(target) + penalty @ weights
        hessian = (design.T * (probability * (1 - probability))) @ design / len(target) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("ridge logistic Hessian is singular") from exc
        weights -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return weights


def _score(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.clip(np.column_stack([np.ones(len(features)), features]) @ weights, -35, 35)
    return 1 / (1 + np.exp(-logits))


def _transform(
    frame: pd.DataFrame,
    *,
    median: pd.Series | None = None,
    mean: pd.Series | None = None,
    scale: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series, pd.Series, pd.Series, list[str]]:
    """Median-impute under train fit and expose each imputation as a feature."""

    missing = frame.isna().astype("float64")
    median = frame.median() if median is None else median
    filled = frame.fillna(median).fillna(0.0)
    design = pd.concat([filled, missing.add_suffix("__missing")], axis=1)
    mean = design.mean() if mean is None else mean
    scale = design.std().replace(0.0, 1.0) if scale is None else scale
    return ((design - mean) / scale).to_numpy(dtype="float64"), median, mean, scale, list(design.columns)


def _metrics(
    candidates: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    base_positions: tuple[int, ...],
    direction: str,
    horizon: int,
    weekly_cap: float,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    candidates = candidates.loc[candidates["position"].isin(base_positions)]
    capped = _weekly_cap(candidates, panel, maximum_signals_per_week=weekly_cap)
    return capped, evaluate_positions(
        panel,
        capped["position"].tolist(),
        direction=direction,
        horizon=horizon,
        base_positions=base_positions,
        labels=labels,
    )


def _eligible(metrics: dict[str, float | int | None], evaluation: dict[str, Any]) -> bool:
    frequency = metrics["signals_per_week"]
    return bool(
        metrics["signal_count"] >= int(evaluation["minimum_training_signals"])
        and frequency is not None
        and float(evaluation["minimum_signals_per_week"]) <= float(frequency) <= float(evaluation["maximum_signals_per_week"])
        and metrics["lift"] is not None
        and float(metrics["lift"]) >= float(evaluation["selection_minimum_lift"])
        and metrics["benefit_fwd_bps"] is not None
        and float(metrics["benefit_fwd_bps"]) > float(evaluation["minimum_training_benefit_bps"])
    )


def _fold_model_options(
    features: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    panel: pd.DataFrame,
    train_positions: tuple[int, ...],
    test_positions: tuple[int, ...],
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fit two scorecards and return their train-selected threshold options."""

    horizon = int(evaluation["horizon_observations"])
    training = features.loc[features["position"].isin(train_positions)].copy()
    labels_by_position = labels.set_index("position")
    options: list[dict[str, Any]] = []
    for direction, target_column in (("favorable", "hit_favorable"), ("window_closing", "hit_closing")):
        train = training.copy()
        train["target"] = train["position"].map(labels_by_position[target_column])
        train = train.dropna(subset=["target"])
        if len(train) < int(evaluation["min_training_observations"]) or train["target"].nunique() < 2:
            continue
        feature_columns = [column for column in features.columns if column not in {"position", "timestamp"}]
        train_matrix, median, mean, scale, design_columns = _transform(train[feature_columns])
        weights = _fit_ridge_logistic(
            train_matrix,
            train["target"].astype(int).to_numpy(),
            l2=float(evaluation["l2"]),
            max_iterations=int(evaluation["max_iterations"]),
        )
        train_scores = _score(train_matrix, weights)
        test = features.loc[features["position"].isin(test_positions)].copy()
        test_matrix, _, _, _, _ = _transform(test[feature_columns], median=median, mean=mean, scale=scale)
        test_scores = _score(test_matrix, weights)
        for quantile in evaluation["score_quantiles"]:
            threshold = float(np.quantile(train_scores, float(quantile)))
            train_candidates = pd.DataFrame(
                {
                    "position": train.loc[train_scores >= threshold, "position"].to_numpy(),
                    "score": train_scores[train_scores >= threshold],
                }
            )
            test_candidates = pd.DataFrame(
                {
                    "position": test.loc[test_scores >= threshold, "position"].to_numpy(),
                    "score": test_scores[test_scores >= threshold],
                }
            )
            train_candidates["timestamp"] = panel.loc[train_candidates["position"], "known_at"].to_numpy()
            test_candidates["timestamp"] = panel.loc[test_candidates["position"], "known_at"].to_numpy()
            capped_train, train_metrics = _metrics(
                train_candidates,
                panel=panel,
                labels=labels,
                base_positions=train_positions,
                direction=direction,
                horizon=horizon,
                weekly_cap=float(evaluation["maximum_signals_per_week"]),
            )
            capped_test, test_metrics = _metrics(
                test_candidates,
                panel=panel,
                labels=labels,
                base_positions=test_positions,
                direction=direction,
                horizon=horizon,
                weekly_cap=float(evaluation["maximum_signals_per_week"]),
            )
            options.append(
                {
                    "direction": direction,
                    "score_quantile": float(quantile),
                    "score_threshold": threshold,
                    "weights": weights,
                    "design_columns": design_columns,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "test_candidates": capped_test,
                }
            )
    return options


def run_interpretable_models(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/interpretable_models.json"),
    artifact_dir: Path | str = Path("artifacts/interpretable_models"),
) -> dict[str, Any]:
    """Run scorecard selection on all factors and write OOT decisions/cards."""

    config = load_model_config(config_path)
    evaluation = config["evaluation"]
    snapshot = resolve_snapshot(snapshot_dir)
    prices = load_universe_prices(snapshot, target_instrument_id=str(evaluation["target_instrument_id"]))
    panel = _target_panel(prices, str(evaluation["target_instrument_id"])).reset_index(drop=True)
    horizon = int(evaluation["horizon_observations"])
    labels = label_observations(panel, horizon)
    folds = quarterly_folds(panel, horizon=horizon, min_training_observations=int(evaluation["min_training_observations"]))
    features = cross_market_features(prices, config=config).reset_index(drop=True)
    features.insert(0, "position", features.index)
    features.insert(1, "timestamp", panel["known_at"].to_numpy())
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for fold in folds:
        options = _fold_model_options(
            features,
            labels=labels,
            panel=panel,
            train_positions=fold.train_positions,
            test_positions=fold.test_positions,
            evaluation=evaluation,
        )
        eligible = [option for option in options if _eligible(option["train_metrics"], evaluation)]
        selected = None
        status = "no_model_meets_train_gate"
        if eligible:
            selected = max(
                eligible,
                key=lambda option: (
                    float(option["train_metrics"]["lift"]),
                    float(option["train_metrics"]["benefit_fwd_bps"]),
                    -float(option["score_quantile"]),
                    option["direction"],
                ),
            )
            status = (
                "promoted"
                if float(selected["train_metrics"]["lift"]) >= float(evaluation["promotion_minimum_lift"])
                else "research_only"
            )
        row: dict[str, object] = {
            "fold": fold.name,
            "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
            "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
            "training_observations": len(fold.train_positions),
            "purged_tail_observations": fold.purged_tail_observations,
            "selection_status": status,
        }
        if selected is not None:
            train_metrics = selected["train_metrics"]
            test_metrics = selected["test_metrics"]
            frequency = test_metrics["signals_per_week"]
            frequency_ok = (
                frequency is not None
                and float(evaluation["minimum_signals_per_week"])
                <= float(frequency)
                <= float(evaluation["maximum_signals_per_week"])
            )
            allowed = status == "promoted"
            row.update(
                {
                    "direction": selected["direction"],
                    "score_quantile": selected["score_quantile"],
                    "score_threshold": selected["score_threshold"],
                    "train_signal_count": train_metrics["signal_count"],
                    "train_signals_per_week": train_metrics["signals_per_week"],
                    "train_hit_rate": train_metrics["hit_rate"],
                    "train_baseline_hit_rate": train_metrics["baseline_hit_rate"],
                    "train_lift": train_metrics["lift"],
                    "train_benefit_fwd_bps": train_metrics["benefit_fwd_bps"],
                    "test_candidate_signal_count": test_metrics["signal_count"],
                    "test_dispatched_signal_count": len(selected["test_candidates"]) if allowed else 0,
                    "test_signals_per_week": test_metrics["signals_per_week"],
                    "test_hit_rate": test_metrics["hit_rate"],
                    "test_baseline_hit_rate": test_metrics["baseline_hit_rate"],
                    "test_lift": test_metrics["lift"],
                    "test_benefit_fwd_bps": test_metrics["benefit_fwd_bps"],
                    "test_frequency_in_policy": frequency_ok,
                }
            )
            for feature, coefficient in zip(selected["design_columns"], selected["weights"][1:], strict=True):
                coefficient_rows.append(
                    {
                        "fold": fold.name,
                        "selection_status": status,
                        "direction": selected["direction"],
                        "feature": feature,
                        "coefficient": float(coefficient),
                        "odds_ratio_per_standard_deviation": float(np.exp(coefficient)),
                        "is_missingness_indicator": feature.endswith("__missing"),
                    }
                )
            for candidate in selected["test_candidates"].itertuples(index=False):
                signal_rows.append(
                    {
                        "fold": fold.name,
                        "selection_status": status,
                        "timestamp": candidate.timestamp,
                        "value_date": str(panel.loc[candidate.position, "value_date"]),
                        "direction": selected["direction"],
                        "score": candidate.score,
                        "communication_allowed": allowed,
                        "details": json.dumps(
                            {
                                "model": "ridge_logistic_scorecard",
                                "score_quantile": selected["score_quantile"],
                                "score_threshold": selected["score_threshold"],
                                "train_lift": train_metrics["lift"],
                            },
                            sort_keys=True,
                        ),
                    }
                )
        fold_rows.append(row)
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    signals_frame = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    coefficients_frame = pd.DataFrame(coefficient_rows, columns=COEFFICIENT_COLUMNS)
    fold_status = {str(row["fold"]): str(row["selection_status"]) for row in fold_rows}
    summary_frame = _summary_rows(
        signals_frame,
        folds=folds,
        fold_status=fold_status,
        panel=panel,
        labels=labels,
        horizon=horizon,
    )
    folds_frame.to_csv(output / "folds.csv", index=False)
    signals_frame.to_csv(output / "signals.csv", index=False)
    coefficients_frame.to_csv(output / "coefficients.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "config_sha256": model_config_sha256(config_path),
        "folds": len(folds),
        "factor_instruments": sorted(column for column in prices if column != evaluation["target_instrument_id"]),
        "feature_count": len([column for column in features if column not in {"position", "timestamp"}]),
        "signals_written": len(signals_frame),
        "dispatched_signals": int(signals_frame["communication_allowed"].sum()) if not signals_frame.empty else 0,
        "method": "expanding quarterly ridge-logistic scorecards; h-observation train purge; one-observation factor lag; train-only median imputation with missingness flags; chronological weekly cap; do-not-send fallback",
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/interpretable_models.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/interpretable_models"))
    args = parser.parse_args(argv)
    meta = run_interpretable_models(snapshot_dir=args.snapshot, config_path=args.config, artifact_dir=args.artifact_dir)
    print(
        f"Wrote {meta['folds']} folds and {meta['signals_written']} candidate test signals "
        f"({meta['dispatched_signals']} communication-eligible) to {args.artifact_dir}"
    )


if __name__ == "__main__":
    main()


__all__ = ["cross_market_features", "load_model_config", "run_interpretable_models"]
