"""Test strict-short plus regret-long targets with refitted boosters.

Each hybrid target is a fixed product hypothesis.  A positive label requires
both (1) a literal local minimum in a short, immediate window and (2) no
economically material improvement in a longer wait window.  Models are never
selected by their outer-test results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from fxpulse.boosting_calibration import BOOSTING_KINDS, _chronological_oof_scores
from fxpulse.labeling import evaluate_positions, label_hybrid_observations
from fxpulse.local_minimum_models import (
    ModelKind,
    _candidate_positions,
    _fit_model,
    _predict,
    load_local_minimum_config,
    local_minimum_features,
    refit_score_quantile_threshold,
)
from fxpulse.rule_selection import _target_panel, _weekly_cap, load_universe_prices, quarterly_folds, resolve_snapshot


FOLD_COLUMNS = (
    "target_name",
    "short_horizon",
    "long_horizon",
    "long_tolerance_bps",
    "fold",
    "test_start",
    "test_end",
    "model",
    "status",
    "outer_training_observations",
    "oof_observations",
    "final_refit_observations",
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
)

PREDICTION_COLUMNS = (
    "target_name",
    "short_horizon",
    "long_horizon",
    "long_tolerance_bps",
    "fold",
    "model",
    "value_date",
    "position",
    "hit_favorable",
    "short_strict_favorable",
    "long_regret_favorable",
    "future_regret_bps",
    "score",
    "dispatched",
)

SUMMARY_COLUMNS = (
    "target_name",
    "short_horizon",
    "long_horizon",
    "long_tolerance_bps",
    "model",
    "fold_count",
    "prediction_count",
    "target_positive_rate",
    "short_strict_positive_rate",
    "long_regret_positive_rate",
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

BOOTSTRAP_COLUMNS = (
    "target_name",
    "short_horizon",
    "long_horizon",
    "long_tolerance_bps",
    "model",
    "block_length_observations",
    "bootstrap_samples",
    "usable_samples",
    "lift_bootstrap_median",
    "lift_ci_95_low",
    "lift_ci_95_high",
)


def load_hybrid_targets_config(path: Path | str = Path("configs/hybrid_targets.json")) -> dict[str, Any]:
    config = load_local_minimum_config(path)
    evaluation = config["evaluation"]
    required = {"hybrid_targets", "calibration_oof_blocks", "calibration_minimum_observations", "policy_score_quantile"}
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks hybrid fields: {', '.join(sorted(missing))}")
    if set(evaluation["models"]) - set(BOOSTING_KINDS) or not evaluation["models"]:
        raise ValueError("hybrid targets support xgboost and gradient_boosting only")
    if not isinstance(evaluation["hybrid_targets"], list) or not evaluation["hybrid_targets"]:
        raise ValueError("hybrid_targets must be a non-empty list")
    names: set[str] = set()
    available_horizons = set(evaluation["horizons"])
    for target in evaluation["hybrid_targets"]:
        if not isinstance(target, dict) or {"name", "short_horizon", "long_horizon", "long_tolerance_bps"} - set(target):
            raise ValueError("each hybrid target needs name, short_horizon, long_horizon and long_tolerance_bps")
        if not isinstance(target["name"], str) or not target["name"] or target["name"] in names:
            raise ValueError("hybrid target names must be non-empty and unique")
        names.add(target["name"])
        if target["short_horizon"] not in available_horizons or target["long_horizon"] not in available_horizons or target["short_horizon"] >= target["long_horizon"]:
            raise ValueError("hybrid target horizons must be configured and satisfy short_horizon < long_horizon")
        if not isinstance(target["long_tolerance_bps"], int | float) or not math.isfinite(float(target["long_tolerance_bps"])) or target["long_tolerance_bps"] < 0:
            raise ValueError("hybrid target long_tolerance_bps must be finite and non-negative")
    for field in ("calibration_oof_blocks", "calibration_minimum_observations"):
        if not isinstance(evaluation[field], int) or evaluation[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(evaluation["policy_score_quantile"], int | float) or not 0 < evaluation["policy_score_quantile"] < 1:
        raise ValueError("policy_score_quantile must be in (0, 1)")
    return config


def _summary(
    predictions: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels_by_name: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    groups = ["target_name", "short_horizon", "long_horizon", "long_tolerance_bps", "model"]
    for values, group in predictions.groupby(groups, sort=True):
        name, short_horizon, long_horizon, tolerance_bps, model = values
        labels = labels_by_name[str(name)]
        base_positions = tuple(sorted(set(group["position"].astype(int))))
        signal_positions = group.loc[group["dispatched"], "position"].astype(int).tolist()
        metrics = evaluate_positions(
            panel,
            signal_positions,
            direction="favorable",
            horizon=int(long_horizon),
            base_positions=base_positions,
            labels=labels,
        )
        lifts: list[float] = []
        for _, fold in group.groupby("fold", sort=False):
            fold_positions = tuple(sorted(set(fold["position"].astype(int))))
            fold_signals = fold.loc[fold["dispatched"], "position"].astype(int).tolist()
            fold_metric = evaluate_positions(
                panel,
                fold_signals,
                direction="favorable",
                horizon=int(long_horizon),
                base_positions=fold_positions,
                labels=labels,
            )
            if fold_metric["lift"] is not None:
                lifts.append(float(fold_metric["lift"]))
        rows.append(
            {
                "target_name": name,
                "short_horizon": short_horizon,
                "long_horizon": long_horizon,
                "long_tolerance_bps": tolerance_bps,
                "model": model,
                "fold_count": int(group["fold"].nunique()),
                "prediction_count": int(len(group)),
                "target_positive_rate": float(group["hit_favorable"].mean()),
                "short_strict_positive_rate": float(group["short_strict_favorable"].mean()),
                "long_regret_positive_rate": float(group["long_regret_favorable"].mean()),
                "signal_count": metrics["signal_count"],
                "eligible_candidate_count": metrics["eligible_base_count"],
                "hit_rate": metrics["hit_rate"],
                "candidate_baseline_hit_rate": metrics["baseline_hit_rate"],
                "lift": metrics["lift"],
                "regret_mean_bps": metrics["regret_mean_bps"],
                "regret_p90_bps": metrics["regret_p90_bps"],
                "benefit_fwd_bps": metrics["benefit_fwd_bps"],
                "signals_per_week": metrics["signals_per_week"],
                "fold_lift_median": float(np.median(lifts)) if lifts else None,
                "fold_lift_worst": float(np.min(lifts)) if lifts else None,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _moving_block_sample(length: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=int)
    block = min(length, block_length)
    starts = rng.integers(0, length - block + 1, size=math.ceil(length / block))
    return np.concatenate([np.arange(start, start + block, dtype=int) for start in starts])[:length]


def _bootstrap_lifts(predictions: pd.DataFrame, *, samples: int = 2_000) -> pd.DataFrame:
    """Conditional moving-block lift intervals for each fixed OOT policy."""

    if predictions.empty:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    groups = ["target_name", "short_horizon", "long_horizon", "long_tolerance_bps", "model"]
    rows: list[dict[str, object]] = []
    for group_index, (values, group) in enumerate(predictions.groupby(groups, sort=True)):
        name, short_horizon, long_horizon, tolerance_bps, model = values
        folds = [
            (
                fold.sort_values("position")["hit_favorable"].astype(float).to_numpy(),
                fold.sort_values("position")["dispatched"].astype(bool).to_numpy(),
            )
            for _, fold in group.groupby("fold", sort=False)
        ]
        rng = np.random.default_rng(20260903 + group_index)
        lifts: list[float] = []
        for _ in range(samples):
            all_days: list[np.ndarray] = []
            signals: list[np.ndarray] = []
            for outcome, dispatched in folds:
                indices = _moving_block_sample(len(outcome), block_length=int(long_horizon), rng=rng)
                all_days.append(outcome[indices])
                signals.append(outcome[indices][dispatched[indices]])
            baseline = np.concatenate(all_days)
            selected = np.concatenate(signals)
            if len(selected) and float(baseline.mean()) > 0:
                lifts.append(float(selected.mean()) / float(baseline.mean()))
        distribution = np.asarray(lifts, dtype="float64")
        rows.append(
            {
                "target_name": name,
                "short_horizon": short_horizon,
                "long_horizon": long_horizon,
                "long_tolerance_bps": tolerance_bps,
                "model": model,
                "block_length_observations": int(long_horizon),
                "bootstrap_samples": samples,
                "usable_samples": int(len(distribution)),
                "lift_bootstrap_median": float(np.median(distribution)) if len(distribution) else None,
                "lift_ci_95_low": float(np.quantile(distribution, 0.025)) if len(distribution) else None,
                "lift_ci_95_high": float(np.quantile(distribution, 0.975)) if len(distribution) else None,
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def run_hybrid_targets(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/hybrid_targets.json"),
    artifact_dir: Path | str = Path("artifacts/hybrid_targets"),
) -> dict[str, Any]:
    """Run every preregistered strict-short/regret-long hypothesis."""

    config = load_hybrid_targets_config(config_path)
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
    labels_by_name: dict[str, pd.DataFrame] = {}
    candidate_policy = str(evaluation["candidate_policy"])
    for target_config in evaluation["hybrid_targets"]:
        name = str(target_config["name"])
        short_horizon = int(target_config["short_horizon"])
        long_horizon = int(target_config["long_horizon"])
        long_tolerance_bps = float(target_config["long_tolerance_bps"])
        labels = label_hybrid_observations(
            panel,
            short_horizon=short_horizon,
            long_horizon=long_horizon,
            long_tolerance_bps=long_tolerance_bps,
        )
        labels_by_name[name] = labels
        labels_by_position = labels.set_index("position")
        folds = quarterly_folds(panel, horizon=long_horizon, min_training_observations=int(evaluation["min_training_observations"]))
        for fold in folds:
            test_positions = _candidate_positions(labels, fold.test_positions, candidate_policy=candidate_policy)
            base = {
                "target_name": name,
                "short_horizon": short_horizon,
                "long_horizon": long_horizon,
                "long_tolerance_bps": long_tolerance_bps,
                "fold": fold.name,
                "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
                "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
                "outer_training_observations": len(fold.train_positions),
                "test_candidate_count": len(test_positions),
            }
            for kind_raw in evaluation["models"]:
                kind = cast(ModelKind, kind_raw)
                oof = _chronological_oof_scores(
                    features=features,
                    labels=labels,
                    positions=fold.train_positions,
                    horizon=long_horizon,
                    kind=kind,
                    evaluation=evaluation,
                )
                if len(oof) < int(evaluation["calibration_minimum_observations"]) or oof["hit_favorable"].nunique() < 2:
                    fold_rows.append({**base, "model": kind, "status": "insufficient_chronological_oof_labels", "oof_observations": len(oof)})
                    continue
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
                    fold_rows.append({**base, "model": kind, "status": "final_refit_failed", "oof_observations": len(oof)})
                    continue
                scores, _ = _predict(final_model, features, test_positions)
                # The numeric cutoff has to be in the final refit's score
                # space. OOF scores are reserved for chronology-safe policy
                # assessment, not reused as an absolute score cutoff.
                threshold = refit_score_quantile_threshold(
                    final_model,
                    float(evaluation["policy_score_quantile"]),
                )
                candidates = pd.DataFrame({"position": np.asarray(test_positions)[scores >= threshold], "score": scores[scores >= threshold]})
                candidates["timestamp"] = panel.loc[candidates["position"], "known_at"].to_numpy()
                dispatched = _weekly_cap(candidates, panel, maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]))
                dispatched_positions = set(dispatched["position"].astype(int))
                metrics = evaluate_positions(
                    panel,
                    dispatched_positions,
                    direction="favorable",
                    horizon=long_horizon,
                    base_positions=test_positions,
                    labels=labels,
                )
                fold_rows.append(
                    {
                        **base,
                        "model": kind,
                        "status": "ok",
                        "oof_observations": len(oof),
                        "final_refit_observations": len(final_model.train_positions),
                        "score_threshold": threshold,
                        "test_signal_count": metrics["signal_count"],
                        "test_signals_per_week": metrics["signals_per_week"],
                        "test_hit_rate": metrics["hit_rate"],
                        "test_candidate_baseline_hit_rate": metrics["baseline_hit_rate"],
                        "test_lift": metrics["lift"],
                        "test_regret_mean_bps": metrics["regret_mean_bps"],
                        "test_regret_p90_bps": metrics["regret_p90_bps"],
                        "test_benefit_fwd_bps": metrics["benefit_fwd_bps"],
                    }
                )
                test_labels = labels_by_position.loc[list(test_positions)]
                for position, score, row in zip(test_positions, scores, test_labels.itertuples(), strict=True):
                    prediction_rows.append(
                        {
                            "target_name": name,
                            "short_horizon": short_horizon,
                            "long_horizon": long_horizon,
                            "long_tolerance_bps": long_tolerance_bps,
                            "fold": fold.name,
                            "model": kind,
                            "value_date": str(panel.loc[position, "value_date"]),
                            "position": position,
                            "hit_favorable": bool(row.hit_favorable),
                            "short_strict_favorable": bool(row.short_strict_favorable),
                            "long_regret_favorable": bool(row.long_regret_favorable),
                            "future_regret_bps": float(row.future_regret_bps),
                            "score": float(score),
                            "dispatched": position in dispatched_positions,
                        }
                    )
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    predictions_frame = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    summary_frame = _summary(predictions_frame, panel=panel, labels_by_name=labels_by_name)
    bootstrap_frame = _bootstrap_lifts(predictions_frame)
    folds_frame.to_csv(output / "folds.csv", index=False)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    bootstrap_frame.to_csv(output / "block_bootstrap_lift.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "target_instrument_id": target,
        "hybrid_targets": evaluation["hybrid_targets"],
        "models": evaluation["models"],
        "final_base_model": "refit on all label-observable outer-train observations after OOF threshold construction",
        "bootstrap": "2,000 moving-block samples, long_horizon-sized blocks, conditional on the fixed OOT policy",
        "method": "separate preregistered hybrid targets; expanding quarterly walk-forward; long-horizon purge; chronological OOF score threshold; fixed 80th score percentile; chronological weekly cap; no OOT target or model selection",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_targets.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/hybrid_targets"))
    args = parser.parse_args(argv)
    meta = run_hybrid_targets(snapshot_dir=args.snapshot, config_path=args.config, artifact_dir=args.artifact_dir)
    print(f"Wrote hybrid target benchmark for {len(meta['hybrid_targets'])} targets to {args.artifact_dir}")


if __name__ == "__main__":
    main()


__all__ = ["load_hybrid_targets_config", "run_hybrid_targets"]
