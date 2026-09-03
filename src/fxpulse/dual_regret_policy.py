"""Leakage-safe exploratory test of a two-horizon future-regret policy.

The target is fixed before this run: buying today must leave at most 10 bps of
missed improvement in the next three sessions and at most 50 bps in the next
twenty.  A final refit model uses a score threshold derived from *its own*
training-score distribution.  It is compared with a frequency-matched random
policy and a no-target-tuned historical-level heuristic.

The target was designed after earlier CNY/RUB results, so this is explicitly an
exploratory benchmark.  It can nominate only a hypothesis for a new corridor or
future hold-out; it cannot establish a product winner on this same history.
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

from fxpulse.labeling import evaluate_positions, label_dual_regret_observations
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
    "fold",
    "test_start",
    "test_end",
    "purge_observations",
    "outer_training_observations",
    "final_refit_observations",
    "final_score_reference_observations",
    "model_score_threshold",
    "heuristic_score_threshold",
    "test_candidate_count",
    "model_signal_count",
    "model_signals_per_week",
    "model_hit_rate",
    "model_candidate_baseline_hit_rate",
    "model_lift",
    "heuristic_signal_count",
    "heuristic_signals_per_week",
    "heuristic_hit_rate",
    "heuristic_lift",
)

PREDICTION_COLUMNS = (
    "fold",
    "value_date",
    "position",
    "week",
    "hit_favorable",
    "short_regret_favorable",
    "long_regret_favorable",
    "short_future_regret_bps",
    "long_future_regret_bps",
    "model_score",
    "heuristic_score",
    "model_dispatched",
    "heuristic_dispatched",
)

SUMMARY_COLUMNS = (
    "policy",
    "eligible_candidate_count",
    "signal_count",
    "hit_rate",
    "candidate_baseline_hit_rate",
    "lift",
    "regret_mean_bps",
    "regret_p90_bps",
    "benefit_fwd_bps",
    "signals_per_week",
    "frequency_requirement_met",
    "fold_lift_median",
    "fold_lift_worst",
)


def load_dual_regret_policy_config(path: Path | str = Path("configs/dual_regret_policy.json")) -> dict[str, Any]:
    """Load the single predeclared target and policy without silent defaults."""

    config = load_local_minimum_config(path)
    evaluation = config["evaluation"]
    required = {
        "dual_regret_target",
        "policy_score_quantile",
        "historical_heuristic_window",
        "matched_random_samples",
        "block_bootstrap_samples",
    }
    missing = required - set(evaluation)
    if missing:
        raise ValueError(f"evaluation lacks dual-regret fields: {', '.join(sorted(missing))}")
    if evaluation["models"] != ["xgboost"]:
        raise ValueError("dual-regret policy fixes one XGBoost model to avoid same-history model selection")
    target = evaluation["dual_regret_target"]
    target_required = {"name", "short_horizon", "short_tolerance_bps", "long_horizon", "long_tolerance_bps"}
    if not isinstance(target, dict) or target_required - set(target):
        raise ValueError("dual_regret_target lacks required fields")
    if not isinstance(target["name"], str) or not target["name"]:
        raise ValueError("dual_regret_target name must be non-empty")
    if target["short_horizon"] not in evaluation["horizons"] or target["long_horizon"] not in evaluation["horizons"]:
        raise ValueError("dual-regret horizons must belong to evaluation horizons")
    if not isinstance(target["short_horizon"], int) or not isinstance(target["long_horizon"], int) or target["short_horizon"] >= target["long_horizon"]:
        raise ValueError("dual-regret requires short_horizon < long_horizon")
    for name in ("short_tolerance_bps", "long_tolerance_bps"):
        value = target[name]
        if not isinstance(value, int | float) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"dual-regret {name} must be finite and non-negative")
    if not isinstance(evaluation["policy_score_quantile"], int | float) or not 0 < float(evaluation["policy_score_quantile"]) < 1:
        raise ValueError("policy_score_quantile must be in (0, 1)")
    for name in ("historical_heuristic_window", "matched_random_samples", "block_bootstrap_samples"):
        if not isinstance(evaluation[name], int) or evaluation[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if evaluation["historical_heuristic_window"] not in evaluation["distance_to_min_windows"]:
        raise ValueError("historical_heuristic_window must be available as a distance-to-min feature")
    return config


def _week_by_position(panel: pd.DataFrame, positions: list[int] | tuple[int, ...]) -> pd.Series:
    dates = pd.to_datetime(panel.loc[list(positions), "value_date"], errors="raise")
    return dates.dt.to_period("W").astype(str).reset_index(drop=True)


def _metrics_for_policy(
    predictions: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    dispatched_column: str,
    long_horizon: int,
) -> tuple[dict[str, float | int | None], list[float]]:
    base_positions = tuple(sorted(predictions["position"].astype(int).unique()))
    dispatched = predictions.loc[predictions[dispatched_column], "position"].astype(int).tolist()
    metrics = evaluate_positions(
        panel,
        dispatched,
        direction="favorable",
        horizon=long_horizon,
        base_positions=base_positions,
        labels=labels,
    )
    fold_lifts: list[float] = []
    for _, fold in predictions.groupby("fold", sort=False):
        fold_positions = tuple(sorted(fold["position"].astype(int).unique()))
        fold_dispatched = fold.loc[fold[dispatched_column], "position"].astype(int).tolist()
        fold_metrics = evaluate_positions(
            panel,
            fold_dispatched,
            direction="favorable",
            horizon=long_horizon,
            base_positions=fold_positions,
            labels=labels,
        )
        if fold_metrics["lift"] is not None:
            fold_lifts.append(float(fold_metrics["lift"]))
    return metrics, fold_lifts


def _summary(predictions: pd.DataFrame, *, panel: pd.DataFrame, labels: pd.DataFrame, long_horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, column in (("xgboost_refit_rank", "model_dispatched"), ("historical_low_20d", "heuristic_dispatched")):
        metrics, fold_lifts = _metrics_for_policy(
            predictions,
            panel=panel,
            labels=labels,
            dispatched_column=column,
            long_horizon=long_horizon,
        )
        frequency = metrics["signals_per_week"]
        rows.append(
            {
                "policy": policy,
                "eligible_candidate_count": metrics["eligible_base_count"],
                "signal_count": metrics["signal_count"],
                "hit_rate": metrics["hit_rate"],
                "candidate_baseline_hit_rate": metrics["baseline_hit_rate"],
                "lift": metrics["lift"],
                "regret_mean_bps": metrics["regret_mean_bps"],
                "regret_p90_bps": metrics["regret_p90_bps"],
                "benefit_fwd_bps": metrics["benefit_fwd_bps"],
                "signals_per_week": frequency,
                "frequency_requirement_met": bool(frequency is not None and 1 <= float(frequency) <= 2),
                "fold_lift_median": float(np.median(fold_lifts)) if fold_lifts else None,
                "fold_lift_worst": float(np.min(fold_lifts)) if fold_lifts else None,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _matched_random_summary(predictions: pd.DataFrame, *, samples: int) -> pd.DataFrame:
    """Compare to random precommitted days with exactly model's weekly counts."""

    if predictions.empty:
        return pd.DataFrame()
    ordered = predictions.sort_values(["fold", "position"], kind="mergesort")
    outcome = ordered["hit_favorable"].astype(float).to_numpy()
    baseline = float(outcome.mean())
    model_selected = ordered.loc[ordered["model_dispatched"], "hit_favorable"].astype(float).to_numpy()
    model_hit_rate = float(model_selected.mean()) if len(model_selected) else None
    model_lift = model_hit_rate / baseline if model_hit_rate is not None and baseline > 0 else None
    rng = np.random.default_rng(20260903)
    random_hit_rates: list[float] = []
    random_lifts: list[float] = []
    groups = list(ordered.groupby(["fold", "week"], sort=False))
    for _ in range(samples):
        chosen: list[int] = []
        for _, group in groups:
            desired = int(group["model_dispatched"].sum())
            positions = group.index.to_numpy()
            if desired:
                chosen.extend(rng.choice(positions, size=desired, replace=False).tolist())
        selected = ordered.loc[chosen, "hit_favorable"].astype(float).to_numpy()
        if len(selected):
            hit_rate = float(selected.mean())
            random_hit_rates.append(hit_rate)
            if baseline > 0:
                random_lifts.append(hit_rate / baseline)
    hit_distribution = np.asarray(random_hit_rates, dtype="float64")
    lift_distribution = np.asarray(random_lifts, dtype="float64")
    return pd.DataFrame(
        [
            {
                "policy": "weekly_frequency_matched_random",
                "samples": samples,
                "signal_count_per_draw": int(len(model_selected)),
                "model_hit_rate": model_hit_rate,
                "model_lift": model_lift,
                "random_hit_rate_mean": float(hit_distribution.mean()) if len(hit_distribution) else None,
                "random_hit_rate_ci_95_low": float(np.quantile(hit_distribution, 0.025)) if len(hit_distribution) else None,
                "random_hit_rate_ci_95_high": float(np.quantile(hit_distribution, 0.975)) if len(hit_distribution) else None,
                "random_lift_mean": float(lift_distribution.mean()) if len(lift_distribution) else None,
                "random_lift_ci_95_low": float(np.quantile(lift_distribution, 0.025)) if len(lift_distribution) else None,
                "random_lift_ci_95_high": float(np.quantile(lift_distribution, 0.975)) if len(lift_distribution) else None,
                "model_vs_random_hit_rate_ratio": model_hit_rate / float(hit_distribution.mean()) if model_hit_rate is not None and len(hit_distribution) and float(hit_distribution.mean()) > 0 else None,
                "random_draw_share_lift_at_least_model": float((lift_distribution >= model_lift).mean()) if model_lift is not None and len(lift_distribution) else None,
            }
        ]
    )


def _moving_block_sample(length: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    block = min(length, block_length)
    starts = rng.integers(0, length - block + 1, size=math.ceil(length / block))
    return np.concatenate([np.arange(start, start + block, dtype=int) for start in starts])[:length]


def _bootstrap_lifts(predictions: pd.DataFrame, *, long_horizon: int, samples: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy_index, (policy, column) in enumerate((("xgboost_refit_rank", "model_dispatched"), ("historical_low_20d", "heuristic_dispatched"))):
        folds = [
            (
                fold.sort_values("position")["hit_favorable"].astype(float).to_numpy(),
                fold.sort_values("position")[column].astype(bool).to_numpy(),
            )
            for _, fold in predictions.groupby("fold", sort=False)
        ]
        rng = np.random.default_rng(20260903 + policy_index)
        lifts: list[float] = []
        for _ in range(samples):
            all_days: list[np.ndarray] = []
            selected: list[np.ndarray] = []
            for outcome, dispatched in folds:
                indices = _moving_block_sample(len(outcome), block_length=long_horizon, rng=rng)
                all_days.append(outcome[indices])
                selected.append(outcome[indices][dispatched[indices]])
            baseline = np.concatenate(all_days)
            signals = np.concatenate(selected)
            if len(signals) and float(baseline.mean()) > 0:
                lifts.append(float(signals.mean()) / float(baseline.mean()))
        distribution = np.asarray(lifts, dtype="float64")
        rows.append(
            {
                "policy": policy,
                "block_length_observations": long_horizon,
                "bootstrap_samples": samples,
                "usable_samples": int(len(distribution)),
                "lift_bootstrap_median": float(np.median(distribution)) if len(distribution) else None,
                "lift_ci_95_low": float(np.quantile(distribution, 0.025)) if len(distribution) else None,
                "lift_ci_95_high": float(np.quantile(distribution, 0.975)) if len(distribution) else None,
            }
        )
    return pd.DataFrame(rows)


def run_dual_regret_policy(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/dual_regret_policy.json"),
    artifact_dir: Path | str = Path("artifacts/dual_regret_policy"),
) -> dict[str, Any]:
    """Run the single two-regret target with fixed model and baselines."""

    config = load_dual_regret_policy_config(config_path)
    evaluation = config["evaluation"]
    target_config = evaluation["dual_regret_target"]
    short_horizon = int(target_config["short_horizon"])
    long_horizon = int(target_config["long_horizon"])
    snapshot = resolve_snapshot(snapshot_dir)
    target_instrument_id = str(evaluation["target_instrument_id"])
    prices = load_universe_prices(snapshot, target_instrument_id=target_instrument_id)
    panel = _target_panel(prices, target_instrument_id).reset_index(drop=True)
    features = local_minimum_features(prices, config=config).reset_index(drop=True)
    labels = label_dual_regret_observations(
        panel,
        short_horizon=short_horizon,
        short_tolerance_bps=float(target_config["short_tolerance_bps"]),
        long_horizon=long_horizon,
        long_tolerance_bps=float(target_config["long_tolerance_bps"]),
    )
    labels_by_position = labels.set_index("position")
    folds = quarterly_folds(panel, horizon=long_horizon, min_training_observations=int(evaluation["min_training_observations"]))
    heuristic_column = f"target__distance_to_min_{int(evaluation['historical_heuristic_window'])}"
    if heuristic_column not in features:
        raise ValueError(f"missing historical heuristic feature {heuristic_column}")
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    candidate_policy = str(evaluation["candidate_policy"])
    for fold in folds:
        test_positions = _candidate_positions(labels, fold.test_positions, candidate_policy=candidate_policy)
        train_positions = _candidate_positions(labels, fold.train_positions, candidate_policy=candidate_policy)
        final_model = _fit_model(
            features,
            labels,
            positions=fold.train_positions,
            kind=cast(ModelKind, "xgboost"),
            evaluation=evaluation,
            candidate_policy=candidate_policy,
        )
        model_threshold = refit_score_quantile_threshold(final_model, float(evaluation["policy_score_quantile"]))
        model_scores, _ = _predict(final_model, features, test_positions)
        model_candidates = pd.DataFrame(
            {"position": np.asarray(test_positions)[model_scores >= model_threshold], "score": model_scores[model_scores >= model_threshold]}
        )
        model_candidates["timestamp"] = panel.loc[model_candidates["position"], "known_at"].to_numpy()
        model_dispatched = _weekly_cap(model_candidates, panel, maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]))
        model_positions = set(model_dispatched["position"].astype(int))

        train_heuristic = pd.to_numeric(features.loc[list(train_positions), heuristic_column], errors="coerce").dropna()
        if train_heuristic.empty:
            raise ValueError("historical heuristic lacks train observations")
        # Model scores are higher-is-better; distance to a rolling minimum is
        # lower-is-better. The same 80th score percentile therefore becomes
        # the lower 20th percentile of the distance feature.
        heuristic_threshold = float(
            np.quantile(
                train_heuristic.to_numpy(),
                1 - float(evaluation["policy_score_quantile"]),
            )
        )
        heuristic_scores = pd.to_numeric(features.loc[list(test_positions), heuristic_column], errors="coerce").to_numpy(dtype="float64")
        heuristic_mask = np.isfinite(heuristic_scores) & (heuristic_scores <= heuristic_threshold)
        heuristic_candidates = pd.DataFrame(
            {"position": np.asarray(test_positions)[heuristic_mask], "score": -heuristic_scores[heuristic_mask]}
        )
        heuristic_candidates["timestamp"] = panel.loc[heuristic_candidates["position"], "known_at"].to_numpy()
        heuristic_dispatched = _weekly_cap(heuristic_candidates, panel, maximum_signals_per_week=float(evaluation["maximum_signals_per_week"]))
        heuristic_positions = set(heuristic_dispatched["position"].astype(int))

        model_metrics = evaluate_positions(panel, model_positions, direction="favorable", horizon=long_horizon, base_positions=test_positions, labels=labels)
        heuristic_metrics = evaluate_positions(panel, heuristic_positions, direction="favorable", horizon=long_horizon, base_positions=test_positions, labels=labels)
        fold_rows.append(
            {
                "fold": fold.name,
                "test_start": str(panel.loc[fold.test_positions[0], "value_date"]),
                "test_end": str(panel.loc[fold.test_positions[-1], "value_date"]),
                "purge_observations": fold.purged_tail_observations,
                "outer_training_observations": len(fold.train_positions),
                "final_refit_observations": len(final_model.train_positions),
                "final_score_reference_observations": len(final_model.train_scores),
                "model_score_threshold": model_threshold,
                "heuristic_score_threshold": heuristic_threshold,
                "test_candidate_count": len(test_positions),
                "model_signal_count": model_metrics["signal_count"],
                "model_signals_per_week": model_metrics["signals_per_week"],
                "model_hit_rate": model_metrics["hit_rate"],
                "model_candidate_baseline_hit_rate": model_metrics["baseline_hit_rate"],
                "model_lift": model_metrics["lift"],
                "heuristic_signal_count": heuristic_metrics["signal_count"],
                "heuristic_signals_per_week": heuristic_metrics["signals_per_week"],
                "heuristic_hit_rate": heuristic_metrics["hit_rate"],
                "heuristic_lift": heuristic_metrics["lift"],
            }
        )
        test_labels = labels_by_position.loc[list(test_positions)]
        weeks = _week_by_position(panel, test_positions)
        for position, score, heuristic_score, week, row in zip(
            test_positions,
            model_scores,
            heuristic_scores,
            weeks,
            test_labels.itertuples(),
            strict=True,
        ):
            prediction_rows.append(
                {
                    "fold": fold.name,
                    "value_date": str(panel.loc[position, "value_date"]),
                    "position": position,
                    "week": week,
                    "hit_favorable": bool(row.hit_favorable),
                    "short_regret_favorable": bool(row.short_regret_favorable),
                    "long_regret_favorable": bool(row.long_regret_favorable),
                    "short_future_regret_bps": float(row.short_future_regret_bps),
                    "long_future_regret_bps": float(row.future_regret_bps),
                    "model_score": float(score),
                    "heuristic_score": float(-heuristic_score) if np.isfinite(heuristic_score) else None,
                    "model_dispatched": position in model_positions,
                    "heuristic_dispatched": position in heuristic_positions,
                }
            )
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    predictions_frame = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    summary_frame = _summary(predictions_frame, panel=panel, labels=labels, long_horizon=long_horizon)
    random_frame = _matched_random_summary(predictions_frame, samples=int(evaluation["matched_random_samples"]))
    bootstrap_frame = _bootstrap_lifts(predictions_frame, long_horizon=long_horizon, samples=int(evaluation["block_bootstrap_samples"]))
    folds_frame.to_csv(output / "folds.csv", index=False)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    random_frame.to_csv(output / "frequency_matched_random.csv", index=False)
    bootstrap_frame.to_csv(output / "block_bootstrap_lift.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "target": target_config,
        "purge_observations": f"max(short_horizon, long_horizon) = {long_horizon}",
        "model": "one fixed XGBoost; no same-history model selection",
        "threshold": "80th percentile of final_refit_model.train_scores; in-sample score reference for rank only, not probability calibration",
        "communication_policy": "no forced sends; at most two chronological candidates per week; readiness requires realized average in [1, 2]",
        "baselines": "weekly-frequency-matched random policy and untuned 20-session historical-low heuristic",
        "status": "exploratory: target was proposed after earlier CNY/RUB evaluation and requires a new corridor or future hold-out for confirmation",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/dual_regret_policy.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/dual_regret_policy"))
    args = parser.parse_args(argv)
    run_dual_regret_policy(snapshot_dir=args.snapshot, config_path=args.config, artifact_dir=args.artifact_dir)
    print(f"Wrote dual-regret policy benchmark to {args.artifact_dir}")


if __name__ == "__main__":
    main()


__all__ = ["load_dual_regret_policy_config", "run_dual_regret_policy"]
