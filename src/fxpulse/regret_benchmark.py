"""Sensitivity benchmark for a cost-aware FX "buy now" recommendation.

The outcome is a future-only regret budget, not an exact hindsight minimum:
``future_regret_bps <= tau`` means that waiting for the next ``h`` sessions
would not have improved the MOEX close by more than ``tau`` basis points.
Each tolerance is a separate preregistered experiment; it is never selected on
an outer test window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.labeling import evaluate_positions, label_observations
from fxpulse.local_minimum_models import ModelKind, load_local_minimum_config, run_local_minimum_models
from fxpulse.rule_selection import _target_panel, load_universe_prices, resolve_snapshot


FOLDWISE_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "candidate_policy",
    "model",
    "selection_status",
    "communication_allowed",
    "fold_count",
    "signal_count",
    "lift_fold_count",
    "lift_median",
    "lift_iqr_low",
    "lift_iqr_high",
    "lift_worst",
    "lift_best",
    "regret_mean_bps_median",
    "regret_p90_bps_worst",
)

BOOTSTRAP_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "candidate_policy",
    "model",
    "selection_status",
    "communication_allowed",
    "block_length_observations",
    "bootstrap_samples",
    "usable_samples",
    "lift_bootstrap_median",
    "lift_ci_95_low",
    "lift_ci_95_high",
)

PRODUCT_COLUMNS = (
    "tolerance_bps",
    "horizon_observations",
    "candidate_policy",
    "policy",
    "promoted_fold_count",
    "signal_count",
    "eligible_candidate_count",
    "hit_rate",
    "candidate_baseline_hit_rate",
    "lift",
    "benefit_fwd_bps",
    "regret_mean_bps",
    "regret_median_bps",
    "regret_p90_bps",
    "baseline_regret_mean_bps",
    "baseline_regret_p90_bps",
    "signals_per_week",
)


def _tolerance_directory(value: float) -> str:
    return f"tau_{value:g}_bps".replace(".", "_").replace("-", "minus_")


def _validate_tolerances(config: dict[str, Any]) -> tuple[float, ...]:
    values = config["evaluation"].get("regret_tolerance_bps")
    if not isinstance(values, list) or not values:
        raise ValueError("evaluation.regret_tolerance_bps must be a non-empty list")
    tolerances = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("regret_tolerance_bps must contain finite non-negative numbers")
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("regret_tolerance_bps must not contain duplicates")
    return tolerances


def _foldwise_summary(folds: pd.DataFrame, *, tolerance_bps: float) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame(columns=FOLDWISE_COLUMNS)
    groups = ["horizon_observations", "candidate_policy", "model", "selection_status"]
    rows: list[dict[str, object]] = []
    decision_rows = folds.dropna(subset=["model", "selection_status"]).copy()
    for values, group in decision_rows.groupby(groups, sort=True):
        horizon, policy, model, status = values
        lifts = pd.to_numeric(group["test_lift"], errors="coerce").dropna()
        regrets = pd.to_numeric(group["test_regret_mean_bps"], errors="coerce").dropna()
        regret_p90 = pd.to_numeric(group["test_regret_p90_bps"], errors="coerce").dropna()
        rows.append(
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": int(horizon),
                "candidate_policy": policy,
                "model": model,
                "selection_status": status,
                "communication_allowed": status == "promoted",
                "fold_count": int(len(group)),
                "signal_count": int(pd.to_numeric(group["test_dispatched_signal_count"], errors="coerce").fillna(0).sum()),
                "lift_fold_count": int(len(lifts)),
                "lift_median": float(lifts.median()) if not lifts.empty else None,
                "lift_iqr_low": float(lifts.quantile(0.25)) if not lifts.empty else None,
                "lift_iqr_high": float(lifts.quantile(0.75)) if not lifts.empty else None,
                "lift_worst": float(lifts.min()) if not lifts.empty else None,
                "lift_best": float(lifts.max()) if not lifts.empty else None,
                "regret_mean_bps_median": float(regrets.median()) if not regrets.empty else None,
                "regret_p90_bps_worst": float(regret_p90.max()) if not regret_p90.empty else None,
            }
        )
    return pd.DataFrame(rows, columns=FOLDWISE_COLUMNS)


def _moving_block_sample(length: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exactly ``length`` observations from non-wrapping moving blocks."""

    if length <= 0:
        return np.empty(0, dtype=int)
    block = min(block_length, length)
    starts = rng.integers(0, length - block + 1, size=math.ceil(length / block))
    sample = np.concatenate([np.arange(start, start + block, dtype=int) for start in starts])
    return sample[:length]


def _bootstrap_lift(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int,
    candidate_policy: str,
    model: str | None,
    selection_status: str,
    communication_allowed: bool,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Moving-block CI conditional on the already selected OOT policy.

    The resampling unit is a consecutive run of eligible test sessions within a
    quarter.  Model fitting and selection are held fixed: this quantifies the
    uncertainty of the observed test lift rather than repeating model search.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    matching_folds = folds.loc[
        (folds["horizon_observations"] == horizon)
        & (folds["candidate_policy"] == candidate_policy)
        & (folds["selection_status"] == selection_status)
    ].copy()
    if model is not None:
        matching_folds = matching_folds.loc[matching_folds["model"] == model].copy()
    if matching_folds.empty:
        return {"usable_samples": 0, "lift_bootstrap_median": None, "lift_ci_95_low": None, "lift_ci_95_high": None}
    group_signals = signals.loc[
        (signals["horizon_observations"] == horizon)
        & (signals["candidate_policy"] == candidate_policy)
        & (signals["selection_status"] == selection_status)
        & (signals["communication_allowed"] == communication_allowed)
    ].copy()
    if model is not None:
        group_signals = group_signals.loc[group_signals["model"] == model].copy()
    sent_dates_by_fold = {
        str(fold): set(group["value_date"].astype(str)) for fold, group in group_signals.groupby("fold", sort=False)
    }
    labels = labels.copy()
    labels["value_date_string"] = labels["value_date"].astype(str)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for row in matching_folds.itertuples(index=False):
        test_labels = labels.loc[
            labels["value_date_string"].between(str(row.test_start), str(row.test_end), inclusive="both")
        ].copy()
        if candidate_policy == "past_min_gate":
            test_labels = test_labels.loc[test_labels["past_min"]].copy()
        sent_dates = sent_dates_by_fold.get(str(row.fold), set())
        baseline = test_labels["hit_favorable"].astype(float).to_numpy()
        dispatched = test_labels["value_date_string"].isin(sent_dates).to_numpy()
        if len(baseline):
            blocks.append((baseline, dispatched))
    if not blocks:
        return {"usable_samples": 0, "lift_bootstrap_median": None, "lift_ci_95_low": None, "lift_ci_95_high": None}
    rng = np.random.default_rng(seed)
    block_length = max(1, horizon)
    lifts: list[float] = []
    for _ in range(samples):
        sampled_baselines: list[np.ndarray] = []
        sampled_signals: list[np.ndarray] = []
        for baseline, dispatched in blocks:
            indices = _moving_block_sample(len(baseline), block_length=block_length, rng=rng)
            sampled_baselines.append(baseline[indices])
            sampled_signals.append(baseline[indices][dispatched[indices]])
        pooled_baseline = np.concatenate(sampled_baselines)
        pooled_signals = np.concatenate(sampled_signals)
        if len(pooled_signals) == 0:
            continue
        baseline_rate = float(pooled_baseline.mean())
        if baseline_rate > 0:
            lifts.append(float(pooled_signals.mean()) / baseline_rate)
    if not lifts:
        return {"usable_samples": 0, "lift_bootstrap_median": None, "lift_ci_95_low": None, "lift_ci_95_high": None}
    distribution = np.asarray(lifts, dtype="float64")
    return {
        "usable_samples": int(len(distribution)),
        "lift_bootstrap_median": float(np.median(distribution)),
        "lift_ci_95_low": float(np.quantile(distribution, 0.025)),
        "lift_ci_95_high": float(np.quantile(distribution, 0.975)),
    }


def _bootstrap_summary(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    signals: pd.DataFrame,
    tolerance_bps: float,
    samples: int,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    groups = ["horizon_observations", "candidate_policy", "model", "selection_status", "communication_allowed"]
    rows: list[dict[str, object]] = []
    for values, _ in signals.groupby(groups, sort=True):
        horizon, policy, model, status, allowed = values
        result = _bootstrap_lift(
            panel=panel,
            labels=labels,
            folds=folds,
            signals=signals,
            horizon=int(horizon),
            candidate_policy=str(policy),
            model=str(model),
            selection_status=str(status),
            communication_allowed=bool(allowed),
            samples=samples,
            seed=20260903 + int(round(tolerance_bps * 100)) + int(horizon),
        )
        rows.append(
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": int(horizon),
                "candidate_policy": policy,
                "model": model,
                "selection_status": status,
                "communication_allowed": bool(allowed),
                "block_length_observations": max(1, int(horizon)),
                "bootstrap_samples": samples,
                **result,
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def _candidate_positions_for_folds(
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    candidate_policy: str,
) -> tuple[int, ...]:
    """Return the eligible OOT positions for a selected set of test quarters."""

    labels = labels.copy()
    labels["value_date_string"] = labels["value_date"].astype(str)
    positions: list[int] = []
    for row in folds.itertuples(index=False):
        group = labels.loc[
            labels["value_date_string"].between(str(row.test_start), str(row.test_end), inclusive="both")
        ].copy()
        if candidate_policy == "past_min_gate":
            group = group.loc[group["past_min"]]
        positions.extend(group["position"].astype(int).tolist())
    return tuple(sorted(set(positions)))


def _product_policy_summary(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
    signals: pd.DataFrame,
    tolerance_bps: float,
    horizon: int,
    candidate_policy: str,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pool the actually dispatchable per-fold winners into one product policy."""

    promoted_folds = folds.loc[
        (folds["horizon_observations"] == horizon)
        & (folds["candidate_policy"] == candidate_policy)
        & (folds["selection_status"] == "promoted")
    ].copy()
    dispatched = signals.loc[
        (signals["horizon_observations"] == horizon)
        & (signals["candidate_policy"] == candidate_policy)
        & (signals["selection_status"] == "promoted")
        & (signals["communication_allowed"])
    ].copy()
    base_positions = _candidate_positions_for_folds(labels, promoted_folds, candidate_policy=candidate_policy)
    label_position_by_date = dict(zip(labels["value_date"].astype(str), labels["position"], strict=True))
    signal_positions = [
        int(label_position_by_date[value_date])
        for value_date in dispatched["value_date"].astype(str)
        if value_date in label_position_by_date
    ]
    metrics = evaluate_positions(
        panel,
        signal_positions,
        direction="favorable",
        horizon=horizon,
        base_positions=base_positions,
        labels=labels,
    )
    product = pd.DataFrame(
        [
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": horizon,
                "candidate_policy": candidate_policy,
                "policy": "per_fold_inner_selected_promoted_models",
                "promoted_fold_count": int(len(promoted_folds)),
                "signal_count": metrics["signal_count"],
                "eligible_candidate_count": metrics["eligible_base_count"],
                "hit_rate": metrics["hit_rate"],
                "candidate_baseline_hit_rate": metrics["baseline_hit_rate"],
                "lift": metrics["lift"],
                "benefit_fwd_bps": metrics["benefit_fwd_bps"],
                "regret_mean_bps": metrics["regret_mean_bps"],
                "regret_median_bps": metrics["regret_median_bps"],
                "regret_p90_bps": metrics["regret_p90_bps"],
                "baseline_regret_mean_bps": metrics["baseline_regret_mean_bps"],
                "baseline_regret_p90_bps": metrics["baseline_regret_p90_bps"],
                "signals_per_week": metrics["signals_per_week"],
            }
        ],
        columns=PRODUCT_COLUMNS,
    )
    lifts = pd.to_numeric(promoted_folds["test_lift"], errors="coerce").dropna()
    regrets = pd.to_numeric(promoted_folds["test_regret_mean_bps"], errors="coerce").dropna()
    regret_p90 = pd.to_numeric(promoted_folds["test_regret_p90_bps"], errors="coerce").dropna()
    foldwise = pd.DataFrame(
        [
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": horizon,
                "candidate_policy": candidate_policy,
                "model": "per_fold_selected",
                "selection_status": "promoted",
                "communication_allowed": True,
                "fold_count": int(len(promoted_folds)),
                "signal_count": int(pd.to_numeric(promoted_folds["test_dispatched_signal_count"], errors="coerce").fillna(0).sum()),
                "lift_fold_count": int(len(lifts)),
                "lift_median": float(lifts.median()) if not lifts.empty else None,
                "lift_iqr_low": float(lifts.quantile(0.25)) if not lifts.empty else None,
                "lift_iqr_high": float(lifts.quantile(0.75)) if not lifts.empty else None,
                "lift_worst": float(lifts.min()) if not lifts.empty else None,
                "lift_best": float(lifts.max()) if not lifts.empty else None,
                "regret_mean_bps_median": float(regrets.median()) if not regrets.empty else None,
                "regret_p90_bps_worst": float(regret_p90.max()) if not regret_p90.empty else None,
            }
        ],
        columns=FOLDWISE_COLUMNS,
    )
    bootstrap_result = _bootstrap_lift(
        panel=panel,
        labels=labels,
        folds=promoted_folds,
        signals=dispatched,
        horizon=horizon,
        candidate_policy=candidate_policy,
        model=None,
        selection_status="promoted",
        communication_allowed=True,
        samples=bootstrap_samples,
        seed=20260903 + int(round(tolerance_bps * 100)) + horizon,
    )
    bootstrap = pd.DataFrame(
        [
            {
                "tolerance_bps": tolerance_bps,
                "horizon_observations": horizon,
                "candidate_policy": candidate_policy,
                "model": "per_fold_selected",
                "selection_status": "promoted",
                "communication_allowed": True,
                "block_length_observations": max(1, horizon),
                "bootstrap_samples": bootstrap_samples,
                **bootstrap_result,
            }
        ],
        columns=BOOTSTRAP_COLUMNS,
    )
    return product, foldwise, bootstrap


def run_regret_benchmark(
    *,
    snapshot_dir: Path | str | None = None,
    config_path: Path | str = Path("configs/regret_models.json"),
    artifact_dir: Path | str = Path("artifacts/regret_models"),
    models: tuple[ModelKind, ...] | None = None,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    """Run the preregistered future-regret sensitivity grid."""

    config = load_local_minimum_config(config_path)
    tolerances = _validate_tolerances(config)
    evaluation = config["evaluation"]
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = resolve_snapshot(snapshot_dir)
    target = str(evaluation["target_instrument_id"])
    prices = load_universe_prices(snapshot, target_instrument_id=target)
    panel = _target_panel(prices, target).reset_index(drop=True)
    all_folds: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_foldwise: list[pd.DataFrame] = []
    all_bootstrap: list[pd.DataFrame] = []
    product_summaries: list[pd.DataFrame] = []
    product_foldwise: list[pd.DataFrame] = []
    product_bootstrap: list[pd.DataFrame] = []
    for tolerance_bps in tolerances:
        tolerance_output = output / _tolerance_directory(tolerance_bps)
        run_local_minimum_models(
            snapshot_dir=snapshot,
            config_path=config_path,
            artifact_dir=tolerance_output,
            models=models,
            tolerance_bps=tolerance_bps,
        )
        folds = pd.read_csv(tolerance_output / "folds.csv")
        signals = pd.read_csv(tolerance_output / "signals.csv")
        summary = pd.read_csv(tolerance_output / "summary.csv")
        for frame in (folds, signals, summary):
            frame.insert(0, "tolerance_bps", tolerance_bps)
        all_folds.append(folds)
        all_signals.append(signals)
        all_summaries.append(summary)
        all_foldwise.append(_foldwise_summary(folds, tolerance_bps=tolerance_bps))
        for horizon in evaluation["horizons"]:
            horizon_signals = signals.loc[signals["horizon_observations"] == int(horizon)].copy()
            horizon_folds = folds.loc[folds["horizon_observations"] == int(horizon)].copy()
            labels = label_observations(panel, int(horizon), tolerance_bps=tolerance_bps)
            all_bootstrap.append(
                _bootstrap_summary(
                    panel=panel,
                    labels=labels,
                    folds=horizon_folds,
                    signals=horizon_signals,
                    tolerance_bps=tolerance_bps,
                    samples=bootstrap_samples,
                )
            )
            product, product_folds, product_ci = _product_policy_summary(
                panel=panel,
                labels=labels,
                folds=horizon_folds,
                signals=horizon_signals,
                tolerance_bps=tolerance_bps,
                horizon=int(horizon),
                candidate_policy=str(evaluation["candidate_policy"]),
                bootstrap_samples=bootstrap_samples,
            )
            product_summaries.append(product)
            product_foldwise.append(product_folds)
            product_bootstrap.append(product_ci)
    folds_frame = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    signals_frame = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    summary_frame = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    foldwise_frame = pd.concat(all_foldwise, ignore_index=True) if all_foldwise else pd.DataFrame(columns=FOLDWISE_COLUMNS)
    bootstrap_frame = pd.concat(all_bootstrap, ignore_index=True) if all_bootstrap else pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    product_summary_frame = pd.concat(product_summaries, ignore_index=True) if product_summaries else pd.DataFrame(columns=PRODUCT_COLUMNS)
    product_foldwise_frame = pd.concat(product_foldwise, ignore_index=True) if product_foldwise else pd.DataFrame(columns=FOLDWISE_COLUMNS)
    product_bootstrap_frame = pd.concat(product_bootstrap, ignore_index=True) if product_bootstrap else pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    folds_frame.to_csv(output / "folds.csv", index=False)
    signals_frame.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    foldwise_frame.to_csv(output / "foldwise_lift_summary.csv", index=False)
    bootstrap_frame.to_csv(output / "block_bootstrap_lift.csv", index=False)
    product_summary_frame.to_csv(output / "product_policy_summary.csv", index=False)
    product_foldwise_frame.to_csv(output / "product_policy_foldwise_lift.csv", index=False)
    product_bootstrap_frame.to_csv(output / "product_policy_block_bootstrap_lift.csv", index=False)
    meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "config_path": str(config_path),
        "target_instrument_id": target,
        "horizons": evaluation["horizons"],
        "tolerances_bps": tolerances,
        "evaluated_models": list(models or tuple(evaluation["models"])),
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": "horizon_observations",
            "scope": "conditional on each OOT-selected policy; moving blocks within outer test quarters",
        },
        "product_policy": "per-fold inner-selected model, only when its inner lift reached the predeclared promotion gate",
        "method": "separate preregistered tau experiments; expanding quarterly OOT walk-forward; h-observation purge; 252-observation inner selection; model and score quantile selected only in inner validation",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="manifest-gated universe snapshot; widest local snapshot is default")
    parser.add_argument("--config", type=Path, default=Path("configs/regret_models.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/regret_models"))
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--models", nargs="+", choices=(
        "ridge_logistic", "gradient_boosted_stumps", "xgboost", "random_forest", "extra_trees", "adaboost", "gradient_boosting", "svc_rbf",
    ))
    args = parser.parse_args(argv)
    meta = run_regret_benchmark(
        snapshot_dir=args.snapshot,
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        models=tuple(args.models) if args.models else None,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(f"Wrote regret benchmark for tau={meta['tolerances_bps']} bps to {args.artifact_dir}")


if __name__ == "__main__":
    main()


__all__ = ["run_regret_benchmark"]
