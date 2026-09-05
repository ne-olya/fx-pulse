"""Post-hoc robustness check of U08 with the exact current CatBoost budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fxpulse.external_untested_hypotheses import _add_evaluation_fields, _purged_train
from fxpulse.training_history_experiment import add_training_labels
from fxpulse.untested_hypotheses_experiment import (
    BASELINE,
    _fit_catboost_ensemble,
    _fold_boundaries,
    add_u08_score_views,
    bootstrap_deltas,
    compare_to_baseline,
    select_standard_signals,
    selection_calibrated_signals,
    summarize,
)


def run(*, config_path: Path | str, artifact_dir: Path | str) -> dict[str, object]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "posthoc_robustness_after_exploratory_results":
        raise ValueError("this run must be explicitly labelled post-hoc robustness")
    frame = pd.read_csv(config["input"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = add_training_labels(frame, config).sort_values(["corridor", "timestamp"], kind="mergesort")
    columns = [column for column in frame if column.startswith(tuple(config["feature_prefixes"]))]
    horizon = 5
    label = "training_target_5"
    outcome = "outcome__regret_5"
    parts = []
    for corridor in config["corridors"]:
        usable = frame.loc[
            frame["corridor"].eq(corridor) & frame[outcome].notna() & frame[label].notna()
        ].copy()
        usable = _add_evaluation_fields(
            usable, horizon=horizon, tolerance_bps=float(config["evaluation_tolerance_bps"])
        )
        for fold_index, (start, end, fold_name) in enumerate(_fold_boundaries(config)):
            train = _purged_train(
                usable, start, horizon, int(config["rolling_training_years"])
            )
            test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
            if (
                len(train) < int(config["minimum_training_observations"])
                or len(test) < 20
                or train[label].nunique() < 2
            ):
                continue
            export = test[
                ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate"]
            ].copy()
            export["variant"] = BASELINE
            export["horizon"] = horizon
            export["test_fold"] = fold_name
            export["score"] = _fit_catboost_ensemble(
                train,
                test,
                columns,
                label,
                config=config,
                seed_offset=30_000 + 100 * fold_index + list(config["corridors"]).index(corridor),
            )
            parts.append(export)
        print(f"completed exact B0 {corridor}", flush=True)
    base_scores = pd.concat(parts, ignore_index=True)
    u08_signals = selection_calibrated_signals(base_scores, config)
    scores = add_u08_score_views(base_scores, config)
    signals = pd.concat([select_standard_signals(scores, config), u08_signals], ignore_index=True, sort=False)
    corridor, aggregate, detail = summarize(scores, signals)
    comparison, gates = compare_to_baseline(corridor, aggregate, detail, config)
    bootstrap = bootstrap_deltas(signals, config=config)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    corridor.to_csv(output / "corridor_summary.csv", index=False)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    detail.to_csv(output / "fold_and_year_summary.csv", index=False)
    comparison.to_csv(output / "paired_comparison.csv", index=False)
    gates.to_csv(output / "success_gates.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_h5.csv", index=False)
    meta = {
        "status": config["status"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(Path(config["input"]).read_bytes()).hexdigest(),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": "Post-hoc robustness, not an independent confirmation; test history had already been viewed.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/u08_exact_baseline_robustness.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/untested_hypotheses/u08_exact_robustness"))
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
