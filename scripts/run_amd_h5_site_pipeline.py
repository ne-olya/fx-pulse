"""Materialize the h=5 AMD demonstration run from the frozen research pipeline.

The browser must never fit a model or derive outcomes from future prices.  This
script performs the chronological OOT run once and writes a versioned artifact
that a separate UI exporter can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fxpulse.innovation_followup import (
    KEY_COLUMNS,
    _add_score_combinations,
    evaluate_followup_scores,
    load_config as load_followup_config,
)
from fxpulse.next_hypotheses import build_datasets
from fxpulse.robust_innovation_experiment import (
    evaluate_innovations,
    load_config as load_robust_config,
    summarize_folds,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbr", required=True, type=Path, help="CBR daily-rate CSV.")
    parser.add_argument("--market", required=True, type=Path, help="Daily MOEX research panel CSV.")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run's artifacts.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for path in (args.cbr, args.market):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)

    robust = load_robust_config()
    robust["horizons"] = [5]
    datasets = build_datasets(
        cbr_path=args.cbr,
        research_panel_path=args.market,
        corridors=list(robust["corridors"]),
        horizons=[5],
        market_lag=1,
    )
    features = pd.concat(datasets.values(), ignore_index=True)
    wave_one_folds, wave_one_scores, _ = evaluate_innovations(features, robust)

    followup = load_followup_config()
    followup["horizons"] = [5]
    source = wave_one_scores.loc[wave_one_scores["variant"].isin(followup["source_variants"])].copy()
    wide = source.pivot(index=KEY_COLUMNS, columns="variant", values="score").reset_index()
    wide = wide.sort_values(["corridor", "horizon", "timestamp"], kind="mergesort").reset_index(drop=True)
    consensus = _add_score_combinations(wide, followup).melt(
        id_vars=KEY_COLUMNS,
        value_vars=["leave_one_corridor_consensus"],
        var_name="variant",
        value_name="score",
    )
    consensus = consensus.dropna(subset=["score"])
    folds, signals = evaluate_followup_scores(consensus, followup)
    summary = summarize_folds(folds)

    features.to_csv(args.output / "features_h5.csv", index=False)
    consensus.to_csv(
        args.output / "consensus_scores_h5.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    signals.to_csv(args.output / "consensus_signals_h5.csv", index=False)
    folds.to_csv(args.output / "consensus_folds_h5.csv", index=False)
    summary.to_csv(args.output / "consensus_summary_h5.csv", index=False)
    (args.output / "run_meta.json").write_text(
        json.dumps(
            {
                "pipeline": "amd_h5_site_pipeline",
                "horizon_observations": 5,
                "evaluation_target": "regret_bps <= 25",
                "training_label": "triple_barrier_25_50_bps",
                "source_variant": "rolling_4y_catboost",
                "policy_variant": "leave_one_corridor_consensus",
                "test_years": [2022, 2023, 2024, 2025, 2026],
                "inputs": {
                    "cbr": {"path": str(args.cbr), "sha256": _sha256(args.cbr)},
                    "market": {"path": str(args.market), "sha256": _sha256(args.market)},
                },
                "row_counts": {
                    "features": int(len(features)),
                    "consensus_scores": int(len(consensus)),
                    "signals": int(len(signals)),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    amd = summary.loc[summary["corridor"].eq("AMD")].iloc[0]
    print(
        "AMD h=5: "
        f"signals={int(amd['signals'])}, raw_lift={amd['lift']:.4f}, "
        f"same_week_lift={amd['lift_vs_matched_random']:.4f}, "
        f"signals_per_week={amd['signals_per_week']:.4f}"
    )


if __name__ == "__main__":
    main()
