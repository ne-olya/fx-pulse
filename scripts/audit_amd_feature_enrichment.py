"""Bootstrap the saved AMD feature-enrichment policies without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.innovation_followup import _weekly_winner_table, moving_block_bootstrap  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--block-weeks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def _weekly(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    feature_set: str,
    consensus_variant: str,
) -> pd.DataFrame:
    target = scores.loc[
        scores["corridor"].eq("AMD") & scores["feature_set"].eq(feature_set),
        ["timestamp", "test_year", "week", "target"],
    ].copy()
    selected = signals.loc[
        signals["corridor"].eq("AMD")
        & signals["feature_set"].eq(feature_set)
        & signals["consensus_variant"].eq(consensus_variant)
    ].copy()
    return _weekly_winner_table(target, selected).sort_values(["test_year", "week"], kind="mergesort")


def _lifts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    totals = values.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (totals[:, 1] / totals[:, 0]) / (totals[:, 4] / totals[:, 3])
        matched = totals[:, 1] / totals[:, 2]
    return raw, matched


def _paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    samples: int,
    block_weeks: int,
    seed: int,
) -> dict[str, float]:
    keys = ["test_year", "week"]
    columns = ["signal_count", "signal_hits", "matched_expected_hits", "base_count", "base_hits"]
    paired = baseline[keys + columns].merge(
        candidate[keys + columns], on=keys, how="inner", validate="one_to_one", suffixes=("_baseline", "_candidate")
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("candidate and baseline weekly support differ")
    n = len(paired)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(samples, math.ceil(n / block_weeks)))
    offsets = np.arange(block_weeks)
    indices = ((starts[..., None] + offsets) % n).reshape(samples, -1)[:, :n]
    baseline_values = paired[[f"{column}_baseline" for column in columns]].to_numpy(dtype=float)[indices]
    candidate_values = paired[[f"{column}_candidate" for column in columns]].to_numpy(dtype=float)[indices]
    baseline_raw, baseline_matched = _lifts(baseline_values)
    candidate_raw, candidate_matched = _lifts(candidate_values)
    raw_delta = candidate_raw - baseline_raw
    matched_delta = candidate_matched - baseline_matched
    return {
        "paired_raw_delta_low": float(np.nanquantile(raw_delta, 0.025)),
        "paired_raw_delta_median": float(np.nanmedian(raw_delta)),
        "paired_raw_delta_high": float(np.nanquantile(raw_delta, 0.975)),
        "paired_matched_delta_low": float(np.nanquantile(matched_delta, 0.025)),
        "paired_matched_delta_median": float(np.nanmedian(matched_delta)),
        "paired_matched_delta_high": float(np.nanquantile(matched_delta, 0.975)),
    }


def main() -> None:
    args = _parse_args()
    scores = pd.read_csv(args.artifacts / "consensus_scores.csv.gz")
    signals = pd.read_csv(args.artifacts / "signals.csv")
    summary = pd.read_csv(args.artifacts / "summary.csv")
    amd = summary.loc[summary["corridor"].eq("AMD")].copy()
    comparisons = len(amd)
    audit_rows: list[dict[str, object]] = []
    weekly_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for index, row in amd.reset_index(drop=True).iterrows():
        key = (str(row["feature_set"]), str(row["consensus_variant"]))
        weekly = _weekly(scores, signals, feature_set=key[0], consensus_variant=key[1])
        weekly_tables[key] = weekly
        result = moving_block_bootstrap(
            weekly,
            samples=args.samples,
            block_weeks=args.block_weeks,
            seed=args.seed + index,
        )
        audit_rows.append(
            {
                "feature_set": key[0],
                "consensus_variant": key[1],
                "signals": int(row["signals"]),
                "raw_lift": float(row["raw_lift"]),
                "same_week_lift": float(row["same_week_lift"]),
                **result,
                "block_p_bonferroni": min(1.0, float(result["block_matched_p_value"]) * comparisons),
                "comparison_count": comparisons,
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values("same_week_lift", ascending=False, kind="mergesort")
    audit.to_csv(args.artifacts / "bootstrap_audit.csv", index=False)

    baseline_key = ("baseline", "official_mean")
    paired_rows: list[dict[str, object]] = []
    for index, key in enumerate(
        [("all_enriched", "official_mean"), ("all_enriched", "agreement_weighted"), ("all_enriched", "agreement_gate")]
    ):
        paired_rows.append(
            {
                "baseline_feature_set": baseline_key[0],
                "baseline_consensus_variant": baseline_key[1],
                "candidate_feature_set": key[0],
                "candidate_consensus_variant": key[1],
                **_paired_bootstrap(
                    weekly_tables[baseline_key],
                    weekly_tables[key],
                    samples=args.samples,
                    block_weeks=args.block_weeks,
                    seed=args.seed + 10_000 + index,
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.artifacts / "paired_bootstrap_vs_baseline.csv", index=False)
    source_paths = [
        REPO_ROOT / "src/fxpulse/feature_enrichment.py",
        REPO_ROOT / "scripts/run_amd_feature_enrichment_experiment.py",
        Path(__file__).resolve(),
    ]
    audit_meta = {
        "samples": args.samples,
        "block_weeks": args.block_weeks,
        "seed": args.seed,
        "comparison_count": comparisons,
        "conditional_on_fixed_saved_policies": True,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
        "run_meta_sha256": _sha256(args.artifacts / "run_meta.json"),
    }
    (args.artifacts / "audit_meta.json").write_text(
        json.dumps(audit_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(audit.head(10).to_string(index=False))
    print("\nPaired deltas vs baseline official_mean")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
