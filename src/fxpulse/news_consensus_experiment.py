"""Test whether recipient-corridor agreement strengthens AI-GPR scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.innovation_followup import causal_percentile
from fxpulse.news_robust_experiment import (
    apply_continuous_policy,
    horizon_summary,
    paired_comparison,
    period_summary,
    summarize,
    yearly_summary,
)

EXPECTED_FEATURE_SETS = [
    "market_rank",
    "ai_gpr_own_rank",
    "ai_gpr_peer25",
    "ai_gpr_peer50",
    "liquidity_ai_gpr_own_rank",
    "liquidity_ai_gpr_peer25",
]
IDENTITY = [
    "timestamp",
    "corridor",
    "horizon",
    "test_year",
    "target",
    "regret_bps",
    "benefit_bps",
]


def load_config(
    path: Path | str = Path("configs/news_consensus_experiment.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "registered_after_ai_gpr_before_consensus_results":
        raise ValueError("news consensus config must be registered before consensus results")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("news consensus feature sets differ from implementation")
    return config


def peer_blend(frame: pd.DataFrame, column: str, *, own_weight: float) -> pd.Series:
    """Blend each corridor rank only with the other corridors on the same date."""

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for indices in frame.groupby(["horizon", "timestamp"], sort=False).groups.values():
        current = frame.loc[indices, column]
        for index in indices:
            own = float(current.loc[index])
            peers = current.drop(index).dropna()
            if np.isfinite(own) and not peers.empty:
                result.loc[index] = own_weight * own + (1 - own_weight) * float(peers.mean())
    return result


def build_scores(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    source = data.loc[data["feature_set"].isin(config["source_feature_sets"])].copy()
    wide = source.pivot(index=IDENTITY, columns="feature_set", values="score").reset_index()
    missing = set(config["source_feature_sets"]) - set(wide)
    if missing:
        raise ValueError(f"parent scores lack {sorted(missing)}")

    for source_name in config["source_feature_sets"]:
        rank_column = f"rank__{source_name}"
        wide[rank_column] = np.nan
        for indices in wide.groupby(["corridor", "horizon"], sort=False).groups.values():
            ordered = wide.loc[indices].sort_values("timestamp")
            wide.loc[ordered.index, rank_column] = causal_percentile(
                ordered[source_name],
                lookback=int(config["causal_rank_lookback"]),
                minimum_history=int(config["minimum_rank_history"]),
            ).to_numpy()

    wide["market_rank"] = wide["rank__market_only"]
    wide["ai_gpr_own_rank"] = wide["rank__plus_all_ai_gpr"]
    wide["ai_gpr_peer25"] = peer_blend(
        wide, "rank__plus_all_ai_gpr", own_weight=0.75
    )
    wide["ai_gpr_peer50"] = peer_blend(
        wide, "rank__plus_all_ai_gpr", own_weight=0.50
    )
    wide["liquidity_ai_gpr_own_rank"] = wide["rank__plus_liquidity_all_ai_gpr"]
    wide["liquidity_ai_gpr_peer25"] = peer_blend(
        wide, "rank__plus_liquidity_all_ai_gpr", own_weight=0.75
    )
    long = wide.melt(
        id_vars=IDENTITY,
        value_vars=config["feature_sets"],
        var_name="feature_set",
        value_name="score",
    )
    return long.loc[long["score"].notna()].sort_values(
        ["feature_set", "corridor", "horizon", "timestamp"], kind="mergesort"
    )


def run(
    *,
    config_path: Path | str = Path("configs/news_consensus_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_consensus"),
) -> dict[str, object]:
    config = load_config(config_path)
    parent = Path(config["parent_scores"])
    scores = build_scores(pd.read_csv(parent), config)
    signals = apply_continuous_policy(scores, config)
    summary_frame = summarize(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    per_year = yearly_summary(scores, signals)
    comparison = paired_comparison(
        summary_frame,
        list(config["feature_sets"]),
        baseline_name="market_rank",
    )
    pseudo = period_summary(scores, signals, list(config["pseudo_holdout_years"]))
    pseudo_by_horizon = horizon_summary(
        scores.loc[scores["test_year"].isin(config["pseudo_holdout_years"])],
        signals.loc[signals["test_year"].isin(config["pseudo_holdout_years"])],
    )
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv.gz", index=False, compression="gzip")
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_2025_2026.csv", index=False)
    pseudo_by_horizon.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    metadata = {
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "parent_scores_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": "Post-hoc cross-corridor news ensemble on reviewed OOT history.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_consensus_experiment.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news_consensus")
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
