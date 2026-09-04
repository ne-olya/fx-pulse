"""Combine pooled shared scores with corridor-specific model heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _apply_policy
from fxpulse.temporal_sequence_experiment import summarize
from fxpulse.value_downside_experiment import causal_percentile


def load_config(path: Path | str = Path("configs/shared_head_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("shared-head config must be preregistered")
    return config


def combine(individual: pd.Series, pooled: pd.Series, method: str) -> pd.Series:
    if method == "individual_only":
        return individual
    if method == "pooled_only":
        return pooled
    if method == "balanced_mean":
        return (individual + pooled) / 2
    if method == "agreement_min":
        return pd.concat([individual, pooled], axis=1).min(axis=1)
    raise ValueError(f"unknown combination {method}")


def evaluate(scores: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    individual = source.loc[source["scope"].ne("pooled")].copy()
    pooled = source.loc[source["scope"].eq("pooled"), [
        "timestamp", "corridor", "model", "horizon", "test_year", "score"
    ]].rename(columns={"score": "pooled_score"})
    data = individual.merge(
        pooled,
        on=["timestamp", "corridor", "model", "horizon", "test_year"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={"score": "individual_score"})
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for keys, fold in data.groupby(["model", "horizon", "test_year"], sort=True):
        model, horizon, year = keys
        fold = fold.sort_values(["corridor", "timestamp"], kind="mergesort").copy()
        for column in ["individual_score", "pooled_score"]:
            fold[f"{column}_rank"] = fold.groupby("corridor", sort=False)[column].transform(
                lambda values: causal_percentile(
                    values,
                    lookback=int(config["lookback_observations"]),
                    minimum_history=int(config["minimum_history_observations"]),
                )
            )
        fold["week"] = fold["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = fold.groupby(["corridor", "week"])["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        exposure = int(fold["corridor"].nunique())
        duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7) * exposure
        for method in config["combinations"]:
            fold["score"] = combine(
                fold["individual_score_rank"], fold["pooled_score_rank"], method
            )
            candidates = pd.concat(
                [
                    adaptive_candidates(
                        group,
                        share=float(config["top_score_share"]),
                        lookback=int(config["lookback_observations"]),
                        minimum_history=int(config["minimum_history_observations"]),
                    )
                    for _, group in fold.groupby("corridor", sort=False)
                ],
                ignore_index=False,
            )
            selected = pd.concat(
                [
                    _apply_policy(
                        group,
                        cooldown_days=int(config["cooldown_days"]),
                        weekly_cap=int(config["weekly_cap"]),
                    )
                    for _, group in candidates.groupby("corridor", sort=False)
                ],
                ignore_index=False,
            )
            count = len(selected)
            base_rate = float(fold["target"].mean())
            hit_rate = float(selected["target"].mean()) if count else np.nan
            identity = {
                "scope": "shared_plus_head",
                "model": str(model),
                "feature_set": method,
                "horizon": int(horizon),
            }
            fold_rows.append(
                {
                    **identity,
                    "test_year": int(year),
                    "test_count": len(fold),
                    "test_hits": int(fold["target"].sum()),
                    "signal_count": count,
                    "signal_hits": int(selected["target"].sum()),
                    "lift": hit_rate / base_rate if base_rate > 0 else np.nan,
                    "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()),
                    "regret_sum_bps": float(selected["regret_bps"].sum()),
                    "benefit_sum_bps": float(selected["benefit_bps"].sum()),
                    "duration_weeks": duration,
                }
            )
            exported = selected[[
                "timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score",
                "score_threshold", "individual_score", "pooled_score", "matched_week_hit_rate",
            ]].copy()
            for key, value in identity.items():
                exported[key] = value
            exported["test_year"] = int(year)
            signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def run(
    *,
    config_path: Path | str = Path("configs/shared_head_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/shared_head"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["scores_input"])
    folds, signals = evaluate(pd.read_csv(input_path), config)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory score ensemble; causal ranks use only earlier scores in each OOT year.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared_head_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/shared_head"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
