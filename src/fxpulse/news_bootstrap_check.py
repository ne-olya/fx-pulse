"""Block-bootstrap the GDELT lift increment over the market policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(
    path: Path | str = Path("configs/news_bootstrap_check.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "registered_after_partial_gdelt_before_bootstrap_results":
        raise ValueError("news bootstrap must be registered before bootstrap results")
    if config.get("candidates") != ["plus_all_news", "plus_russia_levels"]:
        raise ValueError("news bootstrap candidates differ from implementation")
    return config


def weekly_policy_table(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    feature_sets: list[str],
    horizon: int,
) -> pd.DataFrame:
    scores = scores.loc[
        scores["feature_set"].eq(feature_sets[0]) & scores["horizon"].eq(horizon)
    ].copy()
    signals = signals.loc[
        signals["feature_set"].isin(feature_sets) & signals["horizon"].eq(horizon)
    ].copy()
    for frame in (scores, signals):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        iso = frame["timestamp"].dt.isocalendar()
        frame["week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    base = scores.groupby("week", as_index=False).agg(
        base_count=("target", "size"),
        base_hits=("target", "sum"),
    )
    result = base
    for feature_set in feature_sets:
        current = (
            signals.loc[signals["feature_set"].eq(feature_set)]
            .groupby("week", as_index=False)
            .agg(
                signal_count=("target", "size"),
                signal_hits=("target", "sum"),
                matched_expected_hits=("matched_week_hit_rate", "sum"),
            )
            .rename(
                columns={
                    "signal_count": f"{feature_set}__count",
                    "signal_hits": f"{feature_set}__hits",
                    "matched_expected_hits": f"{feature_set}__expected",
                }
            )
        )
        result = result.merge(current, on="week", how="left", validate="one_to_one")
    value_columns = [column for column in result if column != "week"]
    result[value_columns] = result[value_columns].fillna(0.0)
    return result.sort_values("week", kind="mergesort").reset_index(drop=True)


def bootstrap_delta(
    weekly: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    samples: int,
    block_weeks: int,
    seed: int,
    previous_trials: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(weekly)
    block_count = math.ceil(n / block_weeks)
    starts = rng.integers(0, n, size=(samples, block_count))
    offsets = np.arange(block_weeks)
    indices = ((starts[..., None] + offsets) % n).reshape(samples, -1)[:, :n]

    def _lift(prefix: str) -> np.ndarray:
        hits = weekly[f"{prefix}__hits"].to_numpy(dtype=float)[indices].sum(axis=1)
        expected = weekly[f"{prefix}__expected"].to_numpy(dtype=float)[indices].sum(axis=1)
        return np.divide(hits, expected, out=np.full_like(hits, np.nan), where=expected > 0)

    baseline_lift = _lift(baseline)
    candidate_lift = _lift(candidate)
    delta = candidate_lift - baseline_lift
    alpha = 0.05
    adjusted_tail = alpha / (2 * previous_trials)
    return {
        "bootstrap_delta_mean": float(np.nanmean(delta)),
        "bootstrap_delta_ci95_low": float(np.nanquantile(delta, alpha / 2)),
        "bootstrap_delta_ci95_high": float(np.nanquantile(delta, 1 - alpha / 2)),
        "selection_adjusted_delta_low": float(np.nanquantile(delta, adjusted_tail)),
        "selection_adjusted_delta_high": float(np.nanquantile(delta, 1 - adjusted_tail)),
        "bootstrap_probability_delta_positive": float(np.nanmean(delta > 0)),
    }


def run(
    *,
    config_path: Path | str = Path("configs/news_bootstrap_check.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_bootstrap_check"),
) -> pd.DataFrame:
    config = load_config(config_path)
    score_path = Path(config["scores"])
    signal_path = Path(config["signals"])
    feature_sets = [str(config["baseline"]), *config["candidates"]]
    weekly = weekly_policy_table(
        pd.read_csv(score_path),
        pd.read_csv(signal_path),
        feature_sets=feature_sets,
        horizon=int(config["horizon"]),
    )
    rows = []
    for index, candidate in enumerate(config["candidates"]):
        rows.append(
            {
                "candidate": candidate,
                "horizon": int(config["horizon"]),
                "weeks": len(weekly),
                **bootstrap_delta(
                    weekly,
                    baseline=str(config["baseline"]),
                    candidate=str(candidate),
                    samples=int(config["samples"]),
                    block_weeks=int(config["block_weeks"]),
                    seed=int(config["random_seed"]) + index,
                    previous_trials=int(config["previous_aggregate_news_trials"]),
                ),
            }
        )
    result = pd.DataFrame(rows)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(output / "weekly_inputs.csv", index=False)
    result.to_csv(output / "bootstrap_delta.csv", index=False)
    metadata = {
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "signals_sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest(),
        "warning": "Post-hoc block bootstrap cannot replace a new hold-out.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_bootstrap_check.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news_bootstrap_check")
    )
    args = parser.parse_args()
    run(config_path=args.config, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
