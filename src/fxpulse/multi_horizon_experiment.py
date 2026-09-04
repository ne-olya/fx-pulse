"""Combine already out-of-time daily scores across several horizons."""

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


KEYS = ("scope", "model", "combination")


def load_config(path: Path | str = Path("configs/multi_horizon_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("multi-horizon config must be preregistered schema_version 1")
    return config


def combine_scores(frame: pd.DataFrame, combination: str) -> pd.Series:
    required = [f"score_{h}" for h in (3, 5, 10)]
    if not set(required).issubset(frame):
        raise ValueError("combined score frame lacks h=3/5/10")
    if combination == "single_h10":
        return frame["score_10"]
    if combination == "mean_scores":
        return frame[required].mean(axis=1)
    if combination == "minimum_score":
        return frame[required].min(axis=1)
    raise ValueError(f"unknown combination {combination}")


def evaluate(scores: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
    index = ["scope", "model", "test_year", "timestamp", "corridor"]
    score_wide = selected.pivot(index=index, columns="horizon", values="score")
    score_wide.columns = [f"score_{int(column)}" for column in score_wide]
    outcomes = selected.loc[selected["horizon"].eq(10), [*index, "target", "regret_bps", "benefit_bps"]]
    data = outcomes.merge(score_wide.reset_index(), on=index, how="inner", validate="one_to_one")
    policy = config["adaptive_policy"]
    folds: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for keys, fold in data.groupby(["scope", "model", "test_year"], sort=True):
        scope, model, year = keys
        fold["week"] = fold["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = fold.groupby(["corridor", "week"])["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        for combination in config["combinations"]:
            fold["score"] = combine_scores(fold, combination)
            candidate_parts = [
                adaptive_candidates(
                    group,
                    share=float(policy["top_score_share"]),
                    lookback=int(policy["lookback_observations"]),
                    minimum_history=int(policy["minimum_history_observations"]),
                )
                for _, group in fold.groupby("corridor", sort=False)
            ]
            candidates = pd.concat(candidate_parts, ignore_index=False)
            signal_parts = [
                _apply_policy(
                    group,
                    cooldown_days=int(policy["cooldown_days"]),
                    weekly_cap=int(policy["weekly_cap"]),
                )
                for _, group in candidates.groupby("corridor", sort=False)
            ]
            dispatched = pd.concat(signal_parts, ignore_index=False) if signal_parts else candidates.iloc[0:0]
            count = len(dispatched)
            base_rate = float(fold["target"].mean())
            hit_rate = float(dispatched["target"].mean()) if count else np.nan
            exposure = int(fold["corridor"].nunique())
            duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7) * exposure
            identity = {"scope": scope, "model": model, "combination": combination}
            folds.append(
                {
                    **identity,
                    "test_year": int(year),
                    "test_count": len(fold),
                    "test_hits": int(fold["target"].sum()),
                    "signal_count": count,
                    "signal_hits": int(dispatched["target"].sum()) if count else 0,
                    "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                    "matched_expected_hits": float(dispatched["matched_week_hit_rate"].sum()) if count else 0.0,
                    "regret_sum_bps": float(dispatched["regret_bps"].sum()) if count else 0.0,
                    "benefit_sum_bps": float(dispatched["benefit_bps"].sum()) if count else 0.0,
                    "duration_weeks": duration,
                }
            )
            if count:
                exported = dispatched[["timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score", "score_threshold", "matched_week_hit_rate"]].copy()
                for key, value in identity.items():
                    exported[key] = value
                exported["test_year"] = int(year)
                signal_frames.append(exported)
    return pd.DataFrame(folds), pd.concat(signal_frames, ignore_index=True)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(KEYS), sort=True):
        identity = dict(zip(KEYS, keys, strict=True))
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / count if count else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / count if count else np.nan
        duration = float(group["duration_weeks"].sum())
        valid_lift = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / count if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / count if count else np.nan,
                "signals_per_week": count / duration if duration else 0.0,
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/multi_horizon_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/multi_horizon"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    folds, signals = evaluate(pd.read_csv(input_path), config)
    summary = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory score ensemble; no untouched confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multi_horizon_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/multi_horizon"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
