"""Combine a known value measure with an out-of-time downside score."""

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


IDENTITY_KEYS = ("scope", "model", "combination", "horizon")


def load_config(path: Path | str = Path("configs/value_downside_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("value/downside config must be preregistered schema_version 1")
    if config.get("combinations") != ["model_only", "balanced_mean", "conjunction_min"]:
        raise ValueError("value/downside combinations differ from the implementation")
    return config


def causal_percentile(values: pd.Series, *, lookback: int, minimum_history: int) -> pd.Series:
    """Rank each current value only against strictly earlier observations."""

    source = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(source), np.nan)
    for index, current in enumerate(source):
        history = source[max(0, index - lookback) : index]
        history = history[np.isfinite(history)]
        if np.isfinite(current) and len(history) >= minimum_history:
            result[index] = float(np.mean(history <= current))
    return pd.Series(result, index=values.index)


def combine_score(frame: pd.DataFrame, combination: str) -> pd.Series:
    if combination == "model_only":
        return frame["model_score"]
    if combination == "balanced_mean":
        return (frame["model_rank"] + frame["value_score"]) / 2
    if combination == "conjunction_min":
        return frame[["model_rank", "value_score"]].min(axis=1)
    raise ValueError(f"unknown value/downside combination {combination}")


def evaluate(scores: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
    value = features[["timestamp", "corridor", "base__percentile_60"]].copy()
    value["timestamp"] = pd.to_datetime(value["timestamp"], errors="raise")
    data = selected.merge(value, on=["timestamp", "corridor"], how="left", validate="many_to_one")
    data["value_score"] = 1 - data["base__percentile_60"]
    policy = config["adaptive_policy"]
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    fold_keys = ["scope", "model", "horizon", "test_year"]

    for keys, fold in data.groupby(fold_keys, sort=True):
        scope, model, horizon, year = keys
        fold = fold.sort_values(["corridor", "timestamp"], kind="mergesort").copy()
        fold["model_score"] = fold["score"]
        fold["model_rank"] = fold.groupby("corridor", sort=False)["model_score"].transform(
            lambda values: causal_percentile(
                values,
                lookback=int(config["model_rank_lookback"]),
                minimum_history=int(config["minimum_rank_history"]),
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
        for combination in config["combinations"]:
            fold["combined_score"] = combine_score(fold, combination)
            candidate_parts: list[pd.DataFrame] = []
            for _, corridor_fold in fold.groupby("corridor", sort=False):
                candidate_input = corridor_fold.rename(columns={"score": "original_score", "combined_score": "score"})
                candidate_parts.append(
                    adaptive_candidates(
                        candidate_input,
                        share=float(policy["top_score_share"]),
                        lookback=int(policy["lookback_observations"]),
                        minimum_history=int(policy["minimum_history_observations"]),
                    )
                )
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
            identity = {
                "scope": scope,
                "model": model,
                "combination": combination,
                "horizon": int(horizon),
            }
            fold_rows.append(
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
                exported = dispatched[[
                    "timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score",
                    "score_threshold", "value_score", "model_rank", "matched_week_hit_rate",
                ]].copy()
                for key, value_item in identity.items():
                    exported[key] = value_item
                exported["test_year"] = int(year)
                signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(IDENTITY_KEYS), sort=True):
        identity = dict(zip(IDENTITY_KEYS, keys, strict=True))
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / count if count else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / count if count else np.nan
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
                "signals_per_week": count / float(group["duration_weeks"].sum()),
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/value_downside_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/value_downside"),
) -> dict[str, object]:
    config = load_config(config_path)
    scores_path = Path(config["scores"])
    features_path = Path(config["features"])
    folds, signals = evaluate(pd.read_csv(scores_path), pd.read_csv(features_path), config)
    summary = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory soft value/downside comparison; no separate value model is needed for an observed percentile.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/value_downside_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/value_downside"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
