"""Causal adaptive selective policies derived from saved OOT scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _apply_policy


MODEL_KEYS = ("scope", "model", "feature_set", "horizon", "tolerance_bps")
POLICY_KEYS = (*MODEL_KEYS, "top_score_share", "lookback_observations", "cooldown_days")


def load_config(path: Path | str = Path("configs/adaptive_threshold.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_adaptive_results":
        raise ValueError("adaptive threshold config is not preregistered schema_version 1")
    return config


def adaptive_candidates(
    frame: pd.DataFrame, *, share: float, lookback: int, minimum_history: int
) -> pd.DataFrame:
    """Compare each score only with scores observed strictly before it."""

    ordered = frame.sort_values("timestamp", kind="mergesort").copy()
    threshold = (
        ordered["score"]
        .shift(1)
        .rolling(lookback, min_periods=minimum_history)
        .quantile(1 - share)
    )
    selected = ordered.loc[ordered["score"].ge(threshold)].copy()
    selected["score_threshold"] = threshold.loc[selected.index]
    return selected


def evaluate(scores: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], errors="raise")
    required = {*MODEL_KEYS, "test_year", "corridor", "week", "score", "target", "regret_bps", "benefit_bps"}
    missing = required - set(scores)
    if missing:
        raise ValueError(f"scores lack {sorted(missing)}")
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    fold_keys = [*MODEL_KEYS, "test_year"]
    for keys, fold in scores.groupby(fold_keys, sort=True):
        identity = dict(zip(fold_keys, keys, strict=True))
        weekly_rate = fold.groupby(["corridor", "week"], sort=False)["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        corridor_exposure = int(fold["corridor"].nunique())
        duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7)
        duration *= corridor_exposure
        for share in config["top_score_share"]:
            for lookback in config["lookback_observations"]:
                candidate_parts = [
                    adaptive_candidates(
                        group,
                        share=float(share),
                        lookback=int(lookback),
                        minimum_history=int(config["minimum_history_observations"]),
                    )
                    for _, group in fold.groupby("corridor", sort=False)
                ]
                candidates = pd.concat(candidate_parts, ignore_index=False)
                for cooldown in config["cooldown_days"]:
                    selected_parts = [
                        _apply_policy(
                            group,
                            cooldown_days=int(cooldown),
                            weekly_cap=int(config["weekly_cap"]),
                        )
                        for _, group in candidates.groupby("corridor", sort=False)
                    ]
                    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else candidates.iloc[0:0]
                    count = len(selected)
                    base_rate = float(fold["target"].mean())
                    hit_rate = float(selected["target"].mean()) if count else np.nan
                    fold_rows.append(
                        {
                            **identity,
                            "top_score_share": float(share),
                            "lookback_observations": int(lookback),
                            "cooldown_days": int(cooldown),
                            "test_count": len(fold),
                            "test_hits": int(fold["target"].sum()),
                            "signal_count": count,
                            "signal_hits": int(selected["target"].sum()) if count else 0,
                            "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                            "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()) if count else 0.0,
                            "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                            "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                            "duration_weeks": duration,
                        }
                    )
                    if count:
                        exported = selected.copy()
                        for key, value in identity.items():
                            exported[key] = value
                        exported["top_score_share"] = float(share)
                        exported["lookback_observations"] = int(lookback)
                        exported["cooldown_days"] = int(cooldown)
                        signal_frames.append(exported)
    return pd.DataFrame(fold_rows), pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(POLICY_KEYS), sort=True):
        identity = dict(zip(POLICY_KEYS, keys, strict=True))
        count = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / count if count else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / count if count else np.nan
        fold_lifts = group["lift"].dropna()
        duration = float(group["duration_weeks"].sum())
        rows.append(
            {
                **identity,
                "folds": len(group),
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if count and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / count if count else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / count if count else np.nan,
                "signals_per_week": count / duration if duration else 0.0,
                "worst_fold_lift": float(fold_lifts.min()) if len(fold_lifts) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def corridor_stability(signals: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    selected = (
        signals.groupby([*POLICY_KEYS, "corridor"], sort=False)
        .agg(
            signals=("target", "size"),
            hits=("target", "sum"),
            matched_expected_hits=("matched_week_hit_rate", "sum"),
        )
        .reset_index()
    )
    base = (
        scores.groupby([*MODEL_KEYS, "corridor"], sort=False)
        .agg(base_count=("target", "size"), base_hits=("target", "sum"))
        .reset_index()
    )
    result = selected.merge(base, on=[*MODEL_KEYS, "corridor"], how="left", validate="many_to_one")
    result["hit_rate"] = result["hits"] / result["signals"]
    result["baseline_hit_rate"] = result["base_hits"] / result["base_count"]
    result["matched_random_hit_rate"] = result["matched_expected_hits"] / result["signals"]
    result["lift"] = result["hit_rate"] / result["baseline_hit_rate"]
    result["lift_vs_matched_random"] = result["hit_rate"] / result["matched_random_hit_rate"]
    return result


def best_by_horizon(
    summary: pd.DataFrame, config: dict[str, Any], stability: pd.DataFrame | None = None
) -> pd.DataFrame:
    minimum = int(config["minimum_summary_signals"])
    low, high = map(float, config["frequency_range_per_corridor_week"])
    eligible = summary.loc[summary["signals"].ge(minimum) & summary["lift"].notna()].copy()
    eligible["frequency_requirement_met"] = eligible["signals_per_week"].between(low, high, inclusive="both")
    if stability is not None and not stability.empty:
        stable = (
            stability.assign(pass_lift=stability["signals"].ge(20) & stability["lift"].ge(1.3))
            .groupby(list(POLICY_KEYS), sort=False)["pass_lift"]
            .sum()
            .rename("corridors_lift_ge_1_3")
            .reset_index()
        )
        eligible = eligible.merge(stable, on=list(POLICY_KEYS), how="left")
    else:
        eligible["corridors_lift_ge_1_3"] = np.nan
    rows: list[pd.DataFrame] = []
    for _, group in eligible.groupby("horizon", sort=True):
        passing = group.loc[group["frequency_requirement_met"]]
        pool = passing if len(passing) else group
        rows.append(
            pool.sort_values(["lift", "lift_vs_matched_random", "signals"], ascending=False, kind="mergesort").head(1)
        )
    if not rows:
        return pd.DataFrame()
    best = pd.concat(rows, ignore_index=True)
    best["best_model"] = (
        best["scope"].astype(str)
        + "/"
        + best["model"].astype(str)
        + "/"
        + best["feature_set"].astype(str)
        + "/adaptive"
        + best["lookback_observations"].astype(str)
        + "/top"
        + (best["top_score_share"] * 100).round().astype(int).astype(str)
        + "%/cd"
        + best["cooldown_days"].astype(str)
        + "/tau"
        + best["tolerance_bps"].round().astype(int).astype(str)
    )
    return best[
        [
            "horizon",
            "best_model",
            "signals",
            "lift",
            "lift_vs_matched_random",
            "signals_per_week",
            "frequency_requirement_met",
            "corridors_lift_ge_1_3",
            "worst_fold_lift",
        ]
    ]


def run(
    *,
    config_path: Path | str = Path("configs/adaptive_threshold.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/adaptive_threshold"),
) -> dict[str, object]:
    config = load_config(config_path)
    scores_path = Path(config["parent_scores"])
    scores = pd.read_csv(scores_path)
    folds, signals = evaluate(scores, config)
    summary = summarize(folds)
    stability = corridor_stability(signals, scores)
    best = best_by_horizon(summary, config, stability)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    stability.to_csv(output / "corridor_stability.csv", index=False)
    best.to_csv(output / "best_by_horizon.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "parent_scores": str(scores_path),
        "parent_scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory policy comparison; requires untouched confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive_threshold.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/adaptive_threshold"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
