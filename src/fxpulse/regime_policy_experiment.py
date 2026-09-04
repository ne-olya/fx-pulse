"""Apply causal global or regime-specific thresholds to saved OOT scores."""

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


IDENTITY_KEYS = ("scope", "model", "policy", "horizon")


def load_config(path: Path | str = Path("configs/regime_policy_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("regime-policy config must be preregistered schema_version 1")
    if config.get("policies") != ["global", "per_regime", "exclude_shock"]:
        raise ValueError("regime policies differ from the implementation")
    return config


def policy_candidates(frame: pd.DataFrame, policy: str, config: dict[str, Any]) -> pd.DataFrame:
    if policy in {"global", "exclude_shock"}:
        candidates = adaptive_candidates(
            frame,
            share=float(config["top_score_share"]),
            lookback=int(config["lookback_observations"]),
            minimum_history=int(config["minimum_history_observations"]),
        )
        return candidates if policy == "global" else candidates.loc[candidates["regime"].ne("SHOCK")]
    if policy == "per_regime":
        parts = [
            adaptive_candidates(
                group,
                share=float(config["top_score_share"]),
                lookback=int(config["lookback_observations"]),
                minimum_history=int(config["minimum_regime_history_observations"]),
            )
            for _, group in frame.groupby("regime", sort=False)
        ]
        return pd.concat(parts, ignore_index=False) if parts else frame.iloc[0:0]
    raise ValueError(f"unknown regime policy {policy}")


def evaluate(scores: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for keys, fold in data.groupby(["scope", "model", "horizon", "test_year"], sort=True):
        scope, model, horizon, year = keys
        fold["week"] = fold["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = fold.groupby(["corridor", "week"])["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        exposure = int(fold["corridor"].nunique())
        duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7) * exposure
        for policy in config["policies"]:
            parts = [policy_candidates(group, policy, config) for _, group in fold.groupby("corridor", sort=False)]
            candidates = pd.concat(parts, ignore_index=False) if parts else fold.iloc[0:0]
            selected_parts = [
                _apply_policy(
                    group,
                    cooldown_days=int(config["cooldown_days"]),
                    weekly_cap=int(config["weekly_cap"]),
                )
                for _, group in candidates.groupby("corridor", sort=False)
            ]
            selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else candidates.iloc[0:0]
            count = len(selected)
            base_rate = float(fold["target"].mean())
            hit_rate = float(selected["target"].mean()) if count else np.nan
            identity = {"scope": scope, "model": model, "policy": policy, "horizon": int(horizon)}
            fold_rows.append(
                {
                    **identity,
                    "test_year": int(year),
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
                exported = selected[[
                    "timestamp", "corridor", "regime", "target", "regret_bps", "benefit_bps",
                    "score", "score_threshold", "matched_week_hit_rate",
                ]].copy()
                for key, value in identity.items():
                    exported[key] = value
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
    config_path: Path | str = Path("configs/regime_policy_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/regime_policy"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
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
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory regime-policy comparison; thresholds use prior scores but the comparison needs confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/regime_policy_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/regime_policy"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
