"""Gate saved out-of-time model scores with causal price-move events."""

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
from fxpulse.regime_policy_experiment import summarize


IDENTITY_KEYS = ("scope", "model", "gate", "horizon")


def load_config(path: Path | str = Path("configs/event_sampling_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("event-sampling config must be preregistered")
    expected = ["all_days", "absolute_move_1sigma", "cusum_0.5sigma", "cusum_1sigma"]
    if config.get("event_gates") != expected:
        raise ValueError("event gates differ from the implementation")
    return config


def _cusum_events(returns: pd.Series, sigma: pd.Series, multiplier: float) -> pd.Series:
    positive = 0.0
    negative = 0.0
    flags: list[bool] = []
    for change, scale in zip(returns, sigma, strict=True):
        if not np.isfinite(change) or not np.isfinite(scale) or scale <= 0:
            flags.append(False)
            continue
        positive = max(0.0, positive + float(change))
        negative = min(0.0, negative + float(change))
        fired = positive >= multiplier * scale or negative <= -multiplier * scale
        flags.append(fired)
        if fired:
            positive = 0.0
            negative = 0.0
    return pd.Series(flags, index=returns.index, dtype=bool)


def add_event_flags(features: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "corridor", "base__return_1", "base__volatility_20"}
    if missing := required - set(features):
        raise ValueError(f"features lack {sorted(missing)}")
    data = features[list(required)].drop_duplicates(["timestamp", "corridor"]).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values(["corridor", "timestamp"], kind="mergesort")
    data["event__all_days"] = True
    data["event__absolute_move_1sigma"] = data["base__return_1"].abs().ge(data["base__volatility_20"])
    for multiplier, label in [(0.5, "0.5"), (1.0, "1")]:
        data[f"event__cusum_{label}sigma"] = (
            data.groupby("corridor", group_keys=False)
            .apply(
                lambda group: _cusum_events(
                    group["base__return_1"], group["base__volatility_20"], multiplier
                ),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
            .reindex(data.index)
            .fillna(False)
            .astype(bool)
        )
    return data.drop(columns=["base__return_1", "base__volatility_20"])


def evaluate(scores: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = add_event_flags(features)
    data = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["horizons"])
    ].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.merge(flags, on=["timestamp", "corridor"], how="left", validate="many_to_one")
    event_columns = [column for column in data if column.startswith("event__")]
    data[event_columns] = data[event_columns].fillna(False).astype(bool)
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for keys, fold in data.groupby(["scope", "model", "horizon", "test_year"], sort=True):
        scope, model, horizon, year = keys
        fold = fold.sort_values(["corridor", "timestamp"], kind="mergesort").copy()
        fold["week"] = fold["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = fold.groupby(["corridor", "week"])["target"].mean()
        fold["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(fold["corridor"], fold["week"], strict=True)
        ]
        exposure = int(fold["corridor"].nunique())
        duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7) * exposure
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
        for gate in config["event_gates"]:
            gated = candidates.loc[candidates[f"event__{gate}"]]
            selected = pd.concat(
                [
                    _apply_policy(
                        group,
                        cooldown_days=int(config["cooldown_days"]),
                        weekly_cap=int(config["weekly_cap"]),
                    )
                    for _, group in gated.groupby("corridor", sort=False)
                ],
                ignore_index=False,
            ) if len(gated) else gated.copy()
            count = len(selected)
            base_rate = float(fold["target"].mean())
            hit_rate = float(selected["target"].mean()) if count else np.nan
            identity = {"scope": scope, "model": model, "gate": gate, "horizon": int(horizon)}
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
                    "timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score",
                    "score_threshold", "matched_week_hit_rate", f"event__{gate}",
                ]].copy()
                for key, value in identity.items():
                    exported[key] = value
                exported["test_year"] = int(year)
                signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def run(
    *,
    config_path: Path | str = Path("configs/event_sampling_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/event_sampling"),
) -> dict[str, object]:
    config = load_config(config_path)
    scores_path = Path(config["scores_input"])
    features_path = Path(config["features_input"])
    folds, signals = evaluate(pd.read_csv(scores_path), pd.read_csv(features_path), config)
    summary_frame = summarize(folds.rename(columns={"gate": "policy"})).rename(columns={"policy": "gate"})
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory event gate; an event uses only returns and volatility known on the signal date.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/event_sampling_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/event_sampling"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
