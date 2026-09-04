"""Simulate sequential buy-now-or-wait decisions on saved OOT scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.value_downside_experiment import causal_percentile


def load_config(path: Path | str = Path("configs/optimal_stopping_simulation.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("optimal-stopping config must be preregistered")
    expected = ["buy_first", "static_score", "relaxing_score", "random_expected", "oracle"]
    if config.get("policies") != expected:
        raise ValueError("stopping policies differ from implementation")
    return config


def choose_position(window: pd.DataFrame, policy: str, config: dict[str, Any]) -> int | None:
    if policy == "buy_first":
        return 0
    if policy == "oracle":
        return int(np.argmin(window["price"].to_numpy()))
    if policy == "random_expected":
        return None
    ranks = window["score_rank"].to_numpy(dtype=float)
    if policy == "static_score":
        threshold = 1 - float(config["static_top_share"])
        hits = np.flatnonzero(ranks >= threshold)
    elif policy == "relaxing_score":
        initial = 1 - float(config["relaxing_start_top_share"])
        thresholds = np.linspace(initial, 0.0, len(window))
        hits = np.flatnonzero(ranks >= thresholds)
    else:
        raise ValueError(f"unknown stopping policy {policy}")
    return int(hits[0]) if len(hits) else len(window) - 1


def simulate(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = scores.loc[
        scores["feature_set"].eq(config["feature_set"])
        & scores["tolerance_bps"].eq(float(config["tolerance_bps"]))
        & scores["model"].isin(config["models"])
        & scores["horizon"].isin(config["deadlines_observations"])
    ].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    rows: list[dict[str, object]] = []
    keys = ["scope", "model", "horizon", "test_year", "corridor"]
    for key_values, group in data.groupby(keys, sort=True):
        scope, model, deadline, year, corridor = key_values
        group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        group["score_rank"] = causal_percentile(
            group["score"],
            lookback=int(config["score_lookback"]),
            minimum_history=int(config["minimum_score_history"]),
        )
        deadline = int(deadline)
        for start in range(0, len(group) - deadline + 1, deadline):
            window = group.iloc[start : start + deadline].reset_index(drop=True)
            best = float(window["price"].min())
            random_regret = float(((window["price"] / best) - 1).mean() * 10_000)
            for policy in config["policies"]:
                position = choose_position(window, policy, config)
                if position is None:
                    selected_date = pd.NaT
                    regret = random_regret
                    waited = (deadline - 1) / 2
                else:
                    selected = window.iloc[position]
                    selected_date = selected["timestamp"]
                    regret = (float(selected["price"]) / best - 1) * 10_000
                    waited = position
                rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "deadline_observations": deadline,
                        "test_year": int(year),
                        "corridor": corridor,
                        "window_start": window.iloc[0]["timestamp"],
                        "selected_date": selected_date,
                        "policy": policy,
                        "regret_bps": regret,
                        "wait_observations": waited,
                        "within_25bps": regret <= 25,
                    }
                )
    return pd.DataFrame(rows)


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    return (
        decisions.groupby(["scope", "model", "deadline_observations", "policy"], as_index=False)
        .agg(
            windows=("regret_bps", "size"),
            mean_regret_bps=("regret_bps", "mean"),
            median_regret_bps=("regret_bps", "median"),
            p90_regret_bps=("regret_bps", lambda values: values.quantile(0.9)),
            mean_wait_observations=("wait_observations", "mean"),
            within_25bps_rate=("within_25bps", "mean"),
        )
        .sort_values(["scope", "model", "deadline_observations", "mean_regret_bps"])
    )


def run(
    *,
    config_path: Path | str = Path("configs/optimal_stopping_simulation.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/optimal_stopping"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["scores_input"])
    decisions = simulate(pd.read_csv(input_path), config)
    summary_frame = summarize(decisions)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output / "decisions.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "decision_rows": len(decisions),
        "warning": "Illustrative non-overlapping windows, not observed client deadlines; oracle is evaluation only.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/optimal_stopping_simulation.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/optimal_stopping"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
