"""Test whether a longer official-rate history improves the best path-label model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import (
    _add_corridor_dummies,
    _make_model,
    _market_feature_frame,
    _purged_train,
    build_datasets,
)
from fxpulse.path_label_experiment import triple_barrier_label
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


FEATURE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/training_history_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("training-history config must be preregistered")
    if config.get("training_label") != "triple_barrier":
        raise ValueError("only the preregistered triple-barrier label is supported")
    return config


def build_extended_features(config: dict[str, Any]) -> pd.DataFrame:
    datasets = build_datasets(
        cbr_path=Path(config["cbr_input"]),
        research_panel_path=Path(config["market_input"]),
        corridors=list(config["corridors"]),
        horizons=[int(value) for value in config["horizons"]],
        market_lag=1,
    )
    frame = pd.concat(datasets.values(), ignore_index=True)
    output = Path(config["feature_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def add_training_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    barriers = config["triple_barrier"]
    pieces: list[pd.DataFrame] = []
    for _, group in data.sort_values(["corridor", "timestamp"]).groupby("corridor", sort=False):
        group = group.copy()
        for horizon in config["horizons"]:
            group[f"training_target_{int(horizon)}"] = triple_barrier_label(
                group["price"],
                horizon=int(horizon),
                better_price_barrier_bps=float(barriers["better_price_barrier_bps"]),
                worse_price_barrier_bps=float(barriers["worse_price_barrier_bps"]),
            )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_training_labels(data, config)
    feature_columns = [column for column in frame if column.startswith(FEATURE_PREFIXES)]
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for scope_kind in config["scopes"]:
        scopes = ["pooled"] if scope_kind == "pooled" else list(config["corridors"])
        for scope in scopes:
            source = frame if scope == "pooled" else frame.loc[frame["corridor"].eq(scope)].copy()
            for horizon in config["horizons"]:
                regret_column = f"outcome__regret_{int(horizon)}"
                benefit_column = f"outcome__benefit_{int(horizon)}"
                label_column = f"training_target_{int(horizon)}"
                usable = source.loc[source[regret_column].notna() & source[label_column].notna()].copy()
                usable["target"] = usable[regret_column].le(float(config["evaluation_tolerance_bps"])).astype(int)
                usable["training_target"] = usable[label_column].astype(int)
                usable["regret_bps"] = usable[regret_column]
                usable["benefit_bps"] = usable[benefit_column]
                for training_start in config["training_starts"]:
                    history = usable.loc[usable["timestamp"].ge(pd.Timestamp(training_start))].copy()
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(history, start, int(horizon))
                        test = history.loc[history["timestamp"].between(start, end, inclusive="left")].copy()
                        columns = list(feature_columns)
                        if scope == "pooled":
                            train, test, columns = _add_corridor_dummies(
                                train, test, columns, list(config["corridors"])
                            )
                        if (
                            len(train) < int(config["minimum_training_observations"])
                            or len(test) < 20
                            or train["training_target"].nunique() < 2
                        ):
                            continue
                        model = _make_model(
                            str(config["model"]),
                            iterations=int(config["model_iterations"]),
                            seed=int(config["random_seed"]) + int(year),
                        )
                        model.fit(train[columns], train["training_target"])
                        test["score"] = model.predict_proba(test[columns])[:, 1]
                        test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                        weekly_rate = test.groupby(["corridor", "week"])["target"].mean()
                        test["matched_week_hit_rate"] = [
                            float(weekly_rate.loc[(corridor, week)])
                            for corridor, week in zip(test["corridor"], test["week"], strict=True)
                        ]
                        selected = _evaluate_fold(test, policy=config["adaptive_policy"])
                        count = len(selected)
                        base_rate = float(test["target"].mean())
                        hit_rate = float(selected["target"].mean()) if count else np.nan
                        exposure = int(test["corridor"].nunique())
                        duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7) * exposure
                        identity = {
                            "scope": scope,
                            "model": str(config["model"]),
                            "feature_set": f"history_from_{training_start[:4]}",
                            "horizon": int(horizon),
                        }
                        fold_rows.append(
                            {
                                **identity,
                                "test_year": int(year),
                                "test_count": len(test),
                                "test_hits": int(test["target"].sum()),
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
                                "timestamp", "corridor", "target", "training_target", "regret_bps", "benefit_bps",
                                "score", "score_threshold", "matched_week_hit_rate",
                            ]].copy()
                            for key, value in identity.items():
                                exported[key] = value
                            exported["test_year"] = int(year)
                            signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def run(
    *,
    config_path: Path | str = Path("configs/training_history_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/training_history"),
) -> dict[str, object]:
    config = load_config(config_path)
    frame = build_extended_features(config)
    folds, signals = evaluate(frame, config)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    feature_path = Path(config["feature_output"])
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "cbr_input_sha256": hashlib.sha256(Path(config["cbr_input"]).read_bytes()).hexdigest(),
        "feature_output": str(feature_path),
        "feature_output_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "feature_rows": len(frame),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory follow-up after path-label results; older regimes can hurt rather than help.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training_history_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/training_history"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
