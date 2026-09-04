"""Compare regret with triple-barrier and trend-scanning training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _add_corridor_dummies, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


FEATURE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/path_label_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("path-label config must be preregistered schema_version 1")
    if config.get("label_methods") != ["regret", "triple_barrier", "trend_scanning"]:
        raise ValueError("path-label methods differ from the implementation")
    return config


def triple_barrier_label(
    price: pd.Series,
    *,
    horizon: int,
    better_price_barrier_bps: float,
    worse_price_barrier_bps: float,
) -> pd.Series:
    """One means price worsens first, or never becomes materially better."""

    values = price.to_numpy(dtype=float)
    labels = np.full(len(values), np.nan)
    for index in range(len(values) - horizon):
        current = values[index]
        label = 1.0
        for future in values[index + 1 : index + horizon + 1]:
            improvement = (current / future - 1) * 10_000
            worsening = (future / current - 1) * 10_000
            if improvement >= better_price_barrier_bps:
                label = 0.0
                break
            if worsening >= worse_price_barrier_bps:
                label = 1.0
                break
        labels[index] = label
    return pd.Series(labels, index=price.index)


def _slope_t_stat(path: np.ndarray) -> float:
    if not np.isfinite(path).all() or len(path) < 3 or np.any(path <= 0):
        return np.nan
    y = np.log(path)
    x = np.arange(len(y), dtype=float)
    centered = x - x.mean()
    ssx = float(np.square(centered).sum())
    slope = float(np.dot(centered, y - y.mean()) / ssx)
    residual = y - (y.mean() + slope * centered)
    variance = float(np.square(residual).sum() / (len(y) - 2))
    if variance <= 1e-20:
        return math.copysign(float("inf"), slope) if slope else 0.0
    return slope / math.sqrt(variance / ssx)


def trend_scanning_label(price: pd.Series, *, horizon: int, scan_horizons: list[int]) -> pd.Series:
    """Choose the future window with the strongest absolute trend t-stat."""

    candidates = sorted({int(value) for value in scan_horizons if 2 <= int(value) <= horizon})
    if not candidates:
        candidates = [horizon]
    values = price.to_numpy(dtype=float)
    labels = np.full(len(values), np.nan)
    for index in range(len(values) - horizon):
        statistics = [_slope_t_stat(values[index : index + candidate + 1]) for candidate in candidates]
        valid = [value for value in statistics if np.isfinite(value) or np.isinf(value)]
        if valid:
            strongest = max(valid, key=abs)
            labels[index] = float(strongest > 0)
    return pd.Series(labels, index=price.index)


def add_path_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values(["corridor", "timestamp"], kind="mergesort")
    pieces: list[pd.DataFrame] = []
    barriers = config["triple_barrier"]
    for _, group in data.groupby("corridor", sort=False):
        group = group.copy()
        for horizon in config["horizons"]:
            group[f"label__triple_barrier_{horizon}"] = triple_barrier_label(
                group["price"],
                horizon=int(horizon),
                better_price_barrier_bps=float(barriers["better_price_barrier_bps"]),
                worse_price_barrier_bps=float(barriers["worse_price_barrier_bps"]),
            )
            group[f"label__trend_scanning_{horizon}"] = trend_scanning_label(
                group["price"],
                horizon=int(horizon),
                scan_horizons=list(config["trend_scan_horizons"]),
            )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_path_labels(data, config)
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
                usable = source.loc[source[regret_column].notna()].copy()
                usable["target"] = usable[regret_column].le(float(config["evaluation_tolerance_bps"])).astype(int)
                usable["regret_bps"] = usable[regret_column]
                usable["benefit_bps"] = usable[benefit_column]
                for label_method in config["label_methods"]:
                    if label_method == "regret":
                        usable["training_target"] = usable["target"]
                    else:
                        usable["training_target"] = usable[f"label__{label_method}_{int(horizon)}"]
                    model_data = usable.loc[usable["training_target"].notna()].copy()
                    model_data["training_target"] = model_data["training_target"].astype(int)
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(model_data, start, int(horizon))
                        test = model_data.loc[model_data["timestamp"].between(start, end, inclusive="left")].copy()
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
                            "feature_set": label_method,
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
                                "timestamp", "corridor", "target", "training_target", "regret_bps",
                                "benefit_bps", "score", "score_threshold", "matched_week_hit_rate",
                            ]].copy()
                            for key, value in identity.items():
                                exported[key] = value
                            exported["test_year"] = int(year)
                            signal_frames.append(exported)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(fold_rows), signals


def run(
    *,
    config_path: Path | str = Path("configs/path_label_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/path_labels"),
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
        "warning": "Exploratory label comparison; all models are evaluated on the same regret<=25 bps outcome.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/path_label_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/path_labels"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
