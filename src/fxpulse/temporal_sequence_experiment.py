"""Test explicit lag sequences against the same rolling-feature baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import (
    _add_corridor_dummies,
    _apply_policy,
    _feature_importance,
    _make_model,
    _purged_train,
)


IDENTITY_KEYS = ("scope", "model", "feature_set", "horizon")
BASE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/temporal_sequence_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("temporal sequence config must be preregistered schema_version 1")
    if config.get("feature_sets") != ["interpretable", "plus_sequence"]:
        raise ValueError("temporal sequence feature sets differ from the implementation")
    return config


def add_sequence_features(frame: pd.DataFrame, source_columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Add explicit, past-only lags and simple path-shape summaries."""

    missing = set(source_columns) - set(frame)
    if missing:
        raise ValueError(f"sequence input lacks {sorted(missing)}")
    if not lags or min(lags) < 1:
        raise ValueError("sequence lags must be positive")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values(["corridor", "timestamp"], kind="mergesort")
    grouped = data.groupby("corridor", sort=False)
    for source in source_columns:
        short = source.replace("__", "_")
        for lag in lags:
            data[f"sequence__{short}_lag_{int(lag)}"] = grouped[source].shift(int(lag))

    # These summaries distinguish, for example, one shock from five small
    # same-direction moves even when their cumulative return is similar.
    return_columns = ["base__return_1", *[f"sequence__base_return_1_lag_{lag}" for lag in lags]]
    recent = data[return_columns]
    for window in (5, 10):
        path = recent.iloc[:, :window]
        signs = np.sign(path)
        data[f"sequence__positive_share_{window}"] = path.gt(0).mean(axis=1)
        data[f"sequence__negative_share_{window}"] = path.lt(0).mean(axis=1)
        data[f"sequence__sign_changes_{window}"] = signs.diff(axis=1).ne(0).iloc[:, 1:].sum(axis=1)
        data[f"sequence__max_move_{window}"] = path.abs().max(axis=1)
    return data.replace([np.inf, -np.inf], np.nan)


def _evaluate_fold(
    test: pd.DataFrame,
    *,
    policy: dict[str, Any],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, corridor_test in test.groupby("corridor", sort=False):
        candidates = adaptive_candidates(
            corridor_test,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        parts.append(
            _apply_policy(
                candidates,
                cooldown_days=int(policy["cooldown_days"]),
                weekly_cap=int(policy["weekly_cap"]),
            )
        )
    return pd.concat(parts, ignore_index=False) if parts else test.iloc[0:0].copy()


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = add_sequence_features(
        data,
        list(config["sequence_source_columns"]),
        [int(lag) for lag in config["sequence_lags"]],
    )
    base_columns = [column for column in data if column.startswith(BASE_PREFIXES)]
    sequence_columns = [column for column in data if column.startswith("sequence__")]
    folds: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []

    for scope_kind in config["scopes"]:
        scopes = ["pooled"] if scope_kind == "pooled" else list(config["corridors"])
        for scope in scopes:
            source = data if scope == "pooled" else data.loc[data["corridor"].eq(scope)].copy()
            for horizon in config["horizons"]:
                outcome = f"outcome__regret_{int(horizon)}"
                benefit = f"outcome__benefit_{int(horizon)}"
                usable = source.loc[source[outcome].notna()].copy()
                usable["target"] = usable[outcome].le(float(config["tolerance_bps"])).astype(int)
                usable["regret_bps"] = usable[outcome]
                usable["benefit_bps"] = usable[benefit]
                for feature_set in config["feature_sets"]:
                    raw_columns = base_columns if feature_set == "interpretable" else [*base_columns, *sequence_columns]
                    for model_name in config["models"]:
                        for year in config["test_years"]:
                            start = pd.Timestamp(year=int(year), month=1, day=1)
                            end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                            train = _purged_train(usable, start, int(horizon))
                            test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                            columns = list(raw_columns)
                            if scope == "pooled":
                                train, test, columns = _add_corridor_dummies(
                                    train, test, columns, list(config["corridors"])
                                )
                            if (
                                len(train) < int(config["minimum_training_observations"])
                                or len(test) < 20
                                or train["target"].nunique() < 2
                            ):
                                continue
                            model = _make_model(
                                model_name,
                                iterations=int(config["model_iterations"]),
                                seed=int(config["random_seed"]) + int(year),
                            )
                            model.fit(train[columns], train["target"])
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
                                "model": model_name,
                                "feature_set": feature_set,
                                "horizon": int(horizon),
                            }
                            folds.append(
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
                                    "timestamp", "corridor", "target", "regret_bps", "benefit_bps",
                                    "score", "score_threshold", "matched_week_hit_rate",
                                ]].copy()
                                for key, value in identity.items():
                                    exported[key] = value
                                exported["test_year"] = int(year)
                                signal_frames.append(exported)
                            for feature, importance, signed in _feature_importance(model, columns):
                                importance_rows.append(
                                    {
                                        **identity,
                                        "test_year": int(year),
                                        "feature": feature,
                                        "importance": importance,
                                        "signed_effect": signed,
                                    }
                                )
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return pd.DataFrame(folds), signals, pd.DataFrame(importance_rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(IDENTITY_KEYS), sort=True):
        identity = dict(zip(IDENTITY_KEYS, keys, strict=True))
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        hit_rate = hits / signals if signals else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = float(group["matched_expected_hits"].sum()) / signals if signals else np.nan
        valid_lift = group["lift"].dropna()
        rows.append(
            {
                **identity,
                "signals": signals,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if signals and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if signals and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / signals if signals else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / signals if signals else np.nan,
                "signals_per_week": signals / float(group["duration_weeks"].sum()),
                "worst_fold_lift": float(valid_lift.min()) if len(valid_lift) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def comparison(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope", "model", "horizon"]
    metrics = ["lift", "lift_vs_matched_random", "benefit_mean_bps", "signals_per_week", "worst_fold_lift"]
    base = summary.loc[summary["feature_set"].eq("interpretable"), [*keys, *metrics]]
    sequence = summary.loc[summary["feature_set"].eq("plus_sequence"), [*keys, *metrics]]
    result = sequence.merge(base, on=keys, suffixes=("_sequence", "_baseline"), validate="one_to_one")
    for metric in metrics:
        result[f"delta_{metric}"] = result[f"{metric}_sequence"] - result[f"{metric}_baseline"]
    return result


def run(
    *,
    config_path: Path | str = Path("configs/temporal_sequence_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/temporal_sequence"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    folds, signals, importance = evaluate(pd.read_csv(input_path), config)
    summary = summarize(folds)
    compared = comparison(summary)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    compared.to_csv(output / "comparison.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory lag ablation; GRU/TCN is gated on a positive lag result.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/temporal_sequence_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/temporal_sequence"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
