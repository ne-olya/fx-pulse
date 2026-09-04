"""Compare binary, multiclass and direct-regret daily objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _add_corridor_dummies, _apply_policy, _make_model


IDENTITY_KEYS = ("scope", "model", "horizon")


def load_config(path: Path | str = Path("configs/regret_formulation_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("regret formulation config must be preregistered schema_version 1")
    return config


def regret_class(values: pd.Series, boundaries: list[float]) -> pd.Series:
    return pd.Series(np.digitize(values.to_numpy(), boundaries, right=True), index=values.index, dtype="int64")


def _make_regret_model(name: str, *, iterations: int, seed: int) -> Any:
    from catboost import CatBoostClassifier, CatBoostRegressor

    if name == "binary_catboost":
        return _make_model("catboost", iterations=iterations, seed=seed)
    if name == "multiclass_catboost":
        estimator: Any = CatBoostClassifier(
            iterations=iterations,
            depth=5,
            learning_rate=0.05,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    elif name in {"regression_mae", "quantile_p90"}:
        loss = "MAE" if name == "regression_mae" else "Quantile:alpha=0.9"
        estimator = CatBoostRegressor(
            iterations=iterations,
            depth=5,
            learning_rate=0.05,
            loss_function=loss,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    else:
        raise ValueError(f"unknown regret model {name}")
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), estimator)


def _score(model: Any, name: str, features: pd.DataFrame) -> np.ndarray:
    if name == "binary_catboost":
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if name == "multiclass_catboost":
        fitted = model.steps[-1][1]
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        class_zero = int(np.flatnonzero(np.asarray(fitted.classes_) == 0)[0])
        return probabilities[:, class_zero]
    # Regressors learn log1p(regret); lower predicted regret is a better score.
    return -np.asarray(model.predict(features), dtype=float)


def _purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    past = frame.loc[frame["timestamp"] < start].copy()
    keep: list[int] = []
    for _, group in past.groupby("corridor", sort=False):
        keep.extend(group.index[:-horizon] if len(group) > horizon else [])
    return past.loc[sorted(keep)].copy()


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    feature_columns = [
        column
        for column in data
        if column.startswith(("base__", "leg__", "regime__", "market__", "indicator__"))
    ]
    policy = config["adaptive_policy"]
    folds: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for scope_kind in config["scopes"]:
        scopes = ["pooled"] if scope_kind == "pooled" else list(config["corridors"])
        for scope in scopes:
            source = data if scope == "pooled" else data.loc[data["corridor"].eq(scope)].copy()
            for horizon in config["horizons"]:
                outcome = f"outcome__regret_{horizon}"
                benefit = f"outcome__benefit_{horizon}"
                usable = source.loc[source[outcome].notna()].copy()
                usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
                usable["regret_bps"] = usable[outcome]
                usable["benefit_bps"] = usable[benefit]
                usable["regret_class"] = regret_class(usable[outcome], list(config["regret_classes_bps"]))
                usable["log_regret"] = np.log1p(usable[outcome].clip(lower=0))
                for model_name in config["models"]:
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        columns = list(feature_columns)
                        if scope == "pooled":
                            train, test, columns = _add_corridor_dummies(train, test, columns, list(config["corridors"]))
                        if len(train) < int(config["minimum_training_observations"]) or len(test) < 20:
                            continue
                        if model_name == "binary_catboost":
                            train_target = train["target"]
                        elif model_name == "multiclass_catboost":
                            train_target = train["regret_class"]
                        else:
                            train_target = train["log_regret"]
                        if train_target.nunique() < 2:
                            continue
                        model = _make_regret_model(
                            model_name,
                            iterations=int(config["model_iterations"]),
                            seed=int(config["random_seed"]) + int(year),
                        )
                        model.fit(train[columns], train_target)
                        test["score"] = _score(model, model_name, test[columns])
                        selected_parts: list[pd.DataFrame] = []
                        for _, corridor_test in test.groupby("corridor", sort=False):
                            candidates = adaptive_candidates(
                                corridor_test,
                                share=float(policy["top_score_share"]),
                                lookback=int(policy["lookback_observations"]),
                                minimum_history=int(policy["minimum_history_observations"]),
                            )
                            selected_parts.append(
                                _apply_policy(
                                    candidates,
                                    cooldown_days=int(policy["cooldown_days"]),
                                    weekly_cap=int(policy["weekly_cap"]),
                                )
                            )
                        selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else test.iloc[0:0]
                        test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                        selected["week"] = selected["timestamp"].dt.to_period("W").astype(str)
                        weekly_rate = test.groupby(["corridor", "week"])["target"].mean()
                        selected["matched_week_hit_rate"] = [
                            float(weekly_rate.loc[(corridor, week)])
                            for corridor, week in zip(selected["corridor"], selected["week"], strict=True)
                        ]
                        count = len(selected)
                        base_rate = float(test["target"].mean())
                        hit_rate = float(selected["target"].mean()) if count else np.nan
                        exposure = int(test["corridor"].nunique())
                        duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7) * exposure
                        identity = {"scope": scope, "model": model_name, "horizon": int(horizon)}
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
                            exported = selected[["timestamp", "corridor", "target", "regret_bps", "benefit_bps", "score", "score_threshold", "matched_week_hit_rate"]].copy()
                            for key, value in identity.items():
                                exported[key] = value
                            exported["test_year"] = int(year)
                            signal_frames.append(exported)
    return pd.DataFrame(folds), pd.concat(signal_frames, ignore_index=True)


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
        duration = float(group["duration_weeks"].sum())
        fold_lifts = group["lift"].dropna()
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


def run(
    *,
    config_path: Path | str = Path("configs/regret_formulation_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/regret_formulations"),
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
        "warning": "Exploratory target-formulation comparison; requires untouched confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/regret_formulation_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/regret_formulations"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
