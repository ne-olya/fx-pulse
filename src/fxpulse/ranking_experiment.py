"""Compare pointwise classification with pairwise ranking of historical weeks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from fxpulse.next_hypotheses import _add_corridor_dummies, _make_model, _purged_train
from fxpulse.temporal_sequence_experiment import _evaluate_fold, summarize


FEATURE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/ranking_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("ranking config must be preregistered")
    if config.get("models") != ["xgboost_classifier", "xgboost_ranker"]:
        raise ValueError("ranking models differ from the implementation")
    return config


def relevance_from_regret(regret: pd.Series, boundaries: list[float]) -> pd.Series:
    if sorted(boundaries) != boundaries or len(boundaries) != 3:
        raise ValueError("exactly three increasing relevance boundaries are required")
    result = pd.Series(0, index=regret.index, dtype=int)
    result = result.mask(regret.le(boundaries[2]), 1)
    result = result.mask(regret.le(boundaries[1]), 2)
    result = result.mask(regret.le(boundaries[0]), 3)
    return result


def ranking_qid(frame: pd.DataFrame) -> pd.Series:
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    iso = timestamp.dt.isocalendar()
    return frame["corridor"].astype(str) + "__" + iso["year"].astype(str) + "_" + iso["week"].astype(str)


def _ranker_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    from xgboost import XGBRanker

    ordered = train.copy()
    ordered["_qid_text"] = ranking_qid(ordered)
    ordered = ordered.sort_values(["_qid_text", "timestamp"], kind="mergesort")
    qid, _ = pd.factorize(ordered["_qid_text"], sort=True)
    sort_index = np.argsort(qid, kind="stable")
    ordered = ordered.iloc[sort_index]
    qid = qid[sort_index]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_values = imputer.fit_transform(ordered[columns])
    test_values = imputer.transform(test[columns])
    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=iterations,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=2,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train_values, ordered["relevance"].to_numpy(), qid=qid, verbose=False)
    return model.predict(test_values)


def evaluate(data: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
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
                usable["relevance"] = relevance_from_regret(
                    usable[regret_column], [float(value) for value in config["relevance_boundaries_bps"]]
                )
                usable["regret_bps"] = usable[regret_column]
                usable["benefit_bps"] = usable[benefit_column]
                for model_name in config["models"]:
                    for year in config["test_years"]:
                        start = pd.Timestamp(year=int(year), month=1, day=1)
                        end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                        train = _purged_train(usable, start, int(horizon))
                        test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                        columns = list(feature_columns)
                        if scope == "pooled":
                            train, test, columns = _add_corridor_dummies(
                                train, test, columns, list(config["corridors"])
                            )
                        if len(train) < int(config["minimum_training_observations"]) or len(test) < 20:
                            continue
                        if model_name == "xgboost_classifier":
                            model = _make_model(
                                "xgboost",
                                iterations=int(config["model_iterations"]),
                                seed=int(config["random_seed"]) + int(year),
                            )
                            model.fit(train[columns], train["target"])
                            test["score"] = model.predict_proba(test[columns])[:, 1]
                        else:
                            test["score"] = _ranker_scores(
                                train,
                                test,
                                columns,
                                iterations=int(config["model_iterations"]),
                                seed=int(config["random_seed"]) + int(year),
                            )
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
                            "feature_set": "weekly_ranking" if model_name.endswith("ranker") else "pointwise",
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
                                "timestamp", "corridor", "target", "relevance", "regret_bps", "benefit_bps",
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
    config_path: Path | str = Path("configs/ranking_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/ranking"),
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
        "warning": "Exploratory comparison added after earlier results; requires untouched confirmation.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ranking_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/ranking"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
