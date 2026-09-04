"""Small, reproducible daily and hourly signal experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fxpulse.labeling import label_observations


FEATURE_PREFIX = "feature__"
INDICATOR_PREFIX = "indicator__"


def load_config(path: Path | str = Path("configs/experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only experiment schema_version 1 is supported")
    return config


def load_market_data(path: Path | str, *, secid: str, granularity: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    timestamp_column = "trade_date" if granularity == "daily" else "dt_msk"
    required = {timestamp_column, "secid", "open", "high", "low", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} lacks: {', '.join(sorted(missing))}")
    frame = raw.loc[raw["secid"].astype(str).eq(secid)].copy()
    frame["timestamp"] = pd.to_datetime(frame[timestamp_column], errors="raise")
    if granularity == "daily":
        frame["timestamp"] += pd.DateOffset(hours=23, minutes=59)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[["open", "high", "low", "close"]].isna().any(axis=1) | frame["close"].le(0)
    frame = frame.loc[~invalid].sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} has no valid {secid} observations")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{path} has duplicate timestamps for {secid}")
    return frame[["timestamp", "open", "high", "low", "close"]]


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).rank(pct=True)


def build_features(frame: pd.DataFrame, settings: dict[str, Any], *, granularity: str) -> pd.DataFrame:
    """Build features from the current and earlier completed candles only."""

    close = frame["close"]
    one_step_return = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=frame.index)
    for window in settings["return_windows"]:
        features[f"{FEATURE_PREFIX}return_{window}"] = close.pct_change(int(window), fill_method=None)
    for window in settings["volatility_windows"]:
        features[f"{FEATURE_PREFIX}volatility_{window}"] = one_step_return.rolling(
            int(window), min_periods=int(window)
        ).std()
    for window in settings["level_windows"]:
        rolling_min = close.rolling(int(window), min_periods=int(window)).min()
        features[f"{FEATURE_PREFIX}distance_to_min_{window}"] = close / rolling_min - 1
        features[f"{FEATURE_PREFIX}percentile_{window}"] = _rolling_percentile(close, int(window))

    candle_range = (frame["high"] - frame["low"]) / close
    features[f"{FEATURE_PREFIX}candle_range"] = candle_range
    denominator = frame["high"] - frame["low"]
    features[f"{FEATURE_PREFIX}close_location"] = ((close - frame["low"]) / denominator).where(denominator.gt(0), 0.5)
    weekday = frame["timestamp"].dt.weekday
    features[f"{FEATURE_PREFIX}weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features[f"{FEATURE_PREFIX}weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    if granularity == "hourly":
        minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
        features[f"{FEATURE_PREFIX}time_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
        features[f"{FEATURE_PREFIX}time_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

    short_level = int(settings["level_windows"][0])
    middle_level = int(settings["level_windows"][1])
    middle_volatility = int(settings["volatility_windows"][1])
    low_level = features[f"{FEATURE_PREFIX}percentile_{middle_level}"].le(0.10)
    near_low = features[f"{FEATURE_PREFIX}distance_to_min_{short_level}"].le(0.0025)
    rebound = near_low & one_step_return.gt(0) & one_step_return.shift(1).le(0)
    return_window = int(settings["return_windows"][1])
    downside = features[f"{FEATURE_PREFIX}return_{return_window}"].le(
        -features[f"{FEATURE_PREFIX}volatility_{middle_volatility}"] * math.sqrt(return_window)
    )
    features[f"{INDICATOR_PREFIX}low_level"] = low_level
    features[f"{INDICATOR_PREFIX}rebound_from_low"] = rebound
    features[f"{INDICATOR_PREFIX}downside_move"] = downside
    indicator_columns = [column for column in features if column.startswith(INDICATOR_PREFIX)]
    features["candidate"] = features[indicator_columns].any(axis=1)
    reason = pd.Series("", index=features.index, dtype="object")
    for column in indicator_columns:
        name = column.removeprefix(INDICATOR_PREFIX)
        reason = reason.mask(features[column], reason.where(reason.eq(""), reason + "+") + name)
    features["reason"] = reason
    return features.replace([np.inf, -np.inf], np.nan)


def build_labeled_dataset(
    frame: pd.DataFrame,
    settings: dict[str, Any],
    *,
    granularity: str,
    tolerance_bps: float,
) -> pd.DataFrame:
    features = build_features(frame, settings, granularity=granularity)
    dataset = pd.concat([frame, features], axis=1)
    timestamps = frame["timestamp"].dt.tz_localize("Europe/Moscow")
    panel = pd.DataFrame(
        {
            "known_at": timestamps,
            "value_date": frame["timestamp"].dt.date,
            "price": frame["close"],
            "is_carried": False,
        }
    )
    for horizon in settings["horizons"]:
        labels = label_observations(panel, int(horizon), tolerance_bps=tolerance_bps).set_index("position")
        dataset[f"target_good_now_{horizon}"] = labels["hit_favorable"]
        dataset[f"future_regret_bps_{horizon}"] = labels["future_regret_bps"]
        dataset[f"benefit_fwd_bps_{horizon}"] = labels["benefit_fwd_bps"]
    dataset.insert(0, "granularity", granularity)
    return dataset


def _make_model(name: str) -> Any:
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1_000, random_state=20260903),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=30,
            class_weight="balanced_subsample",
            random_state=20260903,
            n_jobs=1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=20260903,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


def _scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype="float64")


def _weekly_cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values("timestamp", kind="mergesort").copy()
    iso = ordered["timestamp"].dt.isocalendar()
    ordered["_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    return ordered.groupby("_week", sort=False).head(cap).drop(columns="_week")


def _metrics(signals: pd.DataFrame, base: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, float | int | None]:
    count = len(signals)
    if len(base):
        duration_weeks = max(
            (base["timestamp"].max() - base["timestamp"].min()).total_seconds() / (7 * 24 * 60 * 60),
            1 / 7,
        )
    else:
        duration_weeks = 0.0
    hit_rate = float(signals["target"].mean()) if count else None
    baseline = float(base["target"].mean()) if len(base) else None
    candidate_baseline = float(candidates["target"].mean()) if len(candidates) else None
    return {
        "signal_count": count,
        "signal_hits": int(signals["target"].sum()) if count else 0,
        "base_count": len(base),
        "base_hits": int(base["target"].sum()),
        "candidate_count": len(candidates),
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline,
        "candidate_baseline_hit_rate": candidate_baseline,
        "lift": hit_rate / baseline if hit_rate is not None and baseline not in {None, 0} else None,
        "benefit_fwd_bps": float(signals["benefit"].mean()) if count else None,
        "benefit_sum_bps": float(signals["benefit"].sum()) if count else 0.0,
        "regret_mean_bps": float(signals["regret"].mean()) if count else None,
        "signals_per_week": count / duration_weeks if duration_weeks else 0.0,
        "test_duration_weeks": duration_weeks,
    }


def _evaluation_frame(dataset: pd.DataFrame, *, horizon: int, feature_columns: list[str]) -> pd.DataFrame:
    target = f"target_good_now_{horizon}"
    regret = f"future_regret_bps_{horizon}"
    benefit = f"benefit_fwd_bps_{horizon}"
    complete = dataset[feature_columns].notna().all(axis=1) & dataset[target].notna()
    frame = dataset.loc[complete, ["timestamp", "candidate", "reason", target, regret, benefit, *feature_columns]].copy()
    return frame.rename(columns={target: "target", regret: "regret", benefit: "benefit"}).reset_index(drop=True)


def _purged_before(frame: pd.DataFrame, boundary: pd.Timestamp, horizon: int) -> pd.DataFrame:
    past = frame.loc[frame["timestamp"] < boundary].copy()
    return past.iloc[:-horizon] if len(past) > horizon else past.iloc[0:0]


def _feature_importance(model: Any, names: list[str]) -> list[tuple[str, float, float | None]]:
    fitted = model.named_steps["logisticregression"] if hasattr(model, "named_steps") else model
    if hasattr(fitted, "coef_"):
        signed = np.asarray(fitted.coef_[0], dtype="float64")
        return [(name, abs(float(value)), float(value)) for name, value in zip(names, signed, strict=True)]
    values = np.asarray(fitted.feature_importances_, dtype="float64")
    return [(name, float(value), None) for name, value in zip(names, values, strict=True)]


def evaluate_dataset(
    dataset: pd.DataFrame,
    *,
    settings: dict[str, Any],
    validation: dict[str, Any],
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in dataset if column.startswith(FEATURE_PREFIX)]
    fold_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    granularity = str(dataset["granularity"].iloc[0])
    for horizon in settings["horizons"]:
        work = _evaluation_frame(dataset, horizon=int(horizon), feature_columns=feature_columns)
        for test_year in validation["test_years"]:
            test_start = pd.Timestamp(year=int(test_year), month=1, day=1)
            test_end = pd.Timestamp(year=int(test_year) + 1, month=1, day=1)
            test = work.loc[work["timestamp"].between(test_start, test_end, inclusive="left")].copy()
            validation_start = test_start - pd.DateOffset(years=1)
            selection = work.loc[work["timestamp"].between(validation_start, test_start, inclusive="left")].copy()
            fit = _purged_before(work, validation_start, int(horizon))
            outer_train = _purged_before(work, test_start, int(horizon))
            if len(test) < 20 or len(selection) < 20 or len(fit) < 100 or fit["target"].nunique() < 2:
                continue

            rule_signals = _weekly_cap(test.loc[test["candidate"]].copy(), int(validation["weekly_cap"]))
            rule_metrics = _metrics(rule_signals, test, test.loc[test["candidate"]])
            fold_rows.append(
                {
                    "granularity": granularity,
                    "horizon": int(horizon),
                    "test_year": int(test_year),
                    "model": "rules_only",
                    "selection_status": "baseline",
                    "score_quantile": None,
                    **rule_metrics,
                }
            )

            for name in model_names:
                model = _make_model(name)
                model.fit(fit[feature_columns], fit["target"].astype(int))
                fit_candidates = fit.loc[fit["candidate"]].copy()
                selection_candidates = selection.loc[selection["candidate"]].copy()
                if fit_candidates.empty or selection_candidates.empty:
                    continue
                fit_scores = _scores(model, fit_candidates[feature_columns])
                selection_scores = _scores(model, selection_candidates[feature_columns])
                options: list[tuple[float, dict[str, float | int | None]]] = []
                for quantile in validation["score_quantiles"]:
                    threshold = float(np.quantile(fit_scores, float(quantile)))
                    selected = selection_candidates.loc[selection_scores >= threshold].copy()
                    selected["score"] = selection_scores[selection_scores >= threshold]
                    selected = _weekly_cap(selected, int(validation["weekly_cap"]))
                    options.append((float(quantile), _metrics(selected, selection, selection_candidates)))
                valid = [
                    option
                    for option in options
                    if option[1]["signal_count"] >= int(validation["minimum_validation_signals"])
                    and float(validation["minimum_signals_per_week"])
                    <= float(option[1]["signals_per_week"])
                    <= float(validation["maximum_signals_per_week"])
                    and option[1]["lift"] is not None
                    and float(option[1]["lift"]) >= float(validation["minimum_validation_lift"])
                    and option[1]["benefit_fwd_bps"] is not None
                    and float(option[1]["benefit_fwd_bps"]) > 0
                ]
                pool = valid or [option for option in options if option[1]["signal_count"]]
                if not pool:
                    continue
                quantile, selection_metrics = max(
                    pool,
                    key=lambda option: (
                        float(option[1]["lift"] or -math.inf),
                        float(option[1]["benefit_fwd_bps"] or -math.inf),
                    ),
                )
                status = "selected" if valid else "diagnostic_only"

                final_model = _make_model(name)
                final_model.fit(outer_train[feature_columns], outer_train["target"].astype(int))
                train_candidates = outer_train.loc[outer_train["candidate"]].copy()
                threshold = float(np.quantile(_scores(final_model, train_candidates[feature_columns]), quantile))
                test_candidates = test.loc[test["candidate"]].copy()
                test_scores = _scores(final_model, test_candidates[feature_columns])
                selected_test = test_candidates.loc[test_scores >= threshold].copy()
                selected_test["score"] = test_scores[test_scores >= threshold]
                selected_test = _weekly_cap(selected_test, int(validation["weekly_cap"]))
                test_metrics = _metrics(selected_test, test, test_candidates)
                fold_rows.append(
                    {
                        "granularity": granularity,
                        "horizon": int(horizon),
                        "test_year": int(test_year),
                        "model": name,
                        "selection_status": status,
                        "score_quantile": quantile,
                        "selection_lift": selection_metrics["lift"],
                        "selection_signals_per_week": selection_metrics["signals_per_week"],
                        **test_metrics,
                    }
                )
                for row in selected_test.itertuples(index=False):
                    signal_rows.append(
                        {
                            "granularity": granularity,
                            "horizon": int(horizon),
                            "test_year": int(test_year),
                            "model": name,
                            "selection_status": status,
                            "communication_allowed": status == "selected",
                            "timestamp": row.timestamp,
                            "score": float(row.score),
                            "reason": row.reason,
                            "target": bool(row.target),
                            "future_regret_bps": float(row.regret),
                            "benefit_fwd_bps": float(row.benefit),
                        }
                    )
                for feature, importance, signed in _feature_importance(final_model, feature_columns):
                    importance_rows.append(
                        {
                            "granularity": granularity,
                            "horizon": int(horizon),
                            "test_year": int(test_year),
                            "model": name,
                            "feature": feature.removeprefix(FEATURE_PREFIX),
                            "importance": importance,
                            "signed_effect": signed,
                        }
                    )
    return pd.DataFrame(fold_rows), pd.DataFrame(signal_rows), pd.DataFrame(importance_rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return folds.copy()
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(["granularity", "horizon", "model", "selection_status"], sort=True):
        granularity, horizon, model, selection_status = keys
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["base_count"].sum())
        base_hits = int(group["base_hits"].sum())
        hit_rate = hits / signals if signals else None
        baseline = base_hits / base_count if base_count else None
        duration_weeks = float(group["test_duration_weeks"].sum())
        rows.append(
            {
                "granularity": granularity,
                "horizon": int(horizon),
                "model": model,
                "selection_status": selection_status,
                "folds": len(group),
                "signals": signals,
                "hit_rate": hit_rate,
                "baseline_hit_rate": baseline,
                "lift": hit_rate / baseline if hit_rate is not None and baseline else None,
                "benefit_fwd_bps": group["benefit_sum_bps"].sum() / signals if signals else None,
                "test_duration_weeks": duration_weeks,
                "signals_per_week": signals / duration_weeks if duration_weeks else 0.0,
                "worst_fold_lift": group["lift"].min(),
                "selected_folds": int(group["selection_status"].eq("selected").sum()),
            }
        )
    return pd.DataFrame(rows)


def best_by_horizon(summary: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Select the highest-lift model and report frequency over its full OOT runtime."""

    columns = (
        "granularity",
        "horizon",
        "best_model",
        "signals",
        "lift",
        "test_duration_weeks",
        "signals_per_week",
    )
    if summary.empty:
        return pd.DataFrame(columns=columns)
    eligible = summary.loc[
        summary["selection_status"].eq("selected")
        & summary["model"].ne("rules_only")
        & summary["signals"].gt(0)
        & summary["lift"].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    best = (
        eligible.sort_values(
            ["granularity", "horizon", "lift", "benefit_fwd_bps", "signals"],
            ascending=[True, True, False, False, False],
            kind="mergesort",
        )
        .groupby(["granularity", "horizon"], sort=True, as_index=False)
        .head(1)
        .rename(columns={"model": "best_model"})
    )
    durations = (
        folds.loc[folds["model"].ne("rules_only")]
        .groupby(["granularity", "horizon", "model"], sort=False)["test_duration_weeks"]
        .sum()
        .rename("policy_test_duration_weeks")
        .reset_index()
        .rename(columns={"model": "best_model"})
    )
    best = best.drop(columns=["test_duration_weeks", "signals_per_week"]).merge(
        durations,
        how="left",
        on=["granularity", "horizon", "best_model"],
        validate="one_to_one",
    )
    best["test_duration_weeks"] = best.pop("policy_test_duration_weeks")
    best["signals_per_week"] = best["signals"] / best["test_duration_weeks"]
    return best.loc[:, columns].reset_index(drop=True)


def run_experiment(
    *,
    config_path: Path | str = Path("configs/experiment.json"),
    processed_dir: Path | str = Path("data/processed"),
    artifact_dir: Path | str = Path("artifacts/experiment"),
) -> dict[str, Any]:
    config = load_config(config_path)
    processed = Path(processed_dir)
    artifacts = Path(artifact_dir)
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    fold_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    for granularity, settings in config["datasets"].items():
        market = load_market_data(settings["path"], secid=config["target"]["secid"], granularity=granularity)
        dataset = build_labeled_dataset(
            market,
            settings,
            granularity=granularity,
            tolerance_bps=float(config["target"]["tolerance_bps"]),
        )
        dataset.to_csv(processed / f"moex_cny_{granularity}_labeled.csv", index=False)
        coverage[granularity] = {
            "rows": len(dataset),
            "from": dataset["timestamp"].min().isoformat(),
            "to": dataset["timestamp"].max().isoformat(),
            "candidates": int(dataset["candidate"].sum()),
        }
        folds, signals, importance = evaluate_dataset(
            dataset,
            settings=settings,
            validation=config["validation"],
            model_names=config["models"],
        )
        fold_frames.append(folds)
        signal_frames.append(signals)
        importance_frames.append(importance)
    folds = pd.concat(fold_frames, ignore_index=True)
    signals = pd.concat(signal_frames, ignore_index=True)
    importance = pd.concat(importance_frames, ignore_index=True)
    summary = summarize(folds)
    best = best_by_horizon(summary, folds)
    folds.to_csv(artifacts / "folds.csv", index=False)
    signals.to_csv(artifacts / "signals.csv", index=False)
    importance.to_csv(artifacts / "feature_importance.csv", index=False)
    summary.to_csv(artifacts / "summary.csv", index=False)
    best.to_csv(artifacts / "best_by_horizon.csv", index=False)
    metadata = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "coverage": coverage,
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    (artifacts / "run_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/experiment"))
    args = parser.parse_args(argv)
    metadata = run_experiment(config_path=args.config, processed_dir=args.processed_dir, artifact_dir=args.artifact_dir)
    print(json.dumps(metadata["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
