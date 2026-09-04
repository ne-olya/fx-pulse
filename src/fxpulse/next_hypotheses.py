"""Exploratory, leakage-safe tests for the next FX Pulse hypotheses.

The run compares several predeclared targets, feature families and selective
dispatch policies on the same expanding yearly folds.  Results from this file
are exploratory: the same history is used to compare many variants, so a
winner must later survive a new corridor or a genuinely untouched period.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = "base"
LEG_FEATURES = "usd_legs"
REGIME_FEATURES = "regimes"
MARKET_FEATURES = "curated_market"
INDICATOR_FEATURES = "interpretable_indicators"
OUTLIER_FEATURES = "outliers"

FEATURE_PREFIXES: dict[str, tuple[str, ...]] = {
    BASE_FEATURES: ("base__",),
    LEG_FEATURES: ("base__", "leg__"),
    REGIME_FEATURES: ("base__", "leg__", "regime__"),
    MARKET_FEATURES: ("base__", "leg__", "regime__", "market__"),
    INDICATOR_FEATURES: ("base__", "leg__", "regime__", "market__", "indicator__"),
    OUTLIER_FEATURES: ("base__", "leg__", "regime__", "market__", "indicator__"),
}

CURATED_MARKETS = {
    "cny": "moex__fx_cny_rub_tom__close",
    "gold": "moex__metal_gold_rub_tom__close",
    "silver": "moex__metal_silver_rub_tom__close",
    "imoex": "moex__index_imoex__close",
    "rtsi": "moex__index_rtsi__close",
}

SUMMARY_KEYS = (
    "scope",
    "model",
    "feature_set",
    "horizon",
    "tolerance_bps",
    "top_score_share",
    "cooldown_days",
    "value_gate",
)


@dataclass(frozen=True, order=True)
class Task:
    scope: str
    model: str
    feature_set: str
    horizon: int
    tolerance_bps: float


def load_config(path: Path | str = Path("configs/next_hypotheses.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or "wave_1" not in config:
        raise ValueError("next hypotheses config must use schema_version 1 and define wave_1")
    expected = {"AMD", "KGS", "KZT", "TJS", "UZS"}
    if set(config.get("corridors", ())) != expected:
        raise ValueError(f"corridors must be exactly {sorted(expected)}")
    if set(config["wave_1"]["feature_sets"]) != set(FEATURE_PREFIXES):
        raise ValueError("wave_1.feature_sets and implemented feature sets differ")
    return config


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).rank(pct=True)


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std()
    return (values - mean) / std.replace(0, np.nan)


def _future_outcomes(price: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Use exactly the next h observations; lower RUB/local is better."""

    future = pd.concat([price.shift(-step) for step in range(1, horizon + 1)], axis=1)
    complete = future.notna().all(axis=1)
    future_min = future.min(axis=1).where(complete)
    future_mean = future.mean(axis=1).where(complete)
    # This is the product definition proposed in the hypothesis note.
    regret = ((price / future_min) - 1).clip(lower=0) * 10_000
    benefit = ((future_mean / price) - 1) * 10_000
    return regret.where(complete), benefit.where(complete)


def _load_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"rate_date", "ccy", "nominal", "rate_rub"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    raw["timestamp"] = pd.to_datetime(raw["rate_date"], errors="raise")
    raw["price"] = pd.to_numeric(raw["rate_rub"], errors="raise") / pd.to_numeric(
        raw["nominal"], errors="raise"
    )
    if raw["price"].isna().any() or raw["price"].le(0).any():
        raise ValueError("CBR prices must be positive and complete")
    if raw.duplicated(["timestamp", "ccy"]).any():
        raise ValueError("CBR data contain duplicate date/currency rows")
    return raw[["timestamp", "ccy", "price"]].sort_values(["ccy", "timestamp"], kind="mergesort")


def _market_feature_frame(path: Path, *, lag: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "trade_date" not in raw:
        raise ValueError(f"{path} lacks trade_date")
    raw["timestamp"] = pd.to_datetime(raw["trade_date"], errors="raise")
    raw = raw.set_index("timestamp").sort_index()
    result = pd.DataFrame(index=raw.index)
    for short, column in CURATED_MARKETS.items():
        if column not in raw:
            raise ValueError(f"{path} lacks {column}")
        raw_price = pd.to_numeric(raw[column], errors="coerce")
        # Rolling observations mean trading observations, not calendar rows.
        # The research panel has weekends/holidays from its outer join; leaving
        # them inside a strict 20-row window would make volatility permanently
        # missing.
        price = raw_price.dropna()
        ret1 = price.pct_change(fill_method=None)
        for window in (1, 3, 5):
            feature = price.pct_change(window, fill_method=None).shift(lag)
            result[f"market__{short}_return_{window}"] = feature.reindex(result.index)
        volatility = ret1.rolling(20, min_periods=20).std().shift(lag)
        zscore = _rolling_zscore(price, 20).shift(lag)
        result[f"market__{short}_volatility_20"] = volatility.reindex(result.index)
        result[f"market__{short}_zscore_20"] = zscore.reindex(result.index)
        result[f"market__{short}_missing"] = raw_price.shift(lag).isna().astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def _corridor_features(
    prices: pd.DataFrame,
    market: pd.DataFrame,
    *,
    corridor: str,
    horizons: list[int],
) -> pd.DataFrame:
    pivot = prices.pivot(index="timestamp", columns="ccy", values="price").sort_index()
    if corridor not in pivot or "USD" not in pivot:
        raise ValueError(f"missing {corridor} or USD CBR series")
    target = pivot[corridor].dropna()
    usd = pivot["USD"].reindex(target.index)
    local_per_usd = usd / target
    ret1 = target.pct_change(fill_method=None)
    data = pd.DataFrame({"timestamp": target.index, "corridor": corridor, "price": target.to_numpy()})
    data = data.set_index("timestamp")

    for window in (1, 3, 5, 10, 20):
        data[f"base__return_{window}"] = target.pct_change(window, fill_method=None)
    for window in (5, 10, 20):
        data[f"base__volatility_{window}"] = ret1.rolling(window, min_periods=window).std()
    for window in (20, 60, 120):
        rolling_min = target.rolling(window, min_periods=window).min()
        rolling_max = target.rolling(window, min_periods=window).max()
        data[f"base__percentile_{window}"] = _rolling_percentile(target, window)
        data[f"base__distance_to_min_{window}"] = target / rolling_min - 1
        data[f"base__distance_from_max_{window}"] = target / rolling_max - 1
        data[f"base__zscore_{window}"] = _rolling_zscore(target, window)

    weekday = target.index.dayofweek
    month = target.index.month
    data["base__weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    data["base__weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    data["base__month_sin"] = np.sin(2 * np.pi * month / 12)
    data["base__month_cos"] = np.cos(2 * np.pi * month / 12)
    data["base__month_end"] = target.index.is_month_end.astype(float)

    usd_ret1 = usd.pct_change(fill_method=None)
    local_ret1 = local_per_usd.pct_change(fill_method=None)
    for window in (1, 3, 5, 10):
        rub_leg = usd.pct_change(window, fill_method=None)
        recipient_leg = local_per_usd.pct_change(window, fill_method=None)
        data[f"leg__usd_rub_return_{window}"] = rub_leg
        data[f"leg__local_per_usd_return_{window}"] = recipient_leg
        data[f"leg__relative_momentum_{window}"] = rub_leg - recipient_leg
    for window in (5, 20):
        rub_vol = usd_ret1.rolling(window, min_periods=window).std()
        local_vol = local_ret1.rolling(window, min_periods=window).std()
        data[f"leg__usd_rub_volatility_{window}"] = rub_vol
        data[f"leg__local_per_usd_volatility_{window}"] = local_vol
        data[f"leg__volatility_gap_{window}"] = rub_vol - local_vol
    data["leg__divergence_3"] = data["leg__usd_rub_return_3"] - data["leg__local_per_usd_return_3"]

    rolling_vol_rank = _rolling_percentile(data["base__volatility_20"], 252)
    trend_strength = data["base__return_20"] / (
        data["base__volatility_20"].replace(0, np.nan) * math.sqrt(20)
    )
    shock_strength = ret1.abs() / data["base__volatility_20"].replace(0, np.nan)
    data["regime__volatility_percentile_252"] = rolling_vol_rank
    data["regime__trend_strength_20"] = trend_strength
    data["regime__shock_strength"] = shock_strength
    data["regime__low_vol"] = rolling_vol_rank.le(0.33).astype(float)
    data["regime__high_vol"] = rolling_vol_rank.ge(0.67).astype(float)
    data["regime__trend_up"] = trend_strength.ge(0.5).astype(float)
    data["regime__trend_down"] = trend_strength.le(-0.5).astype(float)
    data["regime__shock"] = shock_strength.ge(2.0).astype(float)
    data["regime__post_2022_02_24"] = (data.index >= pd.Timestamp("2022-02-24")).astype(float)
    data["regime__post_2024_06_13"] = (data.index >= pd.Timestamp("2024-06-13")).astype(float)
    regime = pd.Series("NORMAL", index=data.index, dtype="object")
    regime = regime.mask(data["regime__low_vol"].eq(1), "LOW_VOL")
    regime = regime.mask(data["regime__trend_up"].eq(1), "TREND_UP")
    regime = regime.mask(data["regime__trend_down"].eq(1), "TREND_DOWN")
    regime = regime.mask(data["regime__high_vol"].eq(1), "HIGH_VOL")
    regime = regime.mask(data["regime__shock"].eq(1), "SHOCK")
    data["regime"] = regime

    # Market closes are lagged before this exact-date alignment. A short carry
    # bridges holidays but never turns a long missing stretch into live data.
    aligned_market = market.reindex(data.index).ffill(limit=3)
    data = data.join(aligned_market, how="left")

    data["indicator__cheap_60"] = data["base__percentile_60"].le(0.30).astype(float)
    data["indicator__near_min_20"] = data["base__distance_to_min_20"].le(0.003).astype(float)
    data["indicator__rebound_from_low"] = (
        data["indicator__near_min_20"].eq(1)
        & data["base__return_1"].gt(0)
        & data["base__return_1"].shift(1).le(0)
    ).astype(float)
    data["indicator__rub_strength_3"] = data["leg__usd_rub_return_3"].lt(0).astype(float)
    data["indicator__recipient_weakness_3"] = data["leg__local_per_usd_return_3"].gt(0).astype(float)
    data["indicator__volatility_falling"] = (
        data["base__volatility_5"] < data["base__volatility_20"]
    ).astype(float)
    data["indicator__value_and_calm"] = (
        data["indicator__cheap_60"].eq(1) & data["regime__shock"].eq(0)
    ).astype(float)

    for horizon in horizons:
        regret, benefit = _future_outcomes(target, horizon)
        data[f"outcome__regret_{horizon}"] = regret
        data[f"outcome__benefit_{horizon}"] = benefit
    return data.reset_index().replace([np.inf, -np.inf], np.nan)


def build_datasets(
    *,
    cbr_path: Path,
    research_panel_path: Path,
    corridors: list[str],
    horizons: list[int],
    market_lag: int,
) -> dict[str, pd.DataFrame]:
    prices = _load_prices(cbr_path)
    market = _market_feature_frame(research_panel_path, lag=market_lag)
    return {
        corridor: _corridor_features(prices, market, corridor=corridor, horizons=horizons)
        for corridor in corridors
    }


def _feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    prefixes = FEATURE_PREFIXES[feature_set]
    return [column for column in frame if column.startswith(prefixes)]


def _purged_train(frame: pd.DataFrame, start: pd.Timestamp, horizon: int) -> pd.DataFrame:
    past = frame.loc[frame["timestamp"] < start].copy()
    if past.empty:
        return past
    keep: list[int] = []
    for _, group in past.groupby("corridor", sort=False):
        keep.extend(group.index[:-horizon] if len(group) > horizon else [])
    return past.loc[sorted(keep)].copy()


def _make_model(name: str, *, iterations: int, seed: int) -> Any:
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=1_000, random_state=seed),
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            CatBoostClassifier(
                iterations=iterations,
                depth=5,
                learning_rate=0.05,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            XGBClassifier(
                n_estimators=iterations,
                max_depth=3,
                learning_rate=0.05,
                min_child_weight=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                eval_metric="logloss",
            ),
        )
    raise ValueError(f"unknown model {name}")


def _outlier_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train[columns]))
    test_values = scaler.transform(imputer.transform(test[columns]))
    detector = IsolationForest(n_estimators=100, contamination="auto", random_state=seed, n_jobs=1)
    detector.fit(train_values)
    train = train.copy()
    test = test.copy()
    train["outlier__isolation_score"] = -detector.score_samples(train_values)
    test["outlier__isolation_score"] = -detector.score_samples(test_values)

    medians = np.median(train_values, axis=0)
    mad = np.median(np.abs(train_values - medians), axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    train["outlier__robust_distance"] = np.sqrt(np.mean(np.square((train_values - medians) / scale), axis=1))
    test["outlier__robust_distance"] = np.sqrt(np.mean(np.square((test_values - medians) / scale), axis=1))
    return train, test, [*columns, "outlier__isolation_score", "outlier__robust_distance"]


def _add_corridor_dummies(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], corridors: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    dummy_columns: list[str] = []
    for corridor in corridors:
        column = f"corridor__{corridor.lower()}"
        train[column] = train["corridor"].eq(corridor).astype(float)
        test[column] = test["corridor"].eq(corridor).astype(float)
        dummy_columns.append(column)
    return train, test, [*columns, *dummy_columns]


def _apply_policy(candidates: pd.DataFrame, *, cooldown_days: int, weekly_cap: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ranked = candidates.sort_values("timestamp", kind="mergesort").copy()
    iso = ranked["timestamp"].dt.isocalendar()
    ranked["_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    selected: list[int] = []
    last: pd.Timestamp | None = None
    week_counts: dict[str, int] = {}
    for index, row in ranked.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        week = str(row["_week"])
        if week_counts.get(week, 0) >= weekly_cap:
            continue
        if last is None or (timestamp - last).days >= cooldown_days:
            selected.append(index)
            last = timestamp
            week_counts[week] = week_counts.get(week, 0) + 1
    return ranked.loc[selected].drop(columns="_week")


def _feature_importance(model: Any, names: list[str]) -> list[tuple[str, float, float | None]]:
    fitted = model.steps[-1][1]
    transformed_names = model.steps[0][1].get_feature_names_out(names)
    if hasattr(fitted, "coef_"):
        signed = np.asarray(fitted.coef_[0], dtype=float)
        return [(str(name), abs(float(value)), float(value)) for name, value in zip(transformed_names, signed, strict=True)]
    values = np.asarray(fitted.feature_importances_, dtype=float)
    return [(str(name), float(value), None) for name, value in zip(transformed_names, values, strict=True)]


def _task_plan(config: dict[str, Any]) -> list[Task]:
    wave = config["wave_1"]
    tasks: set[Task] = set()
    for corridor in config["corridors"]:
        for horizon in config["regret"]["daily_horizons"]:
            for tolerance in config["regret"]["tolerance_bps"]:
                for feature_set in wave["target_grid_feature_sets"]:
                    for model in wave["models"]:
                        tasks.add(Task(corridor, model, feature_set, int(horizon), float(tolerance)))
    for feature_set in wave["feature_sets"]:
        for corridor in [*config["corridors"], "pooled"]:
            tasks.add(
                Task(
                    corridor,
                    wave["pooled_model"] if corridor == "pooled" else "catboost",
                    feature_set,
                    int(wave["ablation_horizon"]),
                    float(wave["ablation_tolerance_bps"]),
                )
            )
    for horizon in config["regret"]["daily_horizons"]:
        for tolerance in config["regret"]["tolerance_bps"]:
            tasks.add(
                Task(
                    "pooled",
                    wave["pooled_model"],
                    INDICATOR_FEATURES,
                    int(horizon),
                    float(tolerance),
                )
            )
    return sorted(tasks)


def _fold_metrics(
    test: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    task: Task,
    year: int,
    share: float,
    cooldown: int,
    value_gate: bool,
) -> dict[str, object]:
    signal_count = len(selected)
    base_rate = float(test["target"].mean()) if len(test) else np.nan
    hit_rate = float(selected["target"].mean()) if signal_count else np.nan
    # Exact expectation of a random policy with the same number of signals in
    # each corridor/week. It is the mean of each selected slot's weekly base rate.
    matched_expected_hits = float(selected["matched_week_hit_rate"].sum()) if signal_count else 0.0
    start = test["timestamp"].min()
    end = test["timestamp"].max()
    corridor_exposures = int(test["corridor"].nunique()) if len(test) else 0
    duration_weeks = (
        max(float((end - start).days) / 7, 1 / 7) * corridor_exposures if len(test) else 0.0
    )
    return {
        **task.__dict__,
        "test_year": year,
        "top_score_share": share,
        "cooldown_days": cooldown,
        "value_gate": value_gate,
        "test_count": len(test),
        "test_hits": int(test["target"].sum()),
        "signal_count": signal_count,
        "signal_hits": int(selected["target"].sum()) if signal_count else 0,
        "base_hit_rate": base_rate,
        "hit_rate": hit_rate,
        "lift": hit_rate / base_rate if signal_count and base_rate > 0 else np.nan,
        "matched_expected_hits": matched_expected_hits,
        "regret_sum_bps": float(selected["regret_bps"].sum()) if signal_count else 0.0,
        "benefit_sum_bps": float(selected["benefit_bps"].sum()) if signal_count else 0.0,
        "duration_weeks": duration_weeks,
    }


def evaluate(
    datasets: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wave = config["wave_1"]
    folds: list[dict[str, object]] = []
    scores: list[pd.DataFrame] = []
    signals: list[pd.DataFrame] = []
    importances: list[dict[str, object]] = []
    tasks = _task_plan(config)
    pooled = pd.concat(datasets.values(), ignore_index=True)

    for task_index, task in enumerate(tasks, start=1):
        source = pooled if task.scope == "pooled" else datasets[task.scope]
        base_columns = _feature_columns(source, task.feature_set)
        outcome_regret = f"outcome__regret_{task.horizon}"
        outcome_benefit = f"outcome__benefit_{task.horizon}"
        usable = source.loc[source[outcome_regret].notna()].copy()
        usable["target"] = usable[outcome_regret].le(task.tolerance_bps).astype(int)
        usable["regret_bps"] = usable[outcome_regret]
        usable["benefit_bps"] = usable[outcome_benefit]

        for year in wave["test_years"]:
            start = pd.Timestamp(year=int(year), month=1, day=1)
            end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
            train = _purged_train(usable, start, task.horizon)
            test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
            if task.scope != "pooled":
                train = train.loc[train["corridor"].eq(task.scope)]
                test = test.loc[test["corridor"].eq(task.scope)]
            if (
                len(train) < int(wave["minimum_training_observations"])
                or len(test) < 20
                or train["target"].nunique() < 2
            ):
                continue

            feature_columns = list(base_columns)
            if task.scope == "pooled":
                train, test, feature_columns = _add_corridor_dummies(
                    train, test, feature_columns, list(config["corridors"])
                )
            if task.feature_set == OUTLIER_FEATURES:
                train, test, feature_columns = _outlier_features(
                    train, test, feature_columns, seed=int(wave["random_seed"]) + int(year)
                )
            model = _make_model(
                task.model,
                iterations=int(wave["model_iterations"]),
                seed=int(wave["random_seed"]) + int(year),
            )
            model.fit(train[feature_columns], train["target"])
            train["score"] = model.predict_proba(train[feature_columns])[:, 1]
            test["score"] = model.predict_proba(test[feature_columns])[:, 1]
            test["week"] = test["timestamp"].dt.to_period("W").astype(str)
            weekly_base = test.groupby(["corridor", "week"], sort=False)["target"].mean()
            test["matched_week_hit_rate"] = [
                float(weekly_base.loc[(corridor, week)]) for corridor, week in zip(test["corridor"], test["week"], strict=True)
            ]

            score_export = test[
                ["timestamp", "corridor", "regime", "price", "target", "regret_bps", "benefit_bps", "score", "week"]
            ].copy()
            for key, value in task.__dict__.items():
                score_export[key] = value
            score_export["test_year"] = int(year)
            scores.append(score_export)

            for share in config["selective_prediction"]["top_score_share"]:
                eligible_parts: list[pd.DataFrame] = []
                for corridor, test_corridor in test.groupby("corridor", sort=False):
                    train_scores = train.loc[train["corridor"].eq(corridor), "score"]
                    if train_scores.empty:
                        continue
                    threshold = float(np.quantile(train_scores, 1 - float(share)))
                    part = test_corridor.loc[test_corridor["score"] >= threshold].copy()
                    part["score_threshold"] = threshold
                    eligible_parts.append(part)
                eligible = pd.concat(eligible_parts, ignore_index=False) if eligible_parts else test.iloc[0:0].copy()
                for cooldown in config["selective_prediction"]["cooldown_days"]:
                    for value_gate in (False, True):
                        candidates = eligible
                        if value_gate:
                            candidates = candidates.loc[
                                candidates["base__percentile_60"] <= float(wave["value_percentile_max"])
                            ]
                        selected_parts = [
                            _apply_policy(
                                group,
                                cooldown_days=int(cooldown),
                                weekly_cap=int(config["selective_prediction"]["weekly_cap"]),
                            )
                            for _, group in candidates.groupby("corridor", sort=False)
                        ]
                        selected = (
                            pd.concat(selected_parts, ignore_index=False)
                            if selected_parts
                            else candidates.iloc[0:0].copy()
                        )
                        folds.append(
                            _fold_metrics(
                                test,
                                selected,
                                task=task,
                                year=int(year),
                                share=float(share),
                                cooldown=int(cooldown),
                                value_gate=value_gate,
                            )
                        )
                        if not selected.empty:
                            exported = selected[
                                [
                                    "timestamp",
                                    "corridor",
                                    "regime",
                                    "price",
                                    "target",
                                    "regret_bps",
                                    "benefit_bps",
                                    "score",
                                    "score_threshold",
                                    "matched_week_hit_rate",
                                ]
                            ].copy()
                            for key, value in task.__dict__.items():
                                exported[key] = value
                            exported["test_year"] = int(year)
                            exported["top_score_share"] = float(share)
                            exported["cooldown_days"] = int(cooldown)
                            exported["value_gate"] = value_gate
                            signals.append(exported)

            for feature, importance, signed in _feature_importance(model, feature_columns):
                importances.append(
                    {
                        **task.__dict__,
                        "test_year": int(year),
                        "feature": feature,
                        "importance": importance,
                        "signed_effect": signed,
                    }
                )

        if task_index % 25 == 0 or task_index == len(tasks):
            print(f"completed {task_index}/{len(tasks)} tasks", flush=True)

    return (
        pd.DataFrame(folds),
        pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
        pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(),
        pd.DataFrame(importances),
    )


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in folds.groupby(list(SUMMARY_KEYS), sort=True, dropna=False):
        values = dict(zip(SUMMARY_KEYS, keys, strict=True))
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        matched_hits = float(group["matched_expected_hits"].sum())
        hit_rate = hits / signals if signals else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = matched_hits / signals if signals else np.nan
        valid_fold_lifts = group["lift"].dropna()
        rows.append(
            {
                **values,
                "folds": len(group),
                "signals": signals,
                "hits": hits,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if signals and base_rate > 0 else np.nan,
                "lift_vs_matched_random": hit_rate / matched_rate if signals and matched_rate > 0 else np.nan,
                "regret_mean_bps": float(group["regret_sum_bps"].sum()) / signals if signals else np.nan,
                "benefit_mean_bps": float(group["benefit_sum_bps"].sum()) / signals if signals else np.nan,
                "test_duration_weeks": float(group["duration_weeks"].sum()),
                "signals_per_week": signals / float(group["duration_weeks"].sum()) if signals else 0.0,
                "worst_fold_lift": float(valid_fold_lifts.min()) if len(valid_fold_lifts) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def corridor_stability(signals: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or scores.empty:
        return pd.DataFrame()
    keys = [*SUMMARY_KEYS, "corridor"]
    base = (
        scores.groupby(["scope", "model", "feature_set", "horizon", "tolerance_bps", "corridor"], sort=False)
        .agg(base_count=("target", "size"), base_hits=("target", "sum"))
        .reset_index()
    )
    selected = (
        signals.groupby(keys, sort=False)
        .agg(signals=("target", "size"), hits=("target", "sum"))
        .reset_index()
    )
    result = selected.merge(
        base,
        on=["scope", "model", "feature_set", "horizon", "tolerance_bps", "corridor"],
        how="left",
        validate="many_to_one",
    )
    result["hit_rate"] = result["hits"] / result["signals"]
    result["baseline_hit_rate"] = result["base_hits"] / result["base_count"]
    result["lift"] = result["hit_rate"] / result["baseline_hit_rate"]
    return result


def best_by_horizon(summary: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    eligible = summary.loc[
        summary["signals"].ge(50)
        & summary["lift"].notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=["horizon", "best_model", "signals", "lift", "signals_per_week", "corridors_lift_ge_1_3"]
        )
    # Do not count a corridor as stable on a handful of lucky dates. Twenty is
    # still only a screening floor; final confirmation needs uncertainty bounds.
    stability_count = (
        stability.assign(pass_lift=stability["lift"].ge(1.3) & stability["signals"].ge(20))
        .groupby(list(SUMMARY_KEYS), sort=False)["pass_lift"]
        .sum()
        .rename("corridors_lift_ge_1_3")
        .reset_index()
    )
    eligible = eligible.merge(stability_count, on=list(SUMMARY_KEYS), how="left")
    eligible["frequency_requirement_met"] = eligible["signals_per_week"].between(1.0, 2.0, inclusive="both")
    chosen: list[pd.DataFrame] = []
    for _, group in eligible.groupby("horizon", sort=True):
        frequency_ok = group.loc[group["frequency_requirement_met"]]
        pool = frequency_ok if not frequency_ok.empty else group
        chosen.append(
            pool.sort_values(
                ["lift", "lift_vs_matched_random", "signals"],
                ascending=[False, False, False],
                kind="mergesort",
            ).head(1)
        )
    best = pd.concat(chosen, ignore_index=True)
    best["best_model"] = (
        best["scope"].astype(str)
        + "/"
        + best["model"].astype(str)
        + "/"
        + best["feature_set"].astype(str)
        + "/top"
        + (best["top_score_share"] * 100).round().astype(int).astype(str)
        + "%/cd"
        + best["cooldown_days"].astype(str)
        + best["value_gate"].map({True: "/value", False: ""})
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
    ].reset_index(drop=True)


def run(
    *,
    config_path: Path | str = Path("configs/next_hypotheses.json"),
    cbr_path: Path | str = Path("data/raw/cbr_daily.csv"),
    research_panel_path: Path | str = Path("data/processed/research_daily_panel.csv"),
    processed_path: Path | str = Path("data/processed/cbr_corridor_features.csv"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/wave_1"),
) -> dict[str, Any]:
    config = load_config(config_path)
    wave = config["wave_1"]
    datasets = build_datasets(
        cbr_path=Path(cbr_path),
        research_panel_path=Path(research_panel_path),
        corridors=list(config["corridors"]),
        horizons=[int(value) for value in config["regret"]["daily_horizons"]],
        market_lag=int(wave["market_factor_lag_observations"]),
    )
    processed = Path(processed_path)
    processed.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(datasets.values(), ignore_index=True).to_csv(processed, index=False)

    folds, scores, signals, importance = evaluate(datasets, config)
    summary = summarize(folds)
    stability = corridor_stability(signals, scores)
    best = best_by_horizon(summary, stability)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    folds.to_csv(artifacts / "folds.csv", index=False)
    scores.to_csv(artifacts / "scores.csv", index=False)
    signals.to_csv(artifacts / "signals.csv", index=False)
    importance.to_csv(artifacts / "feature_importance.csv", index=False)
    summary.to_csv(artifacts / "summary.csv", index=False)
    stability.to_csv(artifacts / "corridor_stability.csv", index=False)
    best.to_csv(artifacts / "best_by_horizon.csv", index=False)
    metadata = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "purpose": wave["purpose"],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "cbr_sha256": hashlib.sha256(Path(cbr_path).read_bytes()).hexdigest(),
        "research_panel_sha256": hashlib.sha256(Path(research_panel_path).read_bytes()).hexdigest(),
        "processed_path": str(processed),
        "tasks": len(_task_plan(config)),
        "fold_rows": len(folds),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "warning": "Exploratory multi-hypothesis result; requires untouched confirmation.",
    }
    (artifacts / "run_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_hypotheses.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/wave_1"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
