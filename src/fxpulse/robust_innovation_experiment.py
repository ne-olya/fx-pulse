"""Preregistered robustness audit and fixed innovation wave for FX Pulse.

The module deliberately separates exploratory model generation from validation:

* every score is produced by a purged expanding-window fit;
* the dispatch threshold sees only earlier out-of-sample scores;
* nested selection chooses a variant using years strictly before its test year;
* the previously reported winners are replayed with five fresh random seeds;
* cluster bootstrap and circular-shift placebos challenge the selected dates.

The 2025-2026 split is called a *pseudo*-holdout because the researchers had
already inspected those years before this file was written.  It is useful as a
stress test but is not represented as a genuinely untouched final test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.brent_experiment import add_brent_features
from fxpulse.next_hypotheses import _apply_policy, _make_model, _purged_train
from fxpulse.path_label_experiment import (
    evaluate as evaluate_path_label,
    triple_barrier_label,
)
from fxpulse.temporal_sequence_experiment import (
    add_sequence_features,
    evaluate as evaluate_temporal_sequence,
)
from fxpulse.training_history_experiment import evaluate as evaluate_training_history


EXPECTED_VARIANTS = [
    "tb_catboost",
    "direct_regret_catboost",
    "sequence_catboost",
    "seed_mean",
    "seed_lcb",
    "cross_model_mean",
    "dynamic_model_average",
    "multi_barrier_mean",
    "vol_scaled_barrier",
    "rolling_4y_catboost",
    "regime_experts",
    "analog_knn",
    "horizon_consensus",
]

SUMMARY_KEYS = ("variant", "corridor", "horizon")


def load_config(
    path: Path | str = Path("configs/robust_innovation_experiment.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("robust innovation config must be preregistered schema_version 1")
    if config.get("variants") != EXPECTED_VARIANTS:
        raise ValueError("registered variants differ from the implementation")
    if set(config.get("corridors", ())) != {"AMD", "KGS", "KZT", "TJS", "UZS"}:
        raise ValueError("all five case corridors are required")
    if config.get("horizons") != [3, 5, 10]:
        raise ValueError("registered horizons must be [3, 5, 10]")
    if not set(config["development_years"]).isdisjoint(config["pseudo_holdout_years"]):
        raise ValueError("development and pseudo-holdout years must not overlap")
    return config


def volatility_scaled_triple_barrier_label(
    price: pd.Series,
    volatility: pd.Series,
    *,
    horizon: int,
    better_sigma_fraction: float,
    worse_sigma_fraction: float,
    better_bps_bounds: tuple[float, float],
    worse_bps_bounds: tuple[float, float],
) -> pd.Series:
    """Path label whose barriers depend only on volatility known at time t."""

    values = price.to_numpy(dtype=float)
    vol = volatility.to_numpy(dtype=float)
    labels = np.full(len(values), np.nan)
    scale = math.sqrt(horizon) * 10_000
    for index in range(len(values) - horizon):
        if not np.isfinite(values[index]) or not np.isfinite(vol[index]):
            continue
        better = float(
            np.clip(
                vol[index] * scale * better_sigma_fraction,
                better_bps_bounds[0],
                better_bps_bounds[1],
            )
        )
        worse = float(
            np.clip(
                vol[index] * scale * worse_sigma_fraction,
                worse_bps_bounds[0],
                worse_bps_bounds[1],
            )
        )
        label = 1.0
        for future in values[index + 1 : index + horizon + 1]:
            improvement = (values[index] / future - 1) * 10_000
            worsening = (future / values[index] - 1) * 10_000
            if improvement >= better:
                label = 0.0
                break
            if worsening >= worse:
                label = 1.0
                break
        labels[index] = label
    return pd.Series(labels, index=price.index)


def add_registered_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.sort_values(["corridor", "timestamp"], kind="mergesort")
    vol_cfg = config["volatility_barrier"]
    pieces: list[pd.DataFrame] = []
    for _, group in data.groupby("corridor", sort=False):
        group = group.copy()
        for horizon in config["horizons"]:
            h = int(horizon)
            barrier = config["triple_barrier"]
            group[f"label__tb_{h}"] = triple_barrier_label(
                group["price"],
                horizon=h,
                better_price_barrier_bps=float(barrier["better_price_barrier_bps"]),
                worse_price_barrier_bps=float(barrier["worse_price_barrier_bps"]),
            )
            group[f"label__vol_tb_{h}"] = volatility_scaled_triple_barrier_label(
                group["price"],
                group[str(vol_cfg["volatility_column"])],
                horizon=h,
                better_sigma_fraction=float(vol_cfg["better_sigma_fraction"]),
                worse_sigma_fraction=float(vol_cfg["worse_sigma_fraction"]),
                better_bps_bounds=tuple(map(float, vol_cfg["better_bps_bounds"])),
                worse_bps_bounds=tuple(map(float, vol_cfg["worse_bps_bounds"])),
            )
            for better, worse in config["multi_barriers_bps"]:
                group[f"label__tb_{h}_{int(better)}_{int(worse)}"] = triple_barrier_label(
                    group["price"],
                    horizon=h,
                    better_price_barrier_bps=float(better),
                    worse_price_barrier_bps=float(worse),
                )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def prepare_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    sequence = config["sequence"]
    sequenced = add_sequence_features(
        frame,
        list(sequence["source_columns"]),
        [int(value) for value in sequence["lags"]],
    )
    return add_registered_labels(sequenced, config)


def _feature_columns(frame: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    return [column for column in frame if column.startswith(tuple(prefixes))]


def _fit_probability(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target_column: str,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    usable = train.loc[train[target_column].notna()].copy()
    usable[target_column] = usable[target_column].astype(int)
    if len(usable) < 100 or usable[target_column].nunique() < 2:
        return np.full(len(test), np.nan)
    model = _make_model(model_name, iterations=iterations, seed=seed)
    model.fit(usable[columns], usable[target_column])
    return np.asarray(model.predict_proba(test[columns])[:, 1], dtype=float)


def _analog_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    columns: list[str],
    target_column: str,
    neighbors: int,
) -> np.ndarray:
    usable = train.loc[train[target_column].notna()].copy()
    if len(usable) < max(100, neighbors):
        return np.full(len(test), np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(usable[columns]))
    test_values = scaler.transform(imputer.transform(test[columns]))
    count = min(int(neighbors), len(usable))
    finder = NearestNeighbors(n_neighbors=count, metric="euclidean", n_jobs=1)
    finder.fit(train_values)
    distances, indices = finder.kneighbors(test_values)
    weights = 1.0 / np.maximum(distances, 1e-6)
    labels = usable[target_column].to_numpy(dtype=float)[indices]
    return (labels * weights).sum(axis=1) / weights.sum(axis=1)


def delayed_dynamic_average(
    predictions: np.ndarray,
    target: np.ndarray,
    *,
    delay: int,
    eta: float,
) -> np.ndarray:
    """Causal exponential weighting; target[t] becomes visible at t + delay."""

    if predictions.ndim != 2 or predictions.shape[0] != len(target):
        raise ValueError("predictions must be n_observations x n_models")
    clipped = np.clip(predictions, 1e-6, 1 - 1e-6)
    weights = np.ones(clipped.shape[1], dtype=float) / clipped.shape[1]
    result = np.full(len(target), np.nan)
    for index in range(len(target)):
        matured = index - int(delay)
        if matured >= 0 and np.isfinite(target[matured]):
            y = float(target[matured])
            loss = -(y * np.log(clipped[matured]) + (1 - y) * np.log(1 - clipped[matured]))
            weights *= np.exp(-float(eta) * loss)
            total = float(weights.sum())
            weights = weights / total if total > 0 else np.ones_like(weights) / len(weights)
        result[index] = float(np.dot(weights, clipped[index]))
    return result


def _regime_expert_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    target_column: str,
    fallback: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> np.ndarray:
    rank_column = "regime__volatility_percentile_252"
    train_bucket = pd.cut(
        train[rank_column], [-np.inf, 1 / 3, 2 / 3, np.inf], labels=["low", "mid", "high"]
    )
    test_bucket = pd.cut(
        test[rank_column], [-np.inf, 1 / 3, 2 / 3, np.inf], labels=["low", "mid", "high"]
    )
    result = np.asarray(fallback, dtype=float).copy()
    for bucket_index, bucket in enumerate(("low", "mid", "high")):
        train_part = train.loc[train_bucket.eq(bucket)].copy()
        test_mask = test_bucket.eq(bucket).to_numpy()
        if test_mask.sum() == 0:
            continue
        predicted = _fit_probability(
            "catboost",
            train_part,
            test.loc[test_mask],
            columns,
            target_column,
            iterations=iterations,
            seed=seed + bucket_index,
        )
        if np.isfinite(predicted).all():
            result[test_mask] = predicted
    return result


def _variant_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    base_columns: list[str],
    sequence_columns: list[str],
    config: dict[str, Any],
    fold_seed: int,
    start: pd.Timestamp,
) -> dict[str, np.ndarray]:
    iterations = int(config["model_iterations"])
    tb_target = f"label__tb_{horizon}"
    direct_target = f"target__direct_{horizon}"
    seed_predictions = np.column_stack(
        [
            _fit_probability(
                "catboost",
                train,
                test,
                base_columns,
                tb_target,
                iterations=iterations,
                seed=int(seed) + start.year,
            )
            for seed in config["ensemble_seeds"]
        ]
    )
    catboost_score = seed_predictions[:, 0]
    xgboost_score = _fit_probability(
        "xgboost",
        train,
        test,
        base_columns,
        tb_target,
        iterations=iterations,
        seed=fold_seed + 101,
    )
    logistic_score = _fit_probability(
        "logistic",
        train,
        test,
        base_columns,
        tb_target,
        iterations=iterations,
        seed=fold_seed + 211,
    )
    direct_score = _fit_probability(
        "catboost",
        train,
        test,
        base_columns,
        direct_target,
        iterations=iterations,
        seed=fold_seed + 307,
    )
    sequence_score = _fit_probability(
        "catboost",
        train,
        test,
        [*base_columns, *sequence_columns],
        tb_target,
        iterations=iterations,
        seed=fold_seed + 401,
    )
    barrier_scores = []
    for barrier_index, (better, worse) in enumerate(config["multi_barriers_bps"]):
        barrier_scores.append(
            _fit_probability(
                "catboost",
                train,
                test,
                base_columns,
                f"label__tb_{horizon}_{int(better)}_{int(worse)}",
                iterations=iterations,
                seed=fold_seed + 503 + barrier_index,
            )
        )
    vol_score = _fit_probability(
        "catboost",
        train,
        test,
        base_columns,
        f"label__vol_tb_{horizon}",
        iterations=iterations,
        seed=fold_seed + 601,
    )
    rolling_start = start - pd.DateOffset(years=int(config["rolling_training_years"]))
    rolling_train = train.loc[train["timestamp"].ge(rolling_start)].copy()
    rolling_score = _fit_probability(
        "catboost",
        rolling_train,
        test,
        base_columns,
        tb_target,
        iterations=iterations,
        seed=fold_seed + 701,
    )
    regime_score = _regime_expert_scores(
        train,
        test,
        base_columns,
        tb_target,
        catboost_score,
        iterations=iterations,
        seed=fold_seed + 809,
    )
    analog_cfg = config["analog_knn"]
    analog_score = _analog_scores(
        train,
        test,
        columns=list(analog_cfg["feature_columns"]),
        target_column=direct_target,
        neighbors=int(analog_cfg["neighbors"]),
    )
    horizon_scores = []
    for sub_horizon in [value for value in config["horizons"] if int(value) <= horizon]:
        sub_horizon = int(sub_horizon)
        if sub_horizon == horizon:
            horizon_scores.append(direct_score)
        else:
            horizon_scores.append(
                _fit_probability(
                    "catboost",
                    train,
                    test,
                    base_columns,
                    f"target__direct_{sub_horizon}",
                    iterations=iterations,
                    seed=fold_seed + 907 + sub_horizon,
                )
            )
    model_predictions = np.column_stack([catboost_score, xgboost_score, logistic_score])
    return {
        "tb_catboost": catboost_score,
        "direct_regret_catboost": direct_score,
        "sequence_catboost": sequence_score,
        "seed_mean": np.nanmean(seed_predictions, axis=1),
        "seed_lcb": np.nanmean(seed_predictions, axis=1)
        - float(config["seed_lcb_penalty"]) * np.nanstd(seed_predictions, axis=1),
        "cross_model_mean": np.nanmean(model_predictions, axis=1),
        "dynamic_model_average": delayed_dynamic_average(
            model_predictions,
            test["target"].to_numpy(dtype=float),
            delay=horizon,
            eta=float(config["dynamic_model_average"]["eta"]),
        ),
        "multi_barrier_mean": np.nanmean(np.column_stack(barrier_scores), axis=1),
        "vol_scaled_barrier": vol_score,
        "rolling_4y_catboost": rolling_score,
        "regime_experts": regime_score,
        "analog_knn": analog_score,
        "horizon_consensus": np.nanmin(np.column_stack(horizon_scores), axis=1),
    }


def _selected_signals(test: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    candidates = adaptive_candidates(
        test,
        share=float(policy["top_score_share"]),
        lookback=int(policy["lookback_observations"]),
        minimum_history=int(policy["minimum_history_observations"]),
    )
    return _apply_policy(
        candidates,
        cooldown_days=int(policy["cooldown_days"]),
        weekly_cap=int(policy["weekly_cap"]),
    )


def evaluate_innovations(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = prepare_frame(frame, config)
    base_columns = _feature_columns(data, config["feature_prefixes"])
    sequence_columns = _feature_columns(data, ["sequence__"])
    folds: list[dict[str, object]] = []
    scores: list[pd.DataFrame] = []
    signals: list[pd.DataFrame] = []

    for corridor in config["corridors"]:
        corridor_data = data.loc[data["corridor"].eq(corridor)].copy()
        for horizon in config["horizons"]:
            h = int(horizon)
            outcome = f"outcome__regret_{h}"
            benefit = f"outcome__benefit_{h}"
            usable = corridor_data.loc[corridor_data[outcome].notna()].copy()
            for target_horizon in config["horizons"]:
                target_outcome = f"outcome__regret_{int(target_horizon)}"
                usable[f"target__direct_{int(target_horizon)}"] = (
                    usable[target_outcome].le(float(config["evaluation_tolerance_bps"]))
                    .where(usable[target_outcome].notna())
                )
            usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
            usable["regret_bps"] = usable[outcome]
            usable["benefit_bps"] = usable[benefit]
            for year in config["test_years"]:
                start = pd.Timestamp(year=int(year), month=1, day=1)
                end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                train = _purged_train(usable, start, h)
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if len(train) < int(config["minimum_training_observations"]) or len(test) < 20:
                    continue
                test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                weekly_rate = test.groupby("week", sort=False)["target"].mean()
                test["matched_week_hit_rate"] = test["week"].map(weekly_rate).astype(float)
                variant_scores = _variant_scores(
                    train,
                    test,
                    horizon=h,
                    base_columns=base_columns,
                    sequence_columns=sequence_columns,
                    config=config,
                    fold_seed=int(config["random_seed"]) + int(year) + h,
                    start=start,
                )
                missing_variants = set(config["variants"]) - set(variant_scores)
                if missing_variants:
                    raise RuntimeError(f"missing registered scores: {sorted(missing_variants)}")
                duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7)
                for variant in config["variants"]:
                    scored = test.copy()
                    scored["score"] = variant_scores[variant]
                    if not np.isfinite(scored["score"]).all():
                        continue
                    selected = _selected_signals(scored, config["adaptive_policy"])
                    count = len(selected)
                    hit_rate = float(selected["target"].mean()) if count else np.nan
                    base_rate = float(scored["target"].mean())
                    folds.append(
                        {
                            "variant": variant,
                            "corridor": corridor,
                            "horizon": h,
                            "test_year": int(year),
                            "test_count": len(scored),
                            "test_hits": int(scored["target"].sum()),
                            "signal_count": count,
                            "signal_hits": int(selected["target"].sum()) if count else 0,
                            "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                            "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()) if count else 0.0,
                            "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                            "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                            "duration_weeks": duration,
                        }
                    )
                    score_export = scored[
                        ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "score", "matched_week_hit_rate"]
                    ].copy()
                    score_export["variant"] = variant
                    score_export["horizon"] = h
                    score_export["test_year"] = int(year)
                    scores.append(score_export)
                    if count:
                        signal_export = selected[
                            ["timestamp", "corridor", "week", "target", "regret_bps", "benefit_bps", "score", "score_threshold", "matched_week_hit_rate"]
                        ].copy()
                        signal_export["variant"] = variant
                        signal_export["horizon"] = h
                        signal_export["test_year"] = int(year)
                        signals.append(signal_export)
            print(f"completed {corridor} h={h}", flush=True)
    return (
        pd.DataFrame(folds),
        pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(),
        pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(),
    )


def summarize_folds(folds: pd.DataFrame, *, years: Iterable[int] | None = None) -> pd.DataFrame:
    source = folds if years is None else folds.loc[folds["test_year"].isin(list(years))]
    rows: list[dict[str, object]] = []
    for keys, group in source.groupby(list(SUMMARY_KEYS), sort=True):
        identity = dict(zip(SUMMARY_KEYS, keys, strict=True))
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        matched_hits = float(group["matched_expected_hits"].sum())
        hit_rate = hits / signals if signals else np.nan
        base_rate = base_hits / base_count if base_count else np.nan
        matched_rate = matched_hits / signals if signals else np.nan
        duration = float(group["duration_weeks"].sum())
        valid_lifts = group["lift"].dropna()
        rows.append(
            {
                **identity,
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
                "signals_per_week": signals / duration if duration else 0.0,
                "worst_year_lift": float(valid_lifts.min()) if len(valid_lifts) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def nested_year_selection(folds: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Choose on earlier OOT years and evaluate once on the next year."""

    rows: list[dict[str, object]] = []
    acceptance = config["acceptance"]
    for corridor in config["corridors"]:
        for horizon in config["horizons"]:
            scope = folds.loc[
                folds["corridor"].eq(corridor) & folds["horizon"].eq(int(horizon))
            ].copy()
            for test_year in config["nested_selection_years"]:
                past = scope.loc[scope["test_year"] < int(test_year)]
                if past["test_year"].nunique() < 2:
                    continue
                candidates = summarize_folds(past)
                candidates = candidates.loc[
                    candidates["signals_per_week"].between(
                        float(acceptance["minimum_signals_per_week"]),
                        float(acceptance["maximum_signals_per_week"]),
                        inclusive="both",
                    )
                ]
                if candidates.empty:
                    candidates = summarize_folds(past)
                chosen = candidates.sort_values(
                    ["lift_vs_matched_random", "lift", "variant"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).iloc[0]
                test = scope.loc[
                    scope["test_year"].eq(int(test_year)) & scope["variant"].eq(chosen["variant"])
                ]
                if test.empty:
                    continue
                result = test.iloc[0]
                count = int(result["signal_count"])
                matched_lift = (
                    float(result["signal_hits"]) / float(result["matched_expected_hits"])
                    if float(result["matched_expected_hits"]) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "corridor": corridor,
                        "horizon": int(horizon),
                        "test_year": int(test_year),
                        "selected_variant": str(chosen["variant"]),
                        "selection_years": ",".join(map(str, sorted(past["test_year"].unique()))),
                        "selection_lift": float(chosen["lift"]),
                        "selection_matched_lift": float(chosen["lift_vs_matched_random"]),
                        "signal_count": count,
                        "signal_hits": int(result["signal_hits"]),
                        "test_count": int(result["test_count"]),
                        "test_hits": int(result["test_hits"]),
                        "lift": float(result["lift"]),
                        "lift_vs_matched_random": matched_lift,
                        "matched_expected_hits": float(result["matched_expected_hits"]),
                        "duration_weeks": float(result["duration_weeks"]),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_metric_rows(frame: pd.DataFrame) -> dict[str, float | int]:
    signals = int(frame["signal_count"].sum())
    hits = int(frame["signal_hits"].sum())
    base_count = int(frame["test_count"].sum())
    base_hits = int(frame["test_hits"].sum())
    matched = float(frame["matched_expected_hits"].sum())
    duration = float(frame["duration_weeks"].sum()) if "duration_weeks" in frame else np.nan
    return {
        "folds": len(frame),
        "signals": signals,
        "hits": hits,
        "lift": (hits / signals) / (base_hits / base_count)
        if signals and base_count and base_hits
        else np.nan,
        "lift_vs_matched_random": hits / matched if matched > 0 else np.nan,
        "signals_per_week": signals / duration if duration and np.isfinite(duration) else np.nan,
        "worst_year_lift": float(frame["lift"].dropna().min())
        if frame["lift"].notna().any()
        else np.nan,
    }


def summarize_nested_selection(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, group in nested.groupby("horizon", sort=True):
        rows.append({"horizon": int(horizon), **_aggregate_metric_rows(group)})
    return pd.DataFrame(rows)


def innovation_gate(
    full: pd.DataFrame,
    pseudo_holdout: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    holdout = pseudo_holdout.rename(
        columns={
            "signals": "holdout_signals",
            "lift": "holdout_lift",
            "lift_vs_matched_random": "holdout_lift_vs_matched_random",
            "worst_year_lift": "holdout_worst_year_lift",
        }
    )
    keep = [
        *SUMMARY_KEYS,
        "holdout_signals",
        "holdout_lift",
        "holdout_lift_vs_matched_random",
        "holdout_worst_year_lift",
    ]
    result = full.merge(holdout[keep], on=list(SUMMARY_KEYS), how="left", validate="one_to_one")
    gate = config["acceptance"]
    result["enough_signals"] = result["signals"].ge(int(gate["minimum_total_signals"]))
    result["frequency_ok"] = result["signals_per_week"].between(
        float(gate["minimum_signals_per_week"]),
        float(gate["maximum_signals_per_week"]),
        inclusive="both",
    )
    result["raw_lift_ok"] = result["lift"].ge(float(gate["minimum_raw_lift"]))
    result["matched_lift_ok"] = result["lift_vs_matched_random"].ge(
        float(gate["minimum_matched_week_lift"])
    )
    result["holdout_ok"] = result["holdout_lift"].ge(
        float(gate["minimum_pseudo_holdout_raw_lift"])
    )
    result["worst_year_ok"] = result["worst_year_lift"].ge(
        float(gate["minimum_worst_year_lift"])
    )
    checks = [
        "enough_signals",
        "frequency_ok",
        "raw_lift_ok",
        "matched_lift_ok",
        "holdout_ok",
        "worst_year_ok",
    ]
    result["passes_all_registered_gates"] = result[checks].all(axis=1)
    return result.sort_values(
        ["passes_all_registered_gates", "horizon", "lift_vs_matched_random", "lift"],
        ascending=[False, True, False, False],
        kind="mergesort",
    )


def best_innovation_by_horizon(gated: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    for _, group in gated.groupby("horizon", sort=True):
        passed = group.loc[group["passes_all_registered_gates"]]
        pool = passed if not passed.empty else group.loc[group["frequency_ok"] & group["enough_signals"]]
        if pool.empty:
            pool = group
        chosen.append(
            pool.sort_values(
                ["lift_vs_matched_random", "lift", "variant", "corridor"],
                ascending=[False, False, True, True],
                kind="mergesort",
            ).head(1)
        )
    return pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()


def corridor_replication(gated: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gate = config["acceptance"]
    rows: list[dict[str, object]] = []
    for (variant, horizon), group in gated.groupby(["variant", "horizon"], sort=True):
        replicated = group.loc[
            group["signals"].ge(30)
            & group["lift"].ge(float(gate["minimum_raw_lift"]))
            & group["lift_vs_matched_random"].ge(float(gate["minimum_matched_week_lift"]))
        ]
        rows.append(
            {
                "variant": variant,
                "horizon": int(horizon),
                "corridors_raw_and_matched_pass": len(replicated),
                "passing_corridors": ",".join(sorted(replicated["corridor"].astype(str))),
                "median_raw_lift": float(group["lift"].median()),
                "median_matched_lift": float(group["lift_vs_matched_random"].median()),
                "minimum_raw_lift": float(group["lift"].min()),
            }
        )
    return pd.DataFrame(rows)


def _winner_artifact_paths(family: str) -> tuple[Path, Path]:
    root = Path("artifacts/next_hypotheses") / family
    return root / "folds.csv", root / "signals.csv"


def _load_winner_rows(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_path, signal_path = _winner_artifact_paths(str(spec["artifact_family"]))
    folds = pd.read_csv(fold_path)
    signals = pd.read_csv(signal_path)
    filters = {
        "scope": spec["corridor"],
        "model": spec["model"],
        "feature_set": spec["feature_set"],
        "horizon": int(spec["horizon"]),
    }
    for column, value in filters.items():
        folds = folds.loc[folds[column].eq(value)]
        signals = signals.loc[signals[column].eq(value)]
    if len(folds) != 5 or signals.empty:
        raise ValueError(f"saved winner {spec['name']} does not resolve uniquely")
    signals = signals.copy()
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="raise")
    signals["week"] = signals["timestamp"].dt.to_period("W").astype(str)
    return folds.reset_index(drop=True), signals.reset_index(drop=True)


def _target_frame_for_winner(spec: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    input_path = Path(config["input"] if spec["artifact_family"] == "training_history" else config["short_input"])
    outcome = "outcome__regret_" + str(int(spec["horizon"]))
    frame = pd.read_csv(input_path, usecols=["timestamp", "corridor", outcome])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.loc[
        frame["corridor"].eq(spec["corridor"])
        & frame[outcome].notna()
        & frame["timestamp"].dt.year.isin(config["test_years"])
    ].copy()
    frame["target"] = frame[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
    frame["test_year"] = frame["timestamp"].dt.year
    frame["week"] = frame["timestamp"].dt.to_period("W").astype(str)
    return frame[["timestamp", "test_year", "week", "target"]].sort_values("timestamp")


def _weekly_winner_table(
    target: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    selected = signals.groupby(["test_year", "week"], sort=True).agg(
        signal_count=("target", "size"),
        signal_hits=("target", "sum"),
        matched_expected_hits=("matched_week_hit_rate", "sum"),
    )
    base = target.groupby(["test_year", "week"], sort=True).agg(
        base_count=("target", "size"), base_hits=("target", "sum")
    )
    weekly = base.join(selected, how="left").fillna(
        {"signal_count": 0, "signal_hits": 0, "matched_expected_hits": 0.0}
    )
    return weekly.reset_index()


def cluster_bootstrap(
    weekly: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    columns = weekly[
        ["signal_count", "signal_hits", "matched_expected_hits", "base_count", "base_hits"]
    ].to_numpy(dtype=float)
    n = len(columns)
    indices = rng.integers(0, n, size=(int(samples), n))
    sampled = columns[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (sampled[:, 1] / sampled[:, 0]) / (sampled[:, 4] / sampled[:, 3])
        matched = sampled[:, 1] / sampled[:, 2]
    observed_hits = float(columns[:, 1].sum())
    observed_expected = float(columns[:, 2].sum())
    delta = columns[:, 1] - columns[:, 2]
    observed_delta = float(delta.mean())
    centered = delta - delta.mean()
    null_means = centered[indices].mean(axis=1)
    p_value = (1 + int(np.sum(null_means >= observed_delta))) / (int(samples) + 1)
    return {
        "bootstrap_raw_lift_low": float(np.nanquantile(raw, 0.025)),
        "bootstrap_raw_lift_high": float(np.nanquantile(raw, 0.975)),
        "bootstrap_matched_lift_low": float(np.nanquantile(matched, 0.025)),
        "bootstrap_matched_lift_high": float(np.nanquantile(matched, 0.975)),
        "bootstrap_matched_p_value": p_value,
        "observed_matched_lift": observed_hits / observed_expected if observed_expected else np.nan,
    }


def circular_shift_placebo(
    target: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    yearly: list[tuple[np.ndarray, np.ndarray]] = []
    for year, year_target in target.groupby("test_year", sort=True):
        ordered = year_target.sort_values("timestamp").reset_index(drop=True)
        lookup = pd.Series(np.arange(len(ordered)), index=ordered["timestamp"])
        selected_dates = signals.loc[signals["test_year"].eq(year), "timestamp"]
        selected_positions = lookup.reindex(selected_dates).dropna().astype(int).to_numpy()
        if len(selected_positions):
            yearly.append((ordered["target"].to_numpy(dtype=int), selected_positions))
    base_rate = float(target["target"].mean())
    observed_lift = float(signals["target"].mean()) / base_rate
    placebo = np.empty(int(samples), dtype=float)
    for draw in range(int(samples)):
        hits = 0
        count = 0
        for labels, positions in yearly:
            minimum_shift = min(max(15, len(labels) // 10), len(labels) - 1)
            if minimum_shift >= len(labels) - minimum_shift:
                shift = int(rng.integers(1, len(labels)))
            else:
                shift = int(rng.integers(minimum_shift, len(labels) - minimum_shift))
            shifted = np.roll(labels, shift)
            hits += int(shifted[positions].sum())
            count += len(positions)
        placebo[draw] = (hits / count) / base_rate if count else np.nan
    p_value = (1 + int(np.sum(placebo >= observed_lift))) / (int(samples) + 1)
    return {
        "placebo_lift_mean": float(np.nanmean(placebo)),
        "placebo_lift_95pct": float(np.nanquantile(placebo, 0.95)),
        "placebo_p_value": p_value,
    }


def _holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        corrected = min(1.0, float(values.iloc[index]) * (total - rank))
        running = max(running, corrected)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def audit_saved_winners(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(config["current_winners"]):
        folds, signals = _load_winner_rows(spec)
        target = _target_frame_for_winner(spec, config)
        merged = signals.merge(
            target[["timestamp", "target"]].rename(columns={"target": "target_rebuilt"}),
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        if not merged["target"].eq(merged["target_rebuilt"]).all():
            raise ValueError(f"saved targets disagree for {spec['name']}")
        full = _aggregate_metric_rows(folds)
        development = _aggregate_metric_rows(
            folds.loc[folds["test_year"].isin(config["development_years"])]
        )
        holdout = _aggregate_metric_rows(
            folds.loc[folds["test_year"].isin(config["pseudo_holdout_years"])]
        )
        weekly = _weekly_winner_table(target, signals)
        bootstrap = cluster_bootstrap(
            weekly,
            samples=int(config["bootstrap_samples"]),
            seed=int(config["random_seed"]) + index,
        )
        placebo = circular_shift_placebo(
            target,
            signals,
            samples=int(config["placebo_samples"]),
            seed=int(config["random_seed"]) + 100 + index,
        )
        rows.append(
            {
                "name": spec["name"],
                "corridor": spec["corridor"],
                "horizon": int(spec["horizon"]),
                "signals": full["signals"],
                "lift": full["lift"],
                "lift_vs_matched_random": full["lift_vs_matched_random"],
                "worst_year_lift": full["worst_year_lift"],
                "development_lift": development["lift"],
                "development_matched_lift": development["lift_vs_matched_random"],
                "pseudo_holdout_lift": holdout["lift"],
                "pseudo_holdout_matched_lift": holdout["lift_vs_matched_random"],
                **bootstrap,
                **placebo,
            }
        )
    result = pd.DataFrame(rows)
    result["bootstrap_matched_p_holm_3"] = _holm_adjust(result["bootstrap_matched_p_value"])
    result["placebo_p_holm_3"] = _holm_adjust(result["placebo_p_value"])
    return result


def _seed_summary_rows(
    name: str,
    seed: int,
    folds: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    periods = {
        "full": config["test_years"],
        "development": config["development_years"],
        "pseudo_holdout": config["pseudo_holdout_years"],
    }
    for period, years in periods.items():
        metrics = _aggregate_metric_rows(folds.loc[folds["test_year"].isin(years)])
        rows.append({"name": name, "seed": int(seed), "period": period, **metrics})
    return rows


def replay_current_winners(config: dict[str, Any]) -> pd.DataFrame:
    short_frame = pd.read_csv(config["short_input"])
    long_frame = pd.read_csv(config["input"])
    brent_frame = add_brent_features(
        short_frame,
        pd.read_csv(config["brent_input"]),
        lag_days=7,
    )
    rows: list[dict[str, object]] = []
    for seed in config["winner_recheck_seeds"]:
        temporal_config = json.loads(Path("configs/temporal_sequence_experiment.json").read_text())
        temporal_config.update(
            {
                "corridors": ["AMD"],
                "scopes": ["per_corridor"],
                "horizons": [3],
                "models": ["xgboost"],
                "feature_sets": ["plus_sequence"],
                "test_years": config["test_years"],
                "random_seed": int(seed),
            }
        )
        h3_folds, _, _ = evaluate_temporal_sequence(short_frame, temporal_config)
        rows.extend(_seed_summary_rows("h3_amd_xgboost_sequence", seed, h3_folds, config))

        history_config = json.loads(Path("configs/training_history_experiment.json").read_text())
        history_config.update(
            {
                "corridors": ["AMD"],
                "scopes": ["per_corridor"],
                "horizons": [5],
                "training_starts": ["2010-01-01"],
                "test_years": config["test_years"],
                "random_seed": int(seed),
            }
        )
        h5_folds, _ = evaluate_training_history(long_frame, history_config)
        rows.extend(_seed_summary_rows("h5_amd_catboost_tb_history2010", seed, h5_folds, config))

        brent_config = json.loads(Path("configs/brent_experiment.json").read_text())
        brent_config.update(
            {
                "corridors": ["TJS"],
                "scopes": ["per_corridor"],
                "horizons": [10],
                "test_years": config["test_years"],
                "random_seed": int(seed),
            }
        )
        h10_folds, _ = evaluate_path_label(brent_frame, brent_config)
        h10_folds["feature_set"] = "plus_brent"
        rows.extend(_seed_summary_rows("h10_tjs_catboost_tb_brent", seed, h10_folds, config))
        print(f"replayed current winners with seed={seed}", flush=True)
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/robust_innovation_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/robust_innovation"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    frame = pd.read_csv(input_path)
    folds, scores, signals = evaluate_innovations(frame, config)
    full = summarize_folds(folds)
    development = summarize_folds(folds, years=config["development_years"])
    pseudo_holdout = summarize_folds(folds, years=config["pseudo_holdout_years"])
    gated = innovation_gate(full, pseudo_holdout, config)
    best = best_innovation_by_horizon(gated)
    replication = corridor_replication(gated, config)
    nested = nested_year_selection(folds, config)
    nested_summary = summarize_nested_selection(nested)
    winner_audit = audit_saved_winners(config)
    seed_rechecks = replay_current_winners(config)

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "innovation_folds.csv", index=False)
    scores.to_csv(output / "innovation_scores.csv.gz", index=False, compression="gzip")
    signals.to_csv(output / "innovation_signals.csv", index=False)
    full.to_csv(output / "innovation_summary.csv", index=False)
    development.to_csv(output / "innovation_development.csv", index=False)
    pseudo_holdout.to_csv(output / "innovation_pseudo_holdout.csv", index=False)
    gated.to_csv(output / "innovation_gate.csv", index=False)
    best.to_csv(output / "innovation_best_by_horizon.csv", index=False)
    replication.to_csv(output / "corridor_replication.csv", index=False)
    nested.to_csv(output / "nested_selection.csv", index=False)
    nested_summary.to_csv(output / "nested_selection_summary.csv", index=False)
    winner_audit.to_csv(output / "winner_robustness_audit.csv", index=False)
    seed_rechecks.to_csv(output / "winner_seed_rechecks.csv", index=False)

    config_path = Path(config_path)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "short_input_sha256": hashlib.sha256(Path(config["short_input"]).read_bytes()).hexdigest(),
        "brent_input_sha256": hashlib.sha256(Path(config["brent_input"]).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "fold_rows": len(folds),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "registered_variants": len(config["variants"]),
        "registered_gates_passed": int(gated["passes_all_registered_gates"].sum()),
        "warning": (
            "2025-2026 is a pseudo-holdout, not unseen data. Nested historical selection and "
            "bootstrap reduce uncertainty but cannot replace future or bank-held-out observations."
        ),
    }
    (output / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/robust_innovation_experiment.json")
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/robust_innovation"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
