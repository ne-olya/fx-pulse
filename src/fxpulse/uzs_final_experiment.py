"""Final, preregistered RUB-to-UZS model comparison.

The experiment deliberately chooses a winner on 2022-2024 and only then
reports 2025-2026.  The latter is still called a pseudo hold-out because the
team inspected those years before this experiment was registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.innovation_followup import (
    _deoverlap_signals,
    _weekly_winner_table,
    causal_percentile,
    moving_block_bootstrap,
)
from fxpulse.next_hypotheses import _apply_policy, _purged_train
from fxpulse.path_label_experiment import triple_barrier_label
from fxpulse.robust_innovation_experiment import circular_shift_placebo


BASE_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")
EXPECTED_FEATURE_SETS = [
    "base_market",
    "base_plus_liquidity",
    "all_market_enrichment",
    "uzs_national",
    "uzs_national_plus_gdelt_russia",
    "uzs_national_plus_gdelt_topics",
    "uzs_national_plus_all_gdelt",
    "uzs_national_plus_ai_gpr",
    "uzs_national_plus_all_news",
    "uzs_national_plus_regional_news",
    "everything",
]
EXPECTED_MODELS = [
    "catboost_tb_unweighted_3seed",
    "catboost_tb_balanced_3seed",
    "catboost_regret_unweighted_3seed",
    "xgboost_tb_3seed",
    "logistic_tb",
]
EXPECTED_POLICIES = [
    "own_score",
    "cross_corridor_mean",
    "agreement_gate",
    "selective_high_confidence",
    "dual_lane",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path | str = Path("configs/uzs_final_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("UZS final experiment must be registered before results")
    if config.get("target_corridor") != "UZS" or int(config.get("horizon", 0)) != 5:
        raise ValueError("the final contract is fixed to UZS h=5")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("feature sets differ from the registered implementation")
    if config.get("model_variants") != EXPECTED_MODELS:
        raise ValueError("model variants differ from the registered implementation")
    if config.get("policies") != EXPECTED_POLICIES:
        raise ValueError("policies differ from the registered implementation")
    development = set(map(int, config["development_years"]))
    pseudo = set(map(int, config["pseudo_holdout_years"]))
    if development & pseudo or development | pseudo != set(map(int, config["test_years"])):
        raise ValueError("development and pseudo-holdout years must be disjoint and exhaustive")
    return config


def _join_external(
    features: pd.DataFrame,
    gdelt: pd.DataFrame,
    ai_gpr: pd.DataFrame,
) -> pd.DataFrame:
    data = features.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise").dt.normalize()
    data["corridor"] = data["corridor"].astype(str).str.upper()
    for external, prefix in ((gdelt, "news__"), (ai_gpr, "aigpr__")):
        current = external.copy()
        current["feature_date"] = pd.to_datetime(current["feature_date"], errors="raise").dt.normalize()
        current["corridor"] = current["corridor"].astype(str).str.upper()
        columns = [column for column in current if column.startswith(prefix)]
        if not columns or current.duplicated(["feature_date", "corridor"]).any():
            raise ValueError(f"invalid external feature table for {prefix}")
        # Leave-one-corridor-out regional context. For the UZS row this is the
        # news background of AMD/KGS/KZT/TJS; the current corridor never leaks
        # into its own aggregate.
        regional_sources = (
            [column for column in columns if column.startswith("news__recipient_")]
            if prefix == "news__"
            else [
                column
                for column in columns
                if column.startswith(("aigpr__country_recipient_", "aigpr__bilateral_"))
            ]
        )
        for column in regional_sources:
            values = pd.to_numeric(current[column], errors="coerce")
            count = values.notna().groupby(current["feature_date"]).transform("sum")
            total = values.fillna(0).groupby(current["feature_date"]).transform("sum")
            other_count = count - values.notna().astype(int)
            other_total = total - values.fillna(0)
            short = column.removeprefix(prefix)
            regional = f"{prefix}regional_other__{short}"
            current[regional] = (other_total / other_count.replace(0, np.nan)).where(
                values.notna() | other_count.gt(0)
            )
            columns.append(regional)
        data = data.merge(
            current[["feature_date", "corridor", *columns]],
            left_on=["timestamp", "corridor"],
            right_on=["feature_date", "corridor"],
            how="inner",
            validate="many_to_one",
        ).drop(columns="feature_date")
    return data.sort_values(["corridor", "timestamp"], kind="mergesort").reset_index(drop=True)


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    base = [column for column in frame if column.startswith(BASE_PREFIXES)]
    liquidity = [column for column in frame if column.startswith(("liquidity__", "extra__liquidity__"))]
    market_extra = [
        column
        for column in frame
        if column.startswith("extra__") and not column.startswith("extra__uzs__")
    ]
    uzs = [column for column in frame if column.startswith("extra__uzs__")]
    regional = [
        column
        for column in frame
        if column.startswith(("news__regional_other__", "aigpr__regional_other__"))
    ]
    news = [
        column
        for column in frame
        if column.startswith("news__") and not column.startswith("news__regional_other__")
    ]
    ai = [
        column
        for column in frame
        if column.startswith("aigpr__") and not column.startswith("aigpr__regional_other__")
    ]
    russia = [column for column in news if column.startswith("news__russia_")]
    topics = [
        column
        for column in news
        if column.startswith(
            ("news__sanctions_", "news__currency_", "news__energy_", "news__recipient_")
        )
    ]
    if not all((base, liquidity, market_extra, uzs, news, ai, russia, topics, regional)):
        raise ValueError("one or more registered feature families are empty")
    values = {
        "base_market": base,
        "base_plus_liquidity": list(dict.fromkeys([*base, *liquidity])),
        "all_market_enrichment": list(dict.fromkeys([*base, *market_extra])),
        "uzs_national": [*base, *uzs],
        "uzs_national_plus_gdelt_russia": [*base, *uzs, *russia],
        "uzs_national_plus_gdelt_topics": [*base, *uzs, *topics],
        "uzs_national_plus_all_gdelt": [*base, *uzs, *news],
        "uzs_national_plus_ai_gpr": [*base, *uzs, *ai],
        "uzs_national_plus_all_news": [*base, *uzs, *news, *ai],
        "uzs_national_plus_regional_news": [*base, *uzs, *regional],
        "everything": list(dict.fromkeys([*base, *market_extra, *uzs, *news, *ai, *regional])),
    }
    if list(values) != EXPECTED_FEATURE_SETS:
        raise AssertionError("feature set order changed")
    return values


def _add_labels(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    horizon = int(config["horizon"])
    barrier = config["triple_barrier"]
    outcome = f"outcome__regret_{horizon}"
    for _, group in frame.groupby("corridor", sort=False):
        current = group.sort_values("timestamp", kind="mergesort").copy()
        current["training_target_tb"] = triple_barrier_label(
            current["price"],
            horizon=horizon,
            better_price_barrier_bps=float(barrier["better_price_barrier_bps"]),
            worse_price_barrier_bps=float(barrier["worse_price_barrier_bps"]),
        )
        current["target"] = current[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
        current["training_target_regret"] = current["target"]
        pieces.append(current)
    return pd.concat(pieces, ignore_index=True)


def _model(name: str, config: dict[str, Any], seed: int):
    if name.startswith("catboost"):
        from catboost import CatBoostClassifier

        parameters = config["catboost"]
        weights = "Balanced" if "balanced" in name and "unweighted" not in name else None
        model = CatBoostClassifier(
            iterations=int(parameters["iterations"]),
            depth=int(parameters["depth"]),
            learning_rate=float(parameters["learning_rate"]),
            l2_leaf_reg=float(parameters["l2_leaf_reg"]),
            loss_function="Logloss",
            auto_class_weights=weights,
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    if name.startswith("xgboost"):
        from xgboost import XGBClassifier

        parameters = config["xgboost"]
        model = XGBClassifier(
            n_estimators=int(parameters["iterations"]),
            max_depth=int(parameters["max_depth"]),
            learning_rate=float(parameters["learning_rate"]),
            min_child_weight=float(parameters["min_child_weight"]),
            subsample=float(parameters["subsample"]),
            colsample_bytree=float(parameters["colsample_bytree"]),
            reg_lambda=float(parameters["reg_lambda"]),
            random_state=int(seed),
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
        )
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    if name == "logistic_tb":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2_000, random_state=int(seed)),
        )
    raise ValueError(f"unknown model {name}")


def _seeds(name: str, config: dict[str, Any]) -> list[int]:
    return [int(config["ensemble_seeds"][0])] if name == "logistic_tb" else list(map(int, config["ensemble_seeds"]))


def _training_column(name: str) -> str:
    return "training_target_regret" if "regret" in name else "training_target_tb"


def _importance_rows(model: Any, columns: list[str], identity: dict[str, object]) -> list[dict[str, object]]:
    estimator = model.steps[-1][1]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coefficients = getattr(estimator, "coef_", None)
        values = np.abs(coefficients[0]) if coefficients is not None else None
    if values is None:
        return []
    imputer = model.steps[0][1]
    indicator = getattr(getattr(imputer, "indicator_", None), "features_", np.array([], dtype=int))
    names = [*columns, *(f"missing__{columns[int(index)]}" for index in indicator)]
    if len(names) != len(values):
        return []
    return [
        {**identity, "feature": feature, "importance": float(importance)}
        for feature, importance in zip(names, values, strict=True)
    ]


def _fit_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    model_name: str,
    config: dict[str, Any],
    year: int,
    identity: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    predictions: list[np.ndarray] = []
    importances: list[dict[str, object]] = []
    target_column = _training_column(model_name)
    for seed_index, seed in enumerate(_seeds(model_name, config)):
        fitted = _model(model_name, config, seed + year + int(config["horizon"]))
        fitted.fit(train[columns], train[target_column].astype(int))
        predictions.append(fitted.predict_proba(test[columns])[:, 1])
        if seed_index == 0:
            importances.extend(_importance_rows(fitted, columns, identity))
    matrix = np.column_stack(predictions)
    return matrix.mean(axis=1), matrix.std(axis=1), importances


def fit_oot_scores(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _add_labels(frame, config)
    data = data.loc[data["timestamp"].ge(pd.Timestamp(config["training_start"]))].copy()
    horizon = int(config["horizon"])
    outcome = f"outcome__regret_{horizon}"
    benefit = f"outcome__benefit_{horizon}"
    sets = feature_sets(data)
    target_corridor = str(config["target_corridor"])
    score_rows: list[pd.DataFrame] = []
    confirmation_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    target_data = data.loc[
        data["corridor"].eq(target_corridor)
        & data[outcome].notna()
        & data["training_target_tb"].notna()
    ].copy()
    combinations = [
        (feature_set, model_name)
        for feature_set in config["feature_sets"]
        for model_name in config["full_feature_models"]
    ]
    combinations.extend(
        ("uzs_national", model_name) for model_name in config["control_models_on_uzs_national"]
    )
    for year_value in config["test_years"]:
        year = int(year_value)
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
        train_all = _purged_train(target_data, start, horizon)
        train_all = train_all.loc[
            train_all["timestamp"].ge(start - pd.DateOffset(years=int(config["rolling_training_years"])))
        ].copy()
        test = target_data.loc[target_data["timestamp"].between(start, end, inclusive="left")].copy()
        if len(train_all) < int(config["minimum_training_observations"]) or len(test) < 20:
            raise ValueError(f"insufficient UZS fold {year}")
        for feature_set, model_name in combinations:
            target_column = _training_column(model_name)
            train = train_all.loc[train_all[target_column].notna()].copy()
            columns = [column for column in sets[feature_set] if train[column].notna().any()]
            identity = {
                "feature_set": feature_set,
                "model_variant": model_name,
                "test_year": year,
            }
            score, dispersion, importance = _fit_one_fold(
                train, test, columns, model_name, config, year, identity
            )
            predicted = test[
                ["timestamp", "corridor", "target", outcome, benefit, "base__percentile_60"]
            ].copy()
            predicted = predicted.rename(columns={outcome: "regret_bps", benefit: "benefit_bps"})
            predicted["score"] = score
            predicted["seed_std"] = dispersion
            predicted["feature_count"] = len(columns)
            predicted["feature_set"] = feature_set
            predicted["model_variant"] = model_name
            predicted["test_year"] = year
            score_rows.append(predicted)
            importance_rows.extend(importance)
            fold_rows.append(
                {
                    **identity,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_start": train["timestamp"].min(),
                    "train_end": train["timestamp"].max(),
                    "test_start": test["timestamp"].min(),
                    "test_end": test["timestamp"].max(),
                    "feature_count": len(columns),
                }
            )
        print(f"completed UZS models for {year}", flush=True)

    base_columns = sets["base_market"]
    other_names = list(config["corridors_for_confirmation"])
    for corridor in other_names:
        current = data.loc[
            data["corridor"].eq(corridor)
            & data[outcome].notna()
            & data["training_target_tb"].notna()
        ].copy()
        for year_value in config["test_years"]:
            year = int(year_value)
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year + 1, month=1, day=1)
            train = _purged_train(current, start, horizon)
            train = train.loc[
                train["timestamp"].ge(start - pd.DateOffset(years=int(config["rolling_training_years"])))
            ].copy()
            test = current.loc[current["timestamp"].between(start, end, inclusive="left")].copy()
            columns = [column for column in base_columns if train[column].notna().any()]
            identity = {"feature_set": "base_market", "model_variant": "confirmation", "test_year": year}
            score, dispersion, _ = _fit_one_fold(
                train, test, columns, "catboost_tb_unweighted_3seed", config, year, identity
            )
            predicted = test[["timestamp", "corridor"]].copy()
            predicted["score"] = score
            predicted["seed_std"] = dispersion
            predicted["test_year"] = year
            confirmation_rows.append(predicted)
        print(f"completed confirmation model {corridor}", flush=True)
    return (
        pd.concat(score_rows, ignore_index=True),
        pd.concat(confirmation_rows, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(importance_rows),
    )


def _with_week_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("timestamp", kind="mergesort").copy()
    iso = data["timestamp"].dt.isocalendar()
    data["week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    rate = data.groupby("week", sort=False)["target"].mean()
    data["matched_week_hit_rate"] = data["week"].map(rate).astype(float)
    return data


def _dual_lane(
    current: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Causal strong-signal lane plus an explicitly informational fallback."""

    product = config["flexible_product_policy"]
    policy = config["adaptive_policy"]
    source = current.rename(columns={"score": "raw_model_score", "policy_score": "score"})
    strong = adaptive_candidates(
        source,
        share=float(product["high_confidence_top_score_share"]),
        lookback=int(policy["lookback_observations"]),
        minimum_history=int(policy["minimum_history_observations"]),
    )
    fallback_pool = adaptive_candidates(
        source,
        share=float(product["fallback_top_score_share"]),
        lookback=int(policy["lookback_observations"]),
        minimum_history=int(policy["minimum_history_observations"]),
    )
    fallback_pool = fallback_pool.loc[
        fallback_pool["timestamp"].dt.weekday.isin(list(product["fallback_weekdays"]))
        & fallback_pool["base__percentile_60"].le(
            float(product["fallback_max_price_percentile_60"])
        )
    ].copy()
    strong_by_date = {pd.Timestamp(row["timestamp"]): row for _, row in strong.iterrows()}
    fallback_by_date = {pd.Timestamp(row["timestamp"]): row for _, row in fallback_pool.iterrows()}
    chosen: list[pd.Series] = []
    week_counts: dict[str, int] = {}
    last: pd.Timestamp | None = None
    # Chronological decisions: Thursday never sees whether Friday will produce
    # a stronger score. A later strong signal also respects the earlier
    # fallback's cooldown and the common weekly cap.
    for timestamp in sorted(set(strong_by_date) | set(fallback_by_date)):
        strong_row = strong_by_date.get(timestamp)
        fallback_row = fallback_by_date.get(timestamp)
        row = strong_row if strong_row is not None else fallback_row
        if row is None:
            continue
        week = str(row["week"])
        if week_counts.get(week, 0) >= int(policy["weekly_cap"]):
            continue
        if last is not None and (timestamp - last).days < int(policy["cooldown_days"]):
            continue
        if strong_row is not None:
            candidate = strong_row.copy()
            candidate["signal_tier"] = "strong_prediction"
        else:
            # Fallback is allowed only if no earlier touch occurred this week.
            if week_counts.get(week, 0) > 0:
                continue
            candidate = fallback_row.copy()
            candidate["signal_tier"] = "informational_value_fallback"
        chosen.append(candidate)
        week_counts[week] = week_counts.get(week, 0) + 1
        last = timestamp
    return pd.DataFrame(chosen).reset_index(drop=True) if chosen else source.iloc[0:0].copy()


def apply_policies(
    scores: pd.DataFrame,
    confirmation: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    consensus = config["consensus"]
    other = confirmation.sort_values(["corridor", "timestamp"], kind="mergesort").copy()
    other["rank"] = np.nan
    for _, indexes in other.groupby("corridor", sort=False).groups.items():
        ordered = other.loc[indexes].sort_values("timestamp", kind="mergesort")
        other.loc[ordered.index, "rank"] = causal_percentile(
            ordered["score"],
            lookback=int(consensus["rank_lookback"]),
            minimum_history=int(consensus["rank_minimum_history"]),
        ).to_numpy()
    other_wide = other.pivot(index="timestamp", columns="corridor", values="rank")
    all_candidates: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    policy = config["adaptive_policy"]
    for (feature_set, model_name), group in scores.groupby(["feature_set", "model_variant"], sort=False):
        data = _with_week_baseline(group)
        data["own_rank"] = causal_percentile(
            data["score"],
            lookback=int(consensus["rank_lookback"]),
            minimum_history=int(consensus["rank_minimum_history"]),
        ).to_numpy()
        aligned = other_wide.reindex(data["timestamp"])
        values = aligned.to_numpy(dtype=float)
        valid = np.isfinite(values)
        counts = valid.sum(axis=1)
        totals = np.where(valid, values, 0.0).sum(axis=1)
        means = np.divide(totals, counts, out=np.full(len(data), np.nan), where=counts > 0)
        centered = np.where(valid, values - means[:, None], 0.0)
        variances = np.divide(
            np.square(centered).sum(axis=1),
            counts,
            out=np.full(len(data), np.nan),
            where=counts > 0,
        )
        positives = np.logical_and(valid, values >= 0.5).sum(axis=1)
        data["other_rank_mean"] = means
        data["other_rank_std"] = np.sqrt(variances)
        data["other_positive_share"] = np.divide(
            positives, counts, out=np.full(len(data), np.nan), where=counts > 0
        )
        data["consensus_score"] = (
            float(consensus["own_weight"]) * data["own_rank"]
            + float(consensus["other_weight"]) * data["other_rank_mean"]
        )
        for policy_name in config["policies"]:
            current = data.copy()
            current["policy"] = policy_name
            current["policy_score"] = (
                current["score"] if policy_name == "own_score" else current["consensus_score"]
            )
            current = current.dropna(subset=["policy_score"])
            if policy_name == "dual_lane":
                chosen = _dual_lane(current, config=config)
                all_candidates.append(current)
                selected_rows.append(chosen)
                continue
            share = (
                float(config["flexible_product_policy"]["high_confidence_top_score_share"])
                if policy_name == "selective_high_confidence"
                else float(policy["top_score_share"])
            )
            candidates = adaptive_candidates(
                current.rename(columns={"score": "raw_model_score", "policy_score": "score"}),
                share=share,
                lookback=int(policy["lookback_observations"]),
                minimum_history=int(policy["minimum_history_observations"]),
            )
            if policy_name == "agreement_gate":
                candidates = candidates.loc[
                    candidates["other_positive_share"].ge(
                        float(consensus["minimum_other_positive_share"])
                    )
                    & candidates["other_rank_std"].le(float(consensus["maximum_other_rank_std"]))
                ]
            chosen = _apply_policy(
                candidates,
                cooldown_days=int(policy["cooldown_days"]),
                weekly_cap=int(policy["weekly_cap"]),
            )
            chosen["signal_tier"] = "strong_prediction"
            all_candidates.append(current)
            selected_rows.append(chosen)
    return pd.concat(all_candidates, ignore_index=True), pd.concat(selected_rows, ignore_index=True)


def _period_metrics(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    years: list[int],
) -> dict[str, float | int]:
    year_set = set(map(int, years))
    test = scores.loc[scores["test_year"].isin(year_set)].copy()
    chosen = signals.loc[signals["test_year"].isin(year_set)].copy()
    count = len(chosen)
    hits = int(chosen["target"].sum()) if count else 0
    base_rate = float(test["target"].mean())
    expected = float(chosen["matched_week_hit_rate"].sum())
    duration = sum(
        max(float((part["timestamp"].max() - part["timestamp"].min()).days) / 7, 1 / 7)
        for _, part in test.groupby("test_year", sort=True)
    )
    test_weeks = test[["test_year", "week"]].drop_duplicates()
    signal_weeks = chosen[["test_year", "week"]].drop_duplicates() if count else chosen
    fallback_count = (
        int(chosen["signal_tier"].eq("informational_value_fallback").sum())
        if count and "signal_tier" in chosen
        else 0
    )
    yearly: list[float] = []
    for _, part in chosen.groupby("test_year", sort=True):
        part_expected = float(part["matched_week_hit_rate"].sum())
        if part_expected:
            yearly.append(float(part["target"].sum()) / part_expected)
    return {
        "oot_rows": len(test),
        "signals": count,
        "hits": hits,
        "hit_rate": hits / count if count else np.nan,
        "baseline_hit_rate": base_rate,
        "raw_lift": (hits / count) / base_rate if count and base_rate else np.nan,
        "same_week_lift": hits / expected if expected else np.nan,
        "signals_per_week": count / duration if duration else np.nan,
        "weeks_covered_share": len(signal_weeks) / len(test_weeks) if len(test_weeks) else np.nan,
        "strong_prediction_signals": count - fallback_count,
        "informational_fallback_signals": fallback_count,
        "mean_regret_bps": float(chosen["regret_bps"].mean()) if count else np.nan,
        "mean_benefit_bps": float(chosen["benefit_bps"].mean()) if count else np.nan,
        "worst_year_same_week_lift": min(yearly) if yearly else np.nan,
        "mean_seed_std": float(test["seed_std"].mean()),
        "feature_count": int(test["feature_count"].median()),
    }


def summarize(scores: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    periods = {
        "development": list(map(int, config["development_years"])),
        "pseudo_holdout": list(map(int, config["pseudo_holdout_years"])),
        "all": list(map(int, config["test_years"])),
    }
    rows: list[dict[str, object]] = []
    keys = ["feature_set", "model_variant", "policy"]
    for identity, test in scores.groupby(keys, sort=True):
        mask = np.logical_and.reduce(
            [signals[column].eq(value) for column, value in zip(keys, identity, strict=True)]
        )
        selected = signals.loc[mask]
        for period, years in periods.items():
            rows.append(
                {
                    **dict(zip(keys, identity, strict=True)),
                    "period": period,
                    **_period_metrics(test, selected, years),
                }
            )
    return pd.DataFrame(rows)


def yearly_summary(scores: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["feature_set", "model_variant", "policy"]
    for identity, test in scores.groupby(keys, sort=True):
        mask = np.logical_and.reduce(
            [signals[column].eq(value) for column, value in zip(keys, identity, strict=True)]
        )
        selected = signals.loc[mask]
        for year in sorted(test["test_year"].unique()):
            rows.append(
                {
                    **dict(zip(keys, identity, strict=True)),
                    "test_year": int(year),
                    **_period_metrics(test, selected, [int(year)]),
                }
            )
    return pd.DataFrame(rows)


def choose_winner(summary: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    rule = config["selection_rule"]
    candidates = summary.loc[
        summary["period"].eq("development")
        & summary["signals"].ge(int(rule["minimum_signals"]))
        & summary["signals_per_week"].between(
            float(rule["minimum_signals_per_week"]), float(rule["maximum_signals_per_week"])
        )
        & summary["worst_year_same_week_lift"].ge(
            float(rule["minimum_worst_year_same_week_lift"])
        )
    ].copy()
    if candidates.empty:
        raise RuntimeError("no candidate passes the preregistered selection constraints")
    return candidates.sort_values(
        ["same_week_lift", "raw_lift", "mean_regret_bps", "feature_count"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]


def _identity_mask(frame: pd.DataFrame, winner: pd.Series) -> pd.Series:
    return (
        frame["feature_set"].eq(winner["feature_set"])
        & frame["model_variant"].eq(winner["model_variant"])
        & frame["policy"].eq(winner["policy"])
    )


def _paired_bootstrap(
    baseline_weekly: pd.DataFrame,
    candidate_weekly: pd.DataFrame,
    *,
    samples: int,
    block_weeks: int,
    seed: int,
) -> dict[str, float]:
    keys = ["test_year", "week"]
    columns = ["signal_count", "signal_hits", "matched_expected_hits", "base_count", "base_hits"]
    paired = baseline_weekly[keys + columns].merge(
        candidate_weekly[keys + columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_base", "_candidate"),
    )
    n = len(paired)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(int(samples), int(np.ceil(n / block_weeks))))
    indices = ((starts[..., None] + np.arange(block_weeks)) % n).reshape(int(samples), -1)[:, :n]

    def lifts(suffix: str) -> tuple[np.ndarray, np.ndarray]:
        values = paired[[f"{column}_{suffix}" for column in columns]].to_numpy(dtype=float)[indices]
        totals = values.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (totals[:, 1] / totals[:, 0]) / (totals[:, 4] / totals[:, 3]),
                totals[:, 1] / totals[:, 2],
            )

    base_raw, base_matched = lifts("base")
    candidate_raw, candidate_matched = lifts("candidate")
    raw = candidate_raw - base_raw
    matched = candidate_matched - base_matched
    return {
        "paired_raw_delta_low": float(np.nanquantile(raw, 0.025)),
        "paired_raw_delta_median": float(np.nanmedian(raw)),
        "paired_raw_delta_high": float(np.nanquantile(raw, 0.975)),
        "paired_same_week_delta_low": float(np.nanquantile(matched, 0.025)),
        "paired_same_week_delta_median": float(np.nanmedian(matched)),
        "paired_same_week_delta_high": float(np.nanquantile(matched, 0.975)),
    }


def audit_winner(
    policy_scores: pd.DataFrame,
    signals: pd.DataFrame,
    winner: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust = config["robustness"]
    selected_scores = policy_scores.loc[_identity_mask(policy_scores, winner)].copy()
    selected_signals = signals.loc[_identity_mask(signals, winner)].copy()
    rows: list[dict[str, object]] = []
    for index, (period, years) in enumerate(
        (
            ("development", config["development_years"]),
            ("pseudo_holdout", config["pseudo_holdout_years"]),
            ("all", config["test_years"]),
        )
    ):
        target = selected_scores.loc[selected_scores["test_year"].isin(years)]
        chosen = selected_signals.loc[selected_signals["test_year"].isin(years)]
        weekly = _weekly_winner_table(target, chosen)
        bootstrap = moving_block_bootstrap(
            weekly,
            samples=int(robust["bootstrap_samples"]),
            block_weeks=int(robust["moving_block_weeks"]),
            seed=int(config["ensemble_seeds"][0]) + index,
        )
        placebo = circular_shift_placebo(
            target,
            chosen,
            samples=int(robust["placebo_samples"]),
            seed=int(config["ensemble_seeds"][0]) + 100 + index,
        )
        deoverlapped = _deoverlap_signals(chosen, int(robust["deoverlap_cooldown_days"]))
        expected = float(deoverlapped["matched_week_hit_rate"].sum())
        duration = sum(
            max(float((part["timestamp"].max() - part["timestamp"].min()).days) / 7, 1 / 7)
            for _, part in target.groupby("test_year", sort=True)
        )
        raw_base = float(target["target"].mean())
        rows.append(
            {
                "period": period,
                **bootstrap,
                **placebo,
                "deoverlap_signals": len(deoverlapped),
                "deoverlap_raw_lift": float(deoverlapped["target"].mean()) / raw_base,
                "deoverlap_same_week_lift": float(deoverlapped["target"].sum()) / expected,
                "deoverlap_signals_per_week": len(deoverlapped) / duration,
            }
        )

    baseline_identity = {
        "feature_set": "base_market",
        "model_variant": "catboost_tb_unweighted_3seed",
        "policy": "own_score",
    }
    baseline_mask_scores = np.logical_and.reduce(
        [policy_scores[key].eq(value) for key, value in baseline_identity.items()]
    )
    baseline_mask_signals = np.logical_and.reduce(
        [signals[key].eq(value) for key, value in baseline_identity.items()]
    )
    paired_rows: list[dict[str, object]] = []
    for index, (period, years) in enumerate(
        (("development", config["development_years"]), ("pseudo_holdout", config["pseudo_holdout_years"]), ("all", config["test_years"]))
    ):
        baseline_target = policy_scores.loc[baseline_mask_scores & policy_scores["test_year"].isin(years)]
        baseline_signals = signals.loc[baseline_mask_signals & signals["test_year"].isin(years)]
        candidate_target = selected_scores.loc[selected_scores["test_year"].isin(years)]
        candidate_signals = selected_signals.loc[selected_signals["test_year"].isin(years)]
        paired_rows.append(
            {
                "period": period,
                **_paired_bootstrap(
                    _weekly_winner_table(baseline_target, baseline_signals),
                    _weekly_winner_table(candidate_target, candidate_signals),
                    samples=int(robust["bootstrap_samples"]),
                    block_weeks=int(robust["moving_block_weeks"]),
                    seed=int(config["ensemble_seeds"][0]) + 200 + index,
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def tolerance_sensitivity(
    policy_scores: pd.DataFrame,
    signals: pd.DataFrame,
    winner: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    test = policy_scores.loc[_identity_mask(policy_scores, winner)].copy()
    chosen = signals.loc[_identity_mask(signals, winner)].copy()
    rows: list[dict[str, object]] = []
    for tolerance in config["robustness"]["evaluation_tolerances_bps"]:
        current_test = test.copy()
        current_test["target"] = current_test["regret_bps"].le(float(tolerance)).astype(int)
        current_test = _with_week_baseline(current_test.drop(columns="matched_week_hit_rate", errors="ignore"))
        current_chosen = chosen.drop(columns=["target", "matched_week_hit_rate"], errors="ignore").merge(
            current_test[["timestamp", "test_year", "target", "matched_week_hit_rate"]],
            on=["timestamp", "test_year"],
            how="left",
            validate="one_to_one",
        )
        for period, years in (("development", config["development_years"]), ("pseudo_holdout", config["pseudo_holdout_years"]), ("all", config["test_years"])):
            rows.append(
                {
                    "tolerance_bps": int(tolerance),
                    "period": period,
                    **_period_metrics(current_test, current_chosen, list(map(int, years))),
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(policy_scores: pd.DataFrame, winner: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    test = policy_scores.loc[_identity_mask(policy_scores, winner)].copy()
    rows: list[dict[str, object]] = []
    for period, years in (("development", config["development_years"]), ("pseudo_holdout", config["pseudo_holdout_years"]), ("all", config["test_years"])):
        current = test.loc[test["test_year"].isin(years)]
        raw = current["raw_model_score"] if "raw_model_score" in current else current["score"]
        rows.append(
            {
                "period": period,
                "roc_auc": float(roc_auc_score(current["target"], raw)),
                "brier_score": float(brier_score_loss(current["target"], raw)),
                "mean_seed_std": float(current["seed_std"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/uzs_final_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/uzs_final_experiment_20260905"),
) -> dict[str, object]:
    config = load_config(config_path)
    paths = {name: Path(value) for name, value in config["inputs"].items() if name != "colleague_result"}
    frame = _join_external(
        pd.read_csv(paths["features"]),
        pd.read_csv(paths["gdelt"]),
        pd.read_csv(paths["ai_gpr"]),
    )
    scores, confirmation, folds, importance = fit_oot_scores(frame, config)
    policy_scores, signals = apply_policies(scores, confirmation, config)
    summary = summarize(policy_scores, signals, config)
    yearly = yearly_summary(policy_scores, signals)
    winner = choose_winner(summary, config)
    winner_identity = winner[["feature_set", "model_variant", "policy"]].to_dict()
    winner_summary = summary.loc[
        summary["feature_set"].eq(winner_identity["feature_set"])
        & summary["model_variant"].eq(winner_identity["model_variant"])
        & summary["policy"].eq(winner_identity["policy"])
    ].copy()
    audit, paired = audit_winner(policy_scores, signals, winner, config)
    sensitivity = tolerance_sensitivity(policy_scores, signals, winner, config)
    calibration = calibration_summary(policy_scores, winner, config)
    winner_importance = importance.loc[
        importance["feature_set"].eq(winner_identity["feature_set"])
        & importance["model_variant"].eq(winner_identity["model_variant"])
    ].copy()
    if len(winner_importance):
        winner_importance = (
            winner_importance.groupby("feature", as_index=False)["importance"].mean()
            .sort_values("importance", ascending=False, kind="mergesort")
        )

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "oot_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    confirmation.to_csv(output / "confirmation_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    policy_scores.to_csv(output / "policy_scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    signals.to_csv(output / "signals.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    folds.to_csv(output / "folds.csv", index=False)
    importance.to_csv(output / "feature_importance.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(output / "summary.csv", index=False)
    yearly.to_csv(output / "yearly_summary.csv", index=False)
    winner_summary.to_csv(output / "winner_summary.csv", index=False)
    winner_importance.to_csv(output / "winner_feature_importance.csv", index=False)
    audit.to_csv(output / "winner_robustness.csv", index=False)
    paired.to_csv(output / "winner_paired_vs_baseline.csv", index=False)
    sensitivity.to_csv(output / "winner_tolerance_sensitivity.csv", index=False)
    calibration.to_csv(output / "winner_calibration.csv", index=False)
    candidate_count = int(summary["period"].eq("development").sum())
    metadata = {
        "config_path": str(config_path),
        "config_sha256": _sha256(Path(config_path)),
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "joined_rows": len(frame),
        "joined_columns": len(frame.columns),
        "candidate_count": candidate_count,
        "winner": winner_identity,
        "selection_period": list(config["development_years"]),
        "pseudo_holdout_period": list(config["pseudo_holdout_years"]),
        "selection_adjusted_block_p_bound": min(
            1.0,
            float(audit.loc[audit["period"].eq("development"), "block_matched_p_value"].iloc[0])
            * candidate_count,
        ),
        "warning": "2025-2026 is a pseudo-holdout already seen by the team; only future or bank-held data can confirm production performance.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(winner_summary.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/uzs_final_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/uzs_final_experiment_20260905"))
    args = parser.parse_args()
    run(config_path=args.config, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
