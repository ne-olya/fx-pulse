"""Score CatBoost parameters for a frozen all-enriched corridor agreement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.adaptive_threshold import adaptive_candidates  # noqa: E402
from fxpulse.innovation_followup import causal_percentile  # noqa: E402
from fxpulse.next_hypotheses import _apply_policy, _purged_train  # noqa: E402
from fxpulse.robust_innovation_experiment import add_registered_labels, load_config as load_robust_config  # noqa: E402


ALLOWED_PARAMETERS = {
    "iterations",
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
    "border_count",
    "auto_class_weights",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("configs/uzs_enriched_catboost_params.json"))
    parser.add_argument("--contract", type=Path, default=Path("configs/uzs_enriched_search_contract.json"))
    parser.add_argument("--evaluation", choices=["search", "pseudo_holdout", "all"], default="search")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate(contract: dict[str, Any], params: dict[str, Any]) -> Path:
    if contract.get("schema_version") != 1 or contract.get("status") != "frozen_before_autoresearch":
        raise ValueError("invalid frozen search contract")
    if contract.get("search_years") != [2022, 2023, 2024] or contract.get("pseudo_holdout_years") != [2025, 2026]:
        raise ValueError("temporal search/holdout boundary changed")
    corridors = contract.get("corridors")
    target_corridor = contract.get("target_corridor")
    if not isinstance(corridors, list) or len(corridors) < 2 or len(corridors) != len(set(corridors)):
        raise ValueError("corridors must contain at least two unique values")
    if target_corridor not in corridors:
        raise ValueError("target_corridor must be one of corridors")
    if contract.get("training_target", "triple_barrier") not in {"triple_barrier", "regret_binary"}:
        raise ValueError("training_target must be triple_barrier or regret_binary")
    if set(params) - ALLOWED_PARAMETERS:
        raise ValueError(f"unsupported CatBoost parameters: {sorted(set(params) - ALLOWED_PARAMETERS)}")
    required = {"iterations", "depth", "learning_rate"}
    if required - set(params):
        raise ValueError(f"missing CatBoost parameters: {sorted(required - set(params))}")
    if not 40 <= int(params["iterations"]) <= 400:
        raise ValueError("iterations must be in [40, 400]")
    if not 3 <= int(params["depth"]) <= 8:
        raise ValueError("depth must be in [3, 8]")
    if not 0.01 <= float(params["learning_rate"]) <= 0.15:
        raise ValueError("learning_rate must be in [0.01, 0.15]")
    for name, lower, upper in (
        ("l2_leaf_reg", 0.1, 50.0),
        ("random_strength", 0.0, 5.0),
        ("bagging_temperature", 0.0, 5.0),
        ("border_count", 16, 254),
    ):
        if name in params and not lower <= float(params[name]) <= upper:
            raise ValueError(f"{name} must be in [{lower}, {upper}]")
    if params.get("auto_class_weights") not in {None, "Balanced", "SqrtBalanced"}:
        raise ValueError("auto_class_weights must be null, Balanced or SqrtBalanced")
    features = Path(str(contract["features_path"]))
    if not features.is_file() or _sha256(features) != str(contract["features_sha256"]):
        raise ValueError("frozen feature artifact is missing or has another SHA-256")
    return features


def _catboost(params: dict[str, Any], *, seed: int):
    from catboost import CatBoostClassifier

    model_params = {
        "loss_function": "Logloss",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
        **params,
    }
    if model_params.get("auto_class_weights") is None:
        model_params.pop("auto_class_weights", None)
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        CatBoostClassifier(**model_params),
    )


def _score_models(frame: pd.DataFrame, contract: dict[str, Any], params: dict[str, Any], years: list[int]) -> pd.DataFrame:
    horizon = int(contract["horizon"])
    robust = load_robust_config()
    robust["horizons"] = [horizon]
    data = add_registered_labels(frame, robust)
    outcome = f"outcome__regret_{horizon}"
    label = f"label__tb_{horizon}"
    requested = [column for column in data if column.startswith(tuple(contract["base_feature_prefixes"]))]
    for prefix in contract["extra_feature_prefixes"]:
        requested.extend(column for column in data if column.startswith(prefix))
    requested = list(dict.fromkeys(requested))
    rows: list[pd.DataFrame] = []
    for corridor in contract["corridors"]:
        corridor_data = data.loc[data["corridor"].eq(corridor) & data[outcome].notna()].copy()
        corridor_data = corridor_data.sort_values("timestamp", kind="mergesort")
        if contract.get("training_target", "triple_barrier") == "regret_binary":
            corridor_data["training_target"] = corridor_data[outcome].le(
                float(contract["evaluation_tolerance_bps"])
            ).astype(int)
        else:
            corridor_data["training_target"] = corridor_data[label]
        for year in years:
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year + 1, month=1, day=1)
            train = _purged_train(corridor_data, start, horizon)
            train = train.loc[
                train["timestamp"].ge(start - pd.DateOffset(years=int(contract["rolling_training_years"])))
                & train["training_target"].notna()
            ].copy()
            test = corridor_data.loc[corridor_data["timestamp"].between(start, end, inclusive="left")].copy()
            columns = [column for column in requested if train[column].notna().any()]
            if (
                len(train) < int(contract["minimum_training_observations"])
                or len(test) < 20
                or train["training_target"].nunique() < 2
            ):
                raise ValueError(f"insufficient fold {corridor}/{year}")
            model = _catboost(
                params,
                seed=int(contract["random_seed"]) + year + horizon + 701,
            )
            model.fit(train[columns], train["training_target"].astype(int))
            scored = test[["timestamp", "corridor", outcome]].copy().rename(columns={outcome: "regret_bps"})
            scored["target"] = scored["regret_bps"].le(float(contract["evaluation_tolerance_bps"])).astype(int)
            scored["score"] = model.predict_proba(test[columns])[:, 1]
            scored["test_year"] = year
            scored["week"] = scored["timestamp"].dt.to_period("W").astype(str)
            weekly = scored.groupby("week", sort=False)["target"].mean()
            scored["matched_week_hit_rate"] = scored["week"].map(weekly).astype(float)
            rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def _agreement_gate(scores: pd.DataFrame, contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    consensus = contract["consensus"]
    data = scores.sort_values(["corridor", "timestamp"], kind="mergesort").copy()
    data["own_rank"] = np.nan
    for _, indices in data.groupby("corridor", sort=False).groups.items():
        ordered = data.loc[indices].sort_values("timestamp", kind="mergesort")
        data.loc[ordered.index, "own_rank"] = causal_percentile(
            ordered["score"],
            lookback=int(consensus["rank_lookback"]),
            minimum_history=int(consensus["rank_minimum_history"]),
        ).to_numpy()
    target_corridor = str(contract["target_corridor"])
    other_corridors = list(
        consensus.get(
            "other_corridors",
            [corridor for corridor in contract["corridors"] if corridor != target_corridor],
        )
    )
    if not other_corridors or target_corridor in other_corridors or set(other_corridors) - set(contract["corridors"]):
        raise ValueError("consensus other_corridors must be a non-empty target-free subset of corridors")
    raw_weights = consensus.get("other_corridor_weights", {})
    weights = np.asarray([float(raw_weights.get(corridor, 1.0)) for corridor in other_corridors], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("consensus other corridor weights must be finite, non-negative and non-zero")
    weights /= weights.sum()

    target = data.loc[data["corridor"].eq(target_corridor)].copy()
    rank_panel = data.pivot(index="timestamp", columns="corridor", values="own_rank")
    other_ranks = rank_panel.reindex(target["timestamp"])[other_corridors].to_numpy(dtype=float)
    valid = np.isfinite(other_ranks).all(axis=1) & np.isfinite(target["own_rank"].to_numpy(dtype=float))
    other_mean = np.full(len(target), np.nan)
    other_std = np.full(len(target), np.nan)
    other_positive_share = np.full(len(target), np.nan)
    if valid.any():
        valid_ranks = other_ranks[valid]
        valid_mean = valid_ranks @ weights
        other_mean[valid] = valid_mean
        other_std[valid] = np.sqrt(((valid_ranks - valid_mean[:, None]) ** 2) @ weights)
        other_positive_share[valid] = (valid_ranks >= 0.5) @ weights
    target["other_rank_mean"] = other_mean
    target["other_rank_std"] = other_std
    target["other_positive_share"] = other_positive_share
    target["consensus_score"] = (
        float(consensus["own_weight"]) * target["own_rank"]
        + float(consensus["other_weight"]) * target["other_rank_mean"]
    )
    target = target.dropna(subset=["consensus_score"])
    policy = contract["policy"]
    candidates = adaptive_candidates(
        target.rename(columns={"score": "raw_model_score", "consensus_score": "score"}),
        share=float(policy["top_score_share"]),
        lookback=int(policy["lookback_observations"]),
        minimum_history=int(policy["minimum_history_observations"]),
    )
    candidates = candidates.loc[
        candidates["other_positive_share"].ge(float(consensus["minimum_other_positive_share"]))
        & candidates["other_rank_std"].le(float(consensus["maximum_other_rank_std"]))
    ]
    selected = _apply_policy(
        candidates,
        cooldown_days=int(policy["cooldown_days"]),
        weekly_cap=int(policy["weekly_cap"]),
    )
    return target, selected


def _metrics(target: pd.DataFrame, selected: pd.DataFrame, years: list[int], contract: dict[str, Any]) -> dict[str, Any]:
    test = target.loc[target["test_year"].isin(years)].copy()
    signals = selected.loc[selected["test_year"].isin(years)].copy()
    count = len(signals)
    hits = int(signals["target"].sum())
    base_rate = float(test["target"].mean())
    expected = float(signals["matched_week_hit_rate"].sum())
    raw_lift = (hits / count) / base_rate if count and base_rate else np.nan
    same_week_lift = hits / expected if expected else np.nan
    duration = sum(
        max(float((part["timestamp"].max() - part["timestamp"].min()).days) / 7, 1 / 7)
        for _, part in test.groupby("test_year", sort=True)
    )
    yearly = []
    for year in years:
        part = signals.loc[signals["test_year"].eq(year)]
        part_expected = float(part["matched_week_hit_rate"].sum())
        yearly.append(float(part["target"].sum()) / part_expected if part_expected else np.nan)
    worst_year = float(np.nanmin(yearly))
    frequency = count / duration
    constraints = contract["constraints"]
    constraint_pass = (
        float(constraints["minimum_signals_per_week"]) <= frequency <= float(constraints["maximum_signals_per_week"])
        and worst_year >= float(constraints["minimum_worst_year_same_week_lift"])
        and raw_lift >= float(constraints["minimum_pooled_raw_lift"])
    )
    return {
        "objective": float(same_week_lift) if constraint_pass else 0.0,
        "same_week_lift": float(same_week_lift),
        "raw_lift": float(raw_lift),
        "signals": count,
        "hits": hits,
        "hit_rate": float(hits / count) if count else np.nan,
        "all_days_hit_rate": base_rate,
        "matched_same_week_expected_hits": expected,
        "signals_per_week": float(frequency),
        "worst_year_same_week_lift": worst_year,
        "yearly_same_week_lift": [float(value) for value in yearly],
        "constraint_pass": bool(constraint_pass),
        "evaluation_years": years,
        "target_corridor": str(contract["target_corridor"]),
        "training_target": str(contract.get("training_target", "triple_barrier")),
    }


def main() -> None:
    args = _parse_args()
    contract = _load_json(args.contract)
    params = _load_json(args.params)
    features_path = _validate(contract, params)
    if args.validate_only:
        print(json.dumps({"valid": True, "params": params}, sort_keys=True))
        return
    search = list(map(int, contract["search_years"]))
    holdout = list(map(int, contract["pseudo_holdout_years"]))
    fit_years = search if args.evaluation == "search" else search + holdout
    evaluation_years = search if args.evaluation == "search" else holdout if args.evaluation == "pseudo_holdout" else fit_years
    frame = pd.read_csv(features_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    scores = _score_models(frame, contract, params, fit_years)
    target, selected = _agreement_gate(scores, contract)
    result = _metrics(target, selected, evaluation_years, contract)
    result["params"] = params
    result["evaluation"] = args.evaluation
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
