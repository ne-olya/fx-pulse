"""Run a robust, leakage-safe ablation of daily GDELT news features."""

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
    _apply_policy,
    _feature_importance,
    _make_model,
    _purged_train,
)
from fxpulse.training_history_experiment import add_training_labels

EXPECTED_FEATURE_SETS = [
    "market_only",
    "news_only",
    "plus_russia_levels",
    "plus_russia_shocks",
    "plus_recipient_levels",
    "plus_recipient_shocks",
    "plus_all_levels",
    "plus_all_shocks",
    "plus_macro_topics",
    "plus_cross_country",
    "plus_all_news",
    "plus_liquidity_all_news",
]
MARKET_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/news_robust_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("robust news experiment must be preregistered schema_version 1")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("robust news feature sets differ from the implementation")
    if config.get("training_label") != "triple_barrier":
        raise ValueError("robust news experiment requires the registered triple-barrier label")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_frame(market: pd.DataFrame, news: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Join the already one-day-lagged news table to exactly the same market dates."""

    market = market.copy()
    news = news.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="raise").dt.normalize()
    news["feature_date"] = pd.to_datetime(news["feature_date"], errors="raise").dt.normalize()
    market["corridor"] = market["corridor"].astype(str).str.upper()
    news["corridor"] = news["corridor"].astype(str).str.upper()
    if news.duplicated(["feature_date", "corridor"]).any():
        raise ValueError("news input must have one row per feature date and corridor")
    news_columns = [column for column in news if column.startswith("news__")]
    if not news_columns:
        raise ValueError("news input contains no news__ feature")
    allowed = set(config["corridors"])
    result = market.loc[market["corridor"].isin(allowed)].merge(
        news[["feature_date", "corridor", *news_columns]],
        left_on=["timestamp", "corridor"],
        right_on=["feature_date", "corridor"],
        how="inner",
        validate="many_to_one",
    )
    return result.drop(columns="feature_date").sort_values(
        ["corridor", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def _is_level(column: str) -> bool:
    return column.endswith(("_count", "_share", "_count_3d", "_count_7d"))


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(MARKET_PREFIXES)]
    liquidity = [column for column in frame if column.startswith("liquidity__")]
    news = [column for column in frame if column.startswith("news__")]
    if not market or not news:
        raise ValueError("market and news feature groups must both be populated")

    cross_country = [column for column in news if column.startswith("news__cross_")]
    russia = [
        column
        for column in news
        if not column.startswith(("news__recipient_", "news__cross_"))
    ]
    recipient = [column for column in news if column.startswith("news__recipient_")]
    russia_levels = [column for column in russia if _is_level(column)]
    recipient_levels = [column for column in recipient if _is_level(column)]
    russia_shocks = [column for column in russia if "zscore" in column]
    recipient_shocks = [column for column in recipient if "zscore" in column]
    macro_topics = [
        column
        for column in news
        if column.startswith(
            ("news__sanctions_", "news__currency_", "news__energy_", "news__recipient_macro_")
        )
    ]
    if not all((russia_levels, recipient_levels, russia_shocks, recipient_shocks)):
        raise ValueError("daily news input lacks a registered level or shock family")
    return {
        "market_only": market,
        "news_only": news,
        "plus_russia_levels": [*market, *russia_levels],
        "plus_russia_shocks": [*market, *russia_shocks],
        "plus_recipient_levels": [*market, *recipient_levels],
        "plus_recipient_shocks": [*market, *recipient_shocks],
        "plus_all_levels": [*market, *russia_levels, *recipient_levels],
        "plus_all_shocks": [*market, *russia_shocks, *recipient_shocks],
        "plus_macro_topics": [*market, *macro_topics],
        "plus_cross_country": [*market, *cross_country],
        "plus_all_news": [*market, *news],
        "plus_liquidity_all_news": [*market, *liquidity, *news],
    }


def fit_oot_scores(
    data: pd.DataFrame,
    config: dict[str, Any],
    *,
    sets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = add_training_labels(data, config)
    frame = frame.loc[frame["timestamp"].ge(pd.Timestamp(config["training_start"]))].copy()
    sets = feature_sets(frame) if sets is None else sets
    if list(sets) != list(config["feature_sets"]):
        raise ValueError("provided feature sets differ from the registered order")
    score_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for corridor in config["corridors"]:
        corridor_data = frame.loc[frame["corridor"].eq(corridor)].copy()
        for horizon_value in config["horizons"]:
            horizon = int(horizon_value)
            outcome = f"outcome__regret_{horizon}"
            benefit = f"outcome__benefit_{horizon}"
            training_label = f"training_target_{horizon}"
            usable = corridor_data.loc[
                corridor_data[outcome].notna() & corridor_data[training_label].notna()
            ].copy()
            usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
            usable["training_target"] = usable[training_label].astype(int)

            for year_value in config["test_years"]:
                year = int(year_value)
                start = pd.Timestamp(year=year, month=1, day=1)
                end = pd.Timestamp(year=year + 1, month=1, day=1)
                train = _purged_train(usable, start, horizon)
                train = train.loc[
                    train["timestamp"].ge(
                        start - pd.DateOffset(years=int(config["rolling_training_years"]))
                    )
                ].copy()
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if (
                    len(train) < int(config["minimum_training_observations"])
                    or len(test) < 20
                    or train["training_target"].nunique() < 2
                ):
                    continue

                for feature_set in config["feature_sets"]:
                    # A sparse external series can be completely empty in an early
                    # training window. Dropping it is causal and avoids pretending
                    # that a future first observation was available to the model.
                    columns = [column for column in sets[feature_set] if train[column].notna().any()]
                    if not columns:
                        raise ValueError(f"no observed training feature for {feature_set}")
                    predictions: list[np.ndarray] = []
                    fitted_models: list[Any] = []
                    for seed_value in config["ensemble_seeds"]:
                        model = _make_model(
                            str(config["model"]),
                            iterations=int(config["model_iterations"]),
                            seed=int(seed_value) + year + horizon,
                        )
                        model.fit(train[columns], train["training_target"])
                        predictions.append(model.predict_proba(test[columns])[:, 1])
                        fitted_models.append(model)
                    predicted = test[["timestamp", "corridor", "target", outcome, benefit]].copy()
                    predicted["score"] = np.mean(np.column_stack(predictions), axis=1)
                    predicted["feature_set"] = feature_set
                    predicted["horizon"] = horizon
                    predicted["test_year"] = year
                    predicted = predicted.rename(columns={outcome: "regret_bps", benefit: "benefit_bps"})
                    score_frames.append(predicted)
                    fold_rows.append(
                        {
                            "corridor": corridor,
                            "horizon": horizon,
                            "test_year": year,
                            "feature_set": feature_set,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "train_start": train["timestamp"].min(),
                            "train_end": train["timestamp"].max(),
                            "test_start": test["timestamp"].min(),
                            "test_end": test["timestamp"].max(),
                        }
                    )
                    for fitted_model in fitted_models:
                        for feature, importance, _ in _feature_importance(fitted_model, columns):
                            importance_rows.append(
                                {
                                    "corridor": corridor,
                                    "horizon": horizon,
                                    "test_year": year,
                                    "feature_set": feature_set,
                                    "feature": feature,
                                    "importance": importance,
                                }
                            )
            print(f"completed external-feature scores {corridor} h={horizon}", flush=True)

    if not score_frames:
        raise RuntimeError("robust news experiment produced no OOT scores")
    scores = pd.concat(score_frames, ignore_index=True)
    identity = ["corridor", "horizon", "test_year", "timestamp"]
    counts = scores.groupby(identity)["feature_set"].nunique()
    if not counts.eq(len(config["feature_sets"])).all():
        raise RuntimeError("news variants were not scored on identical OOT dates")
    return scores, pd.DataFrame(fold_rows), pd.DataFrame(importance_rows)


def apply_continuous_policy(scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply one causal score threshold without resetting it at New Year."""

    policy = config["adaptive_policy"]
    selected: list[pd.DataFrame] = []
    for _, group in scores.groupby(["feature_set", "corridor", "horizon"], sort=False):
        ordered = group.sort_values("timestamp", kind="mergesort").copy()
        iso = ordered["timestamp"].dt.isocalendar()
        ordered["week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
        weekly_rate = ordered.groupby("week", sort=False)["target"].mean()
        ordered["matched_week_hit_rate"] = ordered["week"].map(weekly_rate).astype(float)
        candidates = adaptive_candidates(
            ordered,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        selected.append(
            _apply_policy(
                candidates,
                cooldown_days=int(policy["cooldown_days"]),
                weekly_cap=int(policy["weekly_cap"]),
            )
        )
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def summarize(scores: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["feature_set", "corridor", "horizon"]
    for identity, all_scores in scores.groupby(keys, sort=True):
        feature_set, corridor, horizon = identity
        chosen = signals.loc[
            signals["feature_set"].eq(feature_set)
            & signals["corridor"].eq(corridor)
            & signals["horizon"].eq(horizon)
        ]
        count = len(chosen)
        hit_rate = float(chosen["target"].mean()) if count else np.nan
        base_rate = float(all_scores["target"].mean())
        matched_rate = float(chosen["matched_week_hit_rate"].mean()) if count else np.nan
        duration = max(
            float((all_scores["timestamp"].max() - all_scores["timestamp"].min()).days) / 7,
            1 / 7,
        )
        yearly_lifts: list[float] = []
        for year, year_scores in all_scores.groupby("test_year"):
            year_signals = chosen.loc[chosen["test_year"].eq(year)]
            if len(year_signals) and float(year_scores["target"].mean()) > 0:
                yearly_lifts.append(
                    float(year_signals["target"].mean()) / float(year_scores["target"].mean())
                )
        rows.append(
            {
                "feature_set": feature_set,
                "corridor": corridor,
                "horizon": int(horizon),
                "oot_rows": len(all_scores),
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": (
                    hit_rate / matched_rate if count and matched_rate > 0 else np.nan
                ),
                "signals_per_week": count / duration,
                "regret_mean_bps": float(chosen["regret_bps"].mean()) if count else np.nan,
                "benefit_mean_bps": float(chosen["benefit_bps"].mean()) if count else np.nan,
                "worst_year_lift": min(yearly_lifts) if yearly_lifts else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(
    summary: pd.DataFrame,
    feature_set_names: list[str] | None = None,
    *,
    baseline_name: str = "market_only",
) -> pd.DataFrame:
    metrics = [
        "lift",
        "lift_vs_matched_random",
        "signals_per_week",
        "regret_mean_bps",
        "benefit_mean_bps",
        "worst_year_lift",
    ]
    keys = ["corridor", "horizon"]
    baseline = summary.loc[summary["feature_set"].eq(baseline_name), [*keys, *metrics]]
    rows: list[pd.DataFrame] = []
    candidates = feature_set_names or EXPECTED_FEATURE_SETS
    for feature_set in candidates:
        if feature_set == baseline_name:
            continue
        candidate = summary.loc[summary["feature_set"].eq(feature_set), [*keys, *metrics]]
        compared = candidate.merge(
            baseline, on=keys, suffixes=("_candidate", "_market"), validate="one_to_one"
        )
        compared.insert(0, "feature_set", feature_set)
        for metric in metrics:
            compared[f"delta_{metric}"] = (
                compared[f"{metric}_candidate"] - compared[f"{metric}_market"]
            )
        rows.append(compared)
    return pd.concat(rows, ignore_index=True)


def promotion_check(
    summary: pd.DataFrame,
    by_horizon: pd.DataFrame,
    pseudo_by_horizon: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, object]:
    rule = config["promotion_rule"]
    feature_set = str(rule["primary_feature_set"])
    sliced = paired_comparison(summary, list(config["feature_sets"]))
    sliced = sliced.loc[sliced["feature_set"].eq(feature_set)]

    def _horizon_deltas(frame: pd.DataFrame) -> pd.DataFrame:
        candidate = frame.loc[
            frame["feature_set"].eq(feature_set),
            ["horizon", "lift_vs_matched_random"],
        ]
        baseline = frame.loc[
            frame["feature_set"].eq("market_only"),
            ["horizon", "lift_vs_matched_random"],
        ]
        result = candidate.merge(baseline, on="horizon", suffixes=("_news", "_market"))
        result["delta"] = (
            result["lift_vs_matched_random_news"]
            - result["lift_vs_matched_random_market"]
        )
        return result

    aggregate = _horizon_deltas(by_horizon)
    pseudo = _horizon_deltas(pseudo_by_horizon)
    positive_slices = int(sliced["delta_lift_vs_matched_random"].gt(0).sum())
    positive_horizons = int(aggregate["delta"].gt(0).sum())
    minimum_frequency = float(sliced["signals_per_week_candidate"].min())
    pseudo_nonnegative = bool(pseudo["delta"].ge(0).all()) and len(pseudo) == len(config["horizons"])
    passed = (
        positive_slices >= int(rule["minimum_positive_same_week_slices_out_of_15"])
        and positive_horizons >= int(rule["minimum_positive_aggregate_horizons_out_of_3"])
        and minimum_frequency >= float(rule["minimum_signals_per_week"])
        and (pseudo_nonnegative or not bool(rule["require_nonnegative_pseudo_holdout_delta"]))
    )
    return {
        "primary_feature_set": feature_set,
        "positive_same_week_slices_out_of_15": positive_slices,
        "positive_aggregate_horizons_out_of_3": positive_horizons,
        "minimum_signals_per_week": minimum_frequency,
        "pseudo_holdout_nonnegative_on_all_horizons": pseudo_nonnegative,
        "promotion_rule_passed": bool(passed),
    }


def period_summary(
    scores: pd.DataFrame, signals: pd.DataFrame, years: list[int]
) -> pd.DataFrame:
    year_set = {int(value) for value in years}
    return summarize(
        scores.loc[scores["test_year"].isin(year_set)].copy(),
        signals.loc[signals["test_year"].isin(year_set)].copy(),
    )


def yearly_summary(scores: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in sorted(scores["test_year"].unique()):
        result = period_summary(scores, signals, [int(year)])
        result.insert(0, "test_year", int(year))
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def horizon_summary(scores: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Pool all five corridor exposures without pretending they are one time series."""

    rows: list[dict[str, object]] = []
    for (feature_set, horizon), all_scores in scores.groupby(["feature_set", "horizon"], sort=True):
        chosen = signals.loc[
            signals["feature_set"].eq(feature_set) & signals["horizon"].eq(horizon)
        ]
        count = len(chosen)
        hit_rate = float(chosen["target"].mean()) if count else np.nan
        base_rate = float(all_scores["target"].mean())
        matched_rate = float(chosen["matched_week_hit_rate"].mean()) if count else np.nan
        durations = [
            max(float((group["timestamp"].max() - group["timestamp"].min()).days) / 7, 1 / 7)
            for _, group in all_scores.groupby("corridor")
        ]
        rows.append(
            {
                "feature_set": feature_set,
                "horizon": int(horizon),
                "signals": count,
                "hit_rate": hit_rate,
                "baseline_hit_rate": base_rate,
                "matched_random_hit_rate": matched_rate,
                "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                "lift_vs_matched_random": (
                    hit_rate / matched_rate if count and matched_rate > 0 else np.nan
                ),
                "signals_per_week_per_corridor": count / sum(durations),
                "regret_mean_bps": float(chosen["regret_bps"].mean()) if count else np.nan,
                "benefit_mean_bps": float(chosen["benefit_bps"].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def nested_selection(
    scores: pd.DataFrame, signals: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose the news variant only from earlier completed OOT years."""

    rule = config["nested_selection"]
    selection_rows: list[dict[str, object]] = []
    chosen_scores: list[pd.DataFrame] = []
    chosen_signals: list[pd.DataFrame] = []
    for corridor in config["corridors"]:
        for horizon_value in config["horizons"]:
            horizon = int(horizon_value)
            for year_value in config["test_years"]:
                year = int(year_value)
                if year < int(rule["first_selection_year"]):
                    continue
                previous_scores = scores.loc[
                    scores["corridor"].eq(corridor)
                    & scores["horizon"].eq(horizon)
                    & scores["test_year"].lt(year)
                ]
                previous_signals = signals.loc[
                    signals["corridor"].eq(corridor)
                    & signals["horizon"].eq(horizon)
                    & signals["test_year"].lt(year)
                ]
                candidates = summarize(previous_scores, previous_signals)
                eligible = candidates.loc[
                    candidates["signals"].ge(int(rule["minimum_past_signals"]))
                ].copy()
                if eligible.empty:
                    selected_feature = str(rule["fallback_feature_set"])
                    past_metric = np.nan
                else:
                    metric = str(rule["metric"])
                    best = eligible.sort_values(
                        [metric, "lift", "signals"], ascending=False, kind="mergesort"
                    ).iloc[0]
                    selected_feature = str(best["feature_set"])
                    past_metric = float(best[metric])
                selection_rows.append(
                    {
                        "corridor": corridor,
                        "horizon": horizon,
                        "test_year": year,
                        "selected_feature_set": selected_feature,
                        "past_metric": past_metric,
                    }
                )
                current_scores = scores.loc[
                    scores["corridor"].eq(corridor)
                    & scores["horizon"].eq(horizon)
                    & scores["test_year"].eq(year)
                    & scores["feature_set"].eq(selected_feature)
                ].copy()
                current_scores["selected_feature_set"] = selected_feature
                current_scores["feature_set"] = "nested_past_choice"
                chosen_scores.append(current_scores)
                current = signals.loc[
                    signals["corridor"].eq(corridor)
                    & signals["horizon"].eq(horizon)
                    & signals["test_year"].eq(year)
                    & signals["feature_set"].eq(selected_feature)
                ].copy()
                current["selected_feature_set"] = selected_feature
                current["feature_set"] = "nested_past_choice"
                chosen_signals.append(current)
    selected_scores = pd.concat(chosen_scores, ignore_index=True) if chosen_scores else pd.DataFrame()
    selected = pd.concat(chosen_signals, ignore_index=True) if chosen_signals else pd.DataFrame()
    return pd.DataFrame(selection_rows), selected_scores, selected


def run(
    *,
    config_path: Path | str = Path("configs/news_robust_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_robust"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    news_path = Path(config["news_input"])
    frame = prepare_frame(pd.read_csv(input_path), pd.read_csv(news_path), config)
    scores, folds, importance = fit_oot_scores(frame, config)
    signals = apply_continuous_policy(scores, config)
    summary = summarize(scores, signals)
    per_year = yearly_summary(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    comparison = paired_comparison(summary)
    pseudo_holdout = period_summary(scores, signals, list(config["pseudo_holdout_years"]))
    pseudo_by_horizon = horizon_summary(
        scores.loc[scores["test_year"].isin(config["pseudo_holdout_years"])],
        signals.loc[signals["test_year"].isin(config["pseudo_holdout_years"])],
    )
    promotion = promotion_check(summary, by_horizon, pseudo_by_horizon, config)
    selections, nested_scores, nested_signals = nested_selection(scores, signals, config)
    nested_summary = summarize(nested_scores, nested_signals)
    nested_horizon_summary = horizon_summary(nested_scores, nested_signals)

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "oot_scores.csv.gz", index=False, compression="gzip")
    folds.to_csv(output / "folds.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pseudo_holdout.to_csv(output / "pseudo_holdout_2025_2026.csv", index=False)
    pseudo_by_horizon.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    (output / "promotion_check.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selections.to_csv(output / "nested_selections.csv", index=False)
    nested_scores.to_csv(output / "nested_scores.csv", index=False)
    nested_signals.to_csv(output / "nested_signals.csv", index=False)
    nested_summary.to_csv(output / "nested_summary.csv", index=False)
    nested_horizon_summary.to_csv(output / "nested_horizon_summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": _sha256(Path(config_path)),
        "input_sha256": _sha256(input_path),
        "news_input_sha256": _sha256(news_path),
        "joined_rows": len(frame),
        "news_feature_count": len([column for column in frame if column.startswith("news__")]),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": (
            "News variants are exploratory on reviewed history. Three seeds, rolling windows and "
            "nested selection reduce overfitting risk but do not replace a genuinely new hold-out."
        ),
    }
    (output / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_robust_experiment.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news_robust")
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
