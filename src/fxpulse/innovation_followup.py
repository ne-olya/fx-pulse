"""Targeted second innovation wave and stricter dependence-aware audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fxpulse.adaptive_threshold import adaptive_candidates
from fxpulse.next_hypotheses import _apply_policy, _make_model, _purged_train
from fxpulse.robust_innovation_experiment import (
    _load_winner_rows,
    _target_frame_for_winner,
    _weekly_winner_table,
    delayed_dynamic_average,
    load_config as load_wave_1_config,
    prepare_frame,
    summarize_folds,
)


EXPECTED_VARIANTS = [
    "continuous_tb_catboost",
    "continuous_rolling_4y",
    "recency_dynamic_average",
    "robust_three_mean",
    "robust_three_unanimous",
    "leave_one_corridor_consensus",
    "causal_stacking",
    "economic_loss_catboost",
]

KEY_COLUMNS = [
    "timestamp",
    "corridor",
    "week",
    "target",
    "regret_bps",
    "benefit_bps",
    "matched_week_hit_rate",
    "horizon",
    "test_year",
]


def load_config(path: Path | str = Path("configs/innovation_followup.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("innovation follow-up must be preregistered schema_version 1")
    if config.get("variants") != EXPECTED_VARIANTS:
        raise ValueError("follow-up variants differ from the implementation")
    if set(config.get("corridors", ())) != {"AMD", "KGS", "KZT", "TJS", "UZS"}:
        raise ValueError("all five corridors are required")
    return config


def causal_percentile(
    values: pd.Series,
    *,
    lookback: int,
    minimum_history: int = 20,
) -> pd.Series:
    """Rank each score against strictly earlier scores only."""

    raw = values.to_numpy(dtype=float)
    ranked = np.full(len(raw), np.nan)
    for index, value in enumerate(raw):
        past = raw[max(0, index - int(lookback)) : index]
        past = past[np.isfinite(past)]
        if np.isfinite(value) and len(past) >= int(minimum_history):
            ranked[index] = float(np.mean(past <= value))
    return pd.Series(ranked, index=values.index)


def _wide_wave_1_scores(config: dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(config["wave_1_scores"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise")
    required = set(KEY_COLUMNS) | {"variant", "score"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"wave-1 scores lack {sorted(missing)}")
    source = raw.loc[raw["variant"].isin(config["source_variants"])].copy()
    wide = source.pivot(index=KEY_COLUMNS, columns="variant", values="score").reset_index()
    missing_variants = set(config["source_variants"]) - set(wide)
    if missing_variants:
        raise ValueError(f"wave-1 score variants missing: {sorted(missing_variants)}")
    return wide.sort_values(["corridor", "horizon", "timestamp"], kind="mergesort").reset_index(drop=True)


def _add_score_combinations(wide: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = wide.copy()
    data["continuous_tb_catboost"] = data["tb_catboost"]
    data["continuous_rolling_4y"] = data["rolling_4y_catboost"]
    data["recency_dynamic_average"] = np.nan
    data["robust_three_mean"] = data[list(config["robust_three"])].mean(axis=1)
    data["robust_three_unanimous"] = np.nan
    data["_rolling_rank"] = np.nan

    for (_, horizon), indices in data.groupby(["corridor", "horizon"], sort=False).groups.items():
        ordered_indices = data.loc[indices].sort_values("timestamp").index
        group = data.loc[ordered_indices]
        data.loc[ordered_indices, "recency_dynamic_average"] = delayed_dynamic_average(
            group[["tb_catboost", "rolling_4y_catboost"]].to_numpy(dtype=float),
            group["target"].to_numpy(dtype=float),
            delay=int(horizon),
            eta=float(config["dynamic_model_average"]["eta"]),
        )
        ranks = pd.DataFrame(index=ordered_indices)
        for source in config["robust_three"]:
            ranks[source] = causal_percentile(
                group[source],
                lookback=int(config["causal_rank_lookback"]),
            ).to_numpy()
        data.loc[ordered_indices, "robust_three_unanimous"] = ranks.min(axis=1).to_numpy()
        data.loc[ordered_indices, "_rolling_rank"] = causal_percentile(
            group["rolling_4y_catboost"],
            lookback=int(config["causal_rank_lookback"]),
        ).to_numpy()

    data["leave_one_corridor_consensus"] = np.nan
    for (_, _), indices in data.groupby(["horizon", "timestamp"], sort=False).groups.items():
        current = data.loc[indices, "_rolling_rank"]
        for index in indices:
            others = current.drop(index).dropna()
            own = float(data.at[index, "_rolling_rank"])
            if np.isfinite(own) and not others.empty:
                data.at[index, "leave_one_corridor_consensus"] = 0.5 * own + 0.5 * float(others.mean())
    return data


def _add_causal_stacking(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = data.copy()
    result["causal_stacking"] = np.nan
    columns = list(config["stacking_sources"])
    for (corridor, horizon), indices in result.groupby(["corridor", "horizon"], sort=False).groups.items():
        group = result.loc[indices].sort_values("timestamp")
        for year in config["stacking_test_years"]:
            train = group.loc[group["test_year"] < int(year)].copy()
            test = group.loc[group["test_year"].eq(int(year))].copy()
            if len(train) < 200 or test.empty or train["target"].nunique() < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(C=0.25, class_weight="balanced", max_iter=1_000),
            )
            model.fit(train[columns], train["target"])
            result.loc[test.index, "causal_stacking"] = model.predict_proba(test[columns])[:, 1]
    return result


def economic_loss_scores(config: dict[str, Any]) -> pd.DataFrame:
    wave_1 = load_wave_1_config()
    frame = prepare_frame(pd.read_csv(config["input"]), wave_1)
    prefixes = tuple(wave_1["feature_prefixes"])
    columns = [column for column in frame if column.startswith(prefixes)]
    rows: list[pd.DataFrame] = []
    loss_cfg = config["economic_loss"]
    for corridor in config["corridors"]:
        corridor_data = frame.loc[frame["corridor"].eq(corridor)].copy()
        for horizon in config["horizons"]:
            h = int(horizon)
            outcome = f"outcome__regret_{h}"
            benefit = f"outcome__benefit_{h}"
            label = f"label__tb_{h}"
            usable = corridor_data.loc[corridor_data[outcome].notna() & corridor_data[label].notna()].copy()
            usable["target"] = usable[outcome].le(float(config["evaluation_tolerance_bps"])).astype(int)
            usable["regret_bps"] = usable[outcome]
            usable["benefit_bps"] = usable[benefit]
            usable[label] = usable[label].astype(int)
            for year in config["test_years"]:
                start = pd.Timestamp(year=int(year), month=1, day=1)
                end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
                train = _purged_train(usable, start, h)
                test = usable.loc[usable["timestamp"].between(start, end, inclusive="left")].copy()
                if len(train) < 500 or len(test) < 20 or train[label].nunique() < 2:
                    continue
                bad_weight = 1 + (
                    train["regret_bps"].clip(lower=0)
                    / 100
                    * float(loss_cfg["bad_day_weight_per_100_bps"])
                )
                bad_weight = bad_weight.clip(upper=float(loss_cfg["maximum_bad_day_weight"]))
                weights = np.where(train[label].eq(0), bad_weight, 1.0)
                model = _make_model(
                    "catboost",
                    iterations=int(loss_cfg["model_iterations"]),
                    seed=int(config["random_seed"]) + int(year) + h,
                )
                step = model.steps[-1][0]
                model.fit(train[columns], train[label], **{f"{step}__sample_weight": weights})
                test["score"] = model.predict_proba(test[columns])[:, 1]
                test["week"] = test["timestamp"].dt.to_period("W").astype(str)
                weekly_rate = test.groupby("week", sort=False)["target"].mean()
                test["matched_week_hit_rate"] = test["week"].map(weekly_rate).astype(float)
                test["horizon"] = h
                test["test_year"] = int(year)
                rows.append(test[[*KEY_COLUMNS, "score"]])
            print(f"economic-loss completed {corridor} h={h}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_followup_scores(config: dict[str, Any]) -> pd.DataFrame:
    data = _add_score_combinations(_wide_wave_1_scores(config), config)
    data = _add_causal_stacking(data, config)
    economic = economic_loss_scores(config).rename(columns={"score": "economic_loss_catboost"})
    merge_keys = ["timestamp", "corridor", "horizon", "test_year"]
    data = data.merge(
        economic[[*merge_keys, "economic_loss_catboost"]],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    long = data.melt(
        id_vars=KEY_COLUMNS,
        value_vars=config["variants"],
        var_name="variant",
        value_name="score",
    )
    return long.loc[long["score"].notna()].sort_values(
        ["variant", "corridor", "horizon", "timestamp"], kind="mergesort"
    )


def evaluate_followup_scores(
    scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    policy = config["adaptive_policy"]
    for (variant, corridor, horizon), group in scores.groupby(
        ["variant", "corridor", "horizon"], sort=True
    ):
        ordered = group.sort_values("timestamp", kind="mergesort").copy()
        candidates = adaptive_candidates(
            ordered,
            share=float(policy["top_score_share"]),
            lookback=int(policy["lookback_observations"]),
            minimum_history=int(policy["minimum_history_observations"]),
        )
        selected = _apply_policy(
            candidates,
            cooldown_days=int(policy["cooldown_days"]),
            weekly_cap=int(policy["weekly_cap"]),
        )
        for year, test in ordered.groupby("test_year", sort=True):
            year_selected = selected.loc[selected["test_year"].eq(year)].copy()
            count = len(year_selected)
            base_rate = float(test["target"].mean())
            hit_rate = float(year_selected["target"].mean()) if count else np.nan
            duration = max(float((test["timestamp"].max() - test["timestamp"].min()).days) / 7, 1 / 7)
            folds.append(
                {
                    "variant": variant,
                    "corridor": corridor,
                    "horizon": int(horizon),
                    "test_year": int(year),
                    "test_count": len(test),
                    "test_hits": int(test["target"].sum()),
                    "signal_count": count,
                    "signal_hits": int(year_selected["target"].sum()) if count else 0,
                    "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                    "matched_expected_hits": float(year_selected["matched_week_hit_rate"].sum()) if count else 0.0,
                    "regret_sum_bps": float(year_selected["regret_bps"].sum()) if count else 0.0,
                    "benefit_sum_bps": float(year_selected["benefit_bps"].sum()) if count else 0.0,
                    "duration_weeks": duration,
                }
            )
            if count:
                signal_frames.append(year_selected.copy())
    return pd.DataFrame(folds), pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()


def best_by_horizon(summary: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    for _, group in summary.groupby("horizon", sort=True):
        eligible = group.loc[
            group["signals"].ge(150)
            & group["signals_per_week"].between(0.8, 1.4, inclusive="both")
        ]
        pool = eligible if not eligible.empty else group
        chosen.append(
            pool.sort_values(
                ["lift_vs_matched_random", "lift", "variant", "corridor"],
                ascending=[False, False, True, True],
                kind="mergesort",
            ).head(1)
        )
    return pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()


def nested_followup_selection(folds: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select a fixed follow-up variant on prior OOT years only."""

    rows: list[dict[str, object]] = []
    for corridor in config["corridors"]:
        for horizon in config["horizons"]:
            scope = folds.loc[
                folds["corridor"].eq(corridor) & folds["horizon"].eq(int(horizon))
            ]
            for test_year in [2024, 2025, 2026]:
                past = scope.loc[scope["test_year"] < test_year]
                available = (
                    past.groupby("variant")["test_year"].nunique().loc[lambda value: value.ge(2)].index
                )
                past = past.loc[past["variant"].isin(available)]
                if past.empty:
                    continue
                summary = summarize_folds(past)
                eligible = summary.loc[
                    summary["signals_per_week"].between(0.8, 1.4, inclusive="both")
                ]
                pool = eligible if not eligible.empty else summary
                chosen = pool.sort_values(
                    ["lift_vs_matched_random", "lift", "variant"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).iloc[0]
                test = scope.loc[
                    scope["test_year"].eq(test_year) & scope["variant"].eq(chosen["variant"])
                ]
                if test.empty:
                    continue
                value = test.iloc[0]
                rows.append(
                    {
                        "corridor": corridor,
                        "horizon": int(horizon),
                        "test_year": int(test_year),
                        "selected_variant": chosen["variant"],
                        "selection_years": ",".join(map(str, sorted(past["test_year"].unique()))),
                        "signal_count": int(value["signal_count"]),
                        "signal_hits": int(value["signal_hits"]),
                        "test_count": int(value["test_count"]),
                        "test_hits": int(value["test_hits"]),
                        "matched_expected_hits": float(value["matched_expected_hits"]),
                        "duration_weeks": float(value["duration_weeks"]),
                        "lift": float(value["lift"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_nested(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, group in nested.groupby("horizon", sort=True):
        signals = int(group["signal_count"].sum())
        hits = int(group["signal_hits"].sum())
        base_count = int(group["test_count"].sum())
        base_hits = int(group["test_hits"].sum())
        expected = float(group["matched_expected_hits"].sum())
        rows.append(
            {
                "horizon": int(horizon),
                "folds": len(group),
                "signals": signals,
                "lift": (hits / signals) / (base_hits / base_count),
                "lift_vs_matched_random": hits / expected,
                "signals_per_week": signals / float(group["duration_weeks"].sum()),
                "worst_year_lift": float(group["lift"].min()),
            }
        )
    return pd.DataFrame(rows)


def moving_block_bootstrap(
    weekly: pd.DataFrame,
    *,
    samples: int,
    block_weeks: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = weekly[
        ["signal_count", "signal_hits", "matched_expected_hits", "base_count", "base_hits"]
    ].to_numpy(dtype=float)
    n = len(values)
    block_count = math.ceil(n / int(block_weeks))
    starts = rng.integers(0, n, size=(int(samples), block_count))
    offsets = np.arange(int(block_weeks))
    indices = ((starts[..., None] + offsets) % n).reshape(int(samples), -1)[:, :n]
    sampled = values[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (sampled[:, 1] / sampled[:, 0]) / (sampled[:, 4] / sampled[:, 3])
        matched = sampled[:, 1] / sampled[:, 2]
    delta = values[:, 1] - values[:, 2]
    observed_delta = float(delta.mean())
    centered = delta - delta.mean()
    null_means = centered[indices].mean(axis=1)
    p_value = (1 + int(np.sum(null_means >= observed_delta))) / (int(samples) + 1)
    return {
        "block_raw_lift_low": float(np.nanquantile(raw, 0.025)),
        "block_raw_lift_high": float(np.nanquantile(raw, 0.975)),
        "block_matched_lift_low": float(np.nanquantile(matched, 0.025)),
        "block_matched_lift_high": float(np.nanquantile(matched, 0.975)),
        "block_matched_p_value": p_value,
    }


def _deoverlap_signals(signals: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    ordered = signals.sort_values("timestamp", kind="mergesort")
    keep: list[int] = []
    last: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if last is None or (timestamp - last).days >= int(cooldown_days):
            keep.append(index)
            last = timestamp
    return ordered.loc[keep].copy()


def strict_winner_followup(config: dict[str, Any]) -> pd.DataFrame:
    wave_1 = load_wave_1_config()
    rows: list[dict[str, object]] = []
    trials = int(config["bootstrap"]["conservative_previous_trials"])
    for index, spec in enumerate(wave_1["current_winners"]):
        folds, signals = _load_winner_rows(spec)
        target = _target_frame_for_winner(spec, wave_1)
        weekly = _weekly_winner_table(target, signals)
        block = moving_block_bootstrap(
            weekly,
            samples=int(config["bootstrap"]["samples"]),
            block_weeks=int(config["bootstrap"]["moving_block_weeks"]),
            seed=int(config["random_seed"]) + index,
        )
        cooldown = int(config["deoverlap_cooldown_calendar_days_by_horizon"][str(spec["horizon"])])
        deoverlapped = _deoverlap_signals(signals, cooldown)
        base_rate = float(target["target"].mean())
        duration = float(folds["duration_weeks"].sum())
        matched_expected = float(deoverlapped["matched_week_hit_rate"].sum())
        rows.append(
            {
                "name": spec["name"],
                "corridor": spec["corridor"],
                "horizon": int(spec["horizon"]),
                **block,
                "block_p_bonferroni_13379": min(1.0, block["block_matched_p_value"] * trials),
                "deoverlap_signals": len(deoverlapped),
                "deoverlap_lift": float(deoverlapped["target"].mean()) / base_rate,
                "deoverlap_matched_lift": float(deoverlapped["target"].sum()) / matched_expected,
                "deoverlap_signals_per_week": len(deoverlapped) / duration,
            }
        )
    return pd.DataFrame(rows)


def strict_followup_best(
    best: pd.DataFrame,
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Audit post-selected wave-2 winners with a 120-comparison bound."""

    rows: list[dict[str, object]] = []
    comparisons = len(config["variants"]) * len(config["corridors"]) * len(config["horizons"])
    for index, selected in best.iterrows():
        mask = (
            scores["variant"].eq(selected["variant"])
            & scores["corridor"].eq(selected["corridor"])
            & scores["horizon"].eq(int(selected["horizon"]))
        )
        target = scores.loc[mask, ["timestamp", "test_year", "week", "target"]].copy()
        signal_mask = (
            signals["variant"].eq(selected["variant"])
            & signals["corridor"].eq(selected["corridor"])
            & signals["horizon"].eq(int(selected["horizon"]))
        )
        chosen_signals = signals.loc[signal_mask].copy()
        weekly = _weekly_winner_table(target, chosen_signals)
        block = moving_block_bootstrap(
            weekly,
            samples=int(config["bootstrap"]["samples"]),
            block_weeks=int(config["bootstrap"]["moving_block_weeks"]),
            seed=int(config["random_seed"]) + 1_000 + int(index),
        )
        cooldown = int(
            config["deoverlap_cooldown_calendar_days_by_horizon"][str(int(selected["horizon"]))]
        )
        deoverlapped = _deoverlap_signals(chosen_signals, cooldown)
        expected = float(deoverlapped["matched_week_hit_rate"].sum())
        duration = sum(
            max(float((group["timestamp"].max() - group["timestamp"].min()).days) / 7, 1 / 7)
            for _, group in scores.loc[mask].groupby("test_year", sort=True)
        )
        rows.append(
            {
                "variant": selected["variant"],
                "corridor": selected["corridor"],
                "horizon": int(selected["horizon"]),
                **block,
                "block_p_bonferroni_120": min(1.0, block["block_matched_p_value"] * comparisons),
                "deoverlap_signals": len(deoverlapped),
                "deoverlap_lift": float(deoverlapped["target"].mean()) / float(target["target"].mean()),
                "deoverlap_matched_lift": float(deoverlapped["target"].sum()) / expected,
                "deoverlap_signals_per_week": len(deoverlapped) / duration,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    config_path: Path | str = Path("configs/innovation_followup.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/innovation_followup"),
) -> dict[str, object]:
    config = load_config(config_path)
    scores = build_followup_scores(config)
    folds, signals = evaluate_followup_scores(scores, config)
    summary = summarize_folds(folds)
    pseudo = summarize_folds(folds, years=[2025, 2026])
    best = best_by_horizon(summary)
    strict = strict_winner_followup(config)
    nested = nested_followup_selection(folds, config)
    nested_summary = summarize_nested(nested)
    best_strict = strict_followup_best(best, scores, signals, config)

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv.gz", index=False, compression="gzip")
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout.csv", index=False)
    best.to_csv(output / "best_by_horizon.csv", index=False)
    strict.to_csv(output / "strict_winner_followup.csv", index=False)
    nested.to_csv(output / "nested_selection.csv", index=False)
    nested_summary.to_csv(output / "nested_selection_summary.csv", index=False)
    best_strict.to_csv(output / "best_strict_audit.csv", index=False)

    config_path = Path(config_path)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "wave_1_scores_sha256": hashlib.sha256(Path(config["wave_1_scores"]).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(Path(config["input"]).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "score_rows": len(scores),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Targeted post-wave-1 follow-up; no result is a genuinely untouched holdout.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/innovation_followup.json"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/innovation_followup"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
