"""Freeze and audit the post-experiment UZS research candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.innovation_followup import (
    _deoverlap_signals,
    _weekly_winner_table,
    moving_block_bootstrap,
)
from fxpulse.robust_innovation_experiment import circular_shift_placebo
from fxpulse.uzs_final_experiment import _paired_bootstrap, tolerance_sensitivity


KEYS = ["feature_set", "model_variant", "policy"]


def _load_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mask(frame: pd.DataFrame, identity: dict[str, str]) -> pd.Series:
    return np.logical_and.reduce([frame[key].eq(identity[key]) for key in KEYS])


def _wide_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "same_week_lift",
        "raw_lift",
        "signals",
        "signals_per_week",
        "weeks_covered_share",
        "worst_year_same_week_lift",
        "mean_regret_bps",
        "hit_rate",
        "feature_count",
    ]
    wide = summary.pivot(index=KEYS, columns="period", values=metrics).reset_index()
    wide.columns = ["__".join(str(item) for item in column if item) for column in wide.columns]
    return wide


def reproduce_selection(
    summary: pd.DataFrame,
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    wide = _wide_summary(summary)
    top = wide.sort_values(
        ["same_week_lift__development", "raw_lift__development"],
        ascending=False,
        kind="mergesort",
    ).head(10).copy()
    eligible = top.loc[
        top["same_week_lift__development"].ge(1.20)
        & top["same_week_lift__pseudo_holdout"].ge(1.20)
        & top["raw_lift__pseudo_holdout"].ge(1.20)
        & top["worst_year_same_week_lift__all"].ge(1.10)
    ].copy()
    eligible["stability_score"] = eligible["worst_year_same_week_lift__all"]
    selected = eligible.sort_values(
        ["stability_score", "same_week_lift__development", "mean_regret_bps__all", "feature_count__all"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    expected = {key: str(frozen[key]) for key in KEYS}
    actual = {key: str(selected[key]) for key in KEYS}
    if actual != expected:
        raise RuntimeError(f"frozen identity {expected} does not reproduce; got {actual}")
    return top, selected


def _periods(config: dict[str, Any]) -> list[tuple[str, list[int]]]:
    return [
        ("development", list(map(int, config["development_years"]))),
        ("pseudo_holdout", list(map(int, config["pseudo_holdout_years"]))),
        ("all", list(map(int, config["test_years"]))),
    ]


def robustness(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    identity: dict[str, str],
    config: dict[str, Any],
    *,
    signal_tier: str | None = None,
) -> pd.DataFrame:
    test = scores.loc[_mask(scores, identity)].copy()
    chosen = signals.loc[_mask(signals, identity)].copy()
    if signal_tier is not None:
        chosen = chosen.loc[chosen["signal_tier"].eq(signal_tier)].copy()
    settings = config["robustness"]
    rows: list[dict[str, object]] = []
    for index, (period, years) in enumerate(_periods(config)):
        current_test = test.loc[test["test_year"].isin(years)]
        current = chosen.loc[chosen["test_year"].isin(years)]
        bootstrap = moving_block_bootstrap(
            _weekly_winner_table(current_test, current),
            samples=int(settings["bootstrap_samples"]),
            block_weeks=int(settings["moving_block_weeks"]),
            seed=int(config["ensemble_seeds"][0]) + 300 + index,
        )
        placebo = circular_shift_placebo(
            current_test,
            current,
            samples=int(settings["placebo_samples"]),
            seed=int(config["ensemble_seeds"][0]) + 400 + index,
        )
        deoverlap = _deoverlap_signals(current, int(settings["deoverlap_cooldown_days"]))
        expected = float(deoverlap["matched_week_hit_rate"].sum())
        duration = sum(
            max(float((part["timestamp"].max() - part["timestamp"].min()).days) / 7, 1 / 7)
            for _, part in current_test.groupby("test_year", sort=True)
        )
        rows.append(
            {
                "period": period,
                **bootstrap,
                **placebo,
                "deoverlap_signals": len(deoverlap),
                "deoverlap_raw_lift": float(deoverlap["target"].mean())
                / float(current_test["target"].mean()),
                "deoverlap_same_week_lift": float(deoverlap["target"].sum()) / expected,
                "deoverlap_signals_per_week": len(deoverlap) / duration,
            }
        )
    return pd.DataFrame(rows)


def tier_summary(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    identity: dict[str, str],
    config: dict[str, Any],
) -> pd.DataFrame:
    test = scores.loc[_mask(scores, identity)].copy()
    chosen = signals.loc[_mask(signals, identity)].copy()
    rows: list[dict[str, object]] = []
    for period, years in _periods(config):
        current_test = test.loc[test["test_year"].isin(years)]
        for tier, current in chosen.loc[chosen["test_year"].isin(years)].groupby("signal_tier", sort=True):
            count = len(current)
            base_rate = float(current_test["target"].mean())
            expected = float(current["matched_week_hit_rate"].sum())
            rows.append(
                {
                    "period": period,
                    "signal_tier": tier,
                    "signals": count,
                    "hit_rate": float(current["target"].mean()),
                    "raw_lift": float(current["target"].mean()) / base_rate,
                    "same_week_lift": float(current["target"].sum()) / expected,
                    "mean_regret_bps": float(current["regret_bps"].mean()),
                }
            )
    return pd.DataFrame(rows)


def news_ablation(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    identity: dict[str, str],
    config: dict[str, Any],
) -> pd.DataFrame:
    baseline = {**identity, "feature_set": "uzs_national"}
    rows: list[dict[str, object]] = []
    settings = config["robustness"]
    for index, (period, years) in enumerate(_periods(config)):
        base_scores = scores.loc[_mask(scores, baseline) & scores["test_year"].isin(years)]
        base_signals = signals.loc[_mask(signals, baseline) & signals["test_year"].isin(years)]
        news_scores = scores.loc[_mask(scores, identity) & scores["test_year"].isin(years)]
        news_signals = signals.loc[_mask(signals, identity) & signals["test_year"].isin(years)]
        rows.append(
            {
                "period": period,
                **_paired_bootstrap(
                    _weekly_winner_table(base_scores, base_signals),
                    _weekly_winner_table(news_scores, news_signals),
                    samples=int(settings["bootstrap_samples"]),
                    block_weeks=int(settings["moving_block_weeks"]),
                    seed=int(config["ensemble_seeds"][0]) + 500 + index,
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_regret(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    identity: dict[str, str],
    config: dict[str, Any],
    *,
    signal_tier: str | None = None,
) -> pd.DataFrame:
    test = scores.loc[_mask(scores, identity)].copy()
    chosen = signals.loc[_mask(signals, identity)].copy()
    if signal_tier is not None:
        chosen = chosen.loc[chosen["signal_tier"].eq(signal_tier)].copy()
    week_regret = test.groupby(["test_year", "week"])["regret_bps"].mean()
    chosen["matched_week_mean_regret_bps"] = [
        float(week_regret.loc[(year, week)])
        for year, week in zip(chosen["test_year"], chosen["week"], strict=True)
    ]
    rows = []
    for period, years in _periods(config):
        current_test = test.loc[test["test_year"].isin(years)]
        current = chosen.loc[chosen["test_year"].isin(years)]
        selected = float(current["regret_bps"].mean())
        matched = float(current["matched_week_mean_regret_bps"].mean())
        rows.append(
            {
                "period": period,
                "all_days_mean_regret_bps": float(current_test["regret_bps"].mean()),
                "matched_week_mean_regret_bps": matched,
                "selected_mean_regret_bps": selected,
                "reduction_vs_matched_week_bps": matched - selected,
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    experiment_dir: Path | str = Path("artifacts/uzs_final_experiment_20260905"),
    experiment_config: Path | str = Path("configs/uzs_final_experiment.json"),
    frozen_config: Path | str = Path("configs/uzs_final_frozen_candidate.json"),
) -> dict[str, Any]:
    root = Path(experiment_dir)
    config = _load_json(experiment_config)
    frozen = _load_json(frozen_config)
    identity = {key: str(frozen[key]) for key in KEYS}
    summary = pd.read_csv(root / "summary.csv")
    scores = pd.read_csv(root / "policy_scores.csv.gz", parse_dates=["timestamp"])
    signals = pd.read_csv(root / "signals.csv.gz", parse_dates=["timestamp"])
    top, selected = reproduce_selection(summary, frozen)
    strict = robustness(scores, signals, identity, config)
    strong_strict = robustness(
        scores,
        signals,
        identity,
        config,
        signal_tier="strong_prediction",
    )
    tiers = tier_summary(scores, signals, identity, config)
    ablation = news_ablation(scores, signals, identity, config)
    regret = matched_regret(scores, signals, identity, config)
    strong_regret = matched_regret(
        scores,
        signals,
        identity,
        config,
        signal_tier="strong_prediction",
    )
    sensitivity = tolerance_sensitivity(scores, signals, pd.Series(identity), config)
    top.to_csv(root / "selection_top10.csv", index=False)
    strict.to_csv(root / "frozen_candidate_robustness.csv", index=False)
    strong_strict.to_csv(root / "frozen_candidate_strong_robustness.csv", index=False)
    tiers.to_csv(root / "frozen_candidate_signal_tiers.csv", index=False)
    ablation.to_csv(root / "frozen_candidate_news_ablation.csv", index=False)
    regret.to_csv(root / "frozen_candidate_regret.csv", index=False)
    strong_regret.to_csv(root / "frozen_candidate_strong_regret.csv", index=False)
    sensitivity.to_csv(root / "frozen_candidate_tolerance_sensitivity.csv", index=False)
    result = {
        "registered_winner": {
            "feature_set": "uzs_national",
            "model_variant": "logistic_tb",
            "policy": "selective_high_confidence",
            "decision": "not_promoted_due_to_pseudo_holdout_instability",
        },
        "frozen_research_candidate": identity,
        "primary_metric_scope": str(frozen["primary_signal_tier"]),
        "fallback_metric_status": str(frozen["fallback_metric_status"]),
        "selection_kind": "post_hoc_stability_rule_after_registered_winner_failed",
        "selection_stability_score": float(selected["stability_score"]),
        "future_change_policy": "no feature, threshold or hyperparameter changes before future or bank-held evaluation",
    }
    (root / "final_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts/uzs_final_experiment_20260905"))
    args = parser.parse_args()
    run(experiment_dir=args.experiment_dir)


if __name__ == "__main__":
    main()
