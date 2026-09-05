"""Test LLM-derived geopolitical-news features in the final FX CatBoost."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.news_robust_experiment import (
    apply_continuous_policy,
    fit_oot_scores,
    horizon_summary,
    nested_selection,
    paired_comparison,
    period_summary,
    promotion_check,
    summarize,
    yearly_summary,
)

EXPECTED_FEATURE_SETS = [
    "market_only",
    "plus_daily_ai_gpr",
    "plus_country_ai_gpr",
    "plus_bilateral_ai_gpr",
    "plus_all_ai_gpr",
    "plus_liquidity_all_ai_gpr",
]
MARKET_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")


def load_config(path: Path | str = Path("configs/ai_gpr_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("AI-GPR experiment must be preregistered schema_version 1")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("AI-GPR feature sets differ from the implementation")
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_frame(
    market: pd.DataFrame, external: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    market = market.copy()
    external = external.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], errors="raise").dt.normalize()
    external["feature_date"] = pd.to_datetime(
        external["feature_date"], errors="raise"
    ).dt.normalize()
    market["corridor"] = market["corridor"].astype(str).str.upper()
    external["corridor"] = external["corridor"].astype(str).str.upper()
    feature_columns = [column for column in external if column.startswith("aigpr__")]
    if not feature_columns:
        raise ValueError("AI-GPR input contains no aigpr__ feature")
    if external.duplicated(["feature_date", "corridor"]).any():
        raise ValueError("AI-GPR input must have one row per feature date and corridor")
    result = market.loc[market["corridor"].isin(config["corridors"])].merge(
        external[["feature_date", "corridor", *feature_columns]],
        left_on=["timestamp", "corridor"],
        right_on=["feature_date", "corridor"],
        how="inner",
        validate="many_to_one",
    )
    return result.drop(columns="feature_date").sort_values(
        ["corridor", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(MARKET_PREFIXES)]
    liquidity = [column for column in frame if column.startswith("liquidity__")]
    daily = [column for column in frame if column.startswith("aigpr__daily_")]
    country = [
        column
        for column in frame
        if column.startswith(("aigpr__country_", "aigpr__cross_"))
    ]
    bilateral = [column for column in frame if column.startswith("aigpr__bilateral_")]
    external = [column for column in frame if column.startswith("aigpr__")]
    if not all((market, daily, country, bilateral, external)):
        raise ValueError("AI-GPR feature family is empty")
    return {
        "market_only": market,
        "plus_daily_ai_gpr": [*market, *daily],
        "plus_country_ai_gpr": [*market, *country],
        "plus_bilateral_ai_gpr": [*market, *bilateral],
        "plus_all_ai_gpr": [*market, *external],
        "plus_liquidity_all_ai_gpr": [*market, *liquidity, *external],
    }


def run(
    *,
    config_path: Path | str = Path("configs/ai_gpr_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/ai_gpr"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    feature_path = Path(config["external_feature_input"])
    frame = prepare_frame(pd.read_csv(input_path), pd.read_csv(feature_path), config)
    scores, folds, importance = fit_oot_scores(frame, config, sets=feature_sets(frame))
    signals = apply_continuous_policy(scores, config)
    summary_frame = summarize(scores, signals)
    per_year = yearly_summary(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    comparison = paired_comparison(summary_frame, list(config["feature_sets"]))
    pseudo_mask_scores = scores["test_year"].isin(config["pseudo_holdout_years"])
    pseudo_mask_signals = signals["test_year"].isin(config["pseudo_holdout_years"])
    pseudo = period_summary(scores, signals, list(config["pseudo_holdout_years"]))
    pseudo_by_horizon = horizon_summary(
        scores.loc[pseudo_mask_scores], signals.loc[pseudo_mask_signals]
    )
    promotion = promotion_check(summary_frame, by_horizon, pseudo_by_horizon, config)
    selections, nested_scores, nested_signals = nested_selection(scores, signals, config)
    nested_summary = summarize(nested_scores, nested_signals)
    nested_by_horizon = horizon_summary(nested_scores, nested_signals)

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "oot_scores.csv.gz", index=False, compression="gzip")
    folds.to_csv(output / "folds.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_2025_2026.csv", index=False)
    pseudo_by_horizon.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    selections.to_csv(output / "nested_selections.csv", index=False)
    nested_scores.to_csv(output / "nested_scores.csv", index=False)
    nested_signals.to_csv(output / "nested_signals.csv", index=False)
    nested_summary.to_csv(output / "nested_summary.csv", index=False)
    nested_by_horizon.to_csv(output / "nested_horizon_summary.csv", index=False)
    (output / "promotion_check.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "config_path": str(config_path),
        "config_sha256": _sha256(Path(config_path)),
        "input_sha256": _sha256(input_path),
        "external_feature_sha256": _sha256(feature_path),
        "joined_rows": len(frame),
        "feature_count": len([column for column in frame if column.startswith("aigpr__")]),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": "Retrospective AI-GPR history has no real-time vintages; treat as exploratory.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ai_gpr_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/ai_gpr"))
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
