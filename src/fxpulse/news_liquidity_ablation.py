"""Isolate GDELT's incremental value after MOEX trading activity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.news_robust_experiment import (
    MARKET_PREFIXES,
    apply_continuous_policy,
    fit_oot_scores,
    horizon_summary,
    paired_comparison,
    period_summary,
    prepare_frame,
    summarize,
    yearly_summary,
)

EXPECTED_FEATURE_SETS = [
    "market_only",
    "plus_liquidity",
    "plus_all_news",
    "plus_liquidity_all_news",
]


def load_config(
    path: Path | str = Path("configs/news_liquidity_ablation_partial4.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    valid_statuses = {
        "registered_after_partial4_before_liquidity_ablation_results",
        "registered_after_full_gdelt_before_liquidity_ablation_results",
    }
    if config.get("status") not in valid_statuses:
        raise ValueError("news/liquidity ablation must be registered before results")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("news/liquidity feature sets differ from implementation")
    return config


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(MARKET_PREFIXES)]
    liquidity = [column for column in frame if column.startswith("liquidity__")]
    news = [column for column in frame if column.startswith("news__")]
    if not all((market, liquidity, news)):
        raise ValueError("market, liquidity and news features must be populated")
    return {
        "market_only": market,
        "plus_liquidity": [*market, *liquidity],
        "plus_all_news": [*market, *news],
        "plus_liquidity_all_news": [*market, *liquidity, *news],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    config_path: Path | str = Path("configs/news_liquidity_ablation_partial4.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_liquidity_ablation_partial4"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    news_path = Path(config["news_input"])
    frame = prepare_frame(pd.read_csv(input_path), pd.read_csv(news_path), config)
    scores, folds, importance = fit_oot_scores(frame, config, sets=feature_sets(frame))
    signals = apply_continuous_policy(scores, config)
    summary = summarize(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    per_year = yearly_summary(scores, signals)
    comparison = paired_comparison(summary, list(config["feature_sets"]))
    pseudo = period_summary(scores, signals, list(config["pseudo_holdout_years"]))
    pseudo_by_horizon = horizon_summary(
        scores.loc[scores["test_year"].isin(config["pseudo_holdout_years"])],
        signals.loc[signals["test_year"].isin(config["pseudo_holdout_years"])],
    )
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "oot_scores.csv.gz", index=False, compression="gzip")
    folds.to_csv(output / "folds.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_2025_2026.csv", index=False)
    pseudo_by_horizon.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    metadata = {
        "config_sha256": _sha256(Path(config_path)),
        "input_sha256": _sha256(input_path),
        "news_input_sha256": _sha256(news_path),
        "score_rows": len(scores),
        "warning": "Post-hoc focused ablation on reviewed GDELT history.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/news_liquidity_ablation_partial4.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/news_liquidity_ablation_partial4"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(config_path=args.config, artifact_dir=args.artifact_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
