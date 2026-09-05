"""Compare current GDELT features with a one-year-late temporal placebo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.news_robust_experiment import (
    MARKET_PREFIXES,
    _is_level,
    apply_continuous_policy,
    fit_oot_scores,
    horizon_summary,
    paired_comparison,
    period_summary,
    prepare_frame,
    summarize,
    yearly_summary,
)

RUSSIA_FEATURE_SETS = [
    "market_only",
    "plus_russia_levels_real",
    "plus_russia_levels_lag365",
]

RECIPIENT_FEATURE_SETS = [
    "market_only",
    "plus_recipient_levels_real",
    "plus_recipient_levels_lag365",
]


def load_config(
    path: Path | str = Path("configs/news_placebo_experiment.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    scope = config.get("news_scope", "russia_levels")
    expected = RUSSIA_FEATURE_SETS if scope == "russia_levels" else RECIPIENT_FEATURE_SETS
    valid_statuses = {
        "russia_levels": {
            "registered_after_partial_gdelt_before_placebo_results",
            "registered_after_full_gdelt_before_placebo_results",
        },
        "recipient_levels": {
            "registered_after_partial4_before_recipient_placebo_results",
            "registered_after_full_gdelt_before_recipient_placebo_results",
        },
    }
    if config.get("status") not in valid_statuses.get(scope, set()):
        raise ValueError("news placebo must be registered before its results")
    if config.get("feature_sets") != expected:
        raise ValueError("news placebo feature sets differ from implementation")
    return config


def _news_columns(frame: pd.DataFrame, *, news_scope: str) -> list[str]:
    if news_scope == "russia_levels":
        return [
            column
            for column in frame
            if column.startswith("news__")
            and not column.startswith(("news__recipient_", "news__cross_"))
            and _is_level(column)
        ]
    if news_scope == "recipient_levels":
        return [
            column
            for column in frame
            if column.startswith("news__recipient_") and _is_level(column)
        ]
    raise ValueError(f"unknown news placebo scope: {news_scope}")


def add_lagged_placebo(
    frame: pd.DataFrame,
    *,
    lag_days: int,
    news_scope: str = "russia_levels",
) -> pd.DataFrame:
    news = _news_columns(frame, news_scope=news_scope)
    if not news:
        raise ValueError(f"no {news_scope} news feature for placebo")
    delayed = frame[["timestamp", "corridor", *news]].copy()
    delayed["timestamp"] += pd.Timedelta(lag_days, unit="D")
    delayed = delayed.rename(columns={column: f"placebo365__{column}" for column in news})
    return frame.merge(
        delayed,
        on=["timestamp", "corridor"],
        how="inner",
        validate="one_to_one",
    )


def feature_sets(
    frame: pd.DataFrame,
    *,
    news_scope: str = "russia_levels",
) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(MARKET_PREFIXES)]
    real = _news_columns(frame, news_scope=news_scope)
    placebo = [column for column in frame if column.startswith("placebo365__")]
    if not all((market, real, placebo)):
        raise ValueError("news placebo feature family is empty")
    if news_scope == "recipient_levels":
        return {
            "market_only": market,
            "plus_recipient_levels_real": [*market, *real],
            "plus_recipient_levels_lag365": [*market, *placebo],
        }
    return {
        "market_only": market,
        "plus_russia_levels_real": [*market, *real],
        "plus_russia_levels_lag365": [*market, *placebo],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    config_path: Path | str = Path("configs/news_placebo_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_placebo"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    news_path = Path(config["news_input"])
    news_scope = str(config.get("news_scope", "russia_levels"))
    frame = prepare_frame(pd.read_csv(input_path), pd.read_csv(news_path), config)
    frame = add_lagged_placebo(
        frame,
        lag_days=int(config["placebo_lag_days"]),
        news_scope=news_scope,
    )
    scores, folds, importance = fit_oot_scores(
        frame,
        config,
        sets=feature_sets(frame, news_scope=news_scope),
    )
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
        "news_scope": news_scope,
        "score_rows": len(scores),
        "warning": "Post-hoc temporal placebo on reviewed GDELT history.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_placebo_experiment.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news_placebo")
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
