"""Check AI-GPR features under seven-, fourteen- and thirty-day delays."""

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
    paired_comparison,
    period_summary,
    summarize,
    yearly_summary,
)

MARKET_PREFIXES = ("base__", "leg__", "regime__", "market__", "indicator__")
EXPECTED_FEATURE_SETS = [
    "market_only",
    "plus_ai_gpr_lag7",
    "plus_ai_gpr_lag14",
    "plus_ai_gpr_lag30",
]


def load_config(
    path: Path | str = Path("configs/ai_gpr_lag_sensitivity_experiment.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "registered_after_initial_results_before_lag_results":
        raise ValueError("AI-GPR lag sensitivity must be registered before lag results")
    if config.get("feature_sets") != EXPECTED_FEATURE_SETS:
        raise ValueError("AI-GPR lag feature sets differ from implementation")
    return config


def prepare_frame(
    market: pd.DataFrame, external: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    result = market.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise").dt.normalize()
    result["corridor"] = result["corridor"].astype(str).str.upper()
    result = result.loc[result["corridor"].isin(config["corridors"])].copy()
    source = external.copy()
    source["feature_date"] = pd.to_datetime(source["feature_date"], errors="raise").dt.normalize()
    source["corridor"] = source["corridor"].astype(str).str.upper()
    columns = [column for column in source if column.startswith("aigpr__")]
    base_lag = int(config["base_availability_lag_days"])
    for lag_value in config["tested_total_lags_days"]:
        lag = int(lag_value)
        if lag < base_lag:
            raise ValueError("sensitivity lag cannot be shorter than the source feature lag")
        delayed = source[["feature_date", "corridor", *columns]].copy()
        delayed["feature_date"] += pd.Timedelta(lag - base_lag, unit="D")
        rename = {column: column.replace("aigpr__", f"aigprlag{lag}__", 1) for column in columns}
        delayed = delayed.rename(columns=rename)
        result = result.merge(
            delayed,
            left_on=["timestamp", "corridor"],
            right_on=["feature_date", "corridor"],
            how="inner",
            validate="many_to_one",
        ).drop(columns="feature_date")
    return result.sort_values(["corridor", "timestamp"], kind="mergesort").reset_index(drop=True)


def feature_sets(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    market = [column for column in frame if column.startswith(MARKET_PREFIXES)]
    result = {"market_only": market}
    for lag_value in config["tested_total_lags_days"]:
        lag = int(lag_value)
        external = [column for column in frame if column.startswith(f"aigprlag{lag}__")]
        if not external:
            raise ValueError(f"AI-GPR lag {lag} feature family is empty")
        result[f"plus_ai_gpr_lag{lag}"] = [*market, *external]
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    config_path: Path | str = Path("configs/ai_gpr_lag_sensitivity_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/ai_gpr_lag_sensitivity"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    feature_path = Path(config["external_feature_input"])
    frame = prepare_frame(pd.read_csv(input_path), pd.read_csv(feature_path), config)
    scores, folds, importance = fit_oot_scores(
        frame, config, sets=feature_sets(frame, config)
    )
    signals = apply_continuous_policy(scores, config)
    summary_frame = summarize(scores, signals)
    per_year = yearly_summary(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    comparison = paired_comparison(summary_frame, list(config["feature_sets"]))
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
    summary_frame.to_csv(output / "summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_2025_2026.csv", index=False)
    pseudo_by_horizon.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    metadata = {
        "config_sha256": _sha256(Path(config_path)),
        "input_sha256": _sha256(input_path),
        "external_feature_sha256": _sha256(feature_path),
        "joined_rows": len(frame),
        "score_rows": len(scores),
        "signal_rows": len(signals),
        "warning": "Post-hoc delay sensitivity after the initial AI-GPR result.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ai_gpr_lag_sensitivity_experiment.json")
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/ai_gpr_lag_sensitivity"),
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
