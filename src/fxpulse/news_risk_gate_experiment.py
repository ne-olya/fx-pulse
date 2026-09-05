"""Test an interpretable AI-GPR shock gate on existing market signals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.news_robust_experiment import horizon_summary, summarize, yearly_summary

EXPECTED_VARIANTS = [
    "market_no_gate",
    "exclude_global_gpr_gt1",
    "exclude_oil_gpr_gt1",
    "exclude_global_or_oil_gt1",
    "global_gpr_gt1_only",
]


def load_config(
    path: Path | str = Path("configs/news_risk_gate_experiment.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "registered_after_ai_gpr_before_gate_results":
        raise ValueError("news risk gate must be registered before gate results")
    if config.get("variants") != EXPECTED_VARIANTS:
        raise ValueError("news risk gate variants differ from implementation")
    return config


def gate_masks(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    threshold = float(config["shock_zscore_threshold"])
    global_shock = frame[str(config["global_feature"])].gt(threshold)
    oil_shock = frame[str(config["oil_feature"])].gt(threshold)
    return {
        "market_no_gate": pd.Series(True, index=frame.index),
        "exclude_global_gpr_gt1": ~global_shock,
        "exclude_oil_gpr_gt1": ~oil_shock,
        "exclude_global_or_oil_gt1": ~(global_shock | oil_shock),
        "global_gpr_gt1_only": global_shock,
    }


def expand_variants(
    scores: pd.DataFrame,
    signals: pd.DataFrame,
    external: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = str(config["source_feature_set"])
    scores = scores.loc[scores["feature_set"].eq(source)].copy()
    signals = signals.loc[signals["feature_set"].eq(source)].copy()
    for frame in (scores, signals):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise").dt.normalize()
        frame["corridor"] = frame["corridor"].astype(str).str.upper()
    external = external.copy()
    external["feature_date"] = pd.to_datetime(
        external["feature_date"], errors="raise"
    ).dt.normalize()
    external["corridor"] = external["corridor"].astype(str).str.upper()
    features = [str(config["global_feature"]), str(config["oil_feature"])]
    if external.duplicated(["feature_date", "corridor"]).any():
        raise ValueError("AI-GPR features must contain one row per date and corridor")
    signals = signals.merge(
        external[["feature_date", "corridor", *features]],
        left_on=["timestamp", "corridor"],
        right_on=["feature_date", "corridor"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="feature_date")
    score_variants: list[pd.DataFrame] = []
    signal_variants: list[pd.DataFrame] = []
    masks = gate_masks(signals, config)
    for variant in config["variants"]:
        current_scores = scores.copy()
        current_scores["feature_set"] = variant
        score_variants.append(current_scores)
        current_signals = signals.loc[masks[variant]].copy()
        current_signals["feature_set"] = variant
        signal_variants.append(current_signals)
    return pd.concat(score_variants, ignore_index=True), pd.concat(signal_variants, ignore_index=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    config_path: Path | str = Path("configs/news_risk_gate_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_risk_gate"),
) -> dict[str, object]:
    config = load_config(config_path)
    score_path = Path(config["parent_scores"])
    signal_path = Path(config["parent_signals"])
    feature_path = Path(config["external_feature_input"])
    scores, signals = expand_variants(
        pd.read_csv(score_path),
        pd.read_csv(signal_path),
        pd.read_csv(feature_path),
        config,
    )
    summary = summarize(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    per_year = yearly_summary(scores, signals)
    years = {int(year) for year in config["pseudo_holdout_years"]}
    pseudo_scores = scores.loc[scores["test_year"].isin(years)]
    pseudo_signals = signals.loc[signals["test_year"].isin(years)]
    pseudo = horizon_summary(pseudo_scores, pseudo_signals)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output / "signals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    metadata = {
        "config_sha256": _sha256(Path(config_path)),
        "parent_scores_sha256": _sha256(score_path),
        "parent_signals_sha256": _sha256(signal_path),
        "signal_rows": len(signals),
        "warning": "Post-hoc safety-gate test on previously reviewed OOT scores.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_risk_gate_experiment.json"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/news_risk_gate")
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
