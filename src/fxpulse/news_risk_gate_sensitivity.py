"""Report a fixed threshold curve for the global-or-oil news-risk gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.news_robust_experiment import horizon_summary, summarize, yearly_summary


def load_config(
    path: Path | str = Path("configs/news_risk_gate_sensitivity.json"),
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("status") != "registered_after_gate_before_sensitivity_results":
        raise ValueError("gate sensitivity must be registered before its results")
    thresholds = [float(value) for value in config.get("thresholds", [])]
    if thresholds != [0.5, 1.0, 1.5, 2.0]:
        raise ValueError("gate sensitivity requires the fixed threshold curve")
    return config


def variant_name(threshold: float) -> str:
    return f"exclude_either_gt{threshold:g}".replace(".", "p")


def build_variants(
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
    signals = signals.merge(
        external[["feature_date", "corridor", *features]],
        left_on=["timestamp", "corridor"],
        right_on=["feature_date", "corridor"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="feature_date")

    score_frames: list[pd.DataFrame] = []
    signal_frames: list[pd.DataFrame] = []
    for variant, mask in [("market_no_gate", pd.Series(True, index=signals.index))] + [
        (
            variant_name(float(threshold)),
            signals[features].le(float(threshold)).all(axis=1),
        )
        for threshold in config["thresholds"]
    ]:
        current_scores = scores.copy()
        current_scores["feature_set"] = variant
        score_frames.append(current_scores)
        current_signals = signals.loc[mask].copy()
        current_signals["feature_set"] = variant
        signal_frames.append(current_signals)
    return pd.concat(score_frames, ignore_index=True), pd.concat(signal_frames, ignore_index=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    config_path: Path | str = Path("configs/news_risk_gate_sensitivity.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/news_risk_gate_sensitivity"),
) -> dict[str, object]:
    config = load_config(config_path)
    score_path = Path(config["parent_scores"])
    signal_path = Path(config["parent_signals"])
    feature_path = Path(config["external_feature_input"])
    scores, signals = build_variants(
        pd.read_csv(score_path), pd.read_csv(signal_path), pd.read_csv(feature_path), config
    )
    summary = summarize(scores, signals)
    by_horizon = horizon_summary(scores, signals)
    per_year = yearly_summary(scores, signals)
    years = {int(year) for year in config["pseudo_holdout_years"]}
    pseudo = horizon_summary(
        scores.loc[scores["test_year"].isin(years)],
        signals.loc[signals["test_year"].isin(years)],
    )
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    by_horizon.to_csv(output / "horizon_summary.csv", index=False)
    per_year.to_csv(output / "yearly_summary.csv", index=False)
    pseudo.to_csv(output / "pseudo_holdout_horizon_summary.csv", index=False)
    metadata = {
        "config_sha256": _sha256(Path(config_path)),
        "parent_scores_sha256": _sha256(score_path),
        "signal_rows": len(signals),
        "warning": "Post-hoc threshold sensitivity; do not select a winning threshold on this history.",
    }
    (output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(by_horizon.to_string(index=False), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/news_risk_gate_sensitivity.json"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/next_hypotheses/news_risk_gate_sensitivity"),
    )
    args = parser.parse_args()
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
