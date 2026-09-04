"""Ablate conservatively delayed daily Brent features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _rolling_zscore
from fxpulse.path_label_experiment import evaluate as evaluate_path_label
from fxpulse.temporal_sequence_experiment import summarize


def load_config(path: Path | str = Path("configs/brent_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("Brent config must be preregistered")
    if config.get("label_methods") != ["triple_barrier"]:
        raise ValueError("Brent experiment requires the fixed triple-barrier label")
    return config


def add_brent_features(frame: pd.DataFrame, raw_brent: pd.DataFrame, *, lag_days: int) -> pd.DataFrame:
    brent = raw_brent[["date", "brent_usd_per_barrel"]].copy()
    brent["date"] = pd.to_datetime(brent["date"], errors="raise")
    brent = brent.drop_duplicates("date").sort_values("date")
    price = pd.to_numeric(brent["brent_usd_per_barrel"], errors="raise")
    returns = price.pct_change(fill_method=None)
    for window in (1, 3, 5):
        brent[f"market__brent_return_{window}"] = price.pct_change(window, fill_method=None)
    brent["market__brent_volatility_20"] = returns.rolling(20, min_periods=20).std()
    brent["market__brent_zscore_20"] = _rolling_zscore(price, 20)
    brent["_available_at"] = brent["date"] + pd.to_timedelta(int(lag_days), unit="D")
    brent = brent.drop(columns=["date", "brent_usd_per_barrel"]).sort_values("_available_at")

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    original_order = data.reset_index().rename(columns={"index": "_original_index"})
    merged = pd.merge_asof(
        original_order.sort_values("timestamp"),
        brent,
        left_on="timestamp",
        right_on="_available_at",
        direction="backward",
    )
    merged["market__brent_missing"] = merged["_available_at"].isna().astype(float)
    return (
        merged.drop(columns=["_available_at"])
        .sort_values("_original_index")
        .drop(columns=["_original_index"])
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
    )


def run(
    *,
    config_path: Path | str = Path("configs/brent_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/brent"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    brent_path = Path(config["brent_input"])
    base = pd.read_csv(input_path)
    enriched = add_brent_features(
        base,
        pd.read_csv(brent_path),
        lag_days=int(config["availability_lag_calendar_days"]),
    )
    base_folds, base_signals = evaluate_path_label(base, config)
    brent_folds, brent_signals = evaluate_path_label(enriched, config)
    base_folds["feature_set"] = "without_brent"
    brent_folds["feature_set"] = "plus_brent"
    base_signals["feature_set"] = "without_brent"
    brent_signals["feature_set"] = "plus_brent"
    folds = pd.concat([base_folds, brent_folds], ignore_index=True)
    signals = pd.concat([base_signals, brent_signals], ignore_index=True)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "brent_input_sha256": hashlib.sha256(brent_path.read_bytes()).hexdigest(),
        "brent_rows": len(pd.read_csv(brent_path)),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "FRED/EIA observations are delayed by seven calendar days; exact historical release vintages were not available.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/brent_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/brent"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
