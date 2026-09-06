"""Calculate fast/slow timing and the UZS pilot benefit dispersion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.decision_audit import (  # noqa: E402
    fast_slow_pairs,
    summarize_benefit,
    summarize_fast_slow,
    symmetric_benefit_bps,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/amd_h5_site_run_20260904_v2/features_h5.csv"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    frame = pd.read_csv(args.features, usecols=["timestamp", "corridor", "price"])
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    pairs = fast_slow_pairs(frame)
    timing = summarize_fast_slow(pairs)
    benefit = symmetric_benefit_bps(frame)
    benefit_summary = summarize_benefit(benefit)

    pairs.to_csv(args.output / "fast_slow_pairs.csv", index=False)
    timing.to_csv(args.output / "fast_slow_summary.csv", index=False)
    benefit.to_csv(args.output / "uzs_benefit_bps.csv", index=False)
    benefit_summary.to_csv(args.output / "uzs_benefit_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "input": {"path": str(args.features), "sha256": _sha256(args.features)},
        "input_period": f"{timestamps.min().date().isoformat()}..{timestamps.max().date().isoformat()}",
        "evaluated_years": "2022..2026",
        "fast_indicator": "momentum_streak(n=3), onset only",
        "slow_indicator": "level_percentile(window=60, pct=20), state",
        "confirmation_horizon_trading_days": 5,
        "outcome_horizon_trading_days": 5,
        "evaluation_tolerance_bps": 25,
        "benefit_definition": "mean price of surrounding ±5 observations, excluding today, versus today's price",
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(timing.to_string(index=False))
    print(benefit_summary.to_string(index=False))


if __name__ == "__main__":
    main()
