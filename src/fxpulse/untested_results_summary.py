"""Consolidate all U01-U15 experiment outcomes into small reproducible tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fxpulse.untested_hypotheses_experiment import BASELINE, bootstrap_deltas


EXPERIMENTS = {
    "data_ready": (
        Path("artifacts/untested_hypotheses/data_ready"),
        Path("configs/untested_hypotheses_experiment.json"),
    ),
    "u10_futures": (
        Path("artifacts/untested_hypotheses/external/u10_futures_curve"),
        Path("configs/external_untested_hypotheses_experiment.json"),
    ),
    "u13_pretraining": (
        Path("artifacts/untested_hypotheses/external/u13_external_fx_pretraining"),
        Path("configs/external_untested_hypotheses_experiment.json"),
    ),
    "u08_exact": (
        Path("artifacts/untested_hypotheses/u08_exact_robustness"),
        Path("configs/u08_exact_baseline_robustness.json"),
    ),
}


SELECTED = {
    "U01": ("data_ready", "u01_discrete_hazard", "NOT_CONFIRMED"),
    "U02": ("data_ready", "u02_intraday_path", "NOT_CONFIRMED"),
    "U03": ("data_ready", "u03_error_drift_switch", "NO_EFFECT_ONE_ALARM"),
    "U05": ("data_ready", "u05_residual_factors", "NOT_CONFIRMED"),
    "U07": ("data_ready", "u07_bootstrap_historical_drift", "REJECTED_WORSE"),
    "U08": ("u08_exact", "u08_selection_cal_q80", "INTERESTING_BUT_TOO_RARE_NOT_CONFIRMED"),
    "U09": ("data_ready", "u09_group_dro", "NOT_CONFIRMED"),
    "U10": ("u10_futures", "u10_cny_futures_curve", "PROMISING_H10_BUT_NOT_STABLE"),
    "U13": ("u13_pretraining", "u13_bis_pretrained_pca", "PROMISING_H10_BUT_NOT_STABLE"),
}


def _metric_row(experiment: str, variant: str, status: str) -> dict[str, object]:
    root, _ = EXPERIMENTS[experiment]
    aggregate = pd.read_csv(root / "aggregate_summary.csv")
    corridor = pd.read_csv(root / "corridor_summary.csv")
    bootstrap = pd.read_csv(root / "bootstrap_h5.csv")
    candidate = aggregate.loc[aggregate["variant"].eq(variant) & aggregate["horizon"].eq(5)].iloc[0]
    baseline = aggregate.loc[aggregate["variant"].eq(BASELINE) & aggregate["horizon"].eq(5)].iloc[0]
    candidate_corridor = corridor.loc[corridor["variant"].eq(variant) & corridor["horizon"].eq(5)]
    baseline_corridor = corridor.loc[corridor["variant"].eq(BASELINE) & corridor["horizon"].eq(5), [
        "corridor", "lift_vs_matched_random"
    ]]
    paired = candidate_corridor.merge(
        baseline_corridor, on="corridor", suffixes=("", "_baseline"), validate="one_to_one"
    )
    delta = paired["lift_vs_matched_random"] - paired["lift_vs_matched_random_baseline"]
    uncertainty = bootstrap.loc[bootstrap["variant"].eq(variant)]
    return {
        "hypothesis": next(key for key, value in SELECTED.items() if value[1] == variant),
        "variant": variant,
        "status": status,
        "signals": int(candidate["signals"]),
        "signals_per_week_per_corridor": float(candidate["signals_per_week_per_corridor"]),
        "raw_lift": float(candidate["lift"]),
        "same_week_lift": float(candidate["lift_vs_matched_random"]),
        "aggregate_delta_same_week_lift": float(
            candidate["lift_vs_matched_random"] - baseline["lift_vs_matched_random"]
        ),
        "median_corridor_delta_same_week_lift": float(delta.median()),
        "positive_corridors": int(delta.gt(0).sum()),
        "regret_mean_bps": float(candidate["regret_mean_bps"]),
        "delta_regret_mean_bps": float(candidate["regret_mean_bps"] - baseline["regret_mean_bps"]),
        "severe_error_share": float(candidate["severe_error_share"]),
        "brier_score": float(candidate["brier_score"]),
        "bootstrap_ci_low": float(uncertainty.iloc[0]["ci_low"]) if len(uncertainty) else np.nan,
        "bootstrap_ci_high": float(uncertainty.iloc[0]["ci_high"]) if len(uncertainty) else np.nan,
    }


def build_consolidated() -> pd.DataFrame:
    rows = [_metric_row(experiment, variant, status) for experiment, variant, status in SELECTED.values()]
    u04 = json.loads(Path("artifacts/untested_hypotheses/u04_nbk_plans/summary.json").read_text())
    rows.append(
        {
            "hypothesis": "U04",
            "variant": "nbk_monthly_plan_event_pilot",
            "status": "PILOT_ONLY_INSUFFICIENT_EVENTS",
            "signals": u04["independent_events"],
            "aggregate_delta_same_week_lift": np.nan,
            "median_corridor_delta_same_week_lift": np.nan,
            "positive_corridors": 0,
            "delta_regret_mean_bps": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "note": f"Expected negative Spearman; observed {u04['spearman_sale_vs_usd_kzt_return']:.3f}, p={u04['permutation_p_two_sided']:.3f}.",
        }
    )
    for hypothesis, status, note in (
        ("U06", "BLOCKED_DATA", "No continuous first-version headline/text archive."),
        ("U11", "BLOCKED_DATA", "No point-in-time consensus expectations archive."),
        ("U12", "BLOCKED_DEPENDENCY", "Requires U06 event semantics."),
        ("U14", "BLOCKED_DEPENDENCY", "Requires U06 before historical trade weights can be tested."),
        ("U15", "BLOCKED_DATA_SUBSCRIPTION", "MOEX order-flow/order-book history is subscriber-only."),
    ):
        rows.append({"hypothesis": hypothesis, "variant": "", "status": status, "note": note})
    return pd.DataFrame(rows).sort_values("hypothesis", key=lambda values: values.str[1:].astype(int))


def build_all_horizon_bootstrap() -> pd.DataFrame:
    rows = []
    for experiment, (root, config_path) in EXPERIMENTS.items():
        signals = pd.read_csv(root / "signals.csv")
        signals["timestamp"] = pd.to_datetime(signals["timestamp"], errors="raise")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for horizon in sorted(signals["horizon"].unique()):
            changed = json.loads(json.dumps(config))
            changed["success_gate"]["primary_horizon"] = int(horizon)
            result = bootstrap_deltas(signals, config=changed)
            result.insert(0, "experiment", experiment)
            rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    order = result["p_one_sided"].sort_values(kind="mergesort").index
    ranked = result.loc[order, "p_one_sided"].to_numpy(dtype=float)
    adjusted = np.maximum.accumulate(
        np.minimum(1.0, ranked * np.arange(len(ranked), 0, -1))
    )
    result["global_holm_p"] = np.nan
    result.loc[order, "global_holm_p"] = adjusted
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/untested_hypotheses"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consolidated = build_consolidated()
    bootstrap = build_all_horizon_bootstrap()
    consolidated.to_csv(args.output_dir / "consolidated_status.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_all_horizons.csv", index=False)
    print(consolidated.to_string(index=False))


if __name__ == "__main__":
    main()
