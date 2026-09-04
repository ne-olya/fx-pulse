"""Backtest fixed filters, momentum streaks and moving-average rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxpulse.next_hypotheses import _apply_policy, _future_outcomes
from fxpulse.temporal_sequence_experiment import summarize


def load_config(path: Path | str = Path("configs/technical_rule_experiment.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("technical-rule config must be preregistered")
    return config


def load_prices(config: dict[str, Any]) -> pd.DataFrame:
    cbr = pd.read_csv(config["cbr_input"], usecols=["timestamp", "corridor", "price"]).drop_duplicates()
    cbr["timestamp"] = pd.to_datetime(cbr["timestamp"], errors="raise")
    moex = pd.read_csv(config["moex_input"])
    moex = moex.loc[moex["secid"].eq("CNYRUB_TOM"), ["trade_date", "close"]].copy()
    moex.columns = ["timestamp", "price"]
    moex["timestamp"] = pd.to_datetime(moex["timestamp"], errors="raise")
    moex["corridor"] = "CNY_MOEX"
    return pd.concat([cbr, moex], ignore_index=True).sort_values(["corridor", "timestamp"])


def add_rules(group: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = group.sort_values("timestamp", kind="mergesort").copy()
    price = data["price"]
    for window in config["return_windows"]:
        movement = price.pct_change(int(window), fill_method=None)
        for threshold in config["fixed_move_percent"]:
            fraction = float(threshold) / 100
            suffix = f"{float(threshold):g}pct_{int(window)}d"
            data[f"rule__move_up_{suffix}"] = movement.ge(fraction)
            data[f"rule__move_down_{suffix}"] = movement.le(-fraction)
    one_step_return = price.pct_change(fill_method=None)
    for length in config.get("streak_lengths", []):
        length = int(length)
        data[f"rule__up_streak_{length}d"] = (
            one_step_return.gt(0).rolling(length, min_periods=length).sum().eq(length)
        )
        data[f"rule__down_streak_{length}d"] = (
            one_step_return.lt(0).rolling(length, min_periods=length).sum().eq(length)
        )
    for fast, slow in config["moving_average_pairs"]:
        fast_average = price.rolling(int(fast), min_periods=int(fast)).mean()
        slow_average = price.rolling(int(slow), min_periods=int(slow)).mean()
        above = fast_average.gt(slow_average)
        data[f"rule__ma_above_{int(fast)}_{int(slow)}"] = above
        data[f"rule__ma_cross_up_{int(fast)}_{int(slow)}"] = above & ~above.shift(1, fill_value=False)
    for horizon in config["horizons"]:
        regret, benefit = _future_outcomes(price, int(horizon))
        data[f"regret_{int(horizon)}"] = regret
        data[f"benefit_{int(horizon)}"] = benefit
    return data


def evaluate(prices: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = pd.concat(
        [add_rules(group, config) for _, group in prices.groupby("corridor", sort=True)],
        ignore_index=True,
    )
    prepared = prepared.loc[
        prepared["timestamp"].between(config["evaluation_from"], config["evaluation_to"], inclusive="both")
    ].copy()
    rule_columns = [column for column in prepared if column.startswith("rule__")]
    fold_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []
    for horizon in config["horizons"]:
        usable = prepared.loc[prepared[f"regret_{int(horizon)}"].notna()].copy()
        usable["target"] = usable[f"regret_{int(horizon)}"].le(float(config["tolerance_bps"])).astype(int)
        usable["regret_bps"] = usable[f"regret_{int(horizon)}"]
        usable["benefit_bps"] = usable[f"benefit_{int(horizon)}"]
        usable["year"] = usable["timestamp"].dt.year
        usable["week"] = usable["timestamp"].dt.to_period("W").astype(str)
        weekly_rate = usable.groupby(["corridor", "week"])["target"].mean()
        usable["matched_week_hit_rate"] = [
            float(weekly_rate.loc[(corridor, week)])
            for corridor, week in zip(usable["corridor"], usable["week"], strict=True)
        ]
        for corridor, corridor_data in usable.groupby("corridor", sort=True):
            for year, fold in corridor_data.groupby("year", sort=True):
                duration = max(float((fold["timestamp"].max() - fold["timestamp"].min()).days) / 7, 1 / 7)
                for rule in rule_columns:
                    candidates = fold.loc[fold[rule]].copy()
                    selected = _apply_policy(
                        candidates,
                        cooldown_days=int(config["cooldown_days"]),
                        weekly_cap=int(config["weekly_cap"]),
                    )
                    count = len(selected)
                    base_rate = float(fold["target"].mean())
                    hit_rate = float(selected["target"].mean()) if count else np.nan
                    identity = {
                        "scope": corridor,
                        "model": "fixed_rule",
                        "feature_set": rule.removeprefix("rule__"),
                        "horizon": int(horizon),
                    }
                    fold_rows.append(
                        {
                            **identity,
                            "test_year": int(year),
                            "test_count": len(fold),
                            "test_hits": int(fold["target"].sum()),
                            "signal_count": count,
                            "signal_hits": int(selected["target"].sum()) if count else 0,
                            "lift": hit_rate / base_rate if count and base_rate > 0 else np.nan,
                            "matched_expected_hits": float(selected["matched_week_hit_rate"].sum()) if count else 0.0,
                            "regret_sum_bps": float(selected["regret_bps"].sum()) if count else 0.0,
                            "benefit_sum_bps": float(selected["benefit_bps"].sum()) if count else 0.0,
                            "duration_weeks": duration,
                        }
                    )
                    if count:
                        exported = selected[[
                            "timestamp", "corridor", "target", "regret_bps", "benefit_bps", "matched_week_hit_rate",
                        ]].copy()
                        for key, value in identity.items():
                            exported[key] = value
                        exported["test_year"] = int(year)
                        signal_frames.append(exported)
    folds = pd.DataFrame(fold_rows)
    pooled = (
        folds.groupby(["model", "feature_set", "horizon", "test_year"], as_index=False)
        .agg(
            test_count=("test_count", "sum"),
            test_hits=("test_hits", "sum"),
            signal_count=("signal_count", "sum"),
            signal_hits=("signal_hits", "sum"),
            matched_expected_hits=("matched_expected_hits", "sum"),
            regret_sum_bps=("regret_sum_bps", "sum"),
            benefit_sum_bps=("benefit_sum_bps", "sum"),
            duration_weeks=("duration_weeks", "sum"),
        )
    )
    pooled["scope"] = "pooled"
    pooled["lift"] = (
        (pooled["signal_hits"] / pooled["signal_count"].replace(0, np.nan))
        / (pooled["test_hits"] / pooled["test_count"])
    )
    folds = pd.concat([folds, pooled[folds.columns]], ignore_index=True)
    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    return folds, signals


def run(
    *,
    config_path: Path | str = Path("configs/technical_rule_experiment.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/technical_rules"),
) -> dict[str, object]:
    config = load_config(config_path)
    folds, signals = evaluate(load_prices(config), config)
    summary_frame = summarize(folds)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "folds.csv", index=False)
    signals.to_csv(output / "signals.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "fold_rows": len(folds),
        "signal_rows": len(signals),
        "warning": "Exploratory fixed rules; every registered rule is reported, with no ex-post threshold tuning.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/technical_rule_experiment.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/technical_rules"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
