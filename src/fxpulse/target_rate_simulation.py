"""Simulate client-defined target-rate orders on cached daily corridor prices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_config(path: Path | str = Path("configs/target_rate_simulation.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("target-rate simulation config must be preregistered")
    if not config.get("target_improvements_percent") or not config.get("deadlines_calendar_days"):
        raise ValueError("target and deadline grids must not be empty")
    return config


def simulate_orders(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {"timestamp", "corridor", "price"}
    if missing := required - set(prices):
        raise ValueError(f"prices lack {sorted(missing)}")
    data = prices[list(required)].drop_duplicates().copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data = data.loc[
        data["corridor"].isin(config["corridors"])
        & data["timestamp"].between(config["from_date"], config["to_date"], inclusive="both")
    ].sort_values(["corridor", "timestamp"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for corridor, group in data.groupby("corridor", sort=True):
        dates = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        values = group["price"].to_numpy(dtype=float)
        for improvement in config["target_improvements_percent"]:
            fraction = float(improvement) / 100
            for deadline in config["deadlines_calendar_days"]:
                deadline = int(deadline)
                last_observable_start = dates[-1] - np.timedelta64(deadline, "D")
                for index, (start_date, current) in enumerate(zip(dates, values, strict=True)):
                    if start_date > last_observable_start:
                        continue
                    deadline_date = start_date + np.timedelta64(deadline, "D")
                    stop = int(np.searchsorted(dates, deadline_date, side="right"))
                    future_dates = dates[index + 1 : stop]
                    future_prices = values[index + 1 : stop]
                    if not len(future_prices):
                        continue
                    target = current * (1 - fraction)
                    hits = np.flatnonzero(future_prices <= target)
                    executed = bool(len(hits))
                    final_index = int(hits[0]) if executed else len(future_prices) - 1
                    final_price = float(future_prices[final_index])
                    final_date = future_dates[final_index]
                    rows.append(
                        {
                            "corridor": corridor,
                            "start_date": pd.Timestamp(start_date),
                            "target_improvement_percent": float(improvement),
                            "deadline_calendar_days": deadline,
                            "start_price": current,
                            "target_price": target,
                            "executed": executed,
                            "final_date": pd.Timestamp(final_date),
                            "final_price": final_price,
                            "wait_calendar_days": int((final_date - start_date) / np.timedelta64(1, "D")),
                            "saving_vs_buy_now_bps": (current / final_price - 1) * 10_000,
                            "loss_vs_buy_now_bps": (final_price / current - 1) * 10_000,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(orders: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupers = ["corridor", "target_improvement_percent", "deadline_calendar_days"]
    for keys, group in orders.groupby(groupers, sort=True):
        corridor, improvement, deadline = keys
        executed = group.loc[group["executed"]]
        unfilled = group.loc[~group["executed"]]
        rows.append(
            {
                "corridor": corridor,
                "target_improvement_percent": improvement,
                "deadline_calendar_days": deadline,
                "orders": len(group),
                "executed": len(executed),
                "execution_rate": float(group["executed"].mean()),
                "executed_mean_wait_days": float(executed["wait_calendar_days"].mean()) if len(executed) else np.nan,
                "executed_mean_saving_bps": float(executed["saving_vs_buy_now_bps"].mean()) if len(executed) else np.nan,
                "unfilled_mean_loss_bps": float(unfilled["loss_vs_buy_now_bps"].mean()) if len(unfilled) else np.nan,
                "all_orders_mean_saving_bps": float(group["saving_vs_buy_now_bps"].mean()),
            }
        )
    corridor_summary = pd.DataFrame(rows)
    pooled_rows: list[dict[str, object]] = []
    for keys, group in orders.groupby(["target_improvement_percent", "deadline_calendar_days"], sort=True):
        improvement, deadline = keys
        executed = group.loc[group["executed"]]
        unfilled = group.loc[~group["executed"]]
        pooled_rows.append(
            {
                "corridor": "pooled",
                "target_improvement_percent": improvement,
                "deadline_calendar_days": deadline,
                "orders": len(group),
                "executed": len(executed),
                "execution_rate": float(group["executed"].mean()),
                "executed_mean_wait_days": float(executed["wait_calendar_days"].mean()) if len(executed) else np.nan,
                "executed_mean_saving_bps": float(executed["saving_vs_buy_now_bps"].mean()) if len(executed) else np.nan,
                "unfilled_mean_loss_bps": float(unfilled["loss_vs_buy_now_bps"].mean()) if len(unfilled) else np.nan,
                "all_orders_mean_saving_bps": float(group["saving_vs_buy_now_bps"].mean()),
            }
        )
    return pd.concat([corridor_summary, pd.DataFrame(pooled_rows)], ignore_index=True)


def run(
    *,
    config_path: Path | str = Path("configs/target_rate_simulation.json"),
    artifact_dir: Path | str = Path("artifacts/next_hypotheses/target_rate"),
) -> dict[str, object]:
    config = load_config(config_path)
    input_path = Path(config["input"])
    orders = simulate_orders(pd.read_csv(input_path), config)
    summary = summarize(orders)
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    orders.to_csv(output / "orders.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    meta = {
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "order_rows": len(orders),
        "warning": "Research simulation on official daily cross-rates, without bank spread, fees, execution liquidity or client behavior.",
    }
    (output / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/target_rate_simulation.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/next_hypotheses/target_rate"))
    args = parser.parse_args(argv)
    print(json.dumps(run(config_path=args.config, artifact_dir=args.artifact_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
