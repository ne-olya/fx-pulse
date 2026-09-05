"""Small event-level feasibility test for U04 using announced NBK FX-sale plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_event_panel(plans: pd.DataFrame, cbr: pd.DataFrame) -> pd.DataFrame:
    rates = cbr.copy()
    rates["timestamp"] = pd.to_datetime(rates["rate_date"], errors="raise")
    rates["price"] = pd.to_numeric(rates["rate_rub"], errors="raise") / pd.to_numeric(
        rates["nominal"], errors="raise"
    )
    pivot = rates.pivot(index="timestamp", columns="ccy", values="price").sort_index()
    local_per_usd = (pivot["USD"] / pivot["KZT"]).dropna().rename("usd_kzt")
    rub_per_kzt = pivot["KZT"].dropna().rename("rub_kzt")
    prices = pd.concat([local_per_usd, rub_per_kzt], axis=1).dropna()

    rows = []
    for plan in plans.itertuples(index=False):
        plan_month = pd.Timestamp(plan.plan_month)
        published = pd.Timestamp(plan.published_at)
        month_end = plan_month + pd.offsets.MonthEnd(0)
        # Use the first official observation strictly after publication.  This
        # avoids pretending that a morning/afternoon release was tradable at an
        # already fixed same-day official rate.
        available = prices.loc[
            (prices.index > published.normalize()) & (prices.index <= month_end)
        ]
        if len(available) < 2:
            continue
        start_date = available.index.min()
        end_date = available.index.max()
        start = available.loc[start_date]
        end = available.loc[end_date]
        rows.append(
            {
                "plan_month": plan_month,
                "published_at": published,
                "available_from": start_date,
                "outcome_to": end_date,
                "planned_sale_mid_usd_mn": float(plan.planned_sale_mid_usd_mn),
                "planned_purchase_mid_usd_mn": float(plan.planned_purchase_mid_usd_mn),
                "planned_net_sale_mid_usd_mn": float(plan.planned_net_sale_mid_usd_mn),
                "usd_kzt_return_bps": float((end["usd_kzt"] / start["usd_kzt"] - 1) * 10_000),
                "rub_kzt_return_bps": float((end["rub_kzt"] / start["rub_kzt"] - 1) * 10_000),
                "url": plan.url,
            }
        )
    return pd.DataFrame(rows).sort_values("plan_month", kind="mergesort")


def expanding_linear_check(events: pd.DataFrame, minimum_train: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(minimum_train, len(events)):
        train = events.iloc[:index]
        test = events.iloc[index]
        model = LinearRegression().fit(
            train[["planned_sale_mid_usd_mn"]], train["usd_kzt_return_bps"]
        )
        rows.append(
            {
                "plan_month": test["plan_month"],
                "actual_usd_kzt_return_bps": test["usd_kzt_return_bps"],
                "prediction_plan": float(model.predict(test[["planned_sale_mid_usd_mn"]].to_frame().T)[0]),
                "prediction_history_mean": float(train["usd_kzt_return_bps"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["squared_error_plan"] = np.square(
            result["actual_usd_kzt_return_bps"] - result["prediction_plan"]
        )
        result["squared_error_history_mean"] = np.square(
            result["actual_usd_kzt_return_bps"] - result["prediction_history_mean"]
        )
    return result


def summarize(events: pd.DataFrame, expanding: pd.DataFrame, *, permutations: int, seed: int) -> dict[str, object]:
    observed = float(events["planned_sale_mid_usd_mn"].corr(events["usd_kzt_return_bps"], method="spearman"))
    rng = np.random.default_rng(seed)
    simulated = np.asarray(
        [
            pd.Series(rng.permutation(events["planned_sale_mid_usd_mn"])).corr(
                events["usd_kzt_return_bps"].reset_index(drop=True), method="spearman"
            )
            for _ in range(permutations)
        ]
    )
    median = float(events["planned_sale_mid_usd_mn"].median())
    high = events.loc[events["planned_sale_mid_usd_mn"].ge(median), "usd_kzt_return_bps"]
    low = events.loc[events["planned_sale_mid_usd_mn"].lt(median), "usd_kzt_return_bps"]
    return {
        "hypothesis": "U04",
        "status": "PILOT_ONLY_INSUFFICIENT_FOR_WALK_FORWARD_GATE_G",
        "reason": "Only one corridor and 30 or fewer independent monthly announcements; repeated daily rows would be pseudoreplication.",
        "independent_events": len(events),
        "first_event": events["plan_month"].min().date().isoformat(),
        "last_event": events["plan_month"].max().date().isoformat(),
        "spearman_sale_vs_usd_kzt_return": observed,
        "permutation_p_two_sided": float((1 + np.sum(np.abs(simulated) >= abs(observed))) / (permutations + 1)),
        "high_minus_low_sale_usd_kzt_return_bps": float(high.mean() - low.mean()),
        "expanding_test_events": len(expanding),
        "expanding_mse_plan": float(expanding["squared_error_plan"].mean()) if len(expanding) else None,
        "expanding_mse_history_mean": float(expanding["squared_error_history_mean"].mean()) if len(expanding) else None,
        "interpretation": "A useful sale-supply mechanism predicts a negative association: larger announced USD sales should strengthen KZT, lowering USD/KZT.",
    }


def run(*, plans_path: Path, cbr_path: Path, output: Path, permutations: int = 10_000) -> dict[str, object]:
    events = build_event_panel(pd.read_csv(plans_path), pd.read_csv(cbr_path))
    expanding = expanding_linear_check(events)
    summary = summarize(events, expanding, permutations=permutations, seed=20260905)
    summary["plans_sha256"] = hashlib.sha256(plans_path.read_bytes()).hexdigest()
    summary["cbr_sha256"] = hashlib.sha256(cbr_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "events.csv", index=False)
    expanding.to_csv(output / "expanding_predictions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=Path("data/processed/nbk_fx_plans.csv"))
    parser.add_argument("--cbr", type=Path, default=Path("data/raw/cbr_daily_2010_2026.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/untested_hypotheses/u04_nbk_plans"))
    args = parser.parse_args()
    print(json.dumps(run(plans_path=args.plans, cbr_path=args.cbr, output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
