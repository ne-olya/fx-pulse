"""Record why source-dependent hypotheses can or cannot be tested today."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def build_audit() -> pd.DataFrame:
    event_files = sorted(glob.glob("data/raw/gdelt/events_*.csv.gz"))
    event_months = [Path(path).stem.removeprefix("events_").removesuffix(".csv") for path in event_files]
    event_columns = set(pd.read_csv(event_files[0], nrows=0).columns) if event_files else set()
    plans = pd.read_csv("data/processed/nbk_fx_plans.csv")
    futures = pd.read_csv("data/raw/moex_cny_futures_daily.csv")
    donors = pd.read_csv("data/processed/bis_fx_donors_daily.csv")
    return pd.DataFrame(
        [
            {
                "hypothesis": "U04",
                "status": "PILOT_ONLY",
                "available": f"{len(plans)} point-in-time NBK monthly plans, {plans.plan_month.min()}..{plans.plan_month.max()}",
                "missing_or_limit": "Only KZT and 30 independent events; no 4-year rolling history; Russian announcement archive not yet normalized.",
                "source": "https://nationalbank.kz/ru/news/informacionnye-soobshcheniya",
            },
            {
                "hypothesis": "U06",
                "status": "BLOCKED_DATA",
                "available": f"GDELT event URLs and structured codes for {len(event_months)} sampled months: {', '.join(event_months)}",
                "missing_or_limit": (
                    "No title/full-text/first-version fields in cached exports; continuous 2018-2026 event archive is absent. "
                    f"Cached columns: {', '.join(sorted(event_columns))}"
                ),
                "source": "https://data.gdeltproject.org/gdeltv2/",
            },
            {
                "hypothesis": "U10",
                "status": "TESTED_GATE_FAILED",
                "available": f"{len(futures)} MOEX contract candles, {futures.begin.min()}..{futures.begin.max()}",
                "missing_or_limit": "History starts in April 2022, so the common test begins only after enough futures training rows accumulate.",
                "source": "https://iss.moex.com/iss/engines/futures/markets/forts/securities",
            },
            {
                "hypothesis": "U11",
                "status": "BLOCKED_DATA",
                "available": "CBR key-rate decisions/levels are cached.",
                "missing_or_limit": "No point-in-time archive of pre-release consensus expectations; a realised decision is not a surprise feature.",
                "source": "https://www.cbr.ru/hd_base/KeyRate/",
            },
            {
                "hypothesis": "U12",
                "status": "BLOCKED_DEPENDENCY",
                "available": "Hourly CNY/RUB reaction data are cached.",
                "missing_or_limit": "Requires point-in-time event semantics from U06 before a news-versus-reaction mismatch can be defined.",
                "source": "data/raw/moex_cets_factors_60m.csv",
            },
            {
                "hypothesis": "U13",
                "status": "TESTED_GATE_FAILED",
                "available": f"{donors.area.nunique()} BIS donor FX series and {len(donors)} daily rows.",
                "missing_or_limit": "BIS publishes weekly; the experiment imposes a seven-day donor cutoff and uses donors only to learn a shape basis.",
                "source": "https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip",
            },
            {
                "hypothesis": "U14",
                "status": "BLOCKED_DEPENDENCY",
                "available": "UN Comtrade exposes bilateral trade data.",
                "missing_or_limit": "U06 point-in-time events are missing; trade weights alone cannot test propagation. No historical-vintage trade file is cached.",
                "source": "https://comtradeplus.un.org/",
            },
            {
                "hypothesis": "U15",
                "status": "BLOCKED_DATA_SUBSCRIPTION",
                "available": "Public daily candle/num_trades proxies are cached and already tested elsewhere.",
                "missing_or_limit": "MOEX TradeStats/OBStats/OrderStats history is subscriber-only without team credentials; candles cannot reconstruct aggressor side or book depth.",
                "source": "https://moexalgo.github.io/docs/api/get-fx-tradestats/",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/untested_hypotheses/data_availability.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/untested_hypotheses/data_availability.json"))
    args = parser.parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    args.json_output.write_text(
        json.dumps(audit.to_dict("records"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
