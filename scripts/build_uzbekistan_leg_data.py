"""Normalize downloaded official Uzbekistan data into point-in-time CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.data.uzbekistan import (  # noqa: E402
    normalize_cbu_fx,
    normalize_cpi,
    normalize_external_balance,
    normalize_policy_rate,
    normalize_remittances,
    normalize_reserves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbu-rates", required=True, type=Path)
    parser.add_argument("--policy-rate", required=True, type=Path)
    parser.add_argument("--policy-supplement", type=Path)
    parser.add_argument("--cpi", required=True, action="append", type=Path)
    parser.add_argument("--reserves", required=True, type=Path)
    parser.add_argument("--bop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inflation = pd.concat([normalize_cpi(pd.read_csv(path)) for path in args.cpi], ignore_index=True)
    inflation = inflation.sort_values("observation_date", kind="mergesort").drop_duplicates(
        "observation_date", keep="last"
    )
    policy = normalize_policy_rate(pd.read_csv(args.policy_rate))
    if args.policy_supplement:
        supplement = pd.read_csv(args.policy_supplement)
        supplement["observation_date"] = pd.to_datetime(supplement["observation_date"], errors="raise")
        supplement["known_at"] = pd.to_datetime(supplement["known_at"], errors="raise")
        supplement["policy_rate_pct"] = pd.to_numeric(supplement["policy_rate_pct"], errors="raise")
        combined = pd.concat([policy, supplement], ignore_index=True).sort_values(
            "observation_date", kind="mergesort"
        ).drop_duplicates("observation_date", keep="last")
        combined["rate_change_pp"] = combined["policy_rate_pct"].diff()
        combined["decision_changed_rate"] = combined["rate_change_pp"].fillna(0).ne(0).astype(float)
        policy = combined
    outputs = {
        "fx": normalize_cbu_fx(pd.read_csv(args.cbu_rates)),
        "policy": policy,
        "inflation": inflation,
        "reserves": normalize_reserves(args.reserves),
        "remittances_proxy": normalize_remittances(args.bop),
        "external_balance": normalize_external_balance(args.bop),
    }
    meta = {}
    for name, frame in outputs.items():
        path = args.output / f"{name}.csv"
        frame.to_csv(path, index=False)
        meta[name] = {
            "rows": len(frame),
            "observation_start": str(frame["observation_date" if "observation_date" in frame else "rate_date"].min()),
            "observation_end": str(frame["observation_date" if "observation_date" in frame else "rate_date"].max()),
            "known_at_start": str(frame["known_at"].min()),
            "known_at_end": str(frame["known_at"].max()),
        }
    meta["bank_execution_quotes"] = {
        "rows": 0,
        "status": "not supplied: no public point-in-time bank history was found; no synthetic quotes were created",
    }
    meta["interventions"] = {
        "rows": 0,
        "status": "not modeled: public releases do not form a consistent daily/monthly intervention series",
    }
    (args.output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
