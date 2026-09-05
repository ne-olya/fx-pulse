"""Extract a fixed donor-currency panel from the official BIS XRU bulk archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DONOR_AREAS = (
    "AU", "BR", "CA", "CH", "CL", "CN", "CO", "CZ", "DK", "GB",
    "HK", "HU", "ID", "IL", "IN", "IS", "JP", "KR", "MX", "MY",
    "NO", "NZ", "PH", "PL", "RO", "RS", "SE", "SG", "TH", "TR", "ZA",
)
USECOLS = [
    "FREQ:Frequency",
    "REF_AREA:Reference area",
    "CURRENCY:Currency",
    "COLLECTION:Collection",
    "TIME_PERIOD:Time period or range",
    "OBS_VALUE:Observation Value",
    "OBS_STATUS:Observation Status",
]


def _code(value: pd.Series) -> pd.Series:
    return value.astype(str).str.split(":", n=1).str[0]


def run(*, source: Path, output: Path, meta_output: Path, from_date: str) -> dict[str, object]:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=USECOLS, chunksize=250_000, low_memory=False):
        frequency = _code(chunk["FREQ:Frequency"])
        area = _code(chunk["REF_AREA:Reference area"])
        selected = chunk.loc[frequency.eq("D") & area.isin(DONOR_AREAS)].copy()
        if selected.empty:
            continue
        selected["area"] = _code(selected["REF_AREA:Reference area"])
        selected["currency"] = _code(selected["CURRENCY:Currency"])
        selected["date"] = pd.to_datetime(selected["TIME_PERIOD:Time period or range"], errors="coerce")
        selected["usd_local"] = pd.to_numeric(selected["OBS_VALUE:Observation Value"], errors="coerce")
        pieces.append(selected[["date", "area", "currency", "usd_local", "OBS_STATUS:Observation Status"]])
    data = pd.concat(pieces, ignore_index=True)
    data = data.loc[data["date"].ge(pd.Timestamp(from_date)) & data["usd_local"].gt(0)].copy()
    data = data.rename(columns={"OBS_STATUS:Observation Status": "obs_status"})
    data = data.sort_values(["area", "date"], kind="mergesort")
    if data.duplicated(["area", "date"]).any():
        duplicates = data.loc[data.duplicated(["area", "date"], keep=False), ["area", "date"]]
        raise ValueError(f"duplicate daily BIS series observations: {duplicates.head().to_dict('records')}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    coverage = data.groupby("area")["date"].agg(["min", "max", "count"]).reset_index()
    meta = {
        "source": "BIS bilateral exchange rates bulk CSV (WS_XRU)",
        "source_url": "https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_lag_assumption_days": 7,
        "from": from_date,
        "areas_requested": list(DONOR_AREAS),
        "areas_found": sorted(data["area"].unique()),
        "rows": len(data),
        "coverage": coverage.assign(
            min=coverage["min"].dt.date.astype(str), max=coverage["max"].dt.date.astype(str)
        ).to_dict("records"),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/bis/WS_XRU_csv_flat.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/bis_fx_donors_daily.csv"))
    parser.add_argument("--meta-output", type=Path, default=Path("data/processed/bis_fx_donors_daily.meta.json"))
    parser.add_argument("--from-date", default="2009-01-01")
    args = parser.parse_args()
    print(json.dumps(run(source=args.source, output=args.output, meta_output=args.meta_output, from_date=args.from_date), indent=2))


if __name__ == "__main__":
    main()
