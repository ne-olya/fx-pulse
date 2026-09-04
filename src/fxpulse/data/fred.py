"""Download and cache the public daily Brent spot series distributed by FRED."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

import certifi


SERIES_ID = "DCOILBRENTEU"
BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def parse_fred_csv(payload: bytes, *, source_url: str, fetched_at: str) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    rows: list[dict[str, object]] = []
    for record in reader:
        value = record.get(SERIES_ID) or record.get("VALUE")
        date = record.get("DATE") or record.get("observation_date")
        if not date or value in {None, "", "."}:
            continue
        rows.append(
            {
                "date": dt.date.fromisoformat(date).isoformat(),
                "series_id": SERIES_ID,
                "brent_usd_per_barrel": float(value),
                "fetched_at": fetched_at,
                "source_url": source_url,
            }
        )
    return rows


def fetch_brent(
    *,
    date_from: dt.date,
    date_to: dt.date,
    output_path: Path = Path("data/raw/fred_brent_daily.csv"),
    timeout: int = 60,
) -> int:
    query = urllib.parse.urlencode({"id": SERIES_ID, "cosd": date_from.isoformat(), "coed": date_to.isoformat()})
    url = f"{BASE_URL}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "fx-pulse/0.1"})
    with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
        payload = response.read()
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    rows = parse_fred_csv(payload, source_url=url, fetched_at=fetched_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [
                "date", "series_id", "brent_usd_per_barrel", "fetched_at", "source_url"
            ])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", type=dt.date.fromisoformat, default=dt.date(2010, 1, 1))
    parser.add_argument("--to", dest="date_to", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--output", type=Path, default=Path("data/raw/fred_brent_daily.csv"))
    args = parser.parse_args(argv)
    print(f"Wrote {fetch_brent(date_from=args.date_from, date_to=args.date_to, output_path=args.output)} Brent rows")


if __name__ == "__main__":
    main()
