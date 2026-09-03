"""Downloader for daily official Bank of Russia FX rates.

The CBR endpoint returns effective dates, not publication timestamps. The panel
layer owns the publication-time convention; this module persists source values
without inventing timestamps.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4


CBR_SERIES: dict[str, str] = {
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
    "TJS": "R01670",
    "UZS": "R01717",
    "KGS": "R01370",
    "KZT": "R01335",
    "AMD": "R01060",
}
CBR_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
RAW_COLUMNS = ("rate_date", "ccy", "nominal", "rate_rub", "fetched_at", "source_url")
USER_AGENT = "fx-pulse/0.1 (+https://github.com/ne-olya/fx-pulse)"


def _format_url(currency_id: str, date_from: dt.date, date_to: dt.date) -> str:
    query = urllib.parse.urlencode(
        {
            "date_req1": date_from.strftime("%d/%m/%Y"),
            "date_req2": date_to.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": currency_id,
        }
    )
    return f"{CBR_URL}?{query}"


def parse_cbr_xml(payload: bytes) -> list[tuple[dt.date, int, float]]:
    """Return `(effective_date, nominal, rate_in_rubles)` records from CBR XML."""

    root = ET.fromstring(payload.decode("windows-1251"))
    records: list[tuple[dt.date, int, float]] = []
    for record in root.findall("Record"):
        effective_date = dt.datetime.strptime(record.attrib["Date"], "%d.%m.%Y").date()
        nominal = int(record.findtext("Nominal", "").replace(",", "."))
        rate_rub = float(record.findtext("Value", "").replace(",", "."))
        if nominal <= 0 or rate_rub <= 0:
            raise ValueError(f"CBR returned a non-positive rate for {effective_date}")
        records.append((effective_date, nominal, rate_rub))
    return records


def _download(url: str, *, timeout: int, attempts: int = 3) -> bytes:
    """Read one CBR response with bounded retry for transient failures."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _write_rows(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    """Atomically replace a raw snapshot after the full response set is ready."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f".{output_path.name}.{uuid4().hex}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def fetch_cbr_daily(
    *,
    date_from: dt.date,
    date_to: dt.date,
    currencies: tuple[str, ...] = tuple(CBR_SERIES),
    output_path: Path = Path("data/raw/cbr_daily.csv"),
    timeout: int = 60,
) -> int:
    """Download CBR rows into the raw-data contract and return their count."""

    if date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    unknown = sorted(set(currencies) - set(CBR_SERIES))
    if unknown:
        raise ValueError(f"Unsupported CBR currency codes: {', '.join(unknown)}")

    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    rows: list[dict[str, str | int]] = []
    for ccy in currencies:
        url = _format_url(CBR_SERIES[ccy], date_from, date_to)
        for rate_date, nominal, rate_rub in parse_cbr_xml(_download(url, timeout=timeout)):
            rows.append(
                {
                    "rate_date": rate_date.isoformat(),
                    "ccy": ccy,
                    "nominal": nominal,
                    "rate_rub": f"{rate_rub:.10f}",
                    "fetched_at": fetched_at,
                    "source_url": url,
                }
            )

    rows.sort(key=lambda row: (str(row["rate_date"]), str(row["ccy"])))
    _write_rows(output_path, rows)
    return len(rows)


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a date in YYYY-MM-DD format") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", type=_date, default=dt.date(2018, 1, 1))
    parser.add_argument("--to", dest="date_to", type=_date, default=dt.date.today())
    parser.add_argument("--currencies", nargs="+", choices=sorted(CBR_SERIES), default=list(CBR_SERIES))
    parser.add_argument("--output", type=Path, default=Path("data/raw/cbr_daily.csv"))
    args = parser.parse_args(argv)

    count = fetch_cbr_daily(
        date_from=args.date_from,
        date_to=args.date_to,
        currencies=tuple(args.currencies),
        output_path=args.output,
    )
    print(f"Wrote {count} CBR rows to {args.output}")


if __name__ == "__main__":
    main()
