"""Cached official-rate downloads from recipient-country central banks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import certifi
import pandas as pd


USER_AGENT = "fx-pulse/0.1 (+https://github.com/ne-olya/fx-pulse)"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
QUOTE_CURRENCIES = ("USD", "RUB")
RAW_COLUMNS = (
    "requested_date",
    "rate_date",
    "bank",
    "local_ccy",
    "quote_ccy",
    "nominal",
    "local_per_nominal",
    "is_carried",
    "fetched_at",
    "source_url",
)


def parse_kazakhstan_xml(payload: bytes, *, requested_date: dt.date) -> list[dict[str, object]]:
    root = ET.fromstring(payload)
    rate_date = dt.datetime.strptime(root.findtext("date", ""), "%d.%m.%Y").date()
    if root.find("info") is not None and not root.findall("item"):
        return []
    rows: list[dict[str, object]] = []
    for item in root.findall("item"):
        ccy = item.findtext("title", "").strip().upper()
        if ccy not in QUOTE_CURRENCIES:
            continue
        nominal = int(item.findtext("quant", "0"))
        rate = float(item.findtext("description", "0").replace(",", "."))
        if nominal <= 0 or rate <= 0:
            raise ValueError(f"NBK returned invalid {ccy} rate on {requested_date}")
        rows.append(
            {
                "requested_date": requested_date.isoformat(),
                "rate_date": rate_date.isoformat(),
                "bank": "NBK_KZ",
                "local_ccy": "KZT",
                "quote_ccy": ccy,
                "nominal": nominal,
                "local_per_nominal": rate,
                "is_carried": rate_date != requested_date,
            }
        )
    if {row["quote_ccy"] for row in rows} != set(QUOTE_CURRENCIES):
        raise ValueError(f"NBK response lacks USD or RUB on {requested_date}")
    return rows


def parse_uzbekistan_json(payload: bytes, *, requested_date: dt.date) -> list[dict[str, object]]:
    records = json.loads(payload.decode("utf-8"))
    by_ccy = {str(record.get("Ccy", "")).upper(): record for record in records}
    rows: list[dict[str, object]] = []
    for ccy in QUOTE_CURRENCIES:
        if ccy not in by_ccy:
            raise ValueError(f"CBU UZ response lacks {ccy} on {requested_date}")
        record = by_ccy[ccy]
        nominal = int(record["Nominal"])
        rate = float(str(record["Rate"]).replace(",", "."))
        rate_date = dt.datetime.strptime(record["Date"], "%d.%m.%Y").date()
        if nominal <= 0 or rate <= 0:
            raise ValueError(f"CBU UZ returned invalid {ccy} rate on {requested_date}")
        rows.append(
            {
                "requested_date": requested_date.isoformat(),
                "rate_date": rate_date.isoformat(),
                "bank": "CBU_UZ",
                "local_ccy": "UZS",
                "quote_ccy": ccy,
                "nominal": nominal,
                "local_per_nominal": rate,
                "is_carried": rate_date != requested_date,
            }
        )
    return rows


def _source(bank: str, date: dt.date) -> tuple[str, str]:
    if bank == "NBK_KZ":
        return f"https://nationalbank.kz/rss/get_rates.cfm?fdate={date:%d.%m.%Y}", "xml"
    if bank == "CBU_UZ":
        return f"https://cbu.uz/ru/arkhiv-kursov-valyut/json/all/{date.isoformat()}/", "json"
    raise ValueError(f"unsupported bank {bank}")


def _download(url: str, *, timeout: int, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
                return response.read()
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _payload(bank: str, date: dt.date, *, cache_dir: Path, timeout: int) -> tuple[bytes, str]:
    url, extension = _source(bank, date)
    path = cache_dir / bank.lower() / f"{date.isoformat()}.{extension}"
    if path.is_file():
        return path.read_bytes(), url
    payload = _download(url, timeout=timeout)
    _atomic_bytes(path, payload)
    return payload, url


def _one(bank: str, date: dt.date, *, cache_dir: Path, timeout: int, fetched_at: str) -> list[dict[str, object]]:
    payload, url = _payload(bank, date, cache_dir=cache_dir, timeout=timeout)
    parsed = (
        parse_kazakhstan_xml(payload, requested_date=date)
        if bank == "NBK_KZ"
        else parse_uzbekistan_json(payload, requested_date=date)
    )
    for row in parsed:
        row["fetched_at"] = fetched_at
        row["source_url"] = url
    return parsed


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fetch_recipient_rates(
    *,
    dates: list[dt.date],
    banks: tuple[str, ...] = ("NBK_KZ", "CBU_UZ"),
    output_path: Path = Path("data/raw/recipient_bank_daily.csv"),
    cache_dir: Path = Path("data/raw/recipient_banks/cache"),
    workers: int = 6,
    timeout: int = 60,
) -> int:
    if not dates:
        raise ValueError("at least one date is required")
    if set(banks) - {"NBK_KZ", "CBU_UZ"}:
        raise ValueError("unsupported recipient bank")
    jobs = [(bank, date) for bank in banks for date in sorted(set(dates))]
    fetched_at = dt.datetime.now(dt.UTC).isoformat()

    def run(job: tuple[str, dt.date]) -> list[dict[str, object]]:
        bank, date = job
        return _one(bank, date, cache_dir=cache_dir, timeout=timeout, fetched_at=fetched_at)

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, parsed in enumerate(executor.map(run, jobs), start=1):
            rows.extend(parsed)
            if index % 250 == 0 or index == len(jobs):
                print(f"downloaded or reused {index}/{len(jobs)} bank-date payloads", flush=True)
    rows.sort(key=lambda row: (str(row["requested_date"]), str(row["bank"]), str(row["quote_ccy"])))
    _atomic_csv(output_path, rows)
    return len(rows)


def _dates_from_cbr(path: Path) -> list[dt.date]:
    frame = pd.read_csv(path, usecols=["rate_date"])
    return sorted(pd.to_datetime(frame["rate_date"], errors="raise").dt.date.unique().tolist())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbr-dates", type=Path, default=Path("data/raw/cbr_daily.csv"))
    parser.add_argument("--banks", nargs="+", choices=("NBK_KZ", "CBU_UZ"), default=["NBK_KZ", "CBU_UZ"])
    parser.add_argument("--output", type=Path, default=Path("data/raw/recipient_bank_daily.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/recipient_banks/cache"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    count = fetch_recipient_rates(
        dates=_dates_from_cbr(args.cbr_dates),
        banks=tuple(args.banks),
        output_path=args.output,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )
    print(f"Wrote {count} official recipient-bank rows to {args.output}")


if __name__ == "__main__":
    main()
