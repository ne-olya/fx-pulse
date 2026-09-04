"""Download and cache the official Bank of Russia key-rate history."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import ssl
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4

import certifi


BASE_URL = "https://www.cbr.ru/hd_base/KeyRate/"
USER_AGENT = "fx-pulse/0.1 (+https://github.com/ne-olya/fx-pulse)"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
DATE_PATTERN = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
RATE_PATTERN = re.compile(r"^\d+(?:[,.]\d+)?$")
COLUMNS = ("rate_date", "key_rate_pct", "fetched_at", "source_url")


class _CellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside = False
        self._parts: list[str] = []
        self.cells: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "td":
            self._inside = True
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._inside:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td" and self._inside:
            self.cells.append(" ".join("".join(self._parts).split()))
            self._inside = False
            self._parts = []


def parse_key_rate_html(payload: bytes) -> list[tuple[dt.date, float]]:
    parser = _CellParser()
    parser.feed(payload.decode("utf-8"))
    rows: dict[dt.date, float] = {}
    for index, cell in enumerate(parser.cells[:-1]):
        following = parser.cells[index + 1]
        if not DATE_PATTERN.fullmatch(cell) or not RATE_PATTERN.fullmatch(following):
            continue
        date = dt.datetime.strptime(cell, "%d.%m.%Y").date()
        rate = float(following.replace(",", "."))
        if not 0 < rate < 100:
            raise ValueError(f"invalid CBR key rate {rate} on {date}")
        if date in rows and rows[date] != rate:
            raise ValueError(f"conflicting CBR key rates on {date}")
        rows[date] = rate
    if not rows:
        raise ValueError("CBR key-rate page contains no rate rows")
    return sorted(rows.items())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fetch_key_rate(
    *,
    date_from: dt.date,
    date_to: dt.date,
    raw_path: Path = Path("data/raw/cbr_key_rate.html"),
    output_path: Path = Path("data/raw/cbr_key_rate.csv"),
    timeout: int = 60,
) -> int:
    if date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": date_from.strftime("%d.%m.%Y"),
        "UniDbQuery.To": date_to.strftime("%d.%m.%Y"),
    }
    source_url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    if raw_path.is_file():
        payload = raw_path.read_bytes()
    else:
        request = urllib.request.Request(source_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
            payload = response.read()
        _atomic_bytes(raw_path, payload)
    parsed = [row for row in parse_key_rate_html(payload) if date_from <= row[0] <= date_to]
    if not parsed:
        raise ValueError("cached CBR key-rate page does not cover the requested period")
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    rows = [
        {
            "rate_date": date.isoformat(),
            "key_rate_pct": rate,
            "fetched_at": fetched_at,
            "source_url": source_url,
        }
        for date, rate in parsed
    ]
    _atomic_csv(output_path, rows)
    return len(rows)


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", type=_date, required=True)
    parser.add_argument("--to", dest="date_to", type=_date, required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/cbr_key_rate.html"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/cbr_key_rate.csv"))
    args = parser.parse_args(argv)
    count = fetch_key_rate(
        date_from=args.date_from,
        date_to=args.date_to,
        raw_path=args.raw,
        output_path=args.output,
    )
    print(f"Wrote {count} official CBR key-rate rows to {args.output}")


if __name__ == "__main__":
    main()
