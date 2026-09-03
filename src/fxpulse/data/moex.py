"""Paginated MOEX ISS downloaders for daily history and 10-minute candles."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4


MOEX_BASE = "https://iss.moex.com/iss"
DEFAULT_SECIDS = ("CNYRUB_TOM", "USD000UTSTOM", "KZTRUB_TOM")
DAILY_COLUMNS = (
    "trade_date",
    "secid",
    "open",
    "high",
    "low",
    "close",
    "waprice",
    "volume_rub",
    "num_trades",
    "fetched_at",
    "source_url",
)
CANDLE_COLUMNS = (
    "dt_msk",
    "secid",
    "open",
    "high",
    "low",
    "close",
    "volume_rub",
    "fetched_at",
    "source_url",
)
USER_AGENT = "fx-pulse/0.1 (+https://github.com/ne-olya/fx-pulse)"


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a date in YYYY-MM-DD format") from exc


def _json_rows(payload: dict[str, Any], block: str) -> list[dict[str, Any]]:
    """Convert an ISS table block into records without depending on column order."""

    table = payload.get(block)
    if not table:
        return []
    columns = table.get("columns", [])
    values = table.get("data", [])
    return [dict(zip(columns, row, strict=True)) for row in values]


def _value(record: dict[str, Any], *names: str) -> Any:
    lookup = {str(key).upper(): value for key, value in record.items()}
    for name in names:
        value = lookup.get(name.upper())
        if value is not None:
            return value
    return ""


def _request_json(
    url: str, params: dict[str, str], *, timeout: int, attempts: int = 3
) -> dict[str, Any]:
    """Request one ISS page with bounded retries for transient transport errors."""

    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _paginate(
    *,
    url: str,
    params: dict[str, str],
    block: str,
    timeout: int,
) -> Iterator[tuple[list[dict[str, Any]], str]]:
    """Yield every ISS page; explicit paging prevents silent history truncation."""

    start = 0
    while True:
        page_params = {**params, "start": str(start)}
        request_url = f"{url}?{urllib.parse.urlencode(page_params)}"
        records = _json_rows(_request_json(url, page_params, timeout=timeout), block)
        if not records:
            return
        yield records, request_url
        start += len(records)


def _daily_rows(
    secid: str,
    date_from: dt.date,
    date_to: dt.date,
    *,
    timeout: int,
) -> Iterator[dict[str, Any]]:
    url = f"{MOEX_BASE}/history/engines/currency/markets/selt/boards/CETS/securities/{secid}.json"
    params = {"from": date_from.isoformat(), "till": date_to.isoformat()}
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    for records, request_url in _paginate(url=url, params=params, block="history", timeout=timeout):
        for record in records:
            yield {
                "trade_date": _value(record, "TRADEDATE"),
                "secid": _value(record, "SECID") or secid,
                "open": _value(record, "OPEN"),
                "high": _value(record, "HIGH"),
                "low": _value(record, "LOW"),
                "close": _value(record, "CLOSE", "LEGALCLOSEPRICE", "WAPRICE"),
                "waprice": _value(record, "WAPRICE"),
                # VALUE is the turnover in rubles when available; VOLUME is a
                # fallback for older ISS response schemas.
                "volume_rub": _value(record, "VALUE", "VOLUME"),
                "num_trades": _value(record, "NUMTRADES", "NUM_TRADES"),
                "fetched_at": fetched_at,
                "source_url": request_url,
            }


def _candle_rows(
    secid: str,
    date_from: dt.date,
    date_to: dt.date,
    *,
    interval: int,
    timeout: int,
) -> Iterator[dict[str, Any]]:
    url = f"{MOEX_BASE}/engines/currency/markets/selt/boards/CETS/securities/{secid}/candles.json"
    params = {
        "from": date_from.isoformat(),
        "till": date_to.isoformat(),
        "interval": str(interval),
    }
    fetched_at = dt.datetime.now(dt.UTC).isoformat()
    for records, request_url in _paginate(url=url, params=params, block="candles", timeout=timeout):
        for record in records:
            # END is used deliberately: a candle is only known once it closes.
            yield {
                "dt_msk": _value(record, "END", "BEGIN"),
                "secid": _value(record, "SECID") or secid,
                "open": _value(record, "OPEN"),
                "high": _value(record, "HIGH"),
                "low": _value(record, "LOW"),
                "close": _value(record, "CLOSE"),
                "volume_rub": _value(record, "VALUE", "VOLUME"),
                "fetched_at": fetched_at,
                "source_url": request_url,
            }


def _write_rows(output_path: Path, columns: tuple[str, ...], rows: Iterator[dict[str, Any]]) -> int:
    """Atomically publish a raw snapshot only after every page has been read."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f".{output_path.name}.{uuid4().hex}.tmp"
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def _date_chunks(date_from: dt.date, date_to: dt.date, *, days: int) -> Iterator[tuple[dt.date, dt.date]]:
    if days <= 0:
        raise ValueError("chunk days must be positive")
    start = date_from
    while start <= date_to:
        end = min(start + dt.timedelta(days=days - 1), date_to)
        yield start, end
        start = end + dt.timedelta(days=1)


def fetch_moex_daily(
    *,
    date_from: dt.date,
    date_to: dt.date,
    secids: tuple[str, ...] = DEFAULT_SECIDS,
    output_path: Path = Path("data/raw/moex_daily.csv"),
    timeout: int = 60,
) -> int:
    if date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    rows = (row for secid in secids for row in _daily_rows(secid, date_from, date_to, timeout=timeout))
    return _write_rows(output_path, DAILY_COLUMNS, rows)


def fetch_moex_candles(
    *,
    date_from: dt.date,
    date_to: dt.date,
    secids: tuple[str, ...] = DEFAULT_SECIDS,
    output_path: Path = Path("data/raw/moex_candles.csv"),
    interval: int = 10,
    timeout: int = 60,
    chunk_days: int = 31,
) -> int:
    if interval != 10:
        raise ValueError("The prototype contract supports only 10-minute candles")
    if date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    rows = (
        row
        for secid in secids
        for chunk_from, chunk_to in _date_chunks(date_from, date_to, days=chunk_days)
        for row in _candle_rows(secid, chunk_from, chunk_to, interval=interval, timeout=timeout)
    )
    return _write_rows(output_path, CANDLE_COLUMNS, rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, default_output in (("daily", "data/raw/moex_daily.csv"), ("candles", "data/raw/moex_candles.csv")):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--from", dest="date_from", type=_date, required=True)
        command_parser.add_argument("--to", dest="date_to", type=_date, required=True)
        command_parser.add_argument("--secids", nargs="+", default=list(DEFAULT_SECIDS))
        command_parser.add_argument("--output", type=Path, default=Path(default_output))
        if command == "candles":
            command_parser.add_argument("--chunk-days", type=int, default=31)

    args = parser.parse_args(argv)
    if args.command == "daily":
        count = fetch_moex_daily(
            date_from=args.date_from,
            date_to=args.date_to,
            secids=tuple(args.secids),
            output_path=args.output,
        )
    else:
        count = fetch_moex_candles(
            date_from=args.date_from,
            date_to=args.date_to,
            secids=tuple(args.secids),
            output_path=args.output,
            chunk_days=args.chunk_days,
        )
    print(f"Wrote {count} MOEX {args.command} rows to {args.output}")


if __name__ == "__main__":
    main()
