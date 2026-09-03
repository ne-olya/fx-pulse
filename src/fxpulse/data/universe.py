"""Correct, point-in-time daily download for the registered MOEX universe.

Fixed securities are fetched from their board-scoped historical endpoint. A
continuous future is built differently: at every past close the loader asks
ISS for contracts with the configured ``ASSETCODE``, selects the most liquid
one using that day's completed fields, then applies a forward ratio adjustment
only when the selected contract changes. No present-day contract is projected
back into history.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import uuid4

from fxpulse.data.moex import MOEX_BASE, _json_rows, _paginate, _request_json, _value
from fxpulse.universe import Instrument, load_universe, universe_sha256


DAILY_COLUMNS = (
    "trade_date",
    "instrument_id",
    "asset_class",
    "secid",
    "engine",
    "market",
    "board",
    "open",
    "high",
    "low",
    "close",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "waprice",
    "turnover",
    "volume",
    "num_trades",
    "open_position",
    "facevalue",
    "normalization",
    "roll_event",
    "fetched_at",
    "source_url",
)
METADATA_COLUMNS = (
    "instrument_id",
    "secid",
    "board",
    "facevalue",
    "lotsize",
    "faceunit",
    "currencyid",
    "metadata_fetched_at",
    "source_url",
)


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a date in YYYY-MM-DD format") from exc


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _write_csv_atomic(path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, object]]) -> tuple[int, int]:
    """Publish one source file only after every record has been written.

    The returned tuple is ``(row_count, invalid_close_count)``. A zero or
    missing close remains visible in raw data and in the manifest rather than
    being forward-filled or silently discarded.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    count = 0
    invalid_close_count = 0
    has_close = "close" in columns
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
                if has_close:
                    close = _number(row.get("close"))
                    if close is None or close <= 0:
                        invalid_close_count += 1
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, invalid_close_count


def _metadata_url(instrument: Instrument) -> str:
    source = instrument.source
    return (
        f"{MOEX_BASE}/engines/{source['engine']}/markets/{source['market']}"
        f"/boards/{source['board']}/securities/{source['secid']}.json"
    )


def fetch_metadata(instrument: Instrument, *, timeout: int = 60) -> dict[str, object]:
    """Fetch one metadata snapshot and enforce the registry price-unit contract."""

    if not instrument.has_fixed_security:
        raise ValueError(f"{instrument.id} has no fixed security metadata")
    url = _metadata_url(instrument)
    payload = _request_json(url, {"iss.only": "securities"}, timeout=timeout)
    records = _json_rows(payload, "securities")
    secid = instrument.source["secid"]
    record = next((item for item in records if str(_value(item, "SECID")) == secid), None)
    if record is None:
        raise ValueError(f"{instrument.id}: ISS metadata does not contain {secid}")
    facevalue = _number(_value(record, "FACEVALUE"))
    if instrument.normalization_kind == "divide_by_facevalue":
        # Runs a metadata assertion before any history row is downloaded.
        instrument.normalize_price(1.0, facevalue=facevalue)
    return {
        "instrument_id": instrument.id,
        "secid": secid,
        "board": instrument.source["board"],
        "facevalue": facevalue,
        "lotsize": _value(record, "LOTSIZE"),
        "faceunit": _value(record, "FACEUNIT"),
        "currencyid": _value(record, "CURRENCYID"),
        "metadata_fetched_at": _now(),
        "source_url": url,
    }


def _normalize(instrument: Instrument, value: Any, *, facevalue: float | None) -> float | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    return instrument.normalize_price(number, facevalue=facevalue)


def _canonical_fixed_rows(
    instrument: Instrument,
    *,
    metadata: dict[str, object],
    date_from: dt.date,
    date_to: dt.date,
    timeout: int,
) -> Iterator[dict[str, object]]:
    """Yield canonical rows for one fixed SECID, preserving invalid raw closes."""

    source = instrument.source
    params = {"from": date_from.isoformat(), "till": date_to.isoformat()}
    facevalue = _number(metadata["facevalue"])
    fetched_at = _now()
    for records, source_url in _paginate(
        url=instrument.history_url(), params=params, block="history", timeout=timeout
    ):
        for record in records:
            raw_close = _number(_value(record, "CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "SETTLEPRICE"))
            yield {
                "trade_date": _value(record, "TRADEDATE"),
                "instrument_id": instrument.id,
                "asset_class": instrument.asset_class,
                "secid": _value(record, "SECID") or source["secid"],
                "engine": source["engine"],
                "market": source["market"],
                "board": _value(record, "BOARDID") or source["board"],
                "open": _normalize(instrument, _value(record, "OPEN"), facevalue=facevalue),
                "high": _normalize(instrument, _value(record, "HIGH"), facevalue=facevalue),
                "low": _normalize(instrument, _value(record, "LOW"), facevalue=facevalue),
                "close": _normalize(instrument, raw_close, facevalue=facevalue),
                "raw_open": _number(_value(record, "OPEN")),
                "raw_high": _number(_value(record, "HIGH")),
                "raw_low": _number(_value(record, "LOW")),
                "raw_close": raw_close,
                "waprice": _number(_value(record, "WAPRICE")),
                "turnover": _number(_value(record, "VALUE")),
                "volume": _number(_value(record, "VOLUME")),
                "num_trades": _number(_value(record, "NUMTRADES", "NUM_TRADES")),
                "open_position": _number(_value(record, "OPENPOSITION")),
                "facevalue": facevalue,
                "normalization": instrument.normalization_kind,
                "roll_event": False,
                "fetched_at": fetched_at,
                "source_url": source_url,
            }


def _weekday_dates(date_from: dt.date, date_to: dt.date) -> Iterator[dt.date]:
    day = date_from
    while day <= date_to:
        if day.weekday() < 5:
            yield day
        day += dt.timedelta(days=1)


def _future_records_for_date(instrument: Instrument, date: dt.date, *, timeout: int) -> tuple[list[dict[str, Any]], str]:
    """Return contracts that were visible for one past FORTS close."""

    source = instrument.source
    url = f"{MOEX_BASE}/history/engines/{source['engine']}/markets/{source['market']}/securities.json"
    params = {"date": date.isoformat(), "assetcode": source["continuous_root"]}
    records: list[dict[str, Any]] = []
    source_url = ""
    for page, page_url in _paginate(url=url, params=params, block="history", timeout=timeout):
        records.extend(page)
        source_url = page_url
    return records, source_url


def _future_candidate(record: dict[str, Any], root: str) -> bool:
    return (
        str(_value(record, "ASSETCODE")) == root
        and (_number(_value(record, "CLOSE", "SETTLEPRICE")) or 0) > 0
        and (_number(_value(record, "NUMTRADES")) or 0) > 0
    )


def _continuous_future_rows(
    instrument: Instrument, *, date_from: dt.date, date_to: dt.date, timeout: int
) -> Iterator[dict[str, object]]:
    """Build a forward-adjusted, liquid daily continuous series.

    At a switch the selected new contract is rescaled to yesterday's adjusted
    close. The adjustment uses only yesterday's selected close and the current
    day's completed close, so appending future dates cannot change past rows.
    """

    if instrument.normalization_kind != "continuous_future" or not instrument.continuous_policy:
        raise ValueError(f"{instrument.id} is not a continuous-future instrument")
    source = instrument.source
    root = source["continuous_root"]
    previous_secid: str | None = None
    previous_adjusted_close: float | None = None
    scale = 1.0
    fetched_at = _now()
    for date in _weekday_dates(date_from, date_to):
        records, source_url = _future_records_for_date(instrument, date, timeout=timeout)
        candidates = [record for record in records if _future_candidate(record, root)]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda record: (
                _number(_value(record, "VALUE")) or 0,
                _number(_value(record, "OPENPOSITION")) or 0,
                _number(_value(record, "VOLUME")) or 0,
                _number(_value(record, "NUMTRADES")) or 0,
                str(_value(record, "SECID")),
            ),
        )
        secid = str(_value(selected, "SECID"))
        raw_close = _number(_value(selected, "CLOSE", "SETTLEPRICE"))
        if raw_close is None or raw_close <= 0:
            continue
        rolled = previous_secid is not None and secid != previous_secid
        if rolled and previous_adjusted_close is not None:
            scale = previous_adjusted_close / raw_close
        normalized = lambda field: (_number(_value(selected, field)) or 0) * scale or None
        adjusted_close = raw_close * scale
        yield {
            "trade_date": _value(selected, "TRADEDATE") or date.isoformat(),
            "instrument_id": instrument.id,
            "asset_class": instrument.asset_class,
            "secid": secid,
            "engine": source["engine"],
            "market": source["market"],
            "board": _value(selected, "BOARDID") or source["board"],
            "open": normalized("OPEN"),
            "high": normalized("HIGH"),
            "low": normalized("LOW"),
            "close": adjusted_close,
            "raw_open": _number(_value(selected, "OPEN")),
            "raw_high": _number(_value(selected, "HIGH")),
            "raw_low": _number(_value(selected, "LOW")),
            "raw_close": raw_close,
            "waprice": _number(_value(selected, "WAPRICE")),
            "turnover": _number(_value(selected, "VALUE")),
            "volume": _number(_value(selected, "VOLUME")),
            "num_trades": _number(_value(selected, "NUMTRADES")),
            "open_position": _number(_value(selected, "OPENPOSITION")),
            "facevalue": None,
            "normalization": instrument.continuous_policy["adjustment"],
            "roll_event": rolled,
            "fetched_at": fetched_at,
            "source_url": source_url,
        }
        previous_secid = secid
        previous_adjusted_close = adjusted_close


def _selected_instruments(
    *,
    universe_path: Path | str,
    include_planned: bool,
) -> tuple[Instrument, ...]:
    instruments = load_universe(universe_path)
    return tuple(
        instrument
        for instrument in instruments
        if instrument.status in {"ready", "candidate"} or (include_planned and instrument.status == "planned")
    )


def fetch_universe_daily(
    *,
    date_from: dt.date,
    date_to: dt.date,
    output_dir: Path | str = Path("data/raw/moex_universe"),
    universe_path: Path | str = Path("configs/moex_universe.json"),
    include_planned: bool = False,
    timeout: int = 60,
) -> dict[str, object]:
    """Download all selected registered series into a manifest-gated snapshot.

    Individual CSVs publish atomically. A consumer must use only a snapshot
    directory containing the final manifest; on any exception the manifest is
    absent and the partial files are deliberately unusable.
    """

    if date_from > date_to:
        raise ValueError("date_from must not be later than date_to")
    snapshot_root = Path(output_dir)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    # A failed refresh must never leave a manifest next to a mixture of old and
    # new source files. Every attempt gets its own immutable candidate folder;
    # only the final manifest makes that folder consumable.
    snapshot_id = f"snapshot-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"
    destination = snapshot_root / snapshot_id
    destination.mkdir()
    instruments = _selected_instruments(universe_path=universe_path, include_planned=include_planned)
    metadata_rows: list[dict[str, object]] = []
    series: list[dict[str, object]] = []
    for instrument in instruments:
        output = destination / f"{instrument.id}.csv"
        if instrument.has_fixed_security:
            metadata = fetch_metadata(instrument, timeout=timeout)
            metadata_rows.append(metadata)
            count, invalid_close_count = _write_csv_atomic(
                output,
                DAILY_COLUMNS,
                _canonical_fixed_rows(
                    instrument,
                    metadata=metadata,
                    date_from=date_from,
                    date_to=date_to,
                    timeout=timeout,
                ),
            )
        else:
            count, invalid_close_count = _write_csv_atomic(
                output,
                DAILY_COLUMNS,
                _continuous_future_rows(instrument, date_from=date_from, date_to=date_to, timeout=timeout),
            )
        series.append(
            {
                "instrument_id": instrument.id,
                "rows": count,
                "invalid_close_rows": invalid_close_count,
                "path": output.name,
            }
        )
    _write_csv_atomic(destination / "metadata.csv", METADATA_COLUMNS, metadata_rows)
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": _now(),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "universe_path": str(universe_path),
        "universe_sha256": universe_sha256(universe_path),
        "include_planned": include_planned,
        "series": series,
    }
    temporary_manifest = destination / f".manifest.{uuid4().hex}.tmp"
    try:
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(destination / "manifest.json")
    except BaseException:
        temporary_manifest.unlink(missing_ok=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, type=_date)
    parser.add_argument("--to", dest="date_to", required=True, type=_date)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/moex_universe"))
    parser.add_argument("--universe", type=Path, default=Path("configs/moex_universe.json"))
    parser.add_argument(
        "--include-planned",
        action="store_true",
        help="also build continuous futures; this performs one point-in-time ISS request per weekday and root",
    )
    args = parser.parse_args(argv)
    manifest = fetch_universe_daily(
        date_from=args.date_from,
        date_to=args.date_to,
        output_dir=args.output_dir,
        universe_path=args.universe,
        include_planned=args.include_planned,
    )
    print(f"Wrote {sum(item['rows'] for item in manifest['series'])} rows for {len(manifest['series'])} instruments")


if __name__ == "__main__":
    main()


__all__ = ["DAILY_COLUMNS", "fetch_metadata", "fetch_universe_daily"]
