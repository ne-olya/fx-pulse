"""Download and cache relevant GDELT 2.0 event records without cloud credentials."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


BASE_URL = "https://data.gdeltproject.org/gdeltv2"
ACTOR_CODES = {
    "RUS": "RUS",
    "AMD": "ARM",
    "KGS": "KGZ",
    "KZT": "KAZ",
    "TJS": "TJK",
    "UZS": "UZB",
}
GEO_CODES = {
    "RUS": "RS",
    "AMD": "AM",
    "KGS": "KG",
    "KZT": "KZ",
    "TJS": "TI",
    "UZS": "UZ",
}
CORRIDORS = ("AMD", "KGS", "KZT", "TJS", "UZS")
RAW_COLUMNS = (
    "timestamp_utc",
    "event_id",
    "source_url",
    "source_name",
    "actor_country_codes",
    "geo_country_codes",
    "tone",
    "quad_class",
    "root_code",
    "goldstein_scale",
)


@dataclass(frozen=True)
class DownloadResult:
    timestamp: datetime
    rows: list[dict[str, object]]
    error: str | None = None
    missing: bool = False


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_name(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def parse_export_archive(payload: bytes) -> list[dict[str, object]]:
    """Extract only Russia/recipient-country events from one 15-minute ZIP."""

    relevant_actor = set(ACTOR_CODES.values())
    relevant_geo = set(GEO_CODES.values())
    output: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError("a GDELT export archive must contain exactly one CSV")
        with archive.open(names[0]) as binary:
            reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8"), delimiter="\t")
            for row in reader:
                if len(row) < 61:
                    continue
                actor_codes = {code for code in (row[7], row[17]) if code}
                geo_codes = {code for code in (row[37], row[45], row[53]) if code}
                if not (actor_codes & relevant_actor or geo_codes & relevant_geo):
                    continue
                try:
                    timestamp = datetime.strptime(row[59], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                output.append(
                    {
                        "timestamp_utc": timestamp.isoformat(),
                        "event_id": row[0],
                        "source_url": row[60],
                        "source_name": _source_name(row[60]),
                        "actor_country_codes": ";".join(sorted(actor_codes)),
                        "geo_country_codes": ";".join(sorted(geo_codes)),
                        "tone": _float(row[34]),
                        "quad_class": _integer(row[29]),
                        "root_code": _integer(row[28]),
                        "goldstein_scale": _float(row[30]),
                    }
                )
    return output


def _download_one(timestamp: datetime, *, retries: int, timeout: int) -> DownloadResult:
    stamp = timestamp.strftime("%Y%m%d%H%M%S")
    url = f"{BASE_URL}/{stamp}.export.CSV.zip"
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fx-pulse-research/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return DownloadResult(timestamp=timestamp, rows=parse_export_archive(response.read()))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return DownloadResult(timestamp=timestamp, rows=[], missing=True)
            last_error = f"HTTPError {error.code}: {error.reason}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
        except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return DownloadResult(timestamp=timestamp, rows=[], error=last_error)


def _timestamps(start: date, end: date) -> list[datetime]:
    current = datetime.combine(start, datetime_time.min, tzinfo=timezone.utc)
    stop = datetime.combine(end + timedelta(days=1), datetime_time.min, tzinfo=timezone.utc)
    values: list[datetime] = []
    while current < stop:
        values.append(current)
        current += timedelta(minutes=15)
    return values


def _codes(value: object) -> set[str]:
    return {item for item in str(value).split(";") if item and item != "nan"}


def aggregate_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse many event rows into one point-in-time row per hour/corridor."""

    if raw.empty:
        return pd.DataFrame()
    data = raw.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="raise").dt.floor("h")
    expanded: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        actors = _codes(row.actor_country_codes)
        geos = _codes(row.geo_country_codes)
        has_russia = ACTOR_CODES["RUS"] in actors or GEO_CODES["RUS"] in geos
        for corridor in CORRIDORS:
            has_recipient = ACTOR_CODES[corridor] in actors or GEO_CODES[corridor] in geos
            if not (has_russia or has_recipient):
                continue
            expanded.append(
                {
                    "timestamp_utc": row.timestamp_utc,
                    "corridor": corridor,
                    "article_id": row.source_url or row.event_id,
                    "source_name": row.source_name,
                    "tone": row.tone,
                    "has_russia": has_russia,
                    "has_recipient": has_recipient,
                    "is_conflict": int(row.quad_class is not None and row.quad_class >= 3),
                    "is_material_conflict": int(row.quad_class == 4),
                    "is_cooperation": int(row.quad_class is not None and row.quad_class <= 2),
                    "is_coercion": int(row.root_code in (16, 17)),
                    "is_protest": int(row.root_code == 14),
                    "goldstein_scale": row.goldstein_scale,
                }
            )
    if not expanded:
        return pd.DataFrame()
    events = pd.DataFrame(expanded)
    # An article often yields several GDELT event rows. It must still count as
    # one piece of news inside an hour/corridor.
    articles = (
        events.groupby(["timestamp_utc", "corridor", "article_id"], as_index=False)
        .agg(
            source_name=("source_name", "first"),
            tone=("tone", "mean"),
            has_russia=("has_russia", "max"),
            has_recipient=("has_recipient", "max"),
            is_conflict=("is_conflict", "max"),
            is_material_conflict=("is_material_conflict", "max"),
            is_cooperation=("is_cooperation", "max"),
            is_coercion=("is_coercion", "max"),
            is_protest=("is_protest", "max"),
            goldstein_scale=("goldstein_scale", "mean"),
        )
    )
    articles["is_bilateral"] = articles["has_russia"] & articles["has_recipient"]
    articles["negative"] = articles["tone"].le(-2.0)
    articles["russia_tone"] = articles["tone"].where(articles["has_russia"])
    articles["recipient_tone"] = articles["tone"].where(articles["has_recipient"])
    articles["goldstein_observed"] = articles["goldstein_scale"].notna()

    output = (
        articles.groupby(["timestamp_utc", "corridor"], as_index=False)
        .agg(
            article_count=("article_id", "size"),
            russia_article_count=("has_russia", "sum"),
            recipient_article_count=("has_recipient", "sum"),
            bilateral_article_count=("is_bilateral", "sum"),
            source_count=("source_name", "nunique"),
            tone_sum=("tone", "sum"),
            tone_count=("tone", "count"),
            russia_tone_sum=("russia_tone", "sum"),
            russia_tone_count=("russia_tone", "count"),
            recipient_tone_sum=("recipient_tone", "sum"),
            recipient_tone_count=("recipient_tone", "count"),
            negative_count=("negative", "sum"),
            conflict_count=("is_conflict", "sum"),
            material_conflict_count=("is_material_conflict", "sum"),
            cooperation_count=("is_cooperation", "sum"),
            coercion_count=("is_coercion", "sum"),
            protest_count=("is_protest", "sum"),
            goldstein_sum=("goldstein_scale", "sum"),
            goldstein_count=("goldstein_observed", "sum"),
        )
        .sort_values(["timestamp_utc", "corridor"])
        .reset_index(drop=True)
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _month_bounds(month: pd.Period, start: date, end: date) -> tuple[date, date]:
    month_start = month.start_time.date()
    month_end = month.end_time.date()
    return max(start, month_start), min(end, month_end)


def download_month(
    month: pd.Period,
    *,
    start: date,
    end: date,
    output_dir: Path,
    workers: int,
    retries: int,
    timeout: int,
) -> tuple[Path, Path]:
    """Download one month atomically; an existing manifest is a checkpoint."""

    raw_path = output_dir / f"events_{month}.csv.gz"
    hourly_path = output_dir / f"hourly_{month}.csv.gz"
    meta_path = output_dir / f"events_{month}.meta.json"
    month_start, month_end = _month_bounds(month, start, end)
    if raw_path.exists() and hourly_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("from") == month_start.isoformat() and metadata.get("to") == month_end.isoformat():
            print(f"{month}: cached", flush=True)
            return raw_path, hourly_path

    timestamps = _timestamps(month_start, month_end)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    missing_batches: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda item: _download_one(item, retries=retries, timeout=timeout),
            timestamps,
        )
        for result in results:
            rows.extend(result.rows)
            if result.missing:
                missing_batches.append(result.timestamp.isoformat())
            if result.error:
                failures.append({"timestamp": result.timestamp.isoformat(), "error": result.error})
    if failures:
        sample = "; ".join(f"{item['timestamp']}: {item['error']}" for item in failures[:3])
        raise RuntimeError(f"{month}: {len(failures)} GDELT batches failed; {sample}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_raw(raw_path, rows)
    hourly = aggregate_hourly(pd.DataFrame(rows, columns=RAW_COLUMNS))
    temporary_hourly = hourly_path.with_suffix(hourly_path.suffix + ".tmp")
    hourly.to_csv(temporary_hourly, index=False, compression="gzip")
    temporary_hourly.replace(hourly_path)
    metadata = {
        "source": "GDELT 2.0 Event export files",
        "month": str(month),
        "from": month_start.isoformat(),
        "to": month_end.isoformat(),
        "expected_15m_batches": len(timestamps),
        "available_15m_batches": len(timestamps) - len(missing_batches),
        "missing_15m_batches": len(missing_batches),
        "missing_batch_timestamps": missing_batches,
        "failed_batches": 0,
        "filtered_event_rows": len(rows),
        "hourly_rows": len(hourly),
        "raw_sha256": _sha256(raw_path),
        "hourly_sha256": _sha256(hourly_path),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{month}: {len(rows)} events -> {len(hourly)} hourly rows; "
        f"{len(missing_batches)} source batches absent",
        flush=True,
    )
    return raw_path, hourly_path


def combine_hourly(paths: list[Path], output: Path) -> None:
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    combined = (
        combined.groupby(["timestamp_utc", "corridor"], as_index=False)
        .sum(numeric_only=True)
        .sort_values(["timestamp_utc", "corridor"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat, default=date(2026, 9, 2))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/gdelt"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/gdelt_hourly_news.csv")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if args.date_from > args.date_to:
        raise ValueError("--from must not be later than --to")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    months = pd.period_range(args.date_from, args.date_to, freq="M")
    hourly_paths: list[Path] = []
    for month in months:
        _, hourly_path = download_month(
            month,
            start=args.date_from,
            end=args.date_to,
            output_dir=args.output_dir,
            workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
        )
        hourly_paths.append(hourly_path)
    combine_hourly(hourly_paths, args.output)
    print(f"combined hourly data: {args.output}", flush=True)


if __name__ == "__main__":
    main()
