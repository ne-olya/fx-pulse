"""Download resumable daily news timelines from the lightweight GDELT DOC API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "fx-pulse-research/0.1"


def load_config(path: Path | str = Path("configs/gdelt_doc.json")) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "registered_before_results":
        raise ValueError("GDELT DOC config must be preregistered schema_version 1")
    if not config.get("queries"):
        raise ValueError("GDELT DOC config must contain queries")
    return config


def _request_parameters(query: str, start: date, end: date) -> dict[str, str]:
    return {
        "query": query,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": end.strftime("%Y%m%d235959"),
        "timelinesmooth": "0",
    }


def _request_url(query: str, start: date, end: date) -> str:
    return f"{BASE_URL}?{urllib.parse.urlencode(_request_parameters(query, start, end))}"


def parse_timeline(payload: dict[str, Any], *, series: str) -> pd.DataFrame:
    timelines = payload.get("timeline", [])
    if len(timelines) != 1 or "data" not in timelines[0]:
        raise ValueError("GDELT DOC response lacks one timeline series")
    rows: list[dict[str, object]] = []
    for item in timelines[0]["data"]:
        timestamp = pd.to_datetime(item["date"], utc=True, errors="raise")
        count = int(item["value"])
        norm = int(item["norm"])
        rows.append(
            {
                "date": timestamp.date().isoformat(),
                "series": series,
                "article_count": count,
                "all_article_count": norm,
                "article_share": count / norm if norm > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def _cache_path(cache_dir: Path, series: str, year: int) -> Path:
    return cache_dir / f"{series}_{year}.json"


def _cache_identity(query: str, start: date, end: date) -> str:
    parameters = json.dumps(_request_parameters(query, start, end), sort_keys=True)
    return hashlib.sha256(parameters.encode("utf-8")).hexdigest()


def _read_cached(path: Path, identity: str, *, series: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("request_sha256") != identity:
        return None
    return parse_timeline(payload["response"], series=series)


def download_timeline(
    *,
    query: str,
    series: str,
    start: date,
    end: date,
    cache_dir: Path,
    timeout: int,
    maximum_attempts: int,
) -> pd.DataFrame:
    """Download one annual slice and preserve it as an atomic checkpoint."""

    path = _cache_path(cache_dir, series, start.year)
    identity = _cache_identity(query, start, end)
    cached = _read_cached(path, identity, series=series)
    if cached is not None:
        print(f"{series} {start.year}: cached ({len(cached)} days)", flush=True)
        return cached

    url = _request_url(query, start, end)
    last_error = "unknown error"
    for attempt in range(maximum_attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            frame = parse_timeline(payload, series=series)
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "request_sha256": identity,
                        "query": query,
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                        "response": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            print(f"{series} {start.year}: downloaded ({len(frame)} days)", flush=True)
            return frame
        except urllib.error.HTTPError as error:
            body = error.read(500).decode("utf-8", errors="replace")
            last_error = f"HTTP {error.code}: {body.strip()}"
            retryable = error.code in {429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == maximum_attempts:
                break
            wait = min(30 * (2**attempt), 300)
            print(f"{series} {start.year}: throttled, retry in {wait}s", flush=True)
            time.sleep(wait)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt + 1 == maximum_attempts:
                break
            wait = min(15 * (2**attempt), 120)
            print(f"{series} {start.year}: retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"GDELT DOC failed for {series} {start.year}: {last_error}")


def run(
    *,
    config: dict[str, Any],
    output: Path,
    cache_dir: Path,
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    start = date_from or date.fromisoformat(config["date_from"])
    end = date_to or date.fromisoformat(config["date_to"])
    if start > end:
        raise ValueError("date_from must not be later than date_to")
    interval = float(config["request_interval_seconds"])
    frames: list[pd.DataFrame] = []
    previous_request_started: float | None = None
    for series, query in config["queries"].items():
        for year in range(start.year, end.year + 1):
            slice_start = max(start, date(year, 1, 1))
            slice_end = min(end, date(year, 12, 31))
            path = _cache_path(cache_dir, series, year)
            identity = _cache_identity(query, slice_start, slice_end)
            cached = _read_cached(path, identity, series=series)
            if cached is not None:
                print(f"{series} {year}: cached ({len(cached)} days)", flush=True)
                frames.append(cached)
                continue
            if previous_request_started is not None:
                wait = interval - (time.monotonic() - previous_request_started)
                if wait > 0:
                    time.sleep(wait)
            previous_request_started = time.monotonic()
            frames.append(
                download_timeline(
                    query=query,
                    series=series,
                    start=slice_start,
                    end=slice_end,
                    cache_dir=cache_dir,
                    timeout=int(config["request_timeout_seconds"]),
                    maximum_attempts=int(config["maximum_attempts"]),
                )
            )
    combined = pd.concat(frames, ignore_index=True).sort_values(["series", "date"])
    if combined.duplicated(["series", "date"]).any():
        raise RuntimeError("GDELT DOC output contains duplicate series/date rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"combined {len(combined)} rows -> {output}", flush=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gdelt_doc.json"))
    parser.add_argument("--from", dest="date_from", type=date.fromisoformat)
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/gdelt_doc"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/gdelt_doc_daily.csv"))
    args = parser.parse_args()
    run(
        config=load_config(args.config),
        output=args.output,
        cache_dir=args.cache_dir,
        date_from=args.date_from,
        date_to=args.date_to,
    )


if __name__ == "__main__":
    main()
