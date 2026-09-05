"""Download expired and live CNY/RUB futures candles from the public MOEX ISS API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


BASE_URL = "https://iss.moex.com/iss/engines/futures/markets/forts/securities"
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
COLUMNS = ["begin", "open", "high", "low", "close", "volume", "value"]


def third_thursday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (3 - first.weekday()) % 7
    return first + pd.offsets.Day(int(offset) + 14)


def contract_codes(start_year: int, end_year: int) -> list[tuple[str, pd.Timestamp]]:
    return [
        (f"CR{code}{year % 10}", third_thursday(year, month))
        for year in range(start_year, end_year + 1)
        for month, code in MONTH_CODES.items()
    ]


def _download_json(url: str, *, retries: int = 4, timeout: int = 30) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "fx-pulse-research/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as error:  # network errors differ between Python/OpenSSL builds
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"MOEX request failed after retries: {url}: {last_error}")


def download_contract(code: str, expiry: pd.Timestamp, *, start: date, end: date) -> tuple[pd.DataFrame, str]:
    query = urllib.parse.urlencode(
        {
            "from": start.isoformat(),
            "till": end.isoformat(),
            "interval": 24,
            "iss.meta": "off",
            "iss.only": "candles",
            "candles.columns": ",".join(COLUMNS),
        }
    )
    url = f"{BASE_URL}/{code}/candles.json?{query}"
    payload = _download_json(url)
    block = payload["candles"]
    frame = pd.DataFrame(block["data"], columns=block["columns"])
    frame["secid"] = code
    frame["expiry"] = expiry.date().isoformat()
    return frame, url


def run(*, start: date, end: date, output: Path, meta_output: Path) -> dict[str, object]:
    parts: list[pd.DataFrame] = []
    urls: list[str] = []
    for code, expiry in contract_codes(start.year, end.year + 1):
        frame, url = download_contract(code, expiry, start=start, end=end)
        if not frame.empty:
            parts.append(frame)
        urls.append(url)
        print(f"{code}: {len(frame)} rows", flush=True)
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*COLUMNS, "secid", "expiry"])
    data["begin"] = pd.to_datetime(data["begin"], errors="raise")
    data = data.sort_values(["begin", "expiry", "secid"], kind="mergesort")
    if data.duplicated(["begin", "secid"]).any():
        raise ValueError("MOEX returned duplicate contract candles")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    meta = {
        "source": "MOEX ISS public API",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "from": start.isoformat(),
        "to": end.isoformat(),
        "contracts_requested": len(urls),
        "contracts_with_rows": int(data["secid"].nunique()),
        "rows": len(data),
        "first_date": data["begin"].min().date().isoformat() if len(data) else None,
        "last_date": data["begin"].max().date().isoformat() if len(data) else None,
        "urls": urls,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--to-date", type=date.fromisoformat, default=date(2026, 9, 3))
    parser.add_argument("--output", type=Path, default=Path("data/raw/moex_cny_futures_daily.csv"))
    parser.add_argument("--meta-output", type=Path, default=Path("data/raw/moex_cny_futures_daily.meta.json"))
    args = parser.parse_args()
    print(json.dumps(run(start=args.from_date, end=args.to_date, output=args.output, meta_output=args.meta_output), indent=2))


if __name__ == "__main__":
    main()
