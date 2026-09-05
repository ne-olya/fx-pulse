"""Download an immutable snapshot of official Uzbekistan macro sources."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
import urllib.request

import certifi


USER_AGENT = "fx-pulse/0.1 (+https://github.com/ne-olya/fx-pulse)"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SOURCES = {
    "cbu_policy_rate.csv": {
        "url": "https://cbu.uz/upload/open_data/0008/4-009-0008_uz.csv",
        "landing_page": "https://cbu.uz/uz/services/open_data/portal/",
    },
    "cbu_reserves.xlsx": {
        "url": "https://cbu.uz/sdmx/public/IR_Uzbekistan_MCD_STA.xlsx",
        "landing_page": "https://cbu.uz/en/statistics/e-gdds/data/111574/",
    },
    "cbu_bop.xlsx": {
        "url": "https://cbu.uz/sdmx/public/BOP_Analytical_Uzbekistan.xlsx",
        "landing_page": "https://cbu.uz/en/statistics/e-gdds/data/127982/",
    },
    "uz_cpi_current.csv": {
        "url": "https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_4585.csv",
        "landing_page": "https://siat.stat.uz/data/4585/?lang=en",
    },
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str, *, timeout: float) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:
        payload = response.read()
        final_url = response.geturl()
    if not payload:
        raise ValueError(f"empty response from {url}")
    return payload, final_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        temporary_path = Path(temporary)
        files = []
        for filename, source in SOURCES.items():
            payload, final_url = _download(source["url"], timeout=args.timeout)
            path = temporary_path / filename
            path.write_bytes(payload)
            files.append(
                {
                    "filename": filename,
                    "requested_url": source["url"],
                    "resolved_url": final_url,
                    "landing_page": source["landing_page"],
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            )
        manifest = {
            "schema_version": 1,
            "retrieved_at": retrieved_at,
            "files": files,
            "notes": [
                "Official endpoints are mutable; preserve this snapshot for exact replication.",
                "The SIAT endpoint is the current CPI extract and may not contain the full historical series.",
                "Reserves and balance-of-payments reserve assets are not labeled as interventions.",
            ],
        }
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
