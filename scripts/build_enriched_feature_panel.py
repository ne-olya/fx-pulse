"""Build one immutable enriched feature panel without fitting any model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fxpulse.feature_enrichment import build_enriched_features  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-stable gzip artifact suitable for a frozen SHA contract."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument(
        "--moex-source",
        action="append",
        nargs=2,
        metavar=("SNAPSHOT", "UNIVERSE"),
        required=True,
    )
    parser.add_argument("--key-rate", type=Path)
    parser.add_argument("--uzbekistan-data", required=True, type=Path)
    parser.add_argument("--bank-quotes", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    base = pd.read_csv(args.features, parse_dates=["timestamp"])
    key_rate = pd.read_csv(args.key_rate) if args.key_rate else None
    uzbekistan = {
        name: pd.read_csv(args.uzbekistan_data / f"{name}.csv")
        for name in ("fx", "policy", "inflation", "reserves", "remittances_proxy", "external_balance")
    }
    bank_quotes = pd.read_csv(args.bank_quotes) if args.bank_quotes else None
    result = build_enriched_features(
        base,
        moex_sources=[(Path(snapshot), Path(universe)) for snapshot, universe in args.moex_source],
        key_rate=key_rate,
        uzbekistan_data=uzbekistan,
        bank_quotes=bank_quotes,
    )
    output = args.output / "features.csv.gz"
    _write_deterministic_csv_gzip(result.frame, output)
    uzs_columns = [column for column in result.frame if column.startswith("extra__uzs__")]
    availability = {
        column: float(result.frame.loc[result.frame["corridor"].eq("UZS"), column].notna().mean())
        for column in uzs_columns
    }
    manifest = {
        "schema_version": 1,
        "base_features": {"path": str(args.features), "sha256": _sha256(args.features)},
        "uzbekistan_data": {
            "path": str(args.uzbekistan_data),
            "manifest_sha256": _sha256(args.uzbekistan_data / "manifest.json"),
        },
        "bank_quotes": (
            {"path": str(args.bank_quotes), "sha256": _sha256(args.bank_quotes)}
            if args.bank_quotes
            else {"status": "not supplied; no synthetic execution data used"}
        ),
        "rows": int(len(result.frame)),
        "columns": int(len(result.frame.columns)),
        "uzbekistan_feature_count": len(uzs_columns),
        "uzbekistan_feature_availability": availability,
        "feature_groups": {name: list(columns) for name, columns in result.feature_groups.items()},
        "provenance": result.provenance,
        "output": {"path": str(output), "sha256": _sha256(output)},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("rows", "columns", "uzbekistan_feature_count")}, indent=2))


if __name__ == "__main__":
    main()
