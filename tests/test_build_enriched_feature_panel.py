from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_enriched_feature_panel",
    ROOT / "scripts" / "build_enriched_feature_panel.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_gzip_feature_artifact_is_byte_stable(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["2026-09-05"], "value": [1.25]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    BUILDER._write_deterministic_csv_gzip(frame, first)
    BUILDER._write_deterministic_csv_gzip(frame, second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
