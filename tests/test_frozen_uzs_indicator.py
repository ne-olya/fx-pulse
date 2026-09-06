from __future__ import annotations

import gzip
import hashlib
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fxpulse.signals import Config, IndicatorSpec, signals_as_of


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw(path, *, include_future: bool) -> None:
    dates = pd.date_range("2026-05-01", periods=40 if include_future else 30, freq="B")
    pd.DataFrame(
        {
            "rate_date": dates.date,
            "ccy": "UZS",
            "nominal": 10_000,
            "rate_rub": 65.0 + pd.Series(range(len(dates))) / 10,
        }
    ).to_csv(path / "cbr_daily.csv", index=False)


def _write_replay(path) -> tuple[str, str]:
    replay = path / "replay.csv.gz"
    with gzip.GzipFile(filename=str(replay), mode="wb", mtime=0) as stream:
        stream.write(
            b"date,available_at,fired,strength,signal_tier\n"
            b"2026-06-10,2026-06-10T20:00:00+03:00,True,0.87,strong_prediction\n"
        )
    manifest = path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "indicator": "uzs_frozen_consensus",
                "replay_rows": 1,
                "replay_sha256": _sha256(replay),
            }
        ),
        encoding="utf-8",
    )
    return str(replay), str(manifest)


def _config(raw_dir, replay, manifest) -> Config:
    return Config(
        raw_dir=raw_dir,
        corridors={"RUB->UZS": "CBR:UZS"},
        indicators=(
            IndicatorSpec(
                name="uzs_frozen_consensus",
                params={"model_path": replay, "manifest_path": manifest},
                direction="favorable",
                speed="slow",
                days_to_confirm=None,
                scenario="T1",
            ),
        ),
    )


def test_frozen_model_replay_is_unchanged_when_future_raw_rows_are_appended(tmp_path) -> None:
    replay, manifest = _write_replay(tmp_path)
    short = tmp_path / "short"
    full = tmp_path / "full"
    short.mkdir()
    full.mkdir()
    _write_raw(short, include_future=False)
    _write_raw(full, include_future=True)

    from_short = signals_as_of("2026-06-10 20:00", _config(short, replay, manifest))
    from_full = signals_as_of("2026-06-10 20:00", _config(full, replay, manifest))

    assert from_short["indicator"].tolist() == ["uzs_frozen_consensus"]
    assert_frame_equal(from_short, from_full, check_exact=True)


def test_date_after_raw_history_returns_empty_instead_of_reusing_stale_fixing(tmp_path) -> None:
    replay, manifest = _write_replay(tmp_path)
    _write_raw(tmp_path, include_future=True)

    result = signals_as_of("2027-01-01 20:00", _config(tmp_path, replay, manifest))

    assert result.empty


def test_frozen_replay_rejects_artifact_changed_after_manifest(tmp_path) -> None:
    replay, manifest = _write_replay(tmp_path)
    _write_raw(tmp_path, include_future=True)
    with open(replay, "ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="hash differs"):
        signals_as_of("2026-06-10 20:00", _config(tmp_path, replay, manifest))
