from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from fxpulse.data import cbr, moex
from fxpulse.data.cbr import parse_cbr_xml
from fxpulse.data.moex import _date_chunks, _json_rows, _value, _write_rows


def test_parse_cbr_xml_preserves_nominal_for_panel_normalization() -> None:
    payload = """<?xml version=\"1.0\" encoding=\"windows-1251\"?>
    <ValCurs><Record Date=\"02.09.2026\"><Nominal>100</Nominal><Value>98,7654</Value></Record></ValCurs>""".encode(
        "windows-1251"
    )

    assert parse_cbr_xml(payload) == [(dt.date(2026, 9, 2), 100, 98.7654)]


def test_iss_table_is_read_by_column_name_not_position() -> None:
    payload = {
        "history": {
            "columns": ["CLOSE", "TRADEDATE", "SECID"],
            "data": [[12.34, "2026-09-02", "CNYRUB_TOM"]],
        }
    }
    row = _json_rows(payload, "history")[0]

    assert _value(row, "TRADEDATE") == "2026-09-02"
    assert _value(row, "CLOSE", "WAPRICE") == 12.34


def test_cbr_downloader_writes_to_explicit_path_and_preserves_provenance(tmp_path, monkeypatch) -> None:
    payload = """<?xml version=\"1.0\" encoding=\"windows-1251\"?>
    <ValCurs><Record Date=\"02.09.2026\"><Nominal>100</Nominal><Value>98,7654</Value></Record></ValCurs>""".encode(
        "windows-1251"
    )
    monkeypatch.setattr(cbr, "_download", lambda url, timeout: payload)
    output = tmp_path / "raw" / "cbr_daily.csv"

    count = cbr.fetch_cbr_daily(
        date_from=dt.date(2026, 9, 2),
        date_to=dt.date(2026, 9, 2),
        currencies=("USD",),
        output_path=output,
    )

    saved = pd.read_csv(output)
    assert count == 1
    assert saved.columns.tolist() == list(cbr.RAW_COLUMNS)
    assert saved.loc[0, "ccy"] == "USD"
    assert saved.loc[0, "source_url"].startswith(cbr.CBR_URL)
    assert saved.loc[0, "fetched_at"]


def test_interrupted_cbr_write_keeps_the_previous_snapshot(tmp_path, monkeypatch) -> None:
    output = tmp_path / "cbr.csv"
    output.write_text("previous snapshot\n", encoding="utf-8")

    class InterruptedWriter:
        def writeheader(self) -> None:
            return None

        def writerows(self, _rows) -> None:
            raise TimeoutError("simulated local write interruption")

    monkeypatch.setattr(cbr.csv, "DictWriter", lambda *_args, **_kwargs: InterruptedWriter())

    try:
        cbr._write_rows(output, [{"rate_date": "2026-09-02"}])
    except TimeoutError:
        pass
    else:
        raise AssertionError("The fixture must simulate an interrupted write")

    assert output.read_text(encoding="utf-8") == "previous snapshot\n"
    assert not list(tmp_path.glob(".cbr.csv.*.tmp"))


def test_interrupted_moex_write_keeps_the_previous_snapshot(tmp_path) -> None:
    output = tmp_path / "moex.csv"
    output.write_text("previous snapshot\n", encoding="utf-8")

    def interrupted_rows():
        yield {"value": "first"}
        raise TimeoutError("simulated ISS timeout")

    try:
        _write_rows(output, ("value",), interrupted_rows())
    except TimeoutError:
        pass
    else:
        raise AssertionError("The fixture must simulate an interrupted page stream")

    assert output.read_text(encoding="utf-8") == "previous snapshot\n"
    assert not list(tmp_path.glob(".moex.csv.*.tmp"))


def test_moex_date_chunks_cover_the_requested_range_without_overlap() -> None:
    chunks = list(_date_chunks(dt.date(2026, 1, 1), dt.date(2026, 1, 10), days=4))

    assert chunks == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 4)),
        (dt.date(2026, 1, 5), dt.date(2026, 1, 8)),
        (dt.date(2026, 1, 9), dt.date(2026, 1, 10)),
    ]


def test_moex_daily_cli_does_not_pass_a_candle_only_argument(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(moex, "fetch_moex_daily", lambda **kwargs: calls.append(kwargs) or 1)

    moex.main(
        [
            "daily",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-02",
            "--output",
            str(tmp_path / "daily.csv"),
        ]
    )

    assert calls[0]["date_from"] == dt.date(2026, 1, 1)
    assert "chunk_days" not in calls[0]


def test_moex_candle_cli_passes_hourly_interval(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(moex, "fetch_moex_candles", lambda **kwargs: calls.append(kwargs) or 1)

    moex.main(
        [
            "candles",
            "--from",
            "2026-01-01",
            "--to",
            "2026-01-02",
            "--interval",
            "60",
            "--output",
            str(tmp_path / "hourly.csv"),
        ]
    )

    assert calls[0]["interval"] == 60


def test_moex_candle_downloader_rejects_unsupported_interval(tmp_path) -> None:
    with pytest.raises(ValueError, match="interval"):
        moex.fetch_moex_candles(
            date_from=dt.date(2026, 1, 1),
            date_to=dt.date(2026, 1, 2),
            secids=("CNYRUB_TOM",),
            interval=30,
            output_path=tmp_path / "candles.csv",
        )
