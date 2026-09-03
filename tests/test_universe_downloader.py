from __future__ import annotations

import datetime as dt

import pytest

from fxpulse.data import universe as downloader
from fxpulse.universe import load_universe


def _instrument(instrument_id: str):
    return next(instrument for instrument in load_universe() if instrument.id == instrument_id)


def _without_fetch_timestamp(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compare market observations, not the moment a request was made."""
    return [{key: value for key, value in row.items() if key != "fetched_at"} for row in rows]


def test_atomic_writer_counts_invalid_canonical_closes(tmp_path) -> None:
    rows, invalid = downloader._write_csv_atomic(
        tmp_path / "series.csv",
        downloader.DAILY_COLUMNS,
        [{"close": 12.0}, {"close": None}, {"close": 0.0}],
    )

    assert (rows, invalid) == (3, 2)


def test_fixed_rows_normalize_kzt_close_by_facevalue(monkeypatch) -> None:
    kzt = _instrument("fx_kzt_rub_tom")
    records = [
        {
            "TRADEDATE": "2026-09-02",
            "SECID": "KZTRUB_TOM",
            "BOARDID": "CETS",
            "OPEN": 19.0,
            "HIGH": 19.1,
            "LOW": 18.9,
            "CLOSE": 19.03,
            "NUMTRADES": 10,
        }
    ]

    monkeypatch.setattr(
        downloader,
        "_paginate",
        lambda **_: iter([(records, "https://example.test/history?start=0")]),
    )
    rows = list(
        downloader._canonical_fixed_rows(
            kzt,
            metadata={"facevalue": 100},
            date_from=dt.date(2026, 9, 2),
            date_to=dt.date(2026, 9, 2),
            timeout=1,
        )
    )

    assert rows[0]["raw_close"] == 19.03
    assert rows[0]["close"] == pytest.approx(0.1903)
    assert rows[0]["normalization"] == "divide_by_facevalue"


def test_continuous_future_selects_current_liquidity_and_forward_adjusts_roll(monkeypatch) -> None:
    brent = _instrument("future_brent_liquid")
    by_date = {
        dt.date(2026, 1, 5): [
            {"TRADEDATE": "2026-01-05", "SECID": "BRF6", "ASSETCODE": "BR", "CLOSE": 100, "VALUE": 1000, "NUMTRADES": 10},
            {"TRADEDATE": "2026-01-05", "SECID": "BRG6", "ASSETCODE": "BR", "CLOSE": 105, "VALUE": 100, "NUMTRADES": 2},
        ],
        dt.date(2026, 1, 6): [
            {"TRADEDATE": "2026-01-06", "SECID": "BRF6", "ASSETCODE": "BR", "CLOSE": 101, "VALUE": 10, "NUMTRADES": 1},
            {"TRADEDATE": "2026-01-06", "SECID": "BRG6", "ASSETCODE": "BR", "CLOSE": 110, "VALUE": 1000, "NUMTRADES": 10},
        ],
        dt.date(2026, 1, 7): [
            {"TRADEDATE": "2026-01-07", "SECID": "BRG6", "ASSETCODE": "BR", "CLOSE": 121, "VALUE": 1000, "NUMTRADES": 10},
        ],
    }

    monkeypatch.setattr(
        downloader,
        "_future_records_for_date",
        lambda _instrument, date, timeout: (by_date.get(date, []), f"https://example.test/{date.isoformat()}"),
    )
    rows = list(
        downloader._continuous_future_rows(
            brent, date_from=dt.date(2026, 1, 5), date_to=dt.date(2026, 1, 7), timeout=1
        )
    )

    assert [row["secid"] for row in rows] == ["BRF6", "BRG6", "BRG6"]
    assert [row["roll_event"] for row in rows] == [False, True, False]
    assert [row["close"] for row in rows] == pytest.approx([100.0, 100.0, 110.0])


def test_continuous_future_rows_are_unchanged_when_future_dates_are_appended(monkeypatch) -> None:
    brent = _instrument("future_brent_liquid")
    records = {
        dt.date(2026, 1, 5): [{"TRADEDATE": "2026-01-05", "SECID": "BRF6", "ASSETCODE": "BR", "CLOSE": 100, "VALUE": 1, "NUMTRADES": 1}],
        dt.date(2026, 1, 6): [{"TRADEDATE": "2026-01-06", "SECID": "BRF6", "ASSETCODE": "BR", "CLOSE": 101, "VALUE": 1, "NUMTRADES": 1}],
        dt.date(2026, 1, 7): [{"TRADEDATE": "2026-01-07", "SECID": "BRG6", "ASSETCODE": "BR", "CLOSE": 120, "VALUE": 2, "NUMTRADES": 1}],
    }
    monkeypatch.setattr(
        downloader,
        "_future_records_for_date",
        lambda _instrument, date, timeout: (records.get(date, []), f"https://example.test/{date.isoformat()}"),
    )

    before = list(
        downloader._continuous_future_rows(
            brent, date_from=dt.date(2026, 1, 5), date_to=dt.date(2026, 1, 6), timeout=1
        )
    )
    after = list(
        downloader._continuous_future_rows(
            brent, date_from=dt.date(2026, 1, 5), date_to=dt.date(2026, 1, 7), timeout=1
        )
    )[:2]

    assert _without_fetch_timestamp(before) == _without_fetch_timestamp(after)
