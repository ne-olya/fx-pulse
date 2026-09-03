from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fxpulse.panel import PANEL_COLUMNS, load_panel


def _write_cbr_raw(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "rate_date": "2026-01-05",  # known Friday, 2026-01-02 at 15:30 MSK
                "ccy": "USD",
                "nominal": 1,
                "rate_rub": 75.0,
                "fetched_at": "2026-01-02T16:00:00+00:00",
                "source_url": "https://example.test/cbr",
            },
            {
                "rate_date": "2026-01-06",  # known Monday, 2026-01-05 at 15:30 MSK
                "ccy": "USD",
                "nominal": 1,
                "rate_rub": 76.0,
                "fetched_at": "2026-01-05T16:00:00+00:00",
                "source_url": "https://example.test/cbr",
            },
            {
                "rate_date": "2026-01-06",
                "ccy": "EUR",
                "nominal": 1,
                "rate_rub": 88.0,
                "fetched_at": "2026-01-05T16:00:00+00:00",
                "source_url": "https://example.test/cbr",
            },
            {
                "rate_date": "2026-01-07",  # known Tuesday, 2026-01-06 at 15:30 MSK
                "ccy": "USD",
                "nominal": 1,
                "rate_rub": 77.0,
                "fetched_at": "2026-01-06T16:00:00+00:00",
                "source_url": "https://example.test/cbr",
            },
        ]
    ).to_csv(tmp_path / "cbr_daily.csv", index=False)


def test_load_panel_filters_at_information_boundary(tmp_path) -> None:
    _write_cbr_raw(tmp_path)

    as_of = pd.Timestamp("2026-01-05 16:00", tz="Europe/Moscow")
    panel = load_panel("CBR:USD", raw_dir=tmp_path, as_of=as_of)

    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert panel["price"].tolist() == [75.0, 76.0]
    assert (panel["known_at"] <= as_of).all()
    assert panel["value_date"].astype(str).tolist() == ["2026-01-05", "2026-01-06"]


def test_as_of_result_matches_explicitly_truncated_raw_data(tmp_path) -> None:
    """A caller cannot observe a row that became public only after T."""

    _write_cbr_raw(tmp_path)
    as_of = pd.Timestamp("2026-01-05 16:00", tz="Europe/Moscow")
    from_full_raw = load_panel("CBR:USD", raw_dir=tmp_path, as_of=as_of)

    raw = pd.read_csv(tmp_path / "cbr_daily.csv")
    raw.loc[raw["rate_date"] <= "2026-01-06"].to_csv(tmp_path / "cbr_daily.csv", index=False)
    from_truncated_raw = load_panel("CBR:USD", raw_dir=tmp_path, as_of=as_of)

    assert_frame_equal(from_full_raw, from_truncated_raw)


def test_daily_moex_close_is_not_available_before_session_end(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "trade_date": "2026-01-05",
                "secid": "CNYRUB_TOM",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "waprice": 10.05,
                "volume_rub": 1000.0,
                "num_trades": 10,
                "fetched_at": "2026-01-05T21:00:00+00:00",
                "source_url": "https://example.test/moex",
            }
        ]
    ).to_csv(tmp_path / "moex_daily.csv", index=False)

    before_close = load_panel("MOEX:CNYRUB_TOM", raw_dir=tmp_path, as_of="2026-01-05 23:00")
    after_close = load_panel("MOEX:CNYRUB_TOM", raw_dir=tmp_path, as_of="2026-01-06 00:00")

    assert before_close.empty
    assert after_close["price"].tolist() == [10.1]
    assert after_close.iloc[0]["meta"] == {"waprice": 10.05, "volume_rub": 1000.0, "num_trades": 10.0}


def test_panel_excludes_invalid_raw_market_price_without_carrying_it_forward(tmp_path) -> None:
    pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "secid": "CNYRUB_TOM", "close": 0.0},
            {"trade_date": "2026-01-06", "secid": "CNYRUB_TOM", "close": 10.1},
        ]
    ).to_csv(tmp_path / "moex_daily.csv", index=False)

    with pytest.warns(RuntimeWarning, match="non-positive close"):
        panel = load_panel("MOEX:CNYRUB_TOM", raw_dir=tmp_path)

    assert panel["price"].tolist() == [10.1]
    assert panel["is_carried"].tolist() == [False]


def test_kzt_panel_requires_and_applies_facevalue_metadata(tmp_path) -> None:
    pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "secid": "KZTRUB_TOM", "close": 19.03, "facevalue": 100},
        ]
    ).to_csv(tmp_path / "moex_daily.csv", index=False)

    panel = load_panel("MOEX:KZTRUB_TOM", raw_dir=tmp_path)

    assert panel["price"].tolist() == pytest.approx([0.1903])
    assert panel.iloc[0]["meta"] == {"facevalue": 100}


def test_equity_facevalue_is_not_a_price_scale_when_snapshot_says_raw(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "trade_date": "2026-01-05",
                "secid": "SBER",
                "close": 220.0,
                "facevalue": 3.0,
                "normalization": "raw",
            },
        ]
    ).to_csv(tmp_path / "moex_daily.csv", index=False)

    panel = load_panel("MOEX:SBER", raw_dir=tmp_path)

    assert panel["price"].tolist() == [220.0]
