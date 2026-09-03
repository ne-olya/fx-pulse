from __future__ import annotations

import json

import pytest

from fxpulse.universe import instruments_for_status, load_universe, universe_sha256


def test_registered_universe_has_only_one_ready_fixed_security_instrument() -> None:
    universe = load_universe()
    ready = instruments_for_status("ready")

    assert len(universe) >= 10
    assert {instrument.id for instrument in ready} == {"fx_cny_rub_tom"}
    assert all(instrument.has_fixed_security for instrument in ready)
    assert ready[0].history_url().endswith("/currency/markets/selt/boards/CETS/securities/CNYRUB_TOM.json")
    assert ready[0].candles_url().endswith("/currency/markets/selt/boards/CETS/securities/CNYRUB_TOM/candles.json")


def test_facevalue_normalization_prevents_kzt_hundred_unit_quote_from_leaking() -> None:
    kzt = next(instrument for instrument in load_universe() if instrument.id == "fx_kzt_rub_tom")

    assert kzt.normalize_price(19.03, facevalue=100) == pytest.approx(0.1903)
    with pytest.raises(ValueError, match="FACEVALUE changed"):
        kzt.normalize_price(19.03, facevalue=1)


def test_continuous_future_cannot_be_downloaded_as_one_static_security() -> None:
    brent = next(instrument for instrument in load_universe() if instrument.id == "future_brent_liquid")

    with pytest.raises(ValueError, match="point-in-time future resolver"):
        brent.history_url()
    with pytest.raises(ValueError, match="roll-adjustment adapter"):
        brent.normalize_price(100.0, facevalue=None)


def test_rtsi_uses_its_primary_board_not_ended_sndx_board() -> None:
    rtsi = next(instrument for instrument in load_universe() if instrument.id == "index_rtsi")

    assert "/boards/RTSI/securities/RTSI.json" in rtsi.history_url()


def test_universe_rejects_ready_continuous_future(tmp_path) -> None:
    config = json.loads(open("configs/moex_universe.json", encoding="utf-8").read())
    config["instruments"][-1]["status"] = "ready"
    path = tmp_path / "invalid-universe.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must have a fixed secid"):
        load_universe(path)
    assert len(universe_sha256()) == 64
