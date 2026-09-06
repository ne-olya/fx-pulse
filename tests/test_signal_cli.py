from __future__ import annotations

import pandas as pd

from fxpulse.signal_cli import decision_timestamp, main, render_signals


def _write_history(path) -> None:
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    prices = [10 + index / 100 for index in range(len(dates))]
    pd.DataFrame(
        {
            "rate_date": dates.date,
            "ccy": "UZS",
            "nominal": 1,
            "rate_rub": prices,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "source_url": "https://example.test/cbr",
        }
    ).to_csv(path / "cbr_daily.csv", index=False)


def test_bare_date_uses_evening_moscow_decision_time() -> None:
    timestamp = decision_timestamp("2026-06-10")

    assert timestamp.hour == 20
    assert timestamp.tzinfo is None


def test_render_signals_uses_only_requested_corridor(tmp_path) -> None:
    _write_history(tmp_path)

    frame = render_signals(
        date="2025-04-22",
        raw_dir=tmp_path,
        series=["RUB->UZS=CBR:UZS"],
    )

    assert set(frame["corridor"]) <= {"RUB->UZS"}


def test_cli_prints_stable_csv_and_can_write_it(tmp_path, capsys) -> None:
    _write_history(tmp_path)
    output = tmp_path / "signals.csv"

    main(
        [
            "--date",
            "2025-04-22",
            "--raw-dir",
            str(tmp_path),
            "--series",
            "RUB->UZS=CBR:UZS",
            "--output",
            str(output),
        ]
    )

    printed = capsys.readouterr().out
    assert printed.startswith("date,corridor,series_id,indicator")
    assert output.read_text(encoding="utf-8") == printed


def test_cli_rejects_missing_date(tmp_path) -> None:
    try:
        render_signals(date="", raw_dir=tmp_path)
    except ValueError as exc:
        assert "DATE is required" in str(exc)
    else:
        raise AssertionError("missing date must fail closed")
