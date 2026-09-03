from __future__ import annotations

import random

import pandas as pd
from pandas.testing import assert_frame_equal

from fxpulse.panel import load_panel
from fxpulse.signals import Config, IndicatorSpec, SIGNAL_COLUMNS, signals_as_of


def _write_cbr_history(path) -> None:
    dates = pd.date_range("2025-01-01", periods=140, freq="B")
    prices = [10 + index / 100 for index in range(len(dates))]
    pd.DataFrame(
        {
            "rate_date": dates.date,
            "ccy": "TJS",
            "nominal": 1,
            "rate_rub": prices,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "source_url": "https://example.test/cbr",
        }
    ).to_csv(path / "cbr_daily.csv", index=False)


def _config(raw_dir) -> Config:
    return Config(
        raw_dir=raw_dir,
        corridors={"RUB->TJS": "CBR:TJS"},
        indicators=(
            IndicatorSpec(
                name="level_percentile",
                params={"window": 20, "pct": 100},
                direction="favorable",
                speed="slow",
                days_to_confirm=None,
                scenario="T1",
            ),
        ),
    )


def test_signals_as_of_is_bitwise_equal_for_full_and_truncated_data_on_50_dates(tmp_path) -> None:
    """The mandatory CI guard: future raw rows cannot alter historic signals."""

    _write_cbr_history(tmp_path)
    raw = pd.read_csv(tmp_path / "cbr_daily.csv")
    full_config = _config(tmp_path)
    visible_panel = load_panel("CBR:TJS", raw_dir=tmp_path)
    sampled_indices = random.Random(7).sample(range(25, 115), 50)

    for index in sampled_indices:
        as_of = visible_panel.loc[index, "known_at"] + pd.DateOffset(minutes=1)
        from_full = signals_as_of(as_of, full_config)
        visible_dates = set(
            load_panel("CBR:TJS", raw_dir=tmp_path, as_of=as_of)["value_date"].astype(str)
        )
        truncated_dir = tmp_path / f"truncated-{index}"
        truncated_dir.mkdir()
        raw.loc[raw["rate_date"].isin(visible_dates)].to_csv(truncated_dir / "cbr_daily.csv", index=False)

        from_truncated = signals_as_of(as_of, _config(truncated_dir))
        assert_frame_equal(from_full, from_truncated, check_exact=True)


def test_signal_schema_is_stable_when_no_indicator_has_sufficient_history(tmp_path) -> None:
    _write_cbr_history(tmp_path)
    config = Config(
        raw_dir=tmp_path,
        corridors={"RUB->TJS": "CBR:TJS"},
        indicators=(
            IndicatorSpec(
                name="level_percentile",
                params={"window": 500, "pct": 10},
                direction="favorable",
                speed="slow",
                days_to_confirm=None,
                scenario="T1",
            ),
        ),
    )

    signals = signals_as_of("2025-02-01 16:00", config)

    assert signals.empty
    assert signals.columns.tolist() == list(SIGNAL_COLUMNS)
