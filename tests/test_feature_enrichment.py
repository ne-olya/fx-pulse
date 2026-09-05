from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fxpulse.feature_enrichment import _strict_previous_asof, add_key_rate_features, build_enriched_features
from fxpulse.universe import universe_sha256


def _registry(tmp_path):
    path = tmp_path / "universe.json"
    instruments = []
    specs = [
        ("fx_cny", "fx", "market_control"),
        ("equity_bank", "equity", "domestic_financial_factor"),
        ("future_wheat", "commodity_future", "agriculture_factor"),
        ("ofz_short", "fixed_income_index", "government_bond_short_duration"),
    ]
    for instrument_id, asset_class, role in specs:
        instruments.append(
            {
                "id": instrument_id,
                "status": "candidate",
                "asset_class": asset_class,
                "role": role,
                "label": instrument_id,
                "source": {"engine": "stock", "market": "index", "board": "SNDX", "secid": instrument_id.upper()},
                "price_unit": "fixture",
                "normalization": {"kind": "raw", "expected_facevalue": None},
                "availability": {"daily_from": "2024-01-01", "intraday_10m_from": None},
                "notes": "fixture",
            }
        )
    path.write_text(json.dumps({"schema_version": 1, "instruments": instruments}), encoding="utf-8")
    return path, specs


def _snapshot(tmp_path, registry, specs, dates):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    series = []
    for offset, (instrument_id, asset_class, _) in enumerate(specs):
        values = 100 + offset + np.arange(len(dates)) * (0.02 + offset * 0.001)
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "instrument_id": instrument_id,
                "close": values,
                "high": values * 1.01,
                "low": values * 0.99,
                "waprice": values * 0.999,
                "turnover": 1_000_000 + np.arange(len(dates)) * 1_000,
                "volume": 10_000 + np.arange(len(dates)) * 10,
                "num_trades": 100 + np.arange(len(dates)),
                "open_position": 500 + np.arange(len(dates)),
            }
        )
        frame.to_csv(snapshot / f"{instrument_id}.csv", index=False)
        series.append({"instrument_id": instrument_id, "path": f"{instrument_id}.csv", "rows": len(dates), "invalid_close_rows": 0})
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot_id": "fixture",
                "universe_sha256": universe_sha256(registry),
                "series": series,
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def _base(dates):
    rows = []
    for corridor in ("AMD", "KGS"):
        price = pd.Series(1 + np.arange(len(dates)) * 0.001)
        returns = price.pct_change(fill_method=None)
        for index, date in enumerate(dates):
            rows.append(
                {
                    "timestamp": date,
                    "corridor": corridor,
                    "price": price.iloc[index],
                    "base__return_1": returns.iloc[index],
                    "base__return_5": price.pct_change(5, fill_method=None).iloc[index],
                    "base__return_20": price.pct_change(20, fill_method=None).iloc[index],
                    "base__volatility_20": returns.rolling(20, min_periods=10).std().iloc[index],
                    "base__percentile_60": 0.1 if 80 <= index <= 82 else 0.5,
                    "regime__shock_strength": 0.0,
                    "regime__volatility_percentile_252": 0.5,
                    "leg__usd_rub_return_1": -0.001,
                    "leg__usd_rub_return_5": -0.002,
                    "outcome__regret_5": float(index % 30),
                }
            )
    return pd.DataFrame(rows)


def test_strict_market_alignment_never_uses_same_day_close():
    market = pd.DataFrame({"feature": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    aligned = _strict_previous_asof(pd.Series(pd.to_datetime(["2026-01-06", "2026-01-07"])), market)
    assert aligned["feature"].tolist() == [1.0, 2.0]


def test_enrichment_builds_all_requested_non_client_groups(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=300)
    registry, specs = _registry(tmp_path)
    snapshot = _snapshot(tmp_path, registry, specs, dates)
    base = _base(dates)
    key_rate = pd.DataFrame({"rate_date": ["2023-12-01", "2024-06-01"], "key_rate_pct": [15.0, 16.0]})

    result = build_enriched_features(base, moex_sources=[(snapshot, registry)], key_rate=key_rate)

    assert len(result.frame) == len(base)
    assert result.frame["outcome__regret_5"].tolist() == base["outcome__regret_5"].tolist()
    assert all(
        result.feature_groups[group]
        for group in result.feature_groups
        if group != "uzbekistan"
    )
    numeric = result.frame[[column for column in result.frame if column.startswith("extra__")]].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_persistent_state_emits_only_one_entry_event(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=300)
    registry, specs = _registry(tmp_path)
    snapshot = _snapshot(tmp_path, registry, specs, dates)
    result = build_enriched_features(_base(dates), moex_sources=[(snapshot, registry)])
    amd = result.frame.loc[result.frame["corridor"].eq("AMD")]

    entries = amd.loc[amd["timestamp"].isin(dates[80:83]), "extra__event__favorable_percentile_entry"]
    assert entries.tolist() == [1.0, 0.0, 0.0]


def test_days_since_rate_change_uses_value_change_not_latest_calendar_row():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-02", "2026-01-07")})
    key_rate = pd.DataFrame(
        {
            "rate_date": pd.date_range("2026-01-01", "2026-01-06"),
            "key_rate_pct": [10.0, 10.0, 10.0, 12.0, 12.0, 12.0],
        }
    )

    enriched = add_key_rate_features(frame, key_rate)

    # The new 12% value is conservatively available on 5 January. The age
    # then increases even though the CBR input has a fresh calendar row daily.
    observed = enriched.set_index("timestamp")["extra__rates__days_since_rate_change"]
    assert observed.loc[pd.Timestamp("2026-01-05")] == 0
    assert observed.loc[pd.Timestamp("2026-01-06")] == 1
    assert observed.loc[pd.Timestamp("2026-01-07")] == 2
