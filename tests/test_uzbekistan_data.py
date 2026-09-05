from __future__ import annotations

import pandas as pd
import pytest

from fxpulse.data.uzbekistan import (
    normalize_cbu_fx,
    normalize_cpi,
    normalize_policy_rate,
    validate_bank_quotes,
)
from fxpulse.feature_enrichment import add_uzbekistan_features


def test_normalize_cbu_fx_keeps_three_independent_legs_and_lags_availability():
    rows = []
    for ccy, value in (("USD", 12000), ("RUB", 140), ("CNY", 1700)):
        rows.append(
            {
                "requested_date": "2024-01-10", "rate_date": "2024-01-09", "bank": "CBU_UZ",
                "local_ccy": "UZS", "quote_ccy": ccy, "nominal": 1, "local_per_nominal": value,
            }
        )
    result = normalize_cbu_fx(pd.DataFrame(rows))
    assert result.loc[0, ["usd_uzs", "rub_uzs", "cny_uzs"]].tolist() == [12000, 140, 1700]
    assert result.loc[0, "known_at"] == pd.Timestamp("2024-01-10")


def test_policy_overlap_is_repaired_from_previous_interval_end():
    raw = pd.DataFrame(
        {
            "Amalqilishmuddati": ["from 26.03.2024", "16.03.2023 - 25.07.2024"],
            "Foiz": [13.5, 14.0],
        }
    )
    result = normalize_policy_rate(raw)
    assert result.iloc[-1]["observation_date"] == pd.Timestamp("2024-07-26")
    assert result.iloc[-1]["policy_rate_pct"] == 13.5


def test_cpi_parser_uses_composite_and_conservative_publication_lag():
    raw = pd.DataFrame(
        {
            "Klassifikator_en": ["COMPOSITE INDEX", "FOOD PRODUCTS"],
            "2024-М01": [100.6, 100.8],
            "2024-M02": [100.3, 100.4],
        }
    )
    result = normalize_cpi(raw)
    assert result["cpi_mom_pct"].round(3).tolist() == [0.6, 0.3]
    assert result.iloc[0]["known_at"] == pd.Timestamp("2024-02-10 23:59:59.999999999").normalize()


def test_bank_quote_contract_rejects_lookahead():
    quote = pd.DataFrame(
        {
            "observed_at": ["2026-01-02T12:00:00Z"],
            "available_at": ["2026-01-02T11:59:00Z"],
            "provider": ["fixture"],
            "bid_uzs_per_rub": [150.0],
            "ask_uzs_per_rub": [151.0],
            "fee_fixed_rub": [0.0],
            "fee_rate": [0.01],
            "source_url": ["https://example.test"],
        }
    )
    with pytest.raises(ValueError, match="before"):
        validate_bank_quotes(quote)


def test_uzbek_features_are_causal_and_visible_only_to_uzs():
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-02-10", "2024-02-11", "2024-02-11"]),
            "corridor": ["UZS", "UZS", "AMD"],
            "price": [0.007, 0.007, 0.2],
        }
    )
    fx = pd.DataFrame(
        {
            "rate_date": ["2024-02-09"], "known_at": ["2024-02-10"],
            "usd_uzs": [12000], "rub_uzs": [140], "cny_uzs": [1700],
        }
    )
    policy = pd.DataFrame(
        {
            "observation_date": ["2024-01-01"], "known_at": ["2024-01-02"],
            "policy_rate_pct": [14.0], "rate_change_pp": [0.0], "decision_changed_rate": [0.0],
        }
    )
    monthly = pd.DataFrame(
        {"observation_date": ["2024-01-31"], "known_at": ["2024-02-11"], "value": [1.0]}
    )
    result = add_uzbekistan_features(
        base,
        {
            "fx": fx,
            "policy": policy,
            "inflation": monthly.rename(columns={"value": "cpi_mom_pct"}),
            "reserves": monthly.rename(columns={"value": "official_reserves_usd_mn"}),
            "remittances_proxy": monthly.rename(columns={"value": "secondary_income_credits_usd_mn"}),
        },
    )
    first, second, amd = result.iloc[0], result.iloc[1], result.iloc[2]
    assert first["extra__uzs__inflation__available"] == 0
    assert second["extra__uzs__inflation__available"] == 1
    assert second["extra__uzs__bank__available"] == 0
    assert "extra__uzs__interaction__rub_usd_relative_strength_5" in result
    assert "extra__uzs__macro__annual_transfers_to_reserves" in result
    assert pd.isna(amd["extra__uzs__fx__usd_uzs"])
    assert pd.isna(amd["extra__uzs__interaction__rub_usd_relative_strength_5"])
