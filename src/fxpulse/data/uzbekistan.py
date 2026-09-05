"""Point-in-time normalization for Uzbekistan-specific public data.

The parsers intentionally separate an economic observation date from
``known_at``.  Models may join only on ``known_at``; this prevents a monthly or
quarterly value from appearing in a backtest before its publication window.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


CBU_FX_SOURCE = "https://cbu.uz/ru/arkhiv-kursov-valyut/"
CBU_POLICY_SOURCE = "https://cbu.uz/en/services/open_data/portal/"
CBU_RESERVES_SOURCE = "https://cbu.uz/en/statistics/e-gdds/data/111574/"
CBU_BOP_SOURCE = "https://cbu.uz/en/statistics/e-gdds/data/127982/"
UZSTAT_CPI_SOURCE = "https://siat.stat.uz/data/4585/?lang=en"


def normalize_cbu_fx(rates: pd.DataFrame) -> pd.DataFrame:
    """Return one causal daily row with UZS per USD/RUB/CNY."""

    required = {
        "requested_date", "rate_date", "bank", "local_ccy", "quote_ccy",
        "nominal", "local_per_nominal",
    }
    if missing := required - set(rates):
        raise ValueError(f"CBU rate data lacks {sorted(missing)}")
    data = rates.loc[rates["bank"].eq("CBU_UZ") & rates["local_ccy"].eq("UZS")].copy()
    data = data.loc[data["quote_ccy"].isin(["USD", "RUB", "CNY"])]
    data["rate_date"] = pd.to_datetime(data["rate_date"], errors="raise").dt.normalize()
    data["per_unit"] = pd.to_numeric(data["local_per_nominal"], errors="raise") / pd.to_numeric(
        data["nominal"], errors="raise"
    )
    duplicates = data.duplicated(["rate_date", "quote_ccy"], keep=False)
    if duplicates.any():
        inconsistent = data.loc[duplicates].groupby(["rate_date", "quote_ccy"])["per_unit"].nunique().gt(1)
        if inconsistent.any():
            raise ValueError("CBU archive has inconsistent duplicate rates")
    pivot = data.drop_duplicates(["rate_date", "quote_ccy"], keep="last").pivot(
        index="rate_date", columns="quote_ccy", values="per_unit"
    )
    if not {"USD", "RUB", "CNY"}.issubset(pivot.columns):
        raise ValueError("CBU archive must contain USD, RUB and CNY")
    result = pivot[["USD", "RUB", "CNY"]].rename(
        columns={"USD": "usd_uzs", "RUB": "rub_uzs", "CNY": "cny_uzs"}
    ).reset_index()
    # Daily values are treated as usable on the following calendar day.  This
    # is conservative for a recommendation made after an unspecified cut-off.
    result["known_at"] = result["rate_date"] + pd.to_timedelta(1, unit="D")
    result["source_url"] = CBU_FX_SOURCE
    return result.sort_values("known_at", kind="mergesort").reset_index(drop=True)


def normalize_policy_rate(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the CBU open-data policy-rate intervals into change events.

    The published first row says ``from 26.03.2024`` while the preceding
    interval ends on 25.07.2024.  Instead of trusting that overlapping date,
    the open-ended rate starts one day after the previous interval ends.  The
    correction is deterministic and recorded in ``source_note``.
    """

    required = {"Amalqilishmuddati", "Foiz"}
    if missing := required - set(raw):
        raise ValueError(f"policy-rate file lacks {sorted(missing)}")
    intervals: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        text = str(row["Amalqilishmuddati"]).strip()
        dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", text)
        if not dates:
            continue
        intervals.append(
            {
                "start": pd.to_datetime(dates[0], format="%d.%m.%Y"),
                "end": pd.to_datetime(dates[1], format="%d.%m.%Y") if len(dates) > 1 else pd.NaT,
                "policy_rate_pct": float(row["Foiz"]),
                "open_ended": len(dates) == 1,
            }
        )
    data = pd.DataFrame(intervals).sort_values("start", kind="mergesort").reset_index(drop=True)
    if data.empty:
        raise ValueError("policy-rate file produced no intervals")
    source_note = "published interval"
    open_rows = data.index[data["open_ended"]].tolist()
    if len(open_rows) == 1:
        index = open_rows[0]
        previous_end = data.loc[: index - 1, "end"].max() if index else pd.NaT
        if pd.notna(previous_end) and data.at[index, "start"] <= previous_end:
            data.at[index, "start"] = previous_end + pd.to_timedelta(1, unit="D")
            source_note = "open-ended start repaired as previous interval end + 1 day"
    events = data[["start", "policy_rate_pct"]].rename(columns={"start": "observation_date"})
    events = events.sort_values("observation_date", kind="mergesort").drop_duplicates("observation_date", keep="last")
    events["known_at"] = events["observation_date"] + pd.to_timedelta(1, unit="D")
    events["rate_change_pp"] = events["policy_rate_pct"].diff()
    events["decision_changed_rate"] = events["rate_change_pp"].fillna(0).ne(0).astype(float)
    events["source_url"] = CBU_POLICY_SOURCE
    events["source_note"] = source_note
    return events.reset_index(drop=True)


def normalize_cpi(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the national monthly composite CPI index (month over month)."""

    label = next((column for column in ("Klassifikator_en", "Classifier_en") if column in raw), None)
    if label is None:
        raise ValueError("CPI file lacks an English classifier column")
    candidates = raw.loc[raw[label].astype(str).str.upper().isin(["COMPOSITE INDEX", "CUMULATIVE INDEX"])]
    if len(candidates) != 1:
        raise ValueError("CPI file must contain exactly one national composite row")
    row = candidates.iloc[0]
    values: list[dict[str, object]] = []
    for column, value in row.items():
        match = re.fullmatch(r"(\d{4})-[МM](\d{1,2})", str(column))
        if not match or pd.isna(value):
            continue
        month = pd.Period(f"{match.group(1)}-{int(match.group(2)):02d}", freq="M")
        values.append({"observation_date": month.end_time.normalize(), "cpi_mom_index": float(value)})
    result = pd.DataFrame(values).sort_values("observation_date", kind="mergesort")
    if result.empty:
        raise ValueError("CPI file produced no monthly observations")
    # CPI is normally published at the beginning of the following month; ten
    # days is a conservative generic lag when historical release timestamps
    # are not embedded in SIAT's downloadable table.
    result["known_at"] = result["observation_date"] + pd.to_timedelta(10, unit="D")
    result["cpi_mom_pct"] = result["cpi_mom_index"] - 100.0
    result["cpi_3m_annualized_pct"] = (
        (result["cpi_mom_index"].div(100).rolling(3, min_periods=3).apply(np.prod, raw=True) ** 4 - 1) * 100
    )
    result["cpi_12m_pct"] = (
        result["cpi_mom_index"].div(100).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1
    ) * 100
    result["source_url"] = UZSTAT_CPI_SOURCE
    return result.reset_index(drop=True)


def _wide_sdmx(path: Path | str, label_pattern: str, value_name: str, period_kind: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Dataset", header=None)
    header = raw.iloc[3]
    labels = raw.iloc[:, 0].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    indices = labels.index[labels.str.fullmatch(label_pattern, case=False, na=False)].tolist()
    if len(indices) != 1:
        raise ValueError(f"{path} must contain exactly one row matching {label_pattern!r}")
    row = raw.loc[indices[0]]
    values: list[dict[str, object]] = []
    for column_index in range(1, len(header)):
        period = str(header.iloc[column_index]).strip()
        value = pd.to_numeric(row.iloc[column_index], errors="coerce")
        if not np.isfinite(value):
            continue
        if period_kind == "month" and re.fullmatch(r"\d{4}-\d{2}", period):
            observation = pd.Period(period, freq="M").end_time.normalize()
            known_at = observation + pd.to_timedelta(14, unit="D")
        elif period_kind == "quarter" and re.fullmatch(r"\d{4}-Q[1-4]", period):
            observation = pd.Period(period, freq="Q").end_time.normalize()
            known_at = observation + pd.to_timedelta(92, unit="D")
        else:
            continue
        values.append({"observation_date": observation, "known_at": known_at, value_name: float(value)})
    result = pd.DataFrame(values).sort_values("known_at", kind="mergesort")
    if result.empty:
        raise ValueError(f"{path} produced no {period_kind} observations")
    return result.reset_index(drop=True)


def normalize_reserves(path: Path | str) -> pd.DataFrame:
    result = _wide_sdmx(path, r"A\.\s*Official reserve assets", "official_reserves_usd_mn", "month")
    result["reserves_change_1m_pct"] = result["official_reserves_usd_mn"].pct_change(fill_method=None) * 100
    result["reserves_change_3m_pct"] = result["official_reserves_usd_mn"].pct_change(3, fill_method=None) * 100
    result["source_url"] = CBU_RESERVES_SOURCE
    return result


def normalize_remittances(path: Path | str) -> pd.DataFrame:
    # In the official BOP table, secondary-income credits are a broad and
    # reproducible proxy for inbound current transfers.  They are not claimed
    # to equal retail remittances exactly.
    result = _wide_sdmx(path, r"Secondary income, credits", "secondary_income_credits_usd_mn", "quarter")
    value = result["secondary_income_credits_usd_mn"]
    result["secondary_income_qoq_pct"] = value.pct_change(fill_method=None) * 100
    result["secondary_income_yoy_pct"] = value.pct_change(4, fill_method=None) * 100
    result["secondary_income_share_trailing_4q"] = value / value.rolling(4, min_periods=4).sum()
    quarter = result["observation_date"].dt.quarter
    prior_same_quarter = value.groupby(quarter).transform(lambda series: series.expanding().mean().shift())
    result["secondary_income_seasonal_surprise_pct"] = (value / prior_same_quarter - 1) * 100
    result["source_url"] = CBU_BOP_SOURCE
    return result


def normalize_external_balance(path: Path | str) -> pd.DataFrame:
    """Normalize quarterly trade/current-account factors from the official BOP."""

    requested = {
        "current_account_usd_mn": r"A\.\s*Current account balance",
        "goods_exports_usd_mn": r"Goods, credits \(exports\)",
        "goods_imports_usd_mn": r"Goods, debits \(imports\)",
        "goods_services_balance_usd_mn": r"Balance on goods and services",
        "overall_balance_usd_mn": r"E\.\s*Overall Balance",
        "reserve_asset_flow_usd_mn": r"Reserve assets",
    }
    pieces = [
        _wide_sdmx(path, pattern, value_name, "quarter")
        for value_name, pattern in requested.items()
    ]
    result = pieces[0]
    for piece in pieces[1:]:
        result = result.merge(
            piece.drop(columns=[column for column in ("source_url",) if column in piece]),
            on=["observation_date", "known_at"],
            how="outer",
            validate="one_to_one",
        )
    result = result.sort_values("known_at", kind="mergesort").reset_index(drop=True)
    exports = result["goods_exports_usd_mn"].abs()
    imports = result["goods_imports_usd_mn"].abs()
    result["exports_imports_ratio"] = exports / imports.replace(0, np.nan)
    result["current_account_change_yoy_usd_mn"] = result["current_account_usd_mn"].diff(4)
    result["goods_services_balance_change_yoy_usd_mn"] = result["goods_services_balance_usd_mn"].diff(4)
    result["source_url"] = CBU_BOP_SOURCE
    return result.replace([np.inf, -np.inf], np.nan)


def validate_bank_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate real RUB->UZS execution quotes supplied by a bank.

    No quote is synthesized when this dataset is unavailable.  ``bid`` and
    ``ask`` are UZS received per RUB; a higher client rate is better.
    """

    required = {
        "observed_at", "available_at", "provider", "bid_uzs_per_rub",
        "ask_uzs_per_rub", "fee_fixed_rub", "fee_rate", "source_url",
    }
    if missing := required - set(frame):
        raise ValueError(f"bank quote data lacks {sorted(missing)}")
    data = frame.copy()
    for column in ("observed_at", "available_at"):
        data[column] = pd.to_datetime(data[column], errors="raise", utc=True)
    if (data["available_at"] < data["observed_at"]).any():
        raise ValueError("bank quote cannot be available before it was observed")
    for column in ("bid_uzs_per_rub", "ask_uzs_per_rub", "fee_fixed_rub", "fee_rate"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data[["bid_uzs_per_rub", "ask_uzs_per_rub"]] <= 0).any().any():
        raise ValueError("bank bid/ask must be positive")
    if (data[["fee_fixed_rub", "fee_rate"]] < 0).any().any():
        raise ValueError("bank fees cannot be negative")
    if data.duplicated(["available_at", "provider"]).any():
        raise ValueError("duplicate provider quote availability timestamps")
    return data.sort_values("available_at", kind="mergesort").reset_index(drop=True)


__all__ = [
    "normalize_cbu_fx", "normalize_policy_rate", "normalize_cpi",
    "normalize_reserves", "normalize_remittances", "normalize_external_balance",
    "validate_bank_quotes",
]
