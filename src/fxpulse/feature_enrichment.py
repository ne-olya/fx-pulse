"""Causal feature enrichment for cross-border transfer models.

The module adds only market, macro and model-context data.  It deliberately
uses bank execution quotes only when a real point-in-time history is explicitly
provided; it never synthesizes them.  Every completed MOEX daily bar is joined
to a later target date (strictly earlier source date), and event features
represent state *entries*, not persistent states.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fxpulse.universe import Instrument, load_universe, universe_sha256


EXTRA_PREFIX = "extra__"
FEATURE_GROUP_PREFIXES = {
    "liquidity": "extra__liquidity__",
    "cross_currency": "extra__cross__",
    "rates": "extra__rates__",
    "commodities": "extra__commodity__",
    "equities": "extra__equity__",
    "regimes": "extra__regime__",
    "events": "extra__event__",
    "uzbekistan": "extra__uzs__",
}


@dataclass(frozen=True)
class EnrichmentResult:
    frame: pd.DataFrame
    feature_groups: dict[str, tuple[str, ...]]
    provenance: dict[str, Any]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _rolling_zscore(values: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    minimum = window if minimum is None else minimum
    mean = values.rolling(window, min_periods=minimum).mean()
    scale = values.rolling(window, min_periods=minimum).std().replace(0, np.nan)
    return (values - mean) / scale


def _rolling_percentile(values: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    minimum = window if minimum is None else minimum
    return values.rolling(window, min_periods=minimum).rank(pct=True)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _economic_block(instrument: Instrument) -> tuple[str, str]:
    role = instrument.role
    if instrument.asset_class == "fx":
        return "cross", "fx_market"
    if instrument.asset_class == "fixed_income_index":
        return "rates", "ofz"
    if instrument.asset_class in {"precious_metal", "commodity_future"}:
        if any(token in role for token in ("agriculture", "food", "wheat", "sugar")):
            return "commodity", "agriculture"
        if any(token in role for token in ("oil", "gas", "energy")):
            return "commodity", "energy"
        return "commodity", "metals"
    if instrument.asset_class == "equity_index":
        return "equity", "broad_market"
    if any(token in role for token in ("oil", "gas", "energy")):
        return "equity", "energy_exporters"
    if any(token in role for token in ("metal", "diamond", "fertilizer", "coal", "agriculture")):
        return "equity", "materials_exporters"
    if any(token in role for token in ("financial", "capital_market", "conglomerate")):
        return "equity", "financial_system"
    if any(token in role for token in ("electricity", "grid")):
        return "equity", "utilities"
    if any(token in role for token in ("transport", "port", "industrial")):
        return "equity", "transport_industrial"
    if any(token in role for token in ("consumption", "housing", "health")):
        return "equity", "consumption_housing"
    if any(token in role for token in ("telecom", "digital", "labor", "urban")):
        return "equity", "digital_services"
    raise ValueError(f"no economic block for {instrument.id}: {instrument.asset_class}/{role}")


def _read_snapshot(
    snapshot_dir: Path,
    universe_path: Path,
) -> tuple[dict[str, Instrument], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{snapshot_dir} is not a manifest-gated snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("universe_sha256") != universe_sha256(universe_path):
        raise ValueError(f"{snapshot_dir} was built from another universe registry")
    registry = {instrument.id: instrument for instrument in load_universe(universe_path)}
    rows = manifest.get("series")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest_path} has no series")
    unknown = {str(item.get("instrument_id")) for item in rows} - set(registry)
    if unknown:
        raise ValueError(f"snapshot contains unregistered instruments: {sorted(unknown)}")
    return registry, rows, manifest


def _load_instrument(path: Path, instrument_id: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"trade_date", "instrument_id", "close"}
    if missing := required - set(raw):
        raise ValueError(f"{path} lacks {sorted(missing)}")
    if raw["instrument_id"].nunique() != 1 or str(raw["instrument_id"].iloc[0]) != instrument_id:
        raise ValueError(f"{path} does not match {instrument_id}")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="raise").dt.normalize()
    if raw["trade_date"].duplicated().any():
        raise ValueError(f"{instrument_id} has duplicate trade dates")
    raw = raw.sort_values("trade_date", kind="mergesort").set_index("trade_date")
    for column in ("open", "high", "low", "close", "waprice", "turnover", "volume", "num_trades", "open_position"):
        if column not in raw:
            raw[column] = np.nan
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    # Technical zeros remain in raw snapshots but are never prices or market activity.
    raw["close"] = raw["close"].where(raw["close"].gt(0))
    for column in ("turnover", "volume", "num_trades", "open_position"):
        raw[column] = raw[column].where(raw[column].gt(0))
    return raw


def _instrument_features(raw: pd.DataFrame) -> pd.DataFrame:
    close = raw["close"]
    returns_1 = close.pct_change(fill_method=None)
    # Large unadjusted corporate actions must not dominate a sector median.
    returns_1 = returns_1.where(returns_1.abs().le(0.35))
    result = pd.DataFrame(index=raw.index)
    result["return_1"] = returns_1
    for window in (5, 20):
        result[f"return_{window}"] = close.pct_change(window, fill_method=None).where(lambda value: value.abs().le(1.0))
    result["volatility_20"] = returns_1.rolling(20, min_periods=12).std()
    valid_range = raw["high"].gt(0) & raw["low"].gt(0) & close.gt(0)
    result["range_pct"] = ((raw["high"] - raw["low"]) / close).where(valid_range)
    result["close_vs_wap"] = (close / raw["waprice"] - 1).where(raw["waprice"].gt(0))
    for source, short in (("turnover", "turnover"), ("volume", "volume"), ("num_trades", "trades")):
        logged = np.log1p(raw[source])
        result[f"log_{short}"] = logged
        result[f"{short}_zscore_20"] = _rolling_zscore(logged, 20, 12)
        result[f"{short}_change_5"] = logged.diff(5)
    per_trade = raw["turnover"] / raw["num_trades"]
    result["log_turnover_per_trade"] = np.log1p(per_trade.where(per_trade.gt(0)))
    result["illiquidity"] = (returns_1.abs() / (raw["turnover"] / 1e9)).where(raw["turnover"].gt(0))
    result["open_position_zscore_20"] = _rolling_zscore(np.log1p(raw["open_position"]), 20, 12)
    return result.replace([np.inf, -np.inf], np.nan)


def _eligible(item: dict[str, Any], instrument: Instrument, maximum_rows: int) -> bool:
    rows = int(item.get("rows", 0))
    invalid = int(item.get("invalid_close_rows", rows))
    if rows < 252 or invalid / max(rows, 1) > 0.05:
        return False
    minimum_fraction = 0.35 if instrument.asset_class == "commodity_future" else 0.80
    return rows >= math.ceil(maximum_rows * minimum_fraction)


def _aggregate_block(
    members: dict[str, pd.DataFrame],
    *,
    family: str,
    block: str,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in members.values()))))
    output = pd.DataFrame(index=index)
    price_metrics = ("return_1", "return_5", "return_20", "volatility_20", "range_pct", "close_vs_wap")
    liquidity_metrics = (
        "log_turnover", "turnover_zscore_20", "turnover_change_5",
        "log_volume", "volume_zscore_20", "volume_change_5",
        "log_trades", "trades_zscore_20", "trades_change_5",
        "log_turnover_per_trade", "illiquidity", "open_position_zscore_20",
    )
    for metric in price_metrics:
        values = pd.concat({name: frame[metric] for name, frame in members.items()}, axis=1).reindex(index)
        output[f"extra__{family}__{block}__{metric}"] = values.median(axis=1, skipna=True)
        if metric == "return_1":
            output[f"extra__{family}__{block}__breadth_positive"] = values.gt(0).where(values.notna()).mean(axis=1)
            output[f"extra__{family}__{block}__dispersion"] = values.std(axis=1, skipna=True)
            output[f"extra__{family}__{block}__member_fraction"] = values.notna().sum(axis=1) / len(members)
    for metric in liquidity_metrics:
        values = pd.concat({name: frame[metric] for name, frame in members.items()}, axis=1).reindex(index)
        if values.notna().any().any():
            output[f"extra__liquidity__{block}__{metric}"] = values.median(axis=1, skipna=True)
    return output


def _market_feature_frame(
    sources: Iterable[tuple[Path, Path]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected: dict[str, tuple[Path, Instrument, dict[str, Any], str]] = {}
    snapshot_meta: list[dict[str, Any]] = []
    for snapshot_dir, universe_path in sources:
        registry, items, manifest = _read_snapshot(snapshot_dir, universe_path)
        maximum_rows = max(int(item.get("rows", 0)) for item in items)
        snapshot_meta.append(
            {
                "snapshot_id": manifest["snapshot_id"],
                "snapshot_dir": str(snapshot_dir),
                "manifest_sha256": _hash(snapshot_dir / "manifest.json"),
                "universe": str(universe_path),
                "universe_sha256": universe_sha256(universe_path),
            }
        )
        for item in items:
            instrument_id = str(item["instrument_id"])
            instrument = registry[instrument_id]
            if not _eligible(item, instrument, maximum_rows):
                continue
            candidate = (snapshot_dir / str(item["path"]), instrument, item, str(manifest["snapshot_id"]))
            current = selected.get(instrument_id)
            quality = (int(item.get("invalid_close_rows", 0)) == 0, int(item.get("rows", 0)))
            current_quality = (
                (int(current[2].get("invalid_close_rows", 0)) == 0, int(current[2].get("rows", 0)))
                if current else (False, -1)
            )
            if current is None or quality > current_quality:
                selected[instrument_id] = candidate

    blocks: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    individual: dict[str, pd.DataFrame] = {}
    excluded: dict[str, str] = {}
    for instrument_id, (path, instrument, item, _) in selected.items():
        try:
            features = _instrument_features(_load_instrument(path, instrument_id))
            family, block = _economic_block(instrument)
        except ValueError as error:
            excluded[instrument_id] = str(error)
            continue
        blocks.setdefault((family, block), {})[instrument_id] = features
        if family in {"cross", "rates"}:
            individual[instrument_id] = features

    frames = [
        _aggregate_block(members, family=family, block=block)
        for (family, block), members in sorted(blocks.items())
    ]
    market = pd.concat(frames, axis=1, join="outer").sort_index() if frames else pd.DataFrame()
    for instrument_id, frame in individual.items():
        family, _ = _economic_block(selected[instrument_id][1])
        for metric in ("return_1", "return_5", "return_20", "volatility_20"):
            market[f"extra__{family}__{_safe_name(instrument_id)}__{metric}"] = frame[metric]
    market = market.loc[:, market.notna().any(axis=0)].astype("float64").replace([np.inf, -np.inf], np.nan)
    meta = {
        "snapshots": snapshot_meta,
        "eligible_instruments": sorted(selected),
        "used_instruments": sorted(set().union(*(set(value) for value in blocks.values())) if blocks else set()),
        "excluded_after_read": excluded,
        "members_by_block": {
            f"{family}:{block}": sorted(members) for (family, block), members in sorted(blocks.items())
        },
    }
    return market, meta


def _strict_previous_asof(timestamps: pd.Series, market: pd.DataFrame) -> pd.DataFrame:
    left = pd.DataFrame({"timestamp": pd.to_datetime(timestamps.drop_duplicates(), errors="raise")}).sort_values("timestamp")
    right = market.reset_index(names="source_date").sort_values("source_date")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(7, unit="D"),
    )
    if (aligned["source_date"].dropna() >= aligned.loc[aligned["source_date"].notna(), "timestamp"]).any():
        raise AssertionError("same-day or future MOEX close entered target features")
    return aligned.drop(columns="source_date")


def add_key_rate_features(frame: pd.DataFrame, key_rate: pd.DataFrame) -> pd.DataFrame:
    rates = key_rate[["rate_date", "key_rate_pct"]].copy()
    rates["rate_date"] = pd.to_datetime(rates["rate_date"], errors="raise").dt.normalize()
    rates["key_rate_pct"] = pd.to_numeric(rates["key_rate_pct"], errors="raise")
    rates = rates.sort_values("rate_date", kind="mergesort").drop_duplicates("rate_date", keep="last")
    rates["known_at"] = rates["rate_date"] + pd.Timedelta(1, unit="D")
    dates = pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"].drop_duplicates(), errors="raise")}).sort_values("timestamp")
    daily = pd.merge_asof(dates, rates, left_on="timestamp", right_on="known_at", direction="backward")
    if (daily["known_at"].dropna() > daily.loc[daily["known_at"].notna(), "timestamp"]).any():
        raise AssertionError("future key-rate value entered features")
    value = daily["key_rate_pct"]
    move = value.diff()
    last_change_known_at = daily["known_at"].where(move.ne(0)).ffill()
    daily["extra__rates__key_rate_pct"] = value
    daily["extra__rates__key_rate_change_5"] = value.diff(5)
    daily["extra__rates__key_rate_change_20"] = value.diff(20)
    daily["extra__rates__key_rate_percentile_252"] = _rolling_percentile(value, 252, 60)
    daily["extra__rates__last_rate_move"] = move.where(move.ne(0)).ffill().fillna(0)
    daily["extra__rates__days_since_rate_change"] = (daily["timestamp"] - last_change_known_at).dt.days
    columns = ["timestamp", *[column for column in daily if column.startswith("extra__")]]
    return frame.merge(daily[columns], on="timestamp", how="left", validate="many_to_one")


def _causal_asof_features(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    observation_column: str = "observation_date",
    prefix: str,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    """Join a normalized source using only rows already available at decision time."""

    required = {"known_at", observation_column}
    if missing := required - set(source):
        raise ValueError(f"{prefix} source lacks {sorted(missing)}")
    right = source.copy()
    right["known_at"] = pd.to_datetime(right["known_at"], errors="raise").dt.tz_localize(None).dt.normalize()
    right[observation_column] = pd.to_datetime(right[observation_column], errors="raise").dt.tz_localize(None).dt.normalize()
    if (right["known_at"] < right[observation_column]).any():
        raise ValueError(f"{prefix} source is available before its observation")
    if right["known_at"].duplicated().any():
        raise ValueError(f"{prefix} source has duplicate availability dates")
    value_columns = [
        column for column in right
        if column not in {"known_at", observation_column, "source_url", "source_note"}
        and pd.api.types.is_numeric_dtype(right[column])
    ]
    rename = {column: f"{prefix}{column}" for column in value_columns}
    right = right[["known_at", observation_column, *value_columns]].rename(columns=rename)
    dates = pd.DataFrame({"timestamp": pd.to_datetime(frame["timestamp"].drop_duplicates(), errors="raise")}).sort_values("timestamp")
    aligned = pd.merge_asof(dates, right.sort_values("known_at"), left_on="timestamp", right_on="known_at", direction="backward")
    valid = aligned["known_at"].notna()
    if (aligned.loc[valid, "known_at"] > aligned.loc[valid, "timestamp"]).any():
        raise AssertionError(f"future {prefix} value entered features")
    aligned[f"{prefix}available"] = valid.astype(float)
    aligned[f"{prefix}source_age_days"] = (aligned["timestamp"] - aligned[observation_column]).dt.days
    if max_age_days is not None:
        stale = aligned[f"{prefix}source_age_days"].gt(max_age_days)
        aligned.loc[stale, [*rename.values(), f"{prefix}source_age_days"]] = np.nan
        aligned.loc[stale, f"{prefix}available"] = 0.0
    aligned = aligned.drop(columns=["known_at", observation_column])
    return frame.merge(aligned, on="timestamp", how="left", validate="many_to_one")


def _prepare_uzbek_fx(source: pd.DataFrame) -> pd.DataFrame:
    required = {"rate_date", "known_at", "usd_uzs", "rub_uzs", "cny_uzs"}
    if missing := required - set(source):
        raise ValueError(f"Uzbek FX source lacks {sorted(missing)}")
    data = source.copy().rename(columns={"rate_date": "observation_date"})
    data = data.sort_values("observation_date", kind="mergesort")
    for name in ("usd_uzs", "rub_uzs", "cny_uzs"):
        price = pd.to_numeric(data[name], errors="raise")
        if price.le(0).any():
            raise ValueError(f"Uzbek FX {name} must be positive")
        for window in (1, 3, 5, 10, 20):
            data[f"{name}_return_{window}"] = price.pct_change(window, fill_method=None)
        data[f"{name}_volatility_5"] = price.pct_change(fill_method=None).rolling(5, min_periods=4).std()
        data[f"{name}_volatility_20"] = price.pct_change(fill_method=None).rolling(20, min_periods=12).std()
    data["cny_per_usd"] = data["usd_uzs"] / data["cny_uzs"]
    data["rub_per_usd"] = data["usd_uzs"] / data["rub_uzs"]
    data["cny_per_rub"] = data["rub_uzs"] / data["cny_uzs"]
    data["official_rub_per_uzs"] = 1 / data["rub_uzs"]
    return data.replace([np.inf, -np.inf], np.nan)


def _prepare_policy(source: pd.DataFrame) -> pd.DataFrame:
    required = {"observation_date", "known_at", "policy_rate_pct", "rate_change_pp", "decision_changed_rate"}
    if missing := required - set(source):
        raise ValueError(f"Uzbek policy source lacks {sorted(missing)}")
    data = source.copy().sort_values("known_at", kind="mergesort")
    data["observation_date"] = pd.to_datetime(data["observation_date"], errors="raise")
    data["days_since_previous_change"] = data["observation_date"].diff().dt.days
    return data


def _add_bank_quote_features(frame: pd.DataFrame, bank_quotes: pd.DataFrame | None) -> pd.DataFrame:
    data = frame.copy()
    data["extra__uzs__bank__available"] = 0.0
    if bank_quotes is None or bank_quotes.empty:
        return data
    from fxpulse.data.uzbekistan import validate_bank_quotes

    quotes = validate_bank_quotes(bank_quotes)
    quotes["available_at"] = quotes["available_at"].dt.tz_convert(None).dt.normalize()
    quotes["observed_at"] = quotes["observed_at"].dt.tz_convert(None).dt.normalize()
    quotes["mid_uzs_per_rub"] = (quotes["bid_uzs_per_rub"] + quotes["ask_uzs_per_rub"]) / 2
    quotes["spread_bps"] = (
        (quotes["ask_uzs_per_rub"] - quotes["bid_uzs_per_rub"]) / quotes["mid_uzs_per_rub"] * 10_000
    )
    daily = quotes.groupby("available_at", as_index=False).agg(
        observation_date=("observed_at", "max"),
        bid_uzs_per_rub=("bid_uzs_per_rub", "median"),
        ask_uzs_per_rub=("ask_uzs_per_rub", "median"),
        mid_uzs_per_rub=("mid_uzs_per_rub", "median"),
        spread_bps=("spread_bps", "median"),
        fee_fixed_rub=("fee_fixed_rub", "median"),
        fee_rate=("fee_rate", "median"),
    ).rename(columns={"available_at": "known_at"})
    data = data.drop(columns="extra__uzs__bank__available")
    return _causal_asof_features(data, daily, prefix="extra__uzs__bank__", max_age_days=2)


def add_uzbekistan_features(
    frame: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    *,
    bank_quotes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add UZS-only FX, policy, inflation, reserve and transfer features."""

    required_sources = {"fx", "policy", "inflation", "reserves", "remittances_proxy"}
    if missing := required_sources - set(sources):
        raise ValueError(f"Uzbekistan data bundle lacks {sorted(missing)}")
    data = frame.copy()
    data = _causal_asof_features(
        data, _prepare_uzbek_fx(sources["fx"]), prefix="extra__uzs__fx__", max_age_days=10
    )
    data = _causal_asof_features(data, _prepare_policy(sources["policy"]), prefix="extra__uzs__policy__")
    data = _causal_asof_features(
        data, sources["inflation"], prefix="extra__uzs__inflation__", max_age_days=70
    )
    data = _causal_asof_features(
        data, sources["reserves"], prefix="extra__uzs__reserves__", max_age_days=75
    )
    data = _causal_asof_features(
        data, sources["remittances_proxy"], prefix="extra__uzs__transfers__", max_age_days=190
    )
    if "external_balance" in sources:
        data = _causal_asof_features(
            data, sources["external_balance"], prefix="extra__uzs__external__", max_age_days=190
        )
    data = _add_bank_quote_features(data, bank_quotes)
    data["extra__uzs__calendar__month_sin"] = np.sin(2 * np.pi * data["timestamp"].dt.month / 12)
    data["extra__uzs__calendar__month_cos"] = np.cos(2 * np.pi * data["timestamp"].dt.month / 12)
    data["extra__uzs__calendar__quarter_end_month"] = data["timestamp"].dt.month.isin([3, 6, 9, 12]).astype(float)
    direct = data.get("extra__uzs__fx__official_rub_per_uzs")
    if direct is not None:
        data["extra__uzs__fx__official_vs_target_gap_bps"] = (direct / data["price"] - 1) * 10_000

    # Stationary, interpretable interactions between the independently
    # published Uzbekistan legs and macro series.  Every input has already
    # passed through a backward point-in-time join above.
    for window in (1, 3, 5, 10, 20):
        usd = pd.to_numeric(data[f"extra__uzs__fx__usd_uzs_return_{window}"], errors="coerce")
        rub = pd.to_numeric(data[f"extra__uzs__fx__rub_uzs_return_{window}"], errors="coerce")
        cny = pd.to_numeric(data[f"extra__uzs__fx__cny_uzs_return_{window}"], errors="coerce")
        data[f"extra__uzs__interaction__usd_cny_divergence_{window}"] = usd - cny
        data[f"extra__uzs__interaction__rub_usd_relative_strength_{window}"] = rub - usd
        data[f"extra__uzs__interaction__uzs_depreciation_consensus_{window}"] = (usd + cny) / 2
    data["extra__uzs__interaction__rub_momentum_acceleration_3_20"] = (
        data["extra__uzs__fx__rub_uzs_return_3"]
        - data["extra__uzs__fx__rub_uzs_return_20"] * 3 / 20
    )
    data["extra__uzs__interaction__rub_volatility_ratio_5_20"] = (
        data["extra__uzs__fx__rub_uzs_volatility_5"]
        / data["extra__uzs__fx__rub_uzs_volatility_20"].replace(0, np.nan)
    )
    if "extra__uzs__inflation__cpi_12m_pct" in data:
        data["extra__uzs__macro__real_policy_rate_pct"] = (
            data["extra__uzs__policy__policy_rate_pct"]
            - data["extra__uzs__inflation__cpi_12m_pct"]
        )
    def optional_numeric(name: str) -> pd.Series:
        if name not in data:
            return pd.Series(np.nan, index=data.index, dtype=float)
        return pd.to_numeric(data[name], errors="coerce")

    reserves = optional_numeric("extra__uzs__reserves__official_reserves_usd_mn")
    transfers = optional_numeric("extra__uzs__transfers__secondary_income_credits_usd_mn")
    data["extra__uzs__macro__annual_transfers_to_reserves"] = transfers * 4 / reserves.replace(0, np.nan)
    data["extra__uzs__macro__currency_pressure_20"] = (
        data["extra__uzs__fx__usd_uzs_return_20"] * 100
        - optional_numeric("extra__uzs__reserves__reserves_change_1m_pct")
    )
    data["extra__uzs__macro__flow_support_vs_depreciation_20"] = (
        optional_numeric("extra__uzs__transfers__secondary_income_yoy_pct")
        - data["extra__uzs__fx__usd_uzs_return_20"] * 100
    )
    data["extra__uzs__macro__recent_policy_decision_30d"] = (
        data["extra__uzs__policy__source_age_days"].between(0, 30)
        & data["extra__uzs__policy__rate_change_pp"].fillna(0).ne(0)
    ).astype(float)
    if "extra__uzs__external__current_account_usd_mn" in data:
        data["extra__uzs__macro__current_account_to_reserves_pct"] = (
            data["extra__uzs__external__current_account_usd_mn"]
            / reserves.replace(0, np.nan)
            * 100
        )
        data["extra__uzs__macro__trade_balance_to_reserves_pct"] = (
            data["extra__uzs__external__goods_services_balance_usd_mn"]
            / reserves.replace(0, np.nan)
            * 100
        )
    columns = [column for column in data if column.startswith("extra__uzs__")]
    not_uzs = ~data["corridor"].eq("UZS")
    data.loc[not_uzs, columns] = np.nan
    return data


def _add_cross_currency_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for window in (1, 5, 20):
        candidates: list[pd.Series] = []
        cbr_usd = f"leg__usd_rub_return_{window}"
        if cbr_usd in data:
            candidates.append(-pd.to_numeric(data[cbr_usd], errors="coerce"))
        for column in (
            f"market__cny_return_{window}",
            f"extra__cross__fx_market__return_{window}",
            f"extra__cross__fx_cny_rub_tom__return_{window}",
            f"extra__cross__fx_usd_rub_tom__return_{window}",
            f"extra__cross__fx_kzt_rub_tom__return_{window}",
        ):
            if column in data:
                candidates.append(-pd.to_numeric(data[column], errors="coerce"))
        if not candidates:
            continue
        legs = pd.concat(candidates, axis=1)
        rub = legs.mean(axis=1, skipna=True)
        target_good = -pd.to_numeric(data[f"base__return_{window}"], errors="coerce")
        data[f"extra__cross__rub_strength_{window}"] = rub
        data[f"extra__cross__rub_strength_dispersion_{window}"] = legs.std(axis=1, skipna=True)
        data[f"extra__cross__rub_positive_share_{window}"] = legs.gt(0).where(legs.notna()).mean(axis=1)
        data[f"extra__cross__target_minus_rub_{window}"] = target_good - rub
        data[f"extra__cross__target_rub_agreement_{window}"] = (
            np.sign(target_good).eq(np.sign(rub)).where(target_good.notna() & rub.notna()).astype(float)
        )
    return data


def _add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["corridor", "timestamp"], kind="mergesort").groupby("corridor", sort=False):
        data = group.copy()
        equity = data.filter(regex=r"^extra__equity__.*__return_1$").median(axis=1, skipna=True)
        commodity = data.filter(regex=r"^extra__commodity__.*__return_1$").median(axis=1, skipna=True)
        rub = data.get("extra__cross__rub_strength_1", pd.Series(np.nan, index=data.index))
        target_good = -pd.to_numeric(data["base__return_1"], errors="coerce")
        cross_assets = pd.concat([target_good, rub, equity, commodity], axis=1)
        data["extra__regime__cross_asset_dispersion"] = cross_assets.std(axis=1, skipna=True)
        data["extra__regime__cross_asset_positive_share"] = cross_assets.gt(0).where(cross_assets.notna()).mean(axis=1)
        liquidity = data.filter(regex=r"^extra__liquidity__.*__(turnover|trades)_zscore_20$")
        liquidity_stress = liquidity.abs().median(axis=1, skipna=True) if len(liquidity.columns) else pd.Series(np.nan, index=data.index)
        data["extra__regime__liquidity_stress"] = liquidity_stress
        trend_shift = (
            pd.to_numeric(data["base__return_5"], errors="coerce") / 5
            - pd.to_numeric(data["base__return_20"], errors="coerce") / 20
        ).abs()
        scale = pd.to_numeric(data["base__volatility_20"], errors="coerce").replace(0, np.nan)
        data["extra__regime__change_point_score"] = trend_shift / scale
        if "extra__cross__rub_strength_1" in data:
            correlation = target_good.rolling(20, min_periods=12).corr(rub)
            normal = correlation.shift(1).rolling(120, min_periods=40).mean()
            data["extra__regime__rub_correlation_20"] = correlation
            data["extra__regime__correlation_break"] = (correlation - normal).abs()
        volatility_rank = pd.to_numeric(data["regime__volatility_percentile_252"], errors="coerce")
        equity_stress = (-equity / equity.rolling(20, min_periods=12).std().replace(0, np.nan)).clip(-5, 5)
        data["extra__regime__market_stress_score"] = pd.concat(
            [volatility_rank, equity_stress.clip(lower=0) / 5, liquidity_stress.clip(0, 5) / 5], axis=1
        ).mean(axis=1, skipna=True)
        data["extra__regime__high_market_stress"] = data["extra__regime__market_stress_score"].ge(0.67).astype(float)
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True)


def _entry(state: pd.Series) -> pd.Series:
    clean = state.fillna(False).astype(bool)
    return clean & ~clean.shift(1, fill_value=False)


def _cusum_events(returns: pd.Series, volatility: pd.Series, multiplier: float = 0.75) -> pd.Series:
    positive = 0.0
    negative = 0.0
    result: list[bool] = []
    for change, scale in zip(returns, volatility, strict=True):
        if not np.isfinite(change) or not np.isfinite(scale) or scale <= 0:
            result.append(False)
            continue
        positive = max(0.0, positive + float(change))
        negative = min(0.0, negative + float(change))
        fired = positive >= multiplier * scale or negative <= -multiplier * scale
        result.append(fired)
        if fired:
            positive = 0.0
            negative = 0.0
    return pd.Series(result, index=returns.index, dtype=bool)


def _add_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["corridor", "timestamp"], kind="mergesort").groupby("corridor", sort=False):
        data = group.copy()
        price = pd.to_numeric(data["price"], errors="coerce")
        prior_low = price.shift(1).rolling(20, min_periods=20).min()
        divergence = data.get("extra__cross__target_minus_rub_5", pd.Series(np.nan, index=data.index)).abs()
        divergence_limit = divergence.shift(1).rolling(120, min_periods=40).quantile(0.80)
        liquidity_columns = data.filter(regex=r"^extra__liquidity__.*__(turnover|trades)_zscore_20$")
        liquidity_spike = (
            liquidity_columns.abs().max(axis=1, skipna=True).ge(2.0)
            if len(liquidity_columns.columns) else pd.Series(False, index=data.index)
        )
        states = {
            "favorable_percentile_entry": pd.to_numeric(data["base__percentile_60"], errors="coerce").le(0.20),
            "new_20d_low_entry": price.le(prior_low),
            "shock_entry": pd.to_numeric(data["regime__shock_strength"], errors="coerce").ge(2.0),
            "high_volatility_entry": pd.to_numeric(data["regime__volatility_percentile_252"], errors="coerce").ge(0.80),
            "fx_divergence_entry": divergence.ge(divergence_limit),
            "liquidity_spike_entry": liquidity_spike,
        }
        event_columns: list[str] = []
        for name, state in states.items():
            column = f"extra__event__{name}"
            data[column] = _entry(state).astype(float)
            event_columns.append(column)
        data["extra__event__cusum_075sigma"] = _cusum_events(
            pd.to_numeric(data["base__return_1"], errors="coerce"),
            pd.to_numeric(data["base__volatility_20"], errors="coerce"),
        ).astype(float)
        event_columns.append("extra__event__cusum_075sigma")
        data["extra__event__count"] = data[event_columns].sum(axis=1)
        data["extra__event__any"] = data["extra__event__count"].gt(0).astype(float)
        data["extra__event__days_since_previous"] = (
            data["timestamp"] - data["timestamp"].where(data["extra__event__any"].eq(1)).shift(1).ffill()
        ).dt.days
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True)


def build_enriched_features(
    frame: pd.DataFrame,
    *,
    moex_sources: Iterable[tuple[Path | str, Path | str]],
    key_rate: pd.DataFrame | None = None,
    uzbekistan_data: dict[str, pd.DataFrame] | None = None,
    bank_quotes: pd.DataFrame | None = None,
) -> EnrichmentResult:
    """Add feature groups while preserving all original rows and outcomes."""

    required = {"timestamp", "corridor", "price", "base__return_1", "base__return_5", "base__return_20"}
    if missing := required - set(frame):
        raise ValueError(f"base feature frame lacks {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    if data.duplicated(["timestamp", "corridor"]).any():
        raise ValueError("base feature frame has duplicate timestamp/corridor rows")
    data["_enrichment_row_id"] = np.arange(len(data))
    outcomes = data.filter(regex=r"^(outcome__|label__)").copy()
    market, provenance = _market_feature_frame(
        [(Path(snapshot), Path(universe)) for snapshot, universe in moex_sources]
    )
    aligned = _strict_previous_asof(data["timestamp"], market)
    data = data.merge(aligned, on="timestamp", how="left", validate="many_to_one")
    if key_rate is not None:
        data = add_key_rate_features(data, key_rate)
        provenance["key_rate_rows"] = int(len(key_rate))
    if uzbekistan_data is not None:
        data = add_uzbekistan_features(data, uzbekistan_data, bank_quotes=bank_quotes)
        provenance["uzbekistan_rows"] = {name: int(len(value)) for name, value in uzbekistan_data.items()}
        provenance["uzbekistan_bank_quote_rows"] = int(len(bank_quotes)) if bank_quotes is not None else 0
    data = _add_cross_currency_features(data)
    data = _add_regime_features(data)
    data = _add_event_features(data)
    data = (
        data.sort_values("_enrichment_row_id", kind="mergesort")
        .drop(columns="_enrichment_row_id")
        .reset_index(drop=True)
    )
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    extra = [column for column in data if column.startswith(EXTRA_PREFIX)]
    data = data.drop(columns=[column for column in extra if data[column].isna().all()])
    # Feature construction must never rewrite a precomputed target or label.
    pd.testing.assert_frame_equal(
        outcomes.reset_index(drop=True),
        data[outcomes.columns].reset_index(drop=True),
        check_dtype=False,
    )
    groups = {
        group: tuple(column for column in data if column.startswith(prefix))
        for group, prefix in FEATURE_GROUP_PREFIXES.items()
    }
    provenance["feature_counts"] = {group: len(columns) for group, columns in groups.items()}
    provenance["rows"] = int(len(data))
    return EnrichmentResult(frame=data, feature_groups=groups, provenance=provenance)


__all__ = [
    "EXTRA_PREFIX",
    "FEATURE_GROUP_PREFIXES",
    "EnrichmentResult",
    "add_key_rate_features",
    "add_uzbekistan_features",
    "build_enriched_features",
]
