"""Build an as-of-safe panel from raw public data.

The loader enforces the information-set boundary before returning a DataFrame.
Callers cannot accidentally receive rows whose `known_at` timestamp is later
than their requested cut-off.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import pandas as pd


MSK = "Europe/Moscow"
PANEL_COLUMNS = (
    "known_at",
    "value_date",
    "series_id",
    "price",
    "is_carried",
    "meta",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _as_msk(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(MSK)
    return timestamp.tz_convert(MSK)


def _date_to_known_at(value_dates: pd.Series, *, hour: int, minute: int) -> pd.Series:
    naive = pd.to_datetime(value_dates, errors="raise").dt.normalize()
    return naive.dt.tz_localize(MSK) + pd.DateOffset(hours=hour, minutes=minute)


def _load_cbr(frame: pd.DataFrame, ccy: str, path: Path) -> pd.DataFrame:
    _require_columns(frame, ("rate_date", "ccy", "nominal", "rate_rub"), path)
    selected = frame.loc[frame["ccy"].astype(str) == ccy].copy().reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    value_dates = pd.to_datetime(selected["rate_date"], errors="raise").dt.normalize()
    nominal = pd.to_numeric(selected["nominal"], errors="raise")
    rate_rub = pd.to_numeric(selected["rate_rub"], errors="raise")
    if (nominal <= 0).any() or (rate_rub <= 0).any():
        raise ValueError("CBR nominal and rate_rub must be positive")

    # CBR publishes the value for the next effective business day around 15:30
    # MSK on the preceding business day. Keep this convention isolated so it can
    # later be replaced by an archival publication-time feed.
    previous_business_day = value_dates - pd.offsets.BDay(1)
    known_at = _date_to_known_at(previous_business_day, hour=15, minute=30)
    carried = (
        selected["is_carried"].fillna(False).astype(bool)
        if "is_carried" in selected.columns
        else pd.Series(False, index=selected.index)
    )
    return pd.DataFrame(
        {
            "known_at": known_at,
            "value_date": value_dates.dt.date,
            "series_id": f"CBR:{ccy}",
            "price": rate_rub / nominal,
            "is_carried": carried,
            "meta": [{} for _ in range(len(selected))],
        }
    )


def _moex_meta(frame: pd.DataFrame, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        metadata.append(
            {
                field: row[field]
                for field in fields
                if field in frame.columns and pd.notna(row[field])
            }
        )
    return metadata


def _drop_invalid_market_prices(
    selected: pd.DataFrame, price: pd.Series, *, series_id: str, path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove raw exchange rows that cannot represent an executable market price.

    They remain untouched in `data/raw` for the data-quality report.  The panel
    warns rather than treating a zero close as a real price or carrying it
    forward, both of which would corrupt indicators and the backtest.
    """

    valid = price.notna() & price.gt(0)
    if not valid.all():
        warnings.warn(
            f"{series_id}: excluding {(~valid).sum()} raw rows with a missing or non-positive close from {path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected.loc[valid].reset_index(drop=True), price.loc[valid].reset_index(drop=True)


def _moex_price_per_unit(selected: pd.DataFrame, secid: str, path: Path) -> pd.Series:
    """Return daily/candle close in the panel's economic price unit.

    A universe snapshot explicitly states its normalization contract.  A
    source ``FACEVALUE`` is not itself a request to divide: for example, an
    equity's face value is its legal nominal, while KZT/RUB is quoted for 100
    tenge. Legacy files have no contract, so the known KZT multi-unit quote
    fails closed until it is refreshed with metadata.
    """

    raw_close = pd.to_numeric(selected["close"], errors="raise")
    if "normalization" in selected.columns:
        modes = selected["normalization"].fillna("").astype(str)
        if not modes.isin({"raw", "divide_by_facevalue", "forward_ratio_on_roll"}).all():
            unexpected = sorted(modes.loc[~modes.isin({"raw", "divide_by_facevalue", "forward_ratio_on_roll"})].unique())
            raise ValueError(f"MOEX:{secid} has unsupported normalization {unexpected!r} in {path}")
        if modes.eq("divide_by_facevalue").any() and not modes.eq("divide_by_facevalue").all():
            raise ValueError(f"MOEX:{secid} mixes normalization modes in {path}")
        if not modes.eq("divide_by_facevalue").all():
            return raw_close
    elif secid != "KZTRUB_TOM":
        return raw_close

    if "facevalue" in selected.columns:
        facevalue = pd.to_numeric(selected["facevalue"], errors="coerce")
        if facevalue.isna().any() or facevalue.le(0).any():
            raise ValueError(f"MOEX:{secid} has missing or non-positive FACEVALUE in {path}")
        return raw_close / facevalue
    raise ValueError(f"MOEX:{secid} requires FACEVALUE metadata; refresh {path} with the universe downloader")


def _load_moex_daily(frame: pd.DataFrame, secid: str, path: Path) -> pd.DataFrame:
    _require_columns(frame, ("trade_date", "secid", "close"), path)
    selected = frame.loc[frame["secid"].astype(str) == secid].copy().reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    price = _moex_price_per_unit(selected, secid, path)
    selected, price = _drop_invalid_market_prices(selected, price, series_id=f"MOEX:{secid}", path=path)
    if selected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    value_dates = pd.to_datetime(selected["trade_date"], errors="raise").dt.normalize()

    # Daily close is only exposed after the trading session is complete. Use a
    # conservative end-of-day boundary until raw data carry an exchange timestamp.
    known_at = _date_to_known_at(value_dates, hour=23, minute=59)
    return pd.DataFrame(
        {
            "known_at": known_at,
            "value_date": value_dates.dt.date,
            "series_id": f"MOEX:{secid}",
            "price": price,
            "is_carried": False,
            "meta": _moex_meta(selected, ("waprice", "volume_rub", "num_trades", "facevalue")),
        }
    )


def _load_moex_candles(frame: pd.DataFrame, secid: str, path: Path) -> pd.DataFrame:
    _require_columns(frame, ("dt_msk", "secid", "close"), path)
    selected = frame.loc[frame["secid"].astype(str) == secid].copy().reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    price = _moex_price_per_unit(selected, secid, path)
    selected, price = _drop_invalid_market_prices(selected, price, series_id=f"MOEX10M:{secid}", path=path)
    if selected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    known_at = pd.to_datetime(selected["dt_msk"], errors="raise")
    if known_at.dt.tz is None:
        known_at = known_at.dt.tz_localize(MSK)
    else:
        known_at = known_at.dt.tz_convert(MSK)
    return pd.DataFrame(
        {
            "known_at": known_at,
            "value_date": known_at.dt.date,
            "series_id": f"MOEX10M:{secid}",
            "price": price,
            "is_carried": False,
            "meta": _moex_meta(selected, ("volume_rub", "facevalue")),
        }
    )


def _source_path(source: str, raw_path: Path) -> Path:
    if source == "CBR":
        return raw_path / "cbr_daily.csv"
    if source == "MOEX":
        return raw_path / "moex_daily.csv"
    if source == "MOEX10M":
        return raw_path / "moex_candles.csv"
    raise ValueError("series must be CBR:<CCY>, MOEX:<SECID>, or MOEX10M:<SECID>")


@lru_cache(maxsize=32)
def _load_full_panel(series: str, raw_dir: str, source_fingerprint: tuple[int, int]) -> pd.DataFrame:
    """Build a source panel once per raw-file version.

    The cache holds only the private full panel. Every public `load_panel` call
    below still applies the `as_of` boundary before returning data, so repeated
    walk-forward calls are fast without widening their information set.
    """

    del source_fingerprint  # Its only role is cache invalidation when raw changes.
    source, identifier = series.split(":", maxsplit=1)
    raw_path = Path(raw_dir)
    path = _source_path(source, raw_path)
    frame = pd.read_csv(path)
    if source == "CBR":
        panel = _load_cbr(frame, identifier, path)
    elif source == "MOEX":
        panel = _load_moex_daily(frame, identifier, path)
    else:
        panel = _load_moex_candles(frame, identifier, path)
    return panel.sort_values(["known_at", "value_date", "series_id"], kind="mergesort").reset_index(drop=True)


def load_panel(series: str, *, raw_dir: Path | str = Path("data/raw"), as_of: object | None = None) -> pd.DataFrame:
    """Load one panel series, physically excluding observations unknown at `as_of`.

    Supported IDs are `CBR:<CCY>`, `MOEX:<SECID>` for daily closes, and
    `MOEX10M:<SECID>` for completed 10-minute candles. `known_at` is always a
    timezone-aware Moscow timestamp and price is RUB per unit of foreign currency.
    """

    try:
        source, identifier = series.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError("series must be CBR:<CCY>, MOEX:<SECID>, or MOEX10M:<SECID>") from exc

    raw_path = Path(raw_dir).resolve()
    path = _source_path(source, raw_path)
    status = path.stat()
    frame = _load_full_panel(series, str(raw_path), (status.st_mtime_ns, status.st_size))

    if as_of is not None:
        cut_off = _as_msk(as_of)
        # Apply the information boundary at the loader boundary, before a caller
        # can derive an indicator or normalize a feature from the panel.
        frame = frame.loc[frame["known_at"] <= cut_off].copy()

    if frame.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    result = frame.loc[:, list(PANEL_COLUMNS)].copy().reset_index(drop=True)
    result.attrs["fxpulse_sorted_by_known_at"] = True
    return result
