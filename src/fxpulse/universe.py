"""Registered MOEX cross-market universe and its data-contract gates.

The registry is intentionally a small candidate set, not a request to ingest
every ticker.  A candidate becomes ``ready`` only after its historical range,
metadata, liquidity and point-in-time session convention have been audited.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


MOEX_BASE = "https://iss.moex.com/iss"
STATUSES = frozenset({"ready", "candidate", "blocked", "planned"})
NORMALIZATIONS = frozenset({"raw", "divide_by_facevalue", "continuous_future"})


@dataclass(frozen=True)
class Instrument:
    """One economic series candidate and its source/data-quality contract."""

    id: str
    status: Literal["ready", "candidate", "blocked", "planned"]
    asset_class: str
    role: str
    label: str
    source: dict[str, str]
    price_unit: str
    normalization_kind: str
    expected_facevalue: float | None
    availability: dict[str, str | None]
    continuous_policy: dict[str, str] | None
    notes: str

    @property
    def has_fixed_security(self) -> bool:
        return "secid" in self.source

    def history_url(self) -> str:
        """Return the daily-history endpoint for a fixed, auditable security."""

        if not self.has_fixed_security:
            raise ValueError(f"{self.id} needs a point-in-time future resolver before it has a history URL")
        source = self.source
        return (
            f"{MOEX_BASE}/history/engines/{source['engine']}/markets/{source['market']}"
            f"/boards/{source['board']}/securities/{source['secid']}.json"
        )

    def candles_url(self) -> str:
        """Return the candle endpoint for a fixed, auditable security."""

        if not self.has_fixed_security:
            raise ValueError(f"{self.id} needs a point-in-time future resolver before it has a candles URL")
        source = self.source
        return (
            f"{MOEX_BASE}/engines/{source['engine']}/markets/{source['market']}"
            f"/boards/{source['board']}/securities/{source['secid']}/candles.json"
        )

    def normalize_price(self, raw_close: float, *, facevalue: float | None) -> float:
        """Convert one raw close to the economic unit declared in the registry.

        ``divide_by_facevalue`` fails closed when the live instrument metadata
        disagrees with an audited expectation. This prevents KZT-like 100-unit
        quotations from silently entering a RUB-per-unit panel.
        """

        if raw_close <= 0:
            raise ValueError("raw_close must be positive")
        if self.normalization_kind == "raw":
            return raw_close
        if self.normalization_kind == "continuous_future":
            raise ValueError("continuous futures require a roll-adjustment adapter")
        if facevalue is None or facevalue <= 0:
            raise ValueError(f"{self.id} requires a positive FACEVALUE metadata field")
        if self.expected_facevalue is not None and facevalue != self.expected_facevalue:
            raise ValueError(
                f"{self.id} FACEVALUE changed from expected {self.expected_facevalue} to observed {facevalue}"
            )
        return raw_close / facevalue


def universe_sha256(path: Path | str = Path("configs/moex_universe.json")) -> str:
    """Return the registry hash to record alongside a future dataset snapshot."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_strings(mapping: dict[str, Any], fields: tuple[str, ...], *, context: str) -> None:
    for field in fields:
        if not isinstance(mapping.get(field), str) or not mapping[field]:
            raise ValueError(f"{context} requires non-empty string {field!r}")


def load_universe(path: Path | str = Path("configs/moex_universe.json")) -> tuple[Instrument, ...]:
    """Load and validate the registered candidate universe without I/O to MOEX."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only universe schema_version 1 is supported")
    instruments = config.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("universe requires a non-empty instruments list")
    ids: set[str] = set()
    result: list[Instrument] = []
    for raw in instruments:
        if not isinstance(raw, dict):
            raise ValueError("each instrument must be an object")
        _require_strings(raw, ("id", "status", "asset_class", "role", "label", "price_unit", "notes"), context="instrument")
        instrument_id = raw["id"]
        if instrument_id in ids:
            raise ValueError(f"duplicate instrument id {instrument_id!r}")
        ids.add(instrument_id)
        if raw["status"] not in STATUSES:
            raise ValueError(f"unsupported status {raw['status']!r}")
        source = raw.get("source")
        normalization = raw.get("normalization")
        availability = raw.get("availability")
        if not isinstance(source, dict) or not isinstance(normalization, dict) or not isinstance(availability, dict):
            raise ValueError(f"{instrument_id} source, normalization and availability must be objects")
        _require_strings(source, ("engine", "market", "board"), context=f"{instrument_id}.source")
        kind = normalization.get("kind")
        if kind not in NORMALIZATIONS:
            raise ValueError(f"{instrument_id} has unsupported normalization {kind!r}")
        expected_facevalue = normalization.get("expected_facevalue")
        if expected_facevalue is not None and (
            not isinstance(expected_facevalue, int | float) or expected_facevalue <= 0
        ):
            raise ValueError(f"{instrument_id} expected_facevalue must be positive or null")
        has_secid = isinstance(source.get("secid"), str) and bool(source["secid"])
        has_root = isinstance(source.get("continuous_root"), str) and bool(source["continuous_root"])
        if has_secid == has_root:
            raise ValueError(f"{instrument_id} source requires exactly one of secid or continuous_root")
        if raw["status"] == "ready" and not has_secid:
            raise ValueError(f"ready instrument {instrument_id} must have a fixed secid")
        if kind == "continuous_future" and not has_root:
            raise ValueError(f"{instrument_id} continuous_future requires continuous_root")
        if kind != "continuous_future" and has_root:
            raise ValueError(f"{instrument_id} source root requires continuous_future normalization")
        policy = raw.get("continuous_policy")
        if kind == "continuous_future":
            if not isinstance(policy, dict):
                raise ValueError(f"{instrument_id} continuous_future requires continuous_policy")
            _require_strings(policy, ("selection", "adjustment"), context=f"{instrument_id}.continuous_policy")
            if policy != {"selection": "max_value_then_openposition", "adjustment": "forward_ratio_on_roll"}:
                raise ValueError(f"{instrument_id} has unsupported continuous policy")
        elif policy is not None:
            raise ValueError(f"{instrument_id} only permits continuous_policy for continuous_future")
        if set(availability) != {"daily_from", "intraday_10m_from"}:
            raise ValueError(f"{instrument_id} availability requires daily_from and intraday_10m_from")
        result.append(
            Instrument(
                id=instrument_id,
                status=raw["status"],
                asset_class=raw["asset_class"],
                role=raw["role"],
                label=raw["label"],
                source={key: str(value) for key, value in source.items()},
                price_unit=raw["price_unit"],
                normalization_kind=kind,
                expected_facevalue=float(expected_facevalue) if expected_facevalue is not None else None,
                availability={key: value for key, value in availability.items()},
                continuous_policy={key: str(value) for key, value in policy.items()} if policy else None,
                notes=raw["notes"],
            )
        )
    return tuple(result)


def instruments_for_status(
    status: str, path: Path | str = Path("configs/moex_universe.json")
) -> tuple[Instrument, ...]:
    """Return only instruments in one explicit lifecycle state."""

    if status not in STATUSES:
        raise ValueError(f"unsupported status {status!r}")
    return tuple(instrument for instrument in load_universe(path) if instrument.status == status)


__all__ = ["Instrument", "instruments_for_status", "load_universe", "universe_sha256"]
