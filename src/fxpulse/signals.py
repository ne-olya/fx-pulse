"""The single point-in-time signal entry point used by every future backtest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fxpulse.indicators import evaluate
from fxpulse.panel import MSK, load_panel


SIGNAL_COLUMNS = (
    "date",
    "corridor",
    "series_id",
    "indicator",
    "params",
    "direction",
    "strength",
    "speed",
    "days_to_confirm",
    "scenario",
)
DEFAULT_CORRIDORS = {
    "RUB->TJS": "CBR:TJS",
    "RUB->UZS": "CBR:UZS",
    "RUB->KGS": "CBR:KGS",
    "RUB->AMD": "CBR:AMD",
    "RUB->KZT": "CBR:KZT",
}


@dataclass(frozen=True)
class IndicatorSpec:
    """A preregistered indicator configuration and its product interpretation."""

    name: str
    params: Mapping[str, Any]
    direction: str | None
    speed: str
    days_to_confirm: int | None
    scenario: str

    def __post_init__(self) -> None:
        if self.direction is not None and self.direction not in {"favorable", "window_closing"}:
            raise ValueError("direction must be favorable, window_closing, or None")
        if self.speed not in {"fast", "slow"}:
            raise ValueError("speed must be fast or slow")
        if self.days_to_confirm is not None and self.days_to_confirm <= 0:
            raise ValueError("days_to_confirm must be positive when supplied")
        if self.scenario not in {"T1", "T2", "T3", "T4", "T5"}:
            raise ValueError("scenario must be T1 through T5")


@dataclass(frozen=True)
class Config:
    """All dependencies of `signals_as_of`; no hidden global dataset is read."""

    raw_dir: Path | str = Path("data/raw")
    corridors: Mapping[str, str] = field(default_factory=lambda: DEFAULT_CORRIDORS.copy())
    indicators: tuple[IndicatorSpec, ...] = ()
    max_data_age_days: int | None = 7

    def __post_init__(self) -> None:
        if self.max_data_age_days is not None and self.max_data_age_days < 0:
            raise ValueError("max_data_age_days must be non-negative or None")


def _as_msk(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize(MSK) if timestamp.tzinfo is None else timestamp.tz_convert(MSK)


def _resolved_direction(spec: IndicatorSpec, details: Mapping[str, Any]) -> str:
    direction = spec.direction or details.get("direction")
    if direction not in {"favorable", "window_closing"}:
        raise ValueError(f"{spec.name} requires an explicit signal direction")
    return str(direction)


def signals_as_of(T: object, config: Config) -> pd.DataFrame:
    """Return signals exactly as they were knowable at Moscow time `T`.

    `load_panel(..., as_of=T)` enforces the information-set boundary before an
    indicator receives data. This is intentionally the only path used for both
    online signals and future walk-forward backtests.
    """

    as_of = _as_msk(T)
    rows: list[dict[str, object]] = []
    for corridor, source_series in config.corridors.items():
        panel = load_panel(source_series, raw_dir=config.raw_dir, as_of=as_of)
        if not panel.empty and not bool(panel["known_at"].le(as_of).all()):
            raise AssertionError("load_panel returned a future observation")
        if (
            not panel.empty
            and config.max_data_age_days is not None
            and as_of - panel["known_at"].max() > pd.Timedelta(config.max_data_age_days, unit="D")
        ):
            # A historical fixing must not be silently reused as if it were a
            # fresh signal after the data feed has stopped.
            continue
        for spec in config.indicators:
            output = evaluate(spec.name, panel, **dict(spec.params))
            if not output.fired:
                continue
            rows.append(
                {
                    "date": as_of.date().isoformat(),
                    "corridor": corridor,
                    "series_id": source_series,
                    "indicator": spec.name,
                    "params": json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":")),
                    "direction": _resolved_direction(spec, output.details),
                    "strength": float(output.strength),
                    "speed": spec.speed,
                    "days_to_confirm": spec.days_to_confirm,
                    "scenario": spec.scenario,
                }
            )
    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return (
        pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
        .sort_values(["corridor", "indicator", "params"], kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = ["Config", "IndicatorSpec", "SIGNAL_COLUMNS", "signals_as_of"]
