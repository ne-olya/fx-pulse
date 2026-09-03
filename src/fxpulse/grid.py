"""Read the preregistered indicator grid without selecting a winner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
import hashlib
import json
from pathlib import Path
from typing import Any

from fxpulse.signals import IndicatorSpec


def grid_sha256(path: Path | str = Path("configs/grid.json")) -> str:
    """Return the content hash recorded alongside a backtest run."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _values(name: str, values: object) -> Iterable[object]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"grid parameter {name!r} must have a non-empty list of values")
    return values


def load_grid(path: Path | str = Path("configs/grid.json")) -> tuple[IndicatorSpec, ...]:
    """Expand every registered parameter combination in stable order."""

    config_path = Path(path)
    contents: Mapping[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    if contents.get("schema_version") != 1:
        raise ValueError("Only grid schema_version 1 is supported")
    definitions = contents.get("indicators")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("grid must define a non-empty indicators list")

    specs: list[IndicatorSpec] = []
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError("each grid indicator must be an object")
        try:
            name = definition["name"]
            raw_params = definition["params"]
            direction = definition["direction"]
            speed = definition["speed"]
            days_to_confirm = definition["days_to_confirm"]
            scenario = definition["scenario"]
        except KeyError as exc:
            raise ValueError(f"grid indicator is missing {exc.args[0]!r}") from exc
        if not isinstance(name, str) or not isinstance(raw_params, dict):
            raise ValueError("grid indicator name must be a string and params must be an object")
        keys = tuple(sorted(raw_params))
        parameter_values = tuple(_values(key, raw_params[key]) for key in keys)
        for combination in product(*parameter_values):
            specs.append(
                IndicatorSpec(
                    name=name,
                    params=dict(zip(keys, combination, strict=True)),
                    direction=direction,
                    speed=speed,
                    days_to_confirm=days_to_confirm,
                    scenario=scenario,
                )
            )
    return tuple(specs)


def spec_id(spec: IndicatorSpec) -> str:
    """Stable configuration identifier for joins between signal and metric rows."""

    params = json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":"))
    return f"{spec.name}:{params}"


__all__ = ["grid_sha256", "load_grid", "spec_id"]
