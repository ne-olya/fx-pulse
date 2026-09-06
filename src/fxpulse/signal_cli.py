"""Print the point-in-time signal table for one requested decision date."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fxpulse.grid import load_grid
from fxpulse.signals import Config, DEFAULT_CORRIDORS, signals_as_of


DEFAULT_DECISION_HOUR_MSK = 20


def _parse_series(values: list[str] | None) -> Mapping[str, str]:
    if not values:
        return DEFAULT_CORRIDORS.copy()
    result: dict[str, str] = {}
    for value in values:
        try:
            corridor, series = value.rsplit("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError("--series must look like RUB->UZS=CBR:UZS") from exc
        if not corridor or not series:
            raise ValueError("--series must contain a corridor and a source series")
        result[corridor] = series
    return result


def decision_timestamp(value: str) -> pd.Timestamp:
    """Interpret a bare date as the documented evening decision moment."""

    if not value or not value.strip():
        raise ValueError(
            "DATE is required; example: uv run python -m fxpulse.signal_cli --date 2026-06-10"
        )
    timestamp = pd.Timestamp(value)
    if len(value.strip()) == 10:
        timestamp += pd.Timedelta(DEFAULT_DECISION_HOUR_MSK, unit="h")
    return timestamp


def render_signals(
    *,
    date: str,
    raw_dir: Path | str = Path("data/raw"),
    grid_path: Path | str = Path("configs/grid.json"),
    series: list[str] | None = None,
) -> pd.DataFrame:
    config = Config(
        raw_dir=raw_dir,
        corridors=_parse_series(series),
        indicators=load_grid(grid_path),
    )
    return signals_as_of(decision_timestamp(date), config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD or an explicit timestamp")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--grid", type=Path, default=Path("configs/grid.json"))
    parser.add_argument(
        "--series",
        action="append",
        help="repeatable CORRIDOR=SERIES override, e.g. RUB->UZS=CBR:UZS",
    )
    parser.add_argument("--output", type=Path, help="optional CSV path; stdout is always printed")
    args = parser.parse_args(argv)

    try:
        frame = render_signals(
            date=args.date,
            raw_dir=args.raw_dir,
            grid_path=args.grid,
            series=args.series,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    print(frame.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
