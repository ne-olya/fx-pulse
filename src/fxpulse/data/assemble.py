"""Assemble a research-wide daily panel without inventing missing prices."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from fxpulse.rule_selection import load_universe_prices, resolve_snapshot


def load_cbr_wide(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"rate_date", "ccy", "nominal", "rate_rub"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} lacks: {', '.join(sorted(missing))}")
    raw["trade_date"] = pd.to_datetime(raw["rate_date"], errors="raise").dt.normalize()
    nominal = pd.to_numeric(raw["nominal"], errors="raise")
    if nominal.le(0).any():
        raise ValueError("CBR nominal must be positive")
    raw["rub_per_unit"] = pd.to_numeric(raw["rate_rub"], errors="raise") / nominal
    if raw.duplicated(["trade_date", "ccy"]).any():
        raise ValueError("CBR data has duplicate (rate_date, ccy) rows")
    wide = raw.pivot(index="trade_date", columns="ccy", values="rub_per_unit")
    wide.columns = [f"cbr__{currency.lower()}__rub_per_unit" for currency in wide.columns]
    return wide.sort_index()


def assemble_daily_panel(
    *,
    cbr_path: Path | str = Path("data/raw/cbr_daily.csv"),
    snapshot_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, Path]:
    snapshot = resolve_snapshot(snapshot_dir)
    moex = load_universe_prices(snapshot, target_instrument_id="fx_cny_rub_tom").copy()
    moex.columns = [f"moex__{column}__close" for column in moex.columns]
    cbr = load_cbr_wide(cbr_path)
    panel = moex.join(cbr, how="outer").sort_index().reset_index()
    return panel, snapshot


def write_daily_panel(
    *,
    cbr_path: Path | str = Path("data/raw/cbr_daily.csv"),
    snapshot_dir: Path | str | None = None,
    output_path: Path | str = Path("data/processed/research_daily_panel.csv"),
) -> dict[str, object]:
    panel, snapshot = assemble_daily_panel(cbr_path=cbr_path, snapshot_dir=snapshot_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    metadata = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "snapshot_dir": str(snapshot),
        "cbr_path": str(cbr_path),
        "output_path": str(output),
        "rows": len(panel),
        "date_from": panel["trade_date"].min().isoformat(),
        "date_to": panel["trade_date"].max().isoformat(),
        "columns": panel.columns.tolist(),
        "warning": "Same-date research join only; model code must enforce each source's actual known_at time.",
    }
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbr", type=Path, default=Path("data/raw/cbr_daily.csv"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/research_daily_panel.csv"))
    args = parser.parse_args(argv)
    metadata = write_daily_panel(cbr_path=args.cbr, snapshot_dir=args.snapshot, output_path=args.output)
    print(f"Wrote {metadata['rows']} daily rows and {len(metadata['columns']) - 1} series to {args.output}")


if __name__ == "__main__":
    main()
