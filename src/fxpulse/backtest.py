"""Walk-forward execution of the preregistered FX Pulse signal grid."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from fxpulse.grid import grid_sha256, load_grid, spec_id
from fxpulse.labeling import HORIZONS, evaluate_positions, label_observations
from fxpulse.panel import load_panel
from fxpulse.signals import Config, IndicatorSpec, SIGNAL_COLUMNS, signals_as_of


# The narrow overnight run deliberately has one fixing and one exchange-close
# series. The CLI can accept more `--series CORRIDOR=SERIES` arguments.
DEFAULT_BACKTEST_SERIES = {
    "RUB->TJS": "CBR:TJS",
    "RUB->CNY": "MOEX:CNYRUB_TOM",
}

METRIC_COLUMNS = (
    "corridor",
    "series_id",
    "config_id",
    "indicator",
    "params",
    "direction",
    "horizon",
    "fold",
    "test_start",
    "test_end",
    "regime",
    "is_last_12_months",
    "training_observations",
    "purged_tail_observations",
    "embargo_observations",
    "signal_count",
    "eligible_base_count",
    "hit_rate",
    "baseline_hit_rate",
    "lift",
    "benefit_sym_bps",
    "benefit_fwd_bps",
    "benefit_fwd_newey_west_t",
    "signals_per_week",
    "signals_per_month",
    "cluster_share",
    "interval_cv",
)


@dataclass(frozen=True)
class WalkForwardFold:
    """One chronological OOT block and its leak-free train boundary."""

    name: str
    test_positions: tuple[int, ...]
    training_observations: int
    purged_tail_observations: int
    embargo_observations: int


def walk_forward_folds(
    panel: pd.DataFrame, *, horizon: int, min_train_observations: int = 252
) -> tuple[WalkForwardFold, ...]:
    """Create expanding quarterly OOT blocks with horizon-aware purging.

    Labels for the last `horizon` observations before a test block can touch the
    block, so they are excluded from its training prefix. The equivalent sized
    embargo after the block is recorded in every fold, preventing a future model
    fitting step from using labels that overlap this evaluation window. The
    present prototype has no fitted estimator: thresholds are preregistered.
    """

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if min_train_observations <= 0:
        raise ValueError("min_train_observations must be positive")
    if panel.empty:
        return ()
    value_dates = pd.to_datetime(panel["value_date"], errors="raise")
    quarters = value_dates.dt.to_period("Q")
    folds: list[WalkForwardFold] = []
    for quarter in sorted(quarters.unique()):
        quarter_positions = panel.index[quarters.eq(quarter)].tolist()
        start, end = quarter_positions[0], quarter_positions[-1]
        train_stop_exclusive = start - horizon
        if train_stop_exclusive < min_train_observations:
            continue
        # A label may use observations from the following calendar quarter:
        # that is the outcome we are evaluating, not an input to the signal.
        # Keep every position whose full h-step outcome is observable in the
        # complete evaluation history.  Cutting ``h`` rows from every quarter
        # would silently shorten every non-final OOT fold and bias frequencies.
        last_label_observable = len(panel) - horizon - 1
        test_stop_inclusive = min(end, last_label_observable)
        test_positions = tuple(range(start, test_stop_inclusive + 1))
        if not test_positions:
            continue
        embargo = min(horizon, len(panel) - end - 1)
        folds.append(
            WalkForwardFold(
                name=str(quarter),
                test_positions=test_positions,
                training_observations=train_stop_exclusive,
                purged_tail_observations=horizon,
                embargo_observations=embargo,
            )
        )
    return tuple(folds)


def _period_group(value_date: object, latest: pd.Timestamp) -> tuple[str, bool]:
    date = pd.Timestamp(value_date).normalize()
    if date < pd.Timestamp("2022-01-01"):
        group = "pre_2022"
    elif date < pd.Timestamp("2023-01-01"):
        group = "2022"
    else:
        group = "post_2022"
    return group, bool(date >= latest - pd.DateOffset(months=12))


def _parse_series(values: Iterable[str] | None) -> Mapping[str, str]:
    if not values:
        return DEFAULT_BACKTEST_SERIES.copy()
    parsed: dict[str, str] = {}
    for value in values:
        try:
            corridor, series = value.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError("--series must look like RUB->TJS=CBR:TJS") from exc
        if not corridor or not series:
            raise ValueError("--series must include non-empty corridor and series")
        parsed[corridor] = series
    return parsed


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def _metric_row(
    *,
    corridor: str,
    series_id: str,
    spec: IndicatorSpec,
    direction: str,
    horizon: int,
    fold: WalkForwardFold,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    positions: Iterable[int],
    latest: pd.Timestamp,
) -> dict[str, object]:
    metrics = evaluate_positions(
        panel,
        positions,
        direction=direction,
        horizon=horizon,
        base_positions=fold.test_positions,
        labels=labels,
    )
    start = panel.loc[fold.test_positions[0], "value_date"]
    end = panel.loc[fold.test_positions[-1], "value_date"]
    regime, is_last_12_months = _period_group(end, latest)
    return {
        "corridor": corridor,
        "series_id": series_id,
        "config_id": spec_id(spec),
        "indicator": spec.name,
        "params": json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":")),
        "direction": direction,
        "horizon": horizon,
        "fold": fold.name,
        "test_start": pd.Timestamp(start).date().isoformat(),
        "test_end": pd.Timestamp(end).date().isoformat(),
        "regime": regime,
        "is_last_12_months": is_last_12_months,
        "training_observations": fold.training_observations,
        "purged_tail_observations": fold.purged_tail_observations,
        "embargo_observations": fold.embargo_observations,
        **metrics,
    }


def run_backtest(
    *,
    raw_dir: Path | str = Path("data/raw"),
    grid_path: Path | str = Path("configs/grid.json"),
    artifact_dir: Path | str = Path("artifacts"),
    series: Mapping[str, str] | None = None,
    min_train_observations: int = 252,
) -> dict[str, Any]:
    """Run all preregistered configurations and write reproducible artifacts."""

    raw_path = Path(raw_dir)
    artifact_path = Path(artifact_dir)
    specs = load_grid(grid_path)
    sources = dict(series or DEFAULT_BACKTEST_SERIES)
    artifact_path.mkdir(parents=True, exist_ok=True)

    signal_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    source_ranges: dict[str, dict[str, object]] = {}

    for corridor, series_id in sources.items():
        panel = load_panel(series_id, raw_dir=raw_path)
        if panel.empty:
            raise ValueError(f"{series_id} has no usable rows in {raw_path}")
        panel = panel.reset_index(drop=True)
        latest = pd.Timestamp(panel["value_date"].max())
        source_ranges[series_id] = {
            "corridor": corridor,
            "rows": len(panel),
            "known_at_from": panel["known_at"].min().isoformat(),
            "known_at_to": panel["known_at"].max().isoformat(),
            "value_date_from": str(panel["value_date"].min()),
            "value_date_to": str(panel["value_date"].max()),
        }

        folds_by_horizon = {
            horizon: walk_forward_folds(panel, horizon=horizon, min_train_observations=min_train_observations)
            for horizon in HORIZONS
        }
        # The h=1 set is the broadest eligible OOT set. Signals are computed
        # once at each of these moments, then evaluated at all fixed horizons.
        signal_positions = sorted(
            {position for fold in folds_by_horizon[1] for position in fold.test_positions}
        )
        fired: dict[tuple[str, str], set[int]] = defaultdict(set)
        signal_config = Config(raw_dir=raw_path, corridors={corridor: series_id}, indicators=specs)
        for position in signal_positions:
            snapshot = panel.loc[position, "known_at"]
            point_in_time_signals = signals_as_of(snapshot, signal_config)
            if not point_in_time_signals.empty:
                signal_rows.extend(point_in_time_signals.to_dict(orient="records"))
            for row in point_in_time_signals.itertuples(index=False):
                fired[(f"{row.indicator}:{row.params}", row.direction)].add(position)

        labels_by_horizon = {horizon: label_observations(panel, horizon) for horizon in HORIZONS}
        for spec in specs:
            config_key = spec_id(spec)
            # Seasonality chooses orientation from historical returns at each T;
            # report both possible orientations so a zero-fire result is visible.
            directions = (spec.direction,) if spec.direction else ("favorable", "window_closing")
            for horizon, folds in folds_by_horizon.items():
                for fold in folds:
                    for direction in directions:
                        metric_rows.append(
                            _metric_row(
                                corridor=corridor,
                                series_id=series_id,
                                spec=spec,
                                direction=direction,
                                horizon=horizon,
                                fold=fold,
                                panel=panel,
                                labels=labels_by_horizon[horizon],
                                positions=fired[(config_key, direction)],
                                latest=latest,
                            )
                        )

    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS) if signal_rows else _empty_signals()
    if not signals.empty:
        signals = signals.loc[:, list(SIGNAL_COLUMNS)].sort_values(
            ["date", "corridor", "indicator", "params"], kind="mergesort"
        ).reset_index(drop=True)
    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["series_id", "indicator", "params", "direction", "horizon", "fold"], kind="mergesort"
        ).reset_index(drop=True)

    signals.to_csv(artifact_path / "signals.csv", index=False)
    metrics.to_csv(artifact_path / "metrics.csv", index=False)
    run_meta = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "grid_path": str(grid_path),
        "grid_sha256": grid_sha256(grid_path),
        "registered_configurations": len(specs),
        "min_train_observations": min_train_observations,
        "horizons": list(HORIZONS),
        "series": source_ranges,
        "signals_written": len(signals),
        "metric_rows_written": len(metrics),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "method": "expanding quarterly walk-forward; h-observation purge before each OOT block and recorded h-observation embargo after it",
    }
    (artifact_path / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--grid", type=Path, default=Path("configs/grid.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-train-observations", type=int, default=252)
    parser.add_argument(
        "--series",
        action="append",
        help="repeatable CORRIDOR=SERIES, e.g. RUB->TJS=CBR:TJS; default runs CBR:TJS and MOEX:CNYRUB_TOM",
    )
    args = parser.parse_args(argv)
    summary = run_backtest(
        raw_dir=args.raw_dir,
        grid_path=args.grid,
        artifact_dir=args.artifact_dir,
        series=_parse_series(args.series),
        min_train_observations=args.min_train_observations,
    )
    print(
        f"Wrote {summary['signals_written']} signals and {summary['metric_rows_written']} metric rows "
        f"to {args.artifact_dir} (grid {summary['grid_sha256'][:12]}… )"
    )


if __name__ == "__main__":
    main()


__all__ = ["METRIC_COLUMNS", "WalkForwardFold", "run_backtest", "walk_forward_folds"]
