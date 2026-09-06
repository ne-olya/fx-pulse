from __future__ import annotations

import json

import pandas as pd

from fxpulse.backtest import METRIC_COLUMNS, run_backtest, walk_forward_folds
from fxpulse.grid import grid_sha256, load_grid


def _write_cbr_history(path, periods: int = 420) -> None:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    prices = [10 - index / 1_000 for index in range(periods)]
    pd.DataFrame(
        {
            "rate_date": dates.date,
            "ccy": "TJS",
            "nominal": 1,
            "rate_rub": prices,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "source_url": "https://example.test/cbr",
        }
    ).to_csv(path / "cbr_daily.csv", index=False)


def test_grid_is_preregistered_and_expands_all_combinations() -> None:
    path = "configs/grid.json"

    specs = load_grid(path)

    assert len(specs) == 37
    assert len(grid_sha256(path)) == 64
    assert specs[0].name == "level_percentile"


def test_walk_forward_purges_labels_that_would_cross_into_oot_block() -> None:
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    panel = pd.DataFrame({"value_date": dates.date})

    folds = walk_forward_folds(panel, horizon=5, min_train_observations=100)

    assert folds
    first = folds[0]
    assert first.training_observations + first.purged_tail_observations == first.test_positions[0]
    assert first.embargo_observations == 5


def test_walk_forward_keeps_label_observable_tail_of_non_final_quarter() -> None:
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    panel = pd.DataFrame({"value_date": dates.date})

    folds = walk_forward_folds(panel, horizon=5, min_train_observations=100)

    assert len(folds) >= 2
    first = folds[0]
    first_quarter = pd.Period(first.name, freq="Q")
    expected_last = max(
        index
        for index, value in enumerate(dates)
        if value.to_period("Q") == first_quarter
    )
    assert first.test_positions[-1] == expected_last


def test_backtest_writes_all_required_artifacts_from_central_signal_path(tmp_path) -> None:
    _write_cbr_history(tmp_path)
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "indicators": [
                    {
                        "name": "level_percentile",
                        "params": {"window": [5], "pct": [100]},
                        "direction": "favorable",
                        "speed": "slow",
                        "days_to_confirm": None,
                        "scenario": "T1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts"

    meta = run_backtest(
        raw_dir=tmp_path,
        grid_path=grid_path,
        artifact_dir=artifacts,
        series={"RUB->TJS": "CBR:TJS"},
        min_train_observations=100,
    )

    signals = pd.read_csv(artifacts / "signals.csv")
    metrics = pd.read_csv(artifacts / "metrics.csv")
    saved_meta = json.loads((artifacts / "run_meta.json").read_text(encoding="utf-8"))
    assert signals.columns.tolist() == [
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
    ]
    assert metrics.columns.tolist() == list(METRIC_COLUMNS)
    assert not metrics.empty
    assert saved_meta["grid_sha256"] == meta["grid_sha256"]
    assert saved_meta["registered_configurations"] == 1
