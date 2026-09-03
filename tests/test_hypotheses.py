from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fxpulse.hypotheses import (
    _make_panel,
    daily_candidate_signals,
    daily_model_features,
    daily_ohlc_candidate_signals,
    hypotheses_sha256,
    load_model_hypotheses,
    monthly_logistic_signals,
    run_hypotheses,
)


def _daily_history(path, periods: int = 900) -> None:
    dates = pd.date_range("2018-01-02", periods=periods, freq="B")
    prices = [10.0]
    for index in range(1, periods):
        if index % 50 < 3:
            prices.append(prices[-1] * 0.99)
        else:
            prices.append(prices[-1] * 1.0005)
    pd.DataFrame(
        {
            "trade_date": dates.date,
            "secid": "CNYRUB_TOM",
            "open": prices,
            "high": [price * 1.001 for price in prices],
            "low": [price * 0.999 for price in prices],
            "close": prices,
            "num_trades": [100 + index for index in range(periods)],
        }
    ).to_csv(path, index=False)


def test_daily_hypotheses_do_not_change_when_future_prices_are_appended(tmp_path) -> None:
    raw_path = tmp_path / "moex_daily.csv"
    _daily_history(raw_path)
    original = pd.read_csv(raw_path)
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(original["trade_date"]).dt.normalize() + pd.DateOffset(hours=23, minutes=59),
            "price": original["close"],
        }
    )
    cut = 700
    before = daily_candidate_signals(daily.iloc[:cut].reset_index(drop=True))
    after = daily_candidate_signals(daily).query("position < @cut").reset_index(drop=True)

    pd.testing.assert_frame_equal(before, after)


def test_monthly_logistic_signals_do_not_change_when_future_prices_are_appended(tmp_path) -> None:
    raw_path = tmp_path / "moex_daily.csv"
    _daily_history(raw_path)
    raw = pd.read_csv(raw_path)
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["trade_date"]).dt.normalize() + pd.DateOffset(hours=23, minutes=59),
            "price": raw["close"],
        }
    )
    registry = load_model_hypotheses()
    cut = 700
    model = registry["models"][0]
    before_daily = daily.iloc[:cut].reset_index(drop=True)
    before = monthly_logistic_signals(
        before_daily,
        _make_panel(before_daily["timestamp"], before_daily["price"], "test"),
        test_from=pd.Timestamp("2020-01-01"),
        model_config=model,
        evaluation_config=registry["evaluation"],
    )
    after = monthly_logistic_signals(
        daily,
        _make_panel(daily["timestamp"], daily["price"], "test"),
        test_from=pd.Timestamp("2020-01-01"),
        model_config=model,
        evaluation_config=registry["evaluation"],
    ).query("position < @cut").reset_index(drop=True)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)


def test_ohlc_hypotheses_do_not_change_when_future_prices_are_appended(tmp_path) -> None:
    raw_path = tmp_path / "moex_daily.csv"
    _daily_history(raw_path)
    raw = pd.read_csv(raw_path)
    daily = raw.rename(columns={"trade_date": "timestamp", "close": "price"}).copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"]).dt.normalize() + pd.DateOffset(hours=23, minutes=59)
    registry = load_model_hypotheses()
    cut = 700
    before = daily_ohlc_candidate_signals(daily.iloc[:cut].reset_index(drop=True), registry["daily_rules"])
    after = daily_ohlc_candidate_signals(daily, registry["daily_rules"]).query("position < @cut").reset_index(drop=True)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)


def test_model_features_convert_zero_volatility_zscores_to_missing() -> None:
    daily = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=300, freq="B"),
            "price": [10.0] * 300,
        }
    )
    features = daily_model_features(daily).drop(columns=["position", "timestamp"])
    assert not np.isinf(features.to_numpy(dtype="float64")).any()


def test_runner_marks_h4_and_h5_not_testable_without_their_required_data(tmp_path) -> None:
    daily_path = tmp_path / "moex_daily.csv"
    _daily_history(daily_path)
    artifact_dir = tmp_path / "artifacts"

    meta = run_hypotheses(
        daily_path=daily_path,
        candles_path=tmp_path / "missing-candles.csv",
        app_quotes_path=tmp_path / "missing-app-quotes.csv",
        artifact_dir=artifact_dir,
        test_from="2020-01-01",
    )

    metrics = pd.read_csv(artifact_dir / "metrics.csv")
    assert meta["metrics_written"] == len(metrics)
    assert set(metrics.loc[metrics["status"].eq("not_testable"), "hypothesis"]) == {
        "H4_intraday_session_dip",
        "H5_quote_fidelity",
    }
    meta = json.loads((artifact_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["test_from"] == "2020-01-01"
    assert meta["hypothesis_config_sha256"] == hypotheses_sha256()
