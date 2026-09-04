import numpy as np
import pandas as pd

from fxpulse.liquidity_experiment import (
    attach_liquidity_features,
    build_liquidity_features,
    paired_comparison,
)


def _raw_activity(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for multiplier, secid in enumerate(["CNY", "USD", "KZT"], start=1):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "secid": secid,
                    "num_trades": multiplier * (index + 1),
                    "volume_rub": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_activity_features_are_lagged_and_do_not_use_future_rows() -> None:
    raw = _raw_activity()
    mapping = {"cny": "CNY", "usd": "USD", "kzt": "KZT"}
    cut = pd.Timestamp("2024-03-15")
    before = build_liquidity_features(raw.loc[pd.to_datetime(raw["trade_date"]).le(cut)], instruments=mapping, lag=1)
    after = build_liquidity_features(raw, instruments=mapping, lag=1)
    after = after.loc[after["timestamp"].le(cut)].reset_index(drop=True)

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)
    feature = "liquidity__cny_num_trades_log_lag1"
    assert pd.isna(after.loc[0, feature])
    assert after.loc[1, feature] == np.log1p(1)
    assert not any("volume_rub" in column for column in after.columns)


def test_asof_alignment_does_not_carry_stale_activity() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-10"]),
            "corridor": ["AMD"] * 3,
        }
    )
    liquidity = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "liquidity__cny_num_trades_log_lag1": [2.0],
        }
    )
    result = attach_liquidity_features(frame, liquidity, carry_days=3)

    assert result.loc[0, "liquidity__cny_num_trades_log_lag1"] == 2.0
    assert result.loc[1, "liquidity__cny_num_trades_log_lag1"] == 2.0
    assert pd.isna(result.loc[2, "liquidity__cny_num_trades_log_lag1"])


def test_liquidity_comparison_is_paired() -> None:
    common = {"scope": "AMD", "model": "catboost_3seed_mean", "horizon": 5}
    metrics = {
        "lift_vs_matched_random": 1.0,
        "regret_mean_bps": 100.0,
        "benefit_mean_bps": 10.0,
        "signals_per_week": 1.0,
        "worst_fold_lift": 0.9,
    }
    summary = pd.DataFrame(
        [
            {**common, "feature_set": "baseline", "lift": 1.2, **metrics},
            {**common, "feature_set": "plus_liquidity", "lift": 1.3, **metrics},
        ]
    )

    result = paired_comparison(summary)

    assert abs(float(result.loc[0, "delta_lift"]) - 0.1) < 1e-12
