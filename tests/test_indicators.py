from __future__ import annotations

import pandas as pd

from fxpulse.indicators import (
    filter_rule,
    level_percentile,
    momentum_streak,
    reversal_from_low,
    volatility_regime,
)


def _panel(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "known_at": dates.tz_localize("Europe/Moscow") + pd.DateOffset(hours=23, minutes=59),
            "value_date": dates.date,
            "price": prices,
        }
    )


def test_price_level_and_decline_indicators_are_explainable() -> None:
    panel = _panel([4.0, 3.0, 2.0, 1.0])

    assert level_percentile(panel, window=4, pct=25).fired
    assert momentum_streak(panel, n=3).fired
    assert filter_rule(_panel([10.0, 9.0]), x=5).fired


def test_reversal_uses_only_prior_low_and_trailing_sigma() -> None:
    output = reversal_from_low(_panel([100.0, 90.0, 91.0, 92.0, 100.0]), lookback=4, bounce=0.5)

    assert output.fired
    assert output.details["rebound"] > output.details["threshold"]


def test_high_volatility_is_compared_to_prior_rolling_history() -> None:
    returns = [0.001, -0.001] * 30 + [0.03, -0.03] * 10
    prices = [100.0]
    for value in returns:
        prices.append(prices[-1] * (1 + value))

    output = volatility_regime(_panel(prices), window=20, regime="high")

    assert output.fired
    assert output.details["volatility"] > output.details["high_tercile"]
