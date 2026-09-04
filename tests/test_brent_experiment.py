import pandas as pd

from fxpulse.brent_experiment import add_brent_features


def test_brent_value_is_unavailable_before_conservative_lag() -> None:
    frame = pd.DataFrame(
        {"timestamp": ["2024-01-07", "2024-01-08"], "corridor": ["TJS", "TJS"], "price": [1, 1]}
    )
    brent = pd.DataFrame(
        {
            "date": pd.date_range("2023-12-01", periods=32, freq="D"),
            "brent_usd_per_barrel": range(70, 102),
        }
    )
    result = add_brent_features(frame, brent, lag_days=7)

    # The Jan 1 observation can first appear on Jan 8, never on Jan 7.
    assert result.loc[0, "market__brent_return_1"] != result.loc[1, "market__brent_return_1"]
