import pandas as pd

from fxpulse.training_history_experiment import add_training_labels


def test_extended_history_training_label_keeps_future_out_of_features() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "corridor": ["TJS"] * 5,
            "price": [100, 99, 101, 102, 103],
        }
    )
    config = {
        "horizons": [3],
        "triple_barrier": {"better_price_barrier_bps": 25, "worse_price_barrier_bps": 50},
    }
    result = add_training_labels(frame, config)

    assert result.loc[0, "training_target_3"] == 0
    assert pd.isna(result.loc[2, "training_target_3"])
