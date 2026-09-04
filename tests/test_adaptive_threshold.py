import pandas as pd

from fxpulse.adaptive_threshold import adaptive_candidates


def test_adaptive_threshold_uses_strictly_previous_scores() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5),
            "score": [0.1, 0.2, 0.3, 0.4, 100.0],
        }
    )

    selected = adaptive_candidates(frame, share=0.5, lookback=3, minimum_history=2)

    assert selected["timestamp"].tolist() == frame.loc[[2, 3, 4], "timestamp"].tolist()
    assert selected.loc[selected["timestamp"].eq(frame.loc[4, "timestamp"]), "score_threshold"].item() < 1
