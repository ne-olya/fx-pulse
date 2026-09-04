import pandas as pd

from fxpulse.event_sampling_experiment import _cusum_events, add_event_flags


def test_cusum_uses_current_and_past_moves_only() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, -0.01])
    sigma = pd.Series([0.04] * 4)
    flags = _cusum_events(returns, sigma, 0.5)

    assert flags.tolist() == [False, True, False, False]


def test_event_flags_are_separate_by_corridor() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"] * 2,
            "corridor": ["TJS", "TJS", "UZS", "UZS"],
            "base__return_1": [0.01, 0.01, -0.01, -0.01],
            "base__volatility_20": [0.04] * 4,
        }
    )
    result = add_event_flags(frame)

    assert result["event__cusum_0.5sigma"].tolist() == [False, True, False, True]
