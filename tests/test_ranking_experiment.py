import pandas as pd

from fxpulse.ranking_experiment import ranking_qid, relevance_from_regret


def test_relevance_rewards_low_regret() -> None:
    result = relevance_from_regret(pd.Series([0, 25, 26, 50, 51, 100, 101]), [25, 50, 100])

    assert result.tolist() == [3, 3, 2, 2, 1, 1, 0]


def test_ranking_group_separates_corridor_and_week() -> None:
    frame = pd.DataFrame(
        {"timestamp": ["2024-01-01", "2024-01-02", "2024-01-01"], "corridor": ["TJS", "TJS", "UZS"]}
    )

    assert ranking_qid(frame).nunique() == 2
