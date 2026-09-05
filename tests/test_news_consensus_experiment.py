from __future__ import annotations

import pandas as pd

from fxpulse.news_consensus_experiment import peer_blend


def test_peer_blend_excludes_the_target_corridor() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01"] * 3),
            "horizon": [5, 5, 5],
            "corridor": ["AMD", "KGS", "KZT"],
            "rank": [1.0, 0.0, 0.0],
        }
    )

    result = peer_blend(frame, "rank", own_weight=0.75)

    assert result.iloc[0] == 0.75
    assert result.iloc[1] == 0.125
