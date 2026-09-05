from __future__ import annotations

import pandas as pd

from fxpulse.news_liquidity_ablation import feature_sets


def test_liquidity_ablation_changes_only_registered_feature_blocks() -> None:
    frame = pd.DataFrame(
        {
            "base__return_1": [0.1],
            "liquidity__usd_num_trades": [100.0],
            "news__russia_count": [2.0],
        }
    )

    result = feature_sets(frame)

    assert "liquidity__usd_num_trades" in result["plus_liquidity"]
    assert "news__russia_count" not in result["plus_liquidity"]
    assert "liquidity__usd_num_trades" not in result["plus_all_news"]
    assert set(result["plus_liquidity_all_news"]) == set(frame.columns)
