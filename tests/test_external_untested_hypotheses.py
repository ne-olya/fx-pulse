from __future__ import annotations

import pandas as pd

from fxpulse.external_untested_hypotheses import futures_curve_features


def test_futures_features_are_available_next_session() -> None:
    futures = pd.DataFrame(
        {
            "begin": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "expiry": ["2026-03-19", "2026-06-18"] * 2,
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1000, 100, 1000, 100],
        }
    )
    spot = pd.DataFrame(
        {
            "dt_msk": ["2026-01-01 20:00", "2026-01-02 20:00"],
            "secid": ["CNYRUB_TOM", "CNYRUB_TOM"],
            "close": [100.0, 100.0],
        }
    )
    result = futures_curve_features(
        futures, spot, minimum_days=7, maximum_near_days=120, maximum_far_days=240
    )
    assert pd.isna(result.loc[0, "futures__near_basis"])
    assert abs(result.loc[1, "futures__near_basis"] - 0.01) < 1e-12
