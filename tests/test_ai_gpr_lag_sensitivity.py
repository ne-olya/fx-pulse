from __future__ import annotations

import pandas as pd

from fxpulse.ai_gpr_lag_sensitivity import feature_sets, prepare_frame


def _config() -> dict[str, object]:
    return {
        "corridors": ["AMD"],
        "base_availability_lag_days": 7,
        "tested_total_lags_days": [7, 14, 30],
    }


def test_lag_sensitivity_never_moves_feature_earlier() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", "2026-02-10", freq="D"),
            "corridor": "AMD",
            "base__return_1": 0.0,
        }
    )
    external = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(["2026-01-08", "2026-01-24", "2026-01-31"]),
            "corridor": ["AMD", "AMD", "AMD"],
            "aigpr__daily_gpr_level": [30.0, 14.0, 7.0],
        }
    )

    result = prepare_frame(market, external, _config())

    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-31")
    assert result.loc[0, "aigprlag7__daily_gpr_level"] == 7
    assert result.loc[0, "aigprlag14__daily_gpr_level"] == 14
    assert result.loc[0, "aigprlag30__daily_gpr_level"] == 30
    sets = feature_sets(result, _config())
    assert "aigprlag30__daily_gpr_level" in sets["plus_ai_gpr_lag30"]
