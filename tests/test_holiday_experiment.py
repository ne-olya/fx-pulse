import pandas as pd

from fxpulse.holiday_experiment import add_holiday_features, build_holiday_table


def test_recipient_holiday_and_lead_are_separate_from_russia() -> None:
    table = build_holiday_table([2024], ["RU", "UZ"])
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-03-20", "2024-03-21"],
            "corridor": ["UZS", "UZS"],
        }
    )
    result = add_holiday_features(frame, table, {"UZS": "UZ"})

    assert result.loc[0, "holiday__recipient_in_next_3_days"] == 1.0
    assert result.loc[1, "holiday__recipient_today"] == 1.0
    assert result.loc[1, "holiday__today_mismatch"] == 1.0
