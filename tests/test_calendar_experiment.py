import pandas as pd

from fxpulse.calendar_experiment import add_calendar_features, feature_columns


def test_calendar_features_are_known_from_timestamp() -> None:
    frame = pd.DataFrame({"timestamp": ["2024-03-27"], "base__weekday_sin": [0.0]})
    result = add_calendar_features(frame)

    assert result.loc[0, "calendar__tax_proxy_25_28"] == 1.0
    assert result.loc[0, "calendar__quarter_end_window"] == 1.0
    assert "base__weekday_sin" not in feature_columns(result, "no_calendar")
    assert "calendar__tax_proxy_25_28" in feature_columns(result, "expanded_calendar")
