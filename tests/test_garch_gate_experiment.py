import pandas as pd

from fxpulse.garch_gate_experiment import gate_mask


def test_garch_gates_partition_known_forecasts() -> None:
    frame = pd.DataFrame({"forecast_vol_percentile": [0.2, 0.67, 0.8, None]})

    low = gate_mask(frame, "low_mid_forecast_vol", 0.67)
    high = gate_mask(frame, "high_forecast_vol", 0.67)

    assert low.tolist() == [True, True, False, False]
    assert high.tolist() == [False, False, True, False]
