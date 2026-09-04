import pandas as pd

from fxpulse.path_label_experiment import trend_scanning_label, triple_barrier_label


def test_triple_barrier_respects_first_touch() -> None:
    rises_first = pd.Series([100.0, 101.0, 99.0, 99.0])
    falls_first = pd.Series([100.0, 99.0, 101.0, 101.0])

    assert triple_barrier_label(rises_first, horizon=3, better_price_barrier_bps=25, worse_price_barrier_bps=50).iloc[0] == 1
    assert triple_barrier_label(falls_first, horizon=3, better_price_barrier_bps=25, worse_price_barrier_bps=50).iloc[0] == 0


def test_trend_scanning_detects_upward_path() -> None:
    price = pd.Series([100.0, 101.0, 102.0, 103.0])

    assert trend_scanning_label(price, horizon=3, scan_horizons=[3]).iloc[0] == 1
