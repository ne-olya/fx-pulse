from __future__ import annotations

import pandas as pd

from fxpulse.decision_audit import fast_slow_pairs, symmetric_benefit_bps


def _frame(prices: list[float], corridor: str = "UZS") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2022-01-03", periods=len(prices)),
            "corridor": corridor,
            "price": prices,
        }
    )


def test_fast_signal_is_an_onset_and_waiting_cost_has_client_sign() -> None:
    prices = [10.0] * 6 + [9.9, 9.8, 9.7, 9.6, 9.5, 9.6, 9.7, 9.8, 9.9]

    pairs = fast_slow_pairs(
        _frame(prices),
        start_year=2022,
        end_year=2022,
        fast_streak=3,
        slow_window=5,
        slow_percentile=0.4,
        confirmation_horizon=3,
        outcome_horizon=2,
    )

    assert len(pairs) == 1
    assert bool(pairs.loc[0, "confirmed"])
    assert pairs.loc[0, "waiting_cost_bps"] <= 0


def test_symmetric_benefit_matches_surrounding_window() -> None:
    values = symmetric_benefit_bps(
        _frame([10.0, 8.0, 10.0, 12.0, 10.0]),
        horizon=1,
        start_year=2022,
        end_year=2022,
    )

    middle = values.loc[values["timestamp"].eq(pd.Timestamp("2022-01-05")), "benefit_sym_bps"].iloc[0]
    assert middle == 0.0


def test_duplicate_timestamp_corridor_fails_closed() -> None:
    frame = _frame([10.0, 9.0, 8.0])
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    try:
        symmetric_benefit_bps(duplicate, horizon=1, start_year=2022, end_year=2022)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate panel must fail closed")


def test_infinite_price_fails_closed() -> None:
    frame = _frame([10.0, float("inf"), 8.0])

    try:
        symmetric_benefit_bps(frame, horizon=1, start_year=2022, end_year=2022)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("infinite price must fail closed")
