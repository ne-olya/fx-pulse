import copy

import numpy as np
import pandas as pd

from fxpulse.robust_innovation_experiment import (
    EXPECTED_VARIANTS,
    cluster_bootstrap,
    delayed_dynamic_average,
    load_config,
    nested_year_selection,
    volatility_scaled_triple_barrier_label,
)


def test_registered_config_matches_implementation() -> None:
    config = load_config()
    assert config["variants"] == EXPECTED_VARIANTS
    assert config["development_years"] == [2022, 2023, 2024]
    assert config["pseudo_holdout_years"] == [2025, 2026]


def test_volatility_scaled_barrier_uses_path_order() -> None:
    price = pd.Series([100.0, 99.0, 102.0, 101.0, 101.0])
    volatility = pd.Series([0.01] * len(price))
    label = volatility_scaled_triple_barrier_label(
        price,
        volatility,
        horizon=2,
        better_sigma_fraction=0.01,
        worse_sigma_fraction=0.01,
        better_bps_bounds=(50, 50),
        worse_bps_bounds=(50, 50),
    )
    assert label.iloc[0] == 0  # A better price arrives before the worse barrier.
    assert label.iloc[1] == 1  # The worse barrier arrives first from 99.
    assert label.iloc[-2:].isna().all()


def test_dynamic_average_does_not_read_unmatured_targets() -> None:
    predictions = np.array(
        [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6]], dtype=float
    )
    target_a = np.array([1, 0, 1, 0, 0], dtype=float)
    target_b = np.array([1, 0, 1, 1, 1], dtype=float)
    score_a = delayed_dynamic_average(predictions, target_a, delay=2, eta=0.5)
    score_b = delayed_dynamic_average(predictions, target_b, delay=2, eta=0.5)
    np.testing.assert_allclose(score_a, score_b)


def test_nested_selection_uses_only_years_before_test() -> None:
    config = copy.deepcopy(load_config())
    config["corridors"] = ["AMD"]
    config["horizons"] = [3]
    config["nested_selection_years"] = [2024]
    rows = []
    for variant, past_hits, current_hits in [("past_winner", 8, 1), ("future_winner", 5, 10)]:
        for year in [2022, 2023, 2024]:
            hits = current_hits if year == 2024 else past_hits
            rows.append(
                {
                    "variant": variant,
                    "corridor": "AMD",
                    "horizon": 3,
                    "test_year": year,
                    "test_count": 20,
                    "test_hits": 10,
                    "signal_count": 10,
                    "signal_hits": hits,
                    "lift": (hits / 10) / 0.5,
                    "matched_expected_hits": 5.0,
                    "regret_sum_bps": 0.0,
                    "benefit_sum_bps": 0.0,
                    "duration_weeks": 10.0,
                }
            )
    selected = nested_year_selection(pd.DataFrame(rows), config)
    assert selected.iloc[0]["selected_variant"] == "past_winner"
    assert selected.iloc[0]["lift"] == 0.2


def test_cluster_bootstrap_is_deterministic() -> None:
    weekly = pd.DataFrame(
        {
            "signal_count": [1, 1, 1, 1],
            "signal_hits": [1, 1, 0, 1],
            "matched_expected_hits": [0.5, 0.5, 0.5, 0.5],
            "base_count": [5, 5, 5, 5],
            "base_hits": [2, 3, 2, 3],
        }
    )
    first = cluster_bootstrap(weekly, samples=200, seed=7)
    second = cluster_bootstrap(weekly, samples=200, seed=7)
    assert first == second
    assert first["observed_matched_lift"] == 1.5
