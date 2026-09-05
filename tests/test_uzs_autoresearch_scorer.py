from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_corridor_enriched_hyperparams",
    ROOT / "scripts" / "score_uzs_enriched_hyperparams.py",
)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def test_agreement_gate_selects_configured_target_corridor(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="D")
    corridors = ["AMD", "KGS", "KZT", "TJS", "UZS"]
    scores = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "corridor": corridor,
                "score": 0.5,
                "target": 1,
                "regret_bps": 0.0,
                "test_year": 2026,
                "week": "2026-W01",
                "matched_week_hit_rate": 0.5,
            }
            for timestamp in timestamps
            for corridor in corridors
        ]
    )
    monkeypatch.setattr(
        SCORER,
        "causal_percentile",
        lambda values, **_: pd.Series(np.full(len(values), 0.75), index=values.index),
    )
    monkeypatch.setattr(SCORER, "adaptive_candidates", lambda frame, **_: frame)
    monkeypatch.setattr(SCORER, "_apply_policy", lambda frame, **_: frame)
    contract = {
        "corridors": corridors,
        "target_corridor": "UZS",
        "consensus": {
            "rank_lookback": 60,
            "rank_minimum_history": 20,
            "own_weight": 0.5,
            "other_weight": 0.5,
            "minimum_other_positive_share": 0.75,
            "maximum_other_rank_std": 0.3,
        },
        "policy": {
            "top_score_share": 0.4,
            "lookback_observations": 60,
            "minimum_history_observations": 20,
            "cooldown_days": 3,
            "weekly_cap": 2,
        },
    }

    target, selected = SCORER._agreement_gate(scores, contract)

    assert set(target["corridor"]) == {"UZS"}
    assert set(selected["corridor"]) == {"UZS"}
    assert target["other_positive_share"].eq(1.0).all()


def test_agreement_gate_can_use_target_specific_confirmation_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    corridors = ["AMD", "KGS", "KZT", "TJS", "UZS"]
    scores = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01")] * 5,
            "corridor": corridors,
            "score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [1] * 5,
            "regret_bps": [0.0] * 5,
            "test_year": [2026] * 5,
            "week": ["2026-W01"] * 5,
            "matched_week_hit_rate": [0.5] * 5,
        }
    )
    ranks_by_corridor = {"AMD": 0.1, "KGS": 0.6, "KZT": 0.7, "TJS": 0.8, "UZS": 0.9}

    def fake_rank(values: pd.Series, **_: object) -> pd.Series:
        corridor = scores.loc[values.index[0], "corridor"]
        return pd.Series([ranks_by_corridor[corridor]], index=values.index)

    monkeypatch.setattr(SCORER, "causal_percentile", fake_rank)
    monkeypatch.setattr(SCORER, "adaptive_candidates", lambda frame, **_: frame)
    monkeypatch.setattr(SCORER, "_apply_policy", lambda frame, **_: frame)
    contract = {
        "corridors": corridors,
        "target_corridor": "UZS",
        "consensus": {
            "rank_lookback": 60,
            "rank_minimum_history": 20,
            "own_weight": 0.5,
            "other_weight": 0.5,
            "other_corridors": ["KGS", "TJS"],
            "other_corridor_weights": {"KGS": 0.25, "TJS": 0.75},
            "minimum_other_positive_share": 0.75,
            "maximum_other_rank_std": 0.3,
        },
        "policy": {
            "top_score_share": 0.4,
            "lookback_observations": 60,
            "minimum_history_observations": 20,
            "cooldown_days": 3,
            "weekly_cap": 2,
        },
    }

    target, _ = SCORER._agreement_gate(scores, contract)

    assert target.iloc[0]["other_rank_mean"] == pytest.approx(0.75)
    assert target.iloc[0]["other_positive_share"] == pytest.approx(1.0)


def test_frozen_contract_requires_target_to_be_in_corridor_panel() -> None:
    contract = json.loads((ROOT / "configs" / "uzs_enriched_search_contract.json").read_text())
    contract["target_corridor"] = "RSD"
    params = json.loads((ROOT / "configs" / "uzs_enriched_catboost_params.json").read_text())

    with pytest.raises(ValueError, match="target_corridor"):
        SCORER._validate(contract, params)
