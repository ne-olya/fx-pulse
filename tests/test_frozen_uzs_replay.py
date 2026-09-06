from __future__ import annotations

import pytest

from fxpulse.frozen_uzs_replay import _verify_metrics


def test_replay_export_rejects_a_different_model_result() -> None:
    with pytest.raises(RuntimeError, match="strong signals"):
        _verify_metrics(
            {
                "signals": 139,
                "raw_lift": 1.56,
                "same_week_lift": 1.25,
                "signals_per_week": 0.61,
            }
        )


def test_replay_export_accepts_published_rounded_metrics() -> None:
    _verify_metrics(
        {
            "signals": 140,
            "raw_lift": 1.559,
            "same_week_lift": 1.251,
            "signals_per_week": 0.609,
        }
    )
