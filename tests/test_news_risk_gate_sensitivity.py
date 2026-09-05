from __future__ import annotations

from fxpulse.news_risk_gate_sensitivity import variant_name


def test_variant_name_is_stable_for_decimal_thresholds() -> None:
    assert variant_name(0.5) == "exclude_either_gt0p5"
    assert variant_name(1.0) == "exclude_either_gt1"
