import pandas as pd

from fxpulse.recipient_leg_experiment import build_independent_features


def test_independent_cross_uses_units_and_is_lagged() -> None:
    features = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "corridor": ["KZT"] * 3,
            "price": [0.2, 0.2, 0.2],
        }
    )
    rows = []
    for date, usd, rub in (("2026-01-01", 500, 5), ("2026-01-02", 510, 5.1), ("2026-01-03", 520, 5.2)):
        rows.extend(
            [
                {"requested_date": date, "bank": "NBK_KZ", "local_ccy": "KZT", "quote_ccy": "USD", "nominal": 1, "local_per_nominal": usd, "is_carried": False},
                {"requested_date": date, "bank": "NBK_KZ", "local_ccy": "KZT", "quote_ccy": "RUB", "nominal": 1, "local_per_nominal": rub, "is_carried": False},
            ]
        )

    result = build_independent_features(features, pd.DataFrame(rows), corridor="KZT", bank="NBK_KZ", lag=1)

    assert pd.isna(result.loc[0, "independent__cross_rub_per_local"])
    assert result.loc[1, "independent__cross_rub_per_local"] == 0.2
    assert result.loc[1, "independent__cross_gap_bps"] == 0
