from __future__ import annotations

import json

import pandas as pd

from fxpulse.data.assemble import assemble_daily_panel


def test_assemble_daily_panel_normalizes_cbr_and_keeps_missing_values(tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pd.DataFrame(
        {
            "trade_date": ["2026-01-01", "2026-01-02"],
            "instrument_id": ["fx_cny_rub_tom", "fx_cny_rub_tom"],
            "close": [11.0, 11.1],
        }
    ).to_csv(snapshot / "target.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": ["2026-01-01"],
            "instrument_id": ["factor"],
            "close": [100.0],
        }
    ).to_csv(snapshot / "factor.csv", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "series": [
                    {"instrument_id": "fx_cny_rub_tom", "path": "target.csv"},
                    {"instrument_id": "factor", "path": "factor.csv"},
                ]
            }
        ),
        encoding="utf-8",
    )
    cbr = tmp_path / "cbr.csv"
    pd.DataFrame(
        {
            "rate_date": ["2026-01-03"],
            "ccy": ["KZT"],
            "nominal": [10],
            "rate_rub": [20.0],
        }
    ).to_csv(cbr, index=False)

    panel, used_snapshot = assemble_daily_panel(cbr_path=cbr, snapshot_dir=snapshot)

    assert used_snapshot == snapshot
    assert len(panel) == 3
    assert panel.loc[panel["trade_date"].eq(pd.Timestamp("2026-01-03")), "cbr__kzt__rub_per_unit"].item() == 2
    assert pd.isna(panel.loc[panel["trade_date"].eq(pd.Timestamp("2026-01-02")), "moex__factor__close"].item())
