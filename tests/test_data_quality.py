from __future__ import annotations

import pandas as pd

from fxpulse.data.quality import _moex_hourly_section


def test_hourly_quality_section_reports_range_and_duplicates(tmp_path) -> None:
    timestamps = ["2026-01-05 10:59:59", "2026-01-05 10:59:59", "2026-01-05 11:59:59"]
    pd.DataFrame(
        {
            "dt_msk": timestamps,
            "secid": ["CNYRUB_TOM"] * 3,
            "open": [11.0, 11.0, 11.1],
            "high": [11.2, 11.2, 11.3],
            "low": [10.9, 10.9, 11.0],
            "close": [11.1, 11.1, 11.2],
            "fetched_at": ["2026-01-06T00:00:00+00:00"] * 3,
            "source_url": ["https://iss.moex.com/example"] * 3,
        }
    ).to_csv(tmp_path / "moex_cny_60m.csv", index=False)

    section, summary = _moex_hourly_section(tmp_path)

    assert summary == {"moex_hourly_rows": 3, "moex_hourly_duplicates": 1}
    assert "2026-01-05 10:59:59" in section
    assert "2026-01-05 11:59:59" in section
