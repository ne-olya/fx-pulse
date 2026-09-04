from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd

from fxpulse.data.gdelt import aggregate_hourly, parse_export_archive


def _archive(rows: list[list[str]]) -> bytes:
    text = io.StringIO()
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.export.CSV", text.getvalue())
    return output.getvalue()


def _row(event_id: str, *, actor2: str, geo: str, url: str) -> list[str]:
    row = [""] * 61
    row[0] = event_id
    row[17] = actor2
    row[28] = "17"
    row[29] = "3"
    row[30] = "-5"
    row[34] = "-4"
    row[53] = geo
    row[59] = "20260102031500"
    row[60] = url
    return row


def test_parser_keeps_only_relevant_countries() -> None:
    payload = _archive(
        [
            _row("1", actor2="RUS", geo="RS", url="https://example.com/a"),
            _row("2", actor2="USA", geo="US", url="https://example.com/b"),
        ]
    )

    result = parse_export_archive(payload)

    assert len(result) == 1
    assert result[0]["actor_country_codes"] == "RUS"
    assert result[0]["timestamp_utc"] == "2026-01-02T03:15:00+00:00"


def test_hourly_aggregation_deduplicates_article_events() -> None:
    payload = _archive(
        [
            _row("1", actor2="RUS", geo="AM", url="https://example.com/same"),
            _row("2", actor2="ARM", geo="RS", url="https://example.com/same"),
        ]
    )
    raw = pd.DataFrame(parse_export_archive(payload))

    result = aggregate_hourly(raw)
    amd = result.loc[result["corridor"].eq("AMD")].iloc[0]

    assert amd["article_count"] == 1
    assert amd["bilateral_article_count"] == 1
    assert amd["coercion_count"] == 1
