from fxpulse.data.fred import parse_fred_csv


def test_fred_parser_skips_missing_values() -> None:
    payload = b"observation_date,DCOILBRENTEU\n2024-01-01,.\n2024-01-02,75.25\n"
    result = parse_fred_csv(payload, source_url="https://example.test", fetched_at="now")

    assert len(result) == 1
    assert result[0]["date"] == "2024-01-02"
    assert result[0]["brent_usd_per_barrel"] == 75.25
