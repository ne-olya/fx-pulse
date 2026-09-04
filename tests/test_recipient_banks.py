import datetime as dt
import json

from fxpulse.data.recipient_banks import parse_kazakhstan_xml, parse_uzbekistan_json


def test_parse_kazakhstan_rates() -> None:
    payload = b"""<rates><date>10.01.2024</date><item><title>USD</title><description>453.19</description><quant>1</quant></item><item><title>RUB</title><description>5.02</description><quant>1</quant></item></rates>"""

    rows = parse_kazakhstan_xml(payload, requested_date=dt.date(2024, 1, 10))

    assert [(row["quote_ccy"], row["local_per_nominal"]) for row in rows] == [("USD", 453.19), ("RUB", 5.02)]
    assert not any(row["is_carried"] for row in rows)


def test_kazakhstan_explicit_no_information_is_not_a_rate() -> None:
    payload = """<rates><date>10.01.2018</date><info>на выбранную дату информации нет.</info></rates>""".encode()

    assert parse_kazakhstan_xml(payload, requested_date=dt.date(2018, 1, 10)) == []


def test_parse_uzbekistan_rates_and_activation_date() -> None:
    payload = json.dumps(
        [
            {"Ccy": "USD", "Nominal": "1", "Rate": "12397.04", "Date": "09.01.2024"},
            {"Ccy": "RUB", "Nominal": "1", "Rate": "138.10", "Date": "09.01.2024"},
        ]
    ).encode()

    rows = parse_uzbekistan_json(payload, requested_date=dt.date(2024, 1, 10))

    assert rows[0]["local_per_nominal"] == 12397.04
    assert all(row["is_carried"] for row in rows)
