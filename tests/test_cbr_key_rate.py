import datetime as dt

from fxpulse.data.cbr_key_rate import parse_key_rate_html


def test_parse_key_rate_html() -> None:
    payload = """
    <table><tr><td>02.09.2026</td><td>14,00</td></tr>
    <tr><td><span>01.09.2026</span></td><td>14,25</td></tr></table>
    """.encode()

    assert parse_key_rate_html(payload) == [
        (dt.date(2026, 9, 1), 14.25),
        (dt.date(2026, 9, 2), 14.0),
    ]
