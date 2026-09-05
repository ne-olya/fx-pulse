from fxpulse.data.moex_cny_futures import contract_codes, third_thursday


def test_third_thursday_and_contract_code() -> None:
    assert third_thursday(2024, 3).date().isoformat() == "2024-03-21"
    assert ("CRH4", third_thursday(2024, 3)) in contract_codes(2024, 2024)
