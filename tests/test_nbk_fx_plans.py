from fxpulse.data.nbk_fx_plans import extract_plan


def test_extracts_billion_range_and_purchase() -> None:
    article = {
        "url": "https://example.test/1",
        "published_at": "2024-01-03 10:00:00",
        "text": (
            "По заявкам в январе 2024 года Национальным Банком ожидается продажа валюты "
            "из Национального фонда в размере от 1 до 1,1 млрд долларов США. "
            "Покупка валюты в январе 2024 года ожидается в размере от 100 до 200 млн долларов США."
        ),
    }
    plan = extract_plan(article)
    assert plan is not None
    assert plan["planned_sale_mid_usd_mn"] == 1050
    assert plan["planned_net_sale_mid_usd_mn"] == 900
