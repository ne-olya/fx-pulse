#!/usr/bin/env python3
"""
Выгрузка дневных официальных курсов ЦБ РФ (открытый источник, воспроизводимо).

Источник: https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=DD/MM/YYYY&date_req2=DD/MM/YYYY&VAL_NM_RQ=<ID>
Даты в выгрузке ЦБ — это даты ДЕЙСТВИЯ курса. Курс на дату T публикуется днём T-1.
Выходные и праздники в выгрузке отсутствуют -> заполняем переносом предыдущего значения
и помечаем флагом is_carried, чтобы серии нулевых изменений не считались боковиком рынка.
"""
import urllib.request, xml.etree.ElementTree as ET, csv, datetime as dt, sys, os

VALUTES = [
    ("USD", "R01235", "Доллар США"),
    ("EUR", "R01239", "Евро"),
    ("CNY", "R01375", "Юань"),
    ("TJS", "R01670", "Сомони (Таджикистан)"),
    ("UZS", "R01717", "Сум (Узбекистан)"),
]

DATE_TO   = dt.date(2026, 9, 2)
DATE_FROM = dt.date(2026, 8, 2)
# запрашиваем с запасом назад, чтобы корректно заполнить первый день, если он выходной
PAD = 10

OUT_DIR = "/Users/kirillnemkin/Хакатон/data"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}

def fetch(val_id, d1, d2):
    url = ("https://www.cbr.ru/scripts/XML_dynamic.asp"
           f"?date_req1={d1:%d/%m/%Y}&date_req2={d2:%d/%m/%Y}&VAL_NM_RQ={val_id}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        raw = r.read()
    root = ET.fromstring(raw.decode("windows-1251"))
    rows = {}
    for rec in root.findall("Record"):
        d = dt.datetime.strptime(rec.attrib["Date"], "%d.%m.%Y").date()
        nominal = int(rec.findtext("Nominal").replace(",", "."))
        value = float(rec.findtext("Value").replace(",", "."))
        rows[d] = (nominal, value)
    return url, rows

series = {}
urls = {}
for code, vid, name in VALUTES:
    url, rows = fetch(vid, DATE_FROM - dt.timedelta(days=PAD), DATE_TO)
    if not rows:
        sys.exit(f"пустой ответ по {code}")
    series[code] = rows
    urls[code] = url
    print(f"{code}: {len(rows)} записей ЦБ, {min(rows)} .. {max(rows)}", file=sys.stderr)

all_days = [DATE_FROM + dt.timedelta(days=i) for i in range((DATE_TO - DATE_FROM).days + 1)]

long_rows, wide_rows = [], []
for d in all_days:
    wide = {"date": d.isoformat(), "weekday": d.strftime("%a")}
    for code, vid, name in VALUTES:
        rows = series[code]
        carried = d not in rows
        # ищем последнюю доступную дату <= d
        prev = max((k for k in rows if k <= d), default=None)
        nominal, value = rows[prev]
        unit = value / nominal                      # рублей за 1 единицу валюты
        long_rows.append({
            "date": d.isoformat(), "weekday": d.strftime("%a"),
            "char_code": code, "name": name, "val_id": vid,
            "nominal": nominal, "value_cbr": f"{value:.4f}",
            "rub_per_unit": f"{unit:.8f}",
            "is_carried": int(carried), "rate_date_source": prev.isoformat(),
        })
        wide[code] = f"{unit:.8f}"
        wide[code + "_carried"] = int(carried)
    wide_rows.append(wide)

os.makedirs(OUT_DIR, exist_ok=True)
p_long = os.path.join(OUT_DIR, "cbr_rates_long_2026-08-02_2026-09-02.csv")
p_wide = os.path.join(OUT_DIR, "cbr_rates_wide_2026-08-02_2026-09-02.csv")

with open(p_long, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys())); w.writeheader(); w.writerows(long_rows)
with open(p_wide, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys())); w.writeheader(); w.writerows(wide_rows)

print("\n".join(f"{c}: {u}" for c, u in urls.items()), file=sys.stderr)
print(f"\n{p_long}\n{p_wide}", file=sys.stderr)
