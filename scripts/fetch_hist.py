#!/usr/bin/env python3
"""Полная история дневных курсов ЦБ РФ по 8 валютам, 2018-01-01 .. 2026-09-02."""
import urllib.request, xml.etree.ElementTree as ET, csv, datetime as dt, sys, os, time

VALUTES = [("USD","R01235"),("EUR","R01239"),("CNY","R01375"),
           ("TJS","R01670"),("UZS","R01717"),("KGS","R01370"),
           ("KZT","R01335"),("AMD","R01060")]
D1, D2 = dt.date(2018,1,1), dt.date(2026,9,2)
OUT = "/Users/kirillnemkin/Хакатон/data/cbr_rates_daily_2018-2026.csv"
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}

series={}
for code,vid in VALUTES:
    url=(f"https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={D1:%d/%m/%Y}"
         f"&date_req2={D2:%d/%m/%Y}&VAL_NM_RQ={vid}")
    root=ET.fromstring(urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=120)
                       .read().decode("windows-1251"))
    rows={}
    for rec in root.findall("Record"):
        d=dt.datetime.strptime(rec.attrib["Date"],"%d.%m.%Y").date()
        nom=int(rec.findtext("Nominal").replace(",","."))
        val=float(rec.findtext("Value").replace(",","."))
        rows[d]=val/nom
    series[code]=rows
    print(f"{code}: {len(rows)} записей, {min(rows)} .. {max(rows)}", file=sys.stderr)
    time.sleep(0.4)

# даты публикации ЦБ = объединение дат действия по USD (базовая валюта фиксинга)
dates=sorted(set().union(*[set(v) for v in series.values()]))
codes=[c for c,_ in VALUTES]
with open(OUT,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["date"]+codes)
    for d in dates:
        w.writerow([d.isoformat()]+[f"{series[c][d]:.10f}" if d in series[c] else "" for c in codes])
print(f"\n{OUT}: {len(dates)} дат действия курса", file=sys.stderr)
