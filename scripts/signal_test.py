#!/usr/bin/env python3
"""
Проверяем, годится ли лид-лаг «мажоры -> валюты СНГ» как сигнал по метрикам кейса.

Направление: клиент отдаёт рубли, получает нацвалюту. Курс = рублей за 1 единицу.
Низкий курс = выгодно клиенту.
Попадание для «сейчас выгодно» = курс НЕ стал лучше (ниже) в течение h дней:
    min(P[t+1..t+h]) >= P[t]
Выгода момента, бп = mean(P[t+1..t+h]) / P[t] - 1  (плюс = ждать было хуже)
База «случайного дня» = та же метрика по всем дням периода.
Информация на момент решения: доходности, посчитанные по курсам с датой действия <= t
(курс на дату t публикуется днём t-1), прогноз — про дату t+1 и далее. Заглядывания нет.
"""
import csv, datetime as dt, numpy as np

rows=list(csv.DictReader(open("/Users/kirillnemkin/Хакатон/data/cbr_rates_daily_2018-2026.csv")))
codes=[c for c in rows[0] if c!="date"]
dates=[dt.date.fromisoformat(r["date"]) for r in rows]
X={c:np.array([float(r[c]) for r in rows]) for c in codes}
R={c:np.concatenate([[np.nan],np.diff(np.log(X[c]))]) for c in codes}   # R[c][i] выровнен на dates[i]
CIS=["TJS","UZS","KGS","KZT","AMD"]; HS=[1,3,5,10,20]
N=len(dates)

def hit_and_bp(P,t,h):
    fut=P[t+1:t+1+h]
    return (fut.min()>=P[t]), (fut.mean()/P[t]-1)*1e4

def evaluate(P, fire, mask, h):
    idx=[t for t in range(1,N-h) if mask[t]]
    if not idx: return None
    base=[hit_and_bp(P,t,h) for t in idx]
    sig=[hit_and_bp(P,t,h) for t in idx if fire[t]]
    if len(sig)<30: return None
    bh=np.mean([b[0] for b in base]); sh=np.mean([s[0] for s in sig])
    bbp=np.mean([b[1] for b in base]); sbp=np.mean([s[1] for s in sig])
    sd=np.std([s[1] for s in sig],ddof=1)
    tstat=(sbp-bbp)/(sd/np.sqrt(len(sig)))
    return dict(n=len(sig),freq=len(sig)/len(idx),hit=sh,base=bh,lift=sh/bh if bh else np.nan,
                bp=sbp,bp_base=bbp,t=tstat)

# ---- сигнал: TJS отстал от USD сегодня -> завтра догонит вверх -> переводить сейчас
def gap_signal(c,q):
    g=R["USD"]-R[c]                       # насколько мажор ушёл вверх сильнее коридора
    thr=np.nanquantile(g[np.isfinite(g)],q)
    return g>thr, g

periods=[("весь период 2018-2026",lambda d:True),
         ("до 2022 (2018-2021)",lambda d:d.year<2022),
         ("2022+ (2022-2026)",lambda d:d.year>=2022)]

print("СИГНАЛ: gap[t] = r_USD[t] - r_CIS[t] выше 80-го процентиля")
print("(мажор вырос сильнее коридора -> ждём догоняющий рост коридора -> «переводить сейчас»)\n")
for pname,pf in periods:
    mask=np.array([pf(d) for d in dates])
    print("="*96); print(pname); print("="*96)
    print(f"{'вал':5}{'h':>4}{'сигн.':>7}{'частота':>9}{'hit':>8}{'база':>8}{'lift':>7}"
          f"{'выгода бп':>11}{'база бп':>9}{'t-стат':>8}")
    for c in CIS:
        fire,_=gap_signal(c,0.80)
        for h in HS:
            r=evaluate(X[c],fire,mask,h)
            if r: print(f"{c:5}{h:>4}{r['n']:>7}{r['freq']:>9.1%}{r['hit']:>8.3f}{r['base']:>8.3f}"
                        f"{r['lift']:>7.2f}{r['bp']:>11.1f}{r['bp_base']:>9.1f}{r['t']:>8.1f}")
        print()
