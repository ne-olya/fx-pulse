#!/usr/bin/env python3
"""Walk-forward: обучаем регрессию только на прошлом, сигналим по предсказанию."""
import csv, datetime as dt, numpy as np
rows=list(csv.DictReader(open("/Users/kirillnemkin/Хакатон/data/cbr_rates_daily_2018-2026.csv")))
codes=[c for c in rows[0] if c!="date"]
dates=[dt.date.fromisoformat(r["date"]) for r in rows]
X={c:np.array([float(r[c]) for r in rows]) for c in codes}
R={c:np.concatenate([[np.nan],np.diff(np.log(X[c]))]) for c in codes}
CIS=["TJS","UZS","KGS","KZT","AMD"]; N=len(dates); MIN=500

def walkforward(c):
    """rhat[t] = прогноз r_CIS[t+1], построенный ТОЛЬКО по данным с датой действия <= t."""
    feats=lambda t: [R["USD"][t],R["EUR"][t],R["CNY"][t],R[c][t]]
    rhat=np.full(N,np.nan)
    for t in range(MIN,N-1):
        tr=[(feats(s),R[c][s+1]) for s in range(1,t)]           # только прошлое
        A=np.array([[1]+f for f,_ in tr]); y=np.array([v for _,v in tr])
        ok=np.isfinite(A).all(1)&np.isfinite(y); A,y=A[ok],y[ok]
        b,*_=np.linalg.lstsq(A,y,rcond=None)
        rhat[t]=b@np.array([1]+feats(t))
    return rhat

def metrics(P,fire,h):
    idx=[t for t in range(MIN,N-h) if np.isfinite(P[t])]
    def hb(t):
        f=P[t+1:t+1+h]; return (f.min()>=P[t]), (f.mean()/P[t]-1)*1e4
    base=[hb(t) for t in idx]; sig=[hb(t) for t in idx if fire[t]]
    if len(sig)<25: return None
    bh=np.mean([x[0] for x in base]); sh=np.mean([x[0] for x in sig])
    bbp=np.mean([x[1] for x in base]); sbp=np.mean([x[1] for x in sig])
    sd=np.std([x[1] for x in sig],ddof=1)
    return dict(n=len(sig),freq=len(sig)/len(idx),hit=sh,base=bh,lift=sh/bh,
                bp=sbp,bpb=bbp,t=(sbp-bbp)/(sd/np.sqrt(len(sig))))

print("Walk-forward регрессия r_CIS[t+1] ~ r_USD[t]+r_EUR[t]+r_CNY[t]+r_CIS[t]")
print("Обучение расширяющимся окном (первые 500 дней — только обучение), OOS 2020-01..2026-09")
print("Сигнал «переводить сейчас»: прогноз роста коридора в топ-10% прошлых прогнозов\n")
for c in CIS:
    rhat=walkforward(c)
    thr=np.full(N,np.nan)
    for t in range(MIN,N-1):
        past=rhat[MIN:t]; past=past[np.isfinite(past)]
        if len(past)>=100: thr[t]=np.quantile(past,0.90)
    fire=np.isfinite(rhat)&np.isfinite(thr)&(rhat>thr)
    print(f"--- {c} | сигналов {int(fire.sum())} | ср. прогноз при срабатывании "
          f"{np.nanmean(rhat[fire])*1e4:.0f} бп")
    print(f"{'h':>4}{'n':>6}{'частота':>9}{'hit':>8}{'база':>8}{'lift':>7}{'выгода бп':>11}"
          f"{'база бп':>9}{'t':>7}")
    for h in [1,3,5,10,20]:
        m=metrics(X[c],fire,h)
        if m: print(f"{h:>4}{m['n']:>6}{m['freq']:>9.1%}{m['hit']:>8.3f}{m['base']:>8.3f}"
                    f"{m['lift']:>7.2f}{m['bp']:>11.1f}{m['bpb']:>9.1f}{m['t']:>7.1f}")
    print()

print("="*78)
print("Проверка на всякий случай: реализованная доходность коридора на СЛЕДУЮЩИЙ день")
print("при срабатывании — сравниваем прогноз и факт (TJS)")
print("="*78)
rhat=walkforward("TJS")
ok=np.isfinite(rhat[:-1])
act=R["TJS"][1:][ok[:-1]] if False else None
mask=np.arange(N-1)[np.isfinite(rhat[:-1])]
pred=rhat[mask]; real=R["TJS"][mask+1]
print(f"корреляция прогноз/факт OOS: {np.corrcoef(pred,real)[0,1]:.3f}  (n={len(pred)})")
q=np.quantile(pred,[0,.2,.4,.6,.8,1.0])
for i in range(5):
    m=(pred>=q[i])&(pred<=q[i+1])
    print(f"квинтиль прогноза {i+1}: ср.прогноз {pred[m].mean()*1e4:7.1f} бп | "
          f"ср.факт {real[m].mean()*1e4:7.1f} бп | n={m.sum()}")
