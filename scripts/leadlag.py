#!/usr/bin/env python3
import csv, datetime as dt, numpy as np

P = "/Users/kirillnemkin/Хакатон/data/cbr_rates_daily_2018-2026.csv"
rows=list(csv.DictReader(open(P)))
codes=[c for c in rows[0] if c!="date"]
dates=np.array([dt.date.fromisoformat(r["date"]) for r in rows])
X={c:np.array([float(r[c]) for r in rows]) for c in codes}
R={c:np.diff(np.log(X[c])) for c in codes}          # r[i] относится к переходу dates[i] -> dates[i+1]
rd=dates[1:]
n=len(rd)
MAJ=["USD","EUR","CNY"]; CIS=["TJS","UZS","KGS","KZT","AMD"]

def ols(y,Xm):
    Xd=np.column_stack([np.ones(len(y))]+Xm)
    b,*_=np.linalg.lstsq(Xd,y,rcond=None)
    resid=y-Xd@b; dof=len(y)-Xd.shape[1]
    s2=resid@resid/dof
    cov=s2*np.linalg.pinv(Xd.T@Xd)
    se=np.sqrt(np.diag(cov))
    r2=1-(resid@resid)/((y-y.mean())@(y-y.mean()))
    return b,b/se,r2

print("Период:", rd[0], "..", rd[-1], f"| {n} переходов между датами публикации ЦБ\n")

print("="*78)
print("1. ОДНОВРЕМЕННАЯ корреляция дневных лог-доходностей (лаг 0)")
print("="*78)
print(f"{'':6}"+"".join(f"{m:>10}" for m in MAJ))
for c in CIS:
    print(f"{c:6}"+"".join(f"{np.corrcoef(R[c],R[m])[0,1]:10.3f}" for m in MAJ))

print("\n"+"="*78)
print("2. ЗАПАЗДЫВАЮЩАЯ корреляция: corr( r_CIS[t+1] , r_major[t] )  — есть ли лид-лаг")
print("="*78)
print(f"{'':6}"+"".join(f"{m:>10}" for m in MAJ)+f"{'сам себя':>12}")
for c in CIS:
    line=f"{c:6}"+"".join(f"{np.corrcoef(R[c][1:],R[m][:-1])[0,1]:10.3f}" for m in MAJ)
    line+=f"{np.corrcoef(R[c][1:],R[c][:-1])[0,1]:12.3f}"
    print(line)

print("\n"+"="*78)
print("3. Регрессия r_CIS[t+1] = a + b1*r_USD[t] + b2*r_CIS[t]   (t-стат в скобках)")
print("="*78)
print(f"{'вал':6}{'b1 (USD[t])':>18}{'b2 (сам[t])':>18}{'R^2':>8}")
for c in CIS:
    y=R[c][1:]; b,t,r2=ols(y,[R["USD"][:-1],R[c][:-1]])
    print(f"{c:6}{b[1]:>10.3f} ({t[1]:5.1f}){b[2]:>10.3f} ({t[2]:5.1f}){r2:8.3f}")

print("\n"+"="*78)
print("4. Направление: доля дней, когда знак r_CIS[t+1] совпал со знаком r_USD[t]")
print("   (база = доля дней роста CIS, т.е. точность 'всегда говорим рост')")
print("="*78)
print(f"{'вал':6}{'hit sign':>10}{'база(рост)':>12}{'lift':>8}")
for c in CIS:
    a=np.sign(R[c][1:]); b_=np.sign(R["USD"][:-1])
    m=(a!=0)&(b_!=0)
    hit=(a[m]==b_[m]).mean(); base=max((a[m]>0).mean(),(a[m]<0).mean())
    print(f"{c:6}{hit:10.3f}{base:12.3f}{hit/base:8.3f}")

print("\n"+"="*78)
print("5. Почему так: кросс-курс CIS/USD = (CIS/RUB)/(USD/RUB)")
print("   Если ЦБ считает CIS через USD, то CIS/RUB ~ USD/RUB * почти константа")
print("="*78)
print(f"{'вал':6}{'sd r_CIS,%':>12}{'sd кросса,%':>13}{'доля дисп.':>12}{'AC(1) кросса':>14}")
for c in CIS:
    cross=np.log(X[c]/X["USD"]); rc=np.diff(cross)
    print(f"{c:6}{R[c].std()*100:12.3f}{rc.std()*100:13.3f}"
          f"{1-rc.var()/R[c].var():12.3f}{np.corrcoef(rc[1:],rc[:-1])[0,1]:14.3f}")

print("\n"+"="*78)
print("6. Устойчивость по годам: corr( r_CIS[t+1], r_USD[t] )")
print("="*78)
yrs=sorted({d.year for d in rd})
print(f"{'вал':6}"+"".join(f"{y:>8}" for y in yrs))
for c in CIS:
    line=f"{c:6}"
    for y in yrs:
        m=np.array([d.year==y for d in rd[1:]])
        line+=f"{np.corrcoef(R[c][1:][m],R['USD'][:-1][m])[0,1]:8.2f}" if m.sum()>30 else f"{'-':>8}"
    print(line)
