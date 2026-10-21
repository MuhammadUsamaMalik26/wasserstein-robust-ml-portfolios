from pathlib import Path
import numpy as np
import pandas as pd
from analysis import load, net, metrics

OUT = str(Path(__file__).resolve().parent / "out")
R = load("5")
sig = pd.read_pickle(f"{OUT}/signals.pkl")
# signal known at decision date t applies to return month t+1
H = sig["entropy"].shift(1); V = sig["vol12"].shift(1)
idx = R["1/N"][0].index
Hq = pd.qcut(H.loc[idx], 3, labels=["Low H", "Mid H", "High H"])
Vq = pd.qcut(V.loc[idx], 3, labels=["Low vol", "Mid vol", "High vol"])
sr = lambda x: x.mean() / x.std() * np.sqrt(12)
rows = []
names = ["1/N", "GMV"] + [f"{m}:{k}" for m in ["lasso", "rf", "gb"] for k in ["plugin", "ridge|val", "fixed|val", "vol|val", "ent|val"]]
for nm in names:
    x = net(R[nm][0], 0.001)
    row = {"strategy": nm}
    for q in Hq.cat.categories:
        row[q] = sr(x[Hq == q])
    for q in Vq.cat.categories:
        row[q] = sr(x[Vq == q])
    rows.append(row)
T = pd.DataFrame(rows).set_index("strategy")
pd.set_option("display.width", 220)
print(T.round(3).to_string())
T.to_pickle(f"{OUT}/regime_table.pkl")

# subperiods
sub = []
for a, b in (("1985-07", "2005-12"), ("2006-01", "2026-06")):
    for nm in names:
        df = R[nm][0].loc[a:b]
        mm = metrics(df)
        sub.append({"period": f"{a[:4]}-{b[:4]}", "strategy": nm, "SR_net10": mm["Sharpe(net10)"], "SR_net50": mm["Sharpe(net50)"], "TO": mm["Turnover"]})
S = pd.DataFrame(sub).pivot(index="strategy", columns="period")
print(S.round(3).to_string())
S.to_pickle(f"{OUT}/subperiod_table.pkl")

# average realised radius and radius variability for the adaptive rules (validated), and weight concentration
c = {}
for m in ["lasso", "rf", "gb"]:
    for k in ["plugin", "ridge|val", "fixed|val", "vol|val", "ent|val"]:
        W = R[f"{m}:{k}"][1]
        c[f"{m}:{k}"] = {"N_eff": (1 / (W ** 2).sum(1)).mean(), "max_w": W.max(1).mean()}
for k in ["1/N", "GMV", "MinCVaR"]:
    W = R[k][1]; c[k] = {"N_eff": (1 / (W ** 2).sum(1)).mean(), "max_w": W.max(1).mean()}
C = pd.DataFrame(c).T
print(C.round(2).to_string())
C.to_pickle(f"{OUT}/concentration.pkl")

# Lasso degenerate months
mu = pd.read_pickle(f"{OUT}/mu_lasso.pkl").loc["1985-06":]
print("Lasso months with constant cross-sectional forecast:", int((mu.std(1) < 1e-12).sum()), "of", len(mu))
