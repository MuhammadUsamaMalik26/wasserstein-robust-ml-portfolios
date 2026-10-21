from pathlib import Path
import sys
import numpy as np
import pandas as pd
from analysis import load, net, boot_sharpe_diff

OUT = str(Path(__file__).resolve().parent / "out")
R = load("5")
pairs = []
for m in ["lasso", "rf", "gb"]:
    p = lambda k: f"{m}:{k}"
    pairs += [(p("plugin"), "1/N"), (p("plugin"), "GMV"), (p("fixed|val"), p("plugin")),
              (p("fixed|val"), p("ridge|val")), (p("dro_mean|val"), p("ridge|val")),
              (p("ent|val"), p("fixed|val")), (p("ent|val"), p("vol|val")), (p("fixed|val"), "GMV")]
pairs += [("GMV", "1/N"), ("MinCVaR", "1/N")]
rows = []
for a, b in pairs:
    for kap, lab in ((0.001, "net10"), (0.005, "net50")):
        x = net(R[a][0], kap).values; y = net(R[b][0], kap).values
        d, se, pv, ci = boot_sharpe_diff(x, y, B=3000)
        rows.append(dict(A=a, B=b, cost=lab, dSR=d, se=se, p=pv, lo=ci[0], hi=ci[1]))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.round(3).to_string())
T.to_pickle(f"{OUT}/tests.pkl")
