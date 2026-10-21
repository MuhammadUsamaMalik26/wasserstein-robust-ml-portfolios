from pathlib import Path
"""Mechanism test: is forecast quality lower when regime uncertainty (HMM entropy) is high?
Newey-West (6 lags) regressions of monthly forecast-quality measures on entropy, controlling for volatility.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_io import load_all
from analysis import load, EVAL_START

OUT = str(Path(__file__).resolve().parent / "out")
ind, ex, ff = load_all()
sig = pd.read_pickle(f"{OUT}/signals.pkl")
idx = ex.index
rows = []
for m in ["lasso", "rf", "gb"]:
    mu = pd.read_pickle(f"{OUT}/mu_{m}.pkl")
    for t in mu.index:
        r = ex.iloc[idx.get_loc(t) + 1]
        f = mu.loc[t]
        ic = f.rank().corr(r.rank()) if f.std() > 0 else np.nan
        mae = (r - f).abs().mean()
        # "decision" error: return of forecast-sorted long-short (top vs bottom 10 industries)
        top = f.nlargest(10).index; bot = f.nsmallest(10).index
        ls = r[top].mean() - r[bot].mean() if f.std() > 0 else np.nan
        rows.append(dict(model=m, date=t, ic=ic, mae=mae, ls=ls,
                         H=sig.loc[t, "entropy"], vol=sig.loc[t, "vol12"], phigh=sig.loc[t, "p_high"],
                         cs_disp=r.std()))
D = pd.DataFrame(rows)
D = D[D.date >= pd.Timestamp(EVAL_START) - pd.DateOffset(months=1)]
D.to_pickle(f"{OUT}/mechanism.pkl")

out = []
for m in ["lasso", "rf", "gb"]:
    d = D[D.model == m].dropna()
    for y in ["ic", "ls", "mae"]:
        for spec in (["H"], ["H", "vol"]):
            X = sm.add_constant(d[spec].assign(**{c: (d[c] - d[c].mean()) / d[c].std() for c in spec}))
            fit = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            out.append(dict(model=m, y=y, spec="+".join(spec), bH=fit.params["H"], tH=fit.tvalues["H"],
                            bV=fit.params.get("vol", np.nan), tV=fit.tvalues.get("vol", np.nan), n=int(fit.nobs)))
    # tercile means
O = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(O.round(4).to_string())
D["Hq"] = D.groupby("model")["H"].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]))
print(D.groupby(["model", "Hq"])[["ic", "ls"]].mean().round(4).to_string())
print("corr(H, vol) =", round(D[D.model == "rf"][["H", "vol"]].corr().iloc[0, 1], 3))
O.to_pickle(f"{OUT}/mechanism_reg.pkl")
