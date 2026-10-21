from pathlib import Path
import sys
import numpy as np
import pandas as pd

OUT = str(Path(__file__).resolve().parent / "out")
EVAL_START = "1985-07"
G = sys.argv[1] if len(sys.argv) > 1 else "5"


def load(g=G):
    R = {}
    for m in ["bench", "lasso", "rf", "gb"]:
        d = pd.read_pickle(f"{OUT}/bt_{m}_g{g}.pkl")
        for k, (df, W) in d["res"].items():
            key = k if m == "bench" else f"{m}:{k}"
            R[key] = (df.loc[EVAL_START:], W.loc[EVAL_START:])
        if m != "bench":
            R[f"{m}:__chosen"] = d["chosen"].set_index("date").loc[EVAL_START:]
    return R


def net(df, kappa):
    return df["ex"] - kappa * df["to"]


def metrics(df):
    n10 = net(df, 0.001); n50 = net(df, 0.005)
    wealth = (1 + n10).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    q = n10.quantile(0.05)
    cvar = n10[n10 <= q].mean()
    return {
        "Ann.ex.ret": df["ex"].mean() * 12,
        "Ann.vol": df["ex"].std() * np.sqrt(12),
        "Sharpe(gross)": df["ex"].mean() / df["ex"].std() * np.sqrt(12),
        "Sharpe(net10)": n10.mean() / n10.std() * np.sqrt(12),
        "Sharpe(net50)": n50.mean() / n50.std() * np.sqrt(12),
        "MaxDD": dd,
        "CVaR95(m)": cvar,
        "Turnover": df["to"].mean(),
    }


def boot_sharpe_diff(a, b, B=5000, block=6, seed=1):
    """Stationary (Politis-Romano) block bootstrap for annualised Sharpe(a)-Sharpe(b); two-sided p-value
    via the studentisation-free percentile-of-centred-statistic method (Ledoit & Wolf 2008 spirit)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b); T = len(a)
    sh = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(12)
    d0 = sh(a) - sh(b)
    ds = np.empty(B)
    p = 1.0 / block
    for i in range(B):
        idx = np.empty(T, dtype=int)
        j = rng.integers(T)
        for t in range(T):
            if t == 0 or rng.random() < p:
                j = rng.integers(T)
            else:
                j = (j + 1) % T
            idx[t] = j
        ds[i] = sh(a[idx]) - sh(b[idx])
    se = ds.std(ddof=1)
    pval = np.mean(np.abs(ds - d0) >= abs(d0))
    return d0, se, pval, np.percentile(ds, [2.5, 97.5])


if __name__ == "__main__":
    R = load()
    rows = {k: metrics(v[0]) for k, v in R.items() if not k.endswith("__chosen")}
    T = pd.DataFrame(rows).T
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    show = [k for k in T.index if ("|" not in k) or k.endswith("|val")]
    print("Eval months:", len(R["1/N"][0]), R["1/N"][0].index[0].date(), R["1/N"][0].index[-1].date())
    print(T.loc[show].round(4).to_string())
    T.to_pickle(f"{OUT}/metrics_g{G}.pkl")
