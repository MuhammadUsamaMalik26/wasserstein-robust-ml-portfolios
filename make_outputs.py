from pathlib import Path
"""Builds every table (LaTeX) and figure in the paper directly from backtest outputs.
No number in the paper is typed by hand: each comes from out/*.pkl via this script."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from analysis import load, metrics, net
from data_io import load_all

OUT = str(Path(__file__).resolve().parent / "out")
PAP = str(Path(__file__).resolve().parent / "paper")
import os
os.makedirs(PAP, exist_ok=True)
MODELS = [("lasso", "Lasso"), ("rf", "Random Forest"), ("gb", "Gradient Boosting")]
FAMS = [("plugin", "Plug-in (no robustness)"), ("ridge|val", "Ridge penalty"),
        ("dro_mean|val", "W-DRO, mean only"), ("fixed|val", "W-DRO-MV, fixed radius"),
        ("vol|val", "W-DRO-MV, volatility radius"), ("ent|val", "W-DRO-MV, entropy radius")]
N = {}  # numbers quoted in text


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


def fp(x, d=1):
    return f"{100 * x:.{d}f}".replace("-", "$-$")


R = load("5")
M = {k: metrics(v[0]) for k, v in R.items() if not k.endswith("__chosen")}

# ---------------- Table 2: main performance
cols = ["Ann.ex.ret", "Ann.vol", "Sharpe(gross)", "Sharpe(net10)", "Sharpe(net50)", "MaxDD", "Turnover"]


def row(label, m):
    return (f"{label} & {fp(m['Ann.ex.ret'])} & {fp(m['Ann.vol'])} & {f(m['Sharpe(gross)'])} & "
            f"{f(m['Sharpe(net10)'])} & {f(m['Sharpe(net50)'])} & {fp(m['MaxDD'])} & {f(m['Turnover'])} \\\\")


L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r" & Excess & Vol. & \multicolumn{3}{c}{Sharpe ratio} & Max & Turn- \\",
     r"\cmidrule(lr){4-6}",
     r"Strategy & ret. (\%) & (\%) & Gross & Net 10bp & Net 50bp & DD (\%) & over \\", r"\midrule",
     r"\multicolumn{8}{l}{\textit{Panel A: Benchmarks without forecasts}} \\"]
for k, lab in (("1/N", "Equal weight (1/N)"), ("GMV", "Minimum variance (GMV)"), ("MinCVaR", "Minimum CVaR (95\\%)"),
               ("HistMean-MV", "Mean-variance, historical mean")):
    L.append(row(lab, M[k]))
for p, (m, name) in enumerate(MODELS):
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{8}}{{l}}{{\textit{{Panel {'BCD'[p]}: {name} forecasts}}}} \\")
    for k, lab in FAMS:
        L.append(row(lab, M[f"{m}:{k}"]))
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_main.tex", "w").write("\n".join(L))

# ---------------- Table 1: forecast quality
ind, ex, ff = load_all()
mech = pd.read_pickle(f"{OUT}/mechanism.pkl")
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r"Model & Forecast dispersion (\%) & OOS $R^2$ (\%) & Mean rank IC & $t$(IC) & Constant months \\", r"\midrule"]
for m, name in MODELS:
    mu = pd.read_pickle(f"{OUT}/mu_{m}.pkl").loc["1985-06":]
    idx = ex.index
    nxt = [idx[idx.get_loc(t) + 1] for t in mu.index]
    r = ex.loc[nxt].values; fc = mu.values
    r2 = 1 - ((r - fc) ** 2).sum() / (r ** 2).sum()
    d = mech[mech.model == m]["ic"]
    const = int((mu.std(axis=1) < 1e-12).sum())
    icm = d.mean(); ict = icm / d.std() * np.sqrt(d.notna().sum())
    N[f"ic_{m}"] = icm; N[f"r2_{m}"] = r2; N[f"const_{m}"] = const; N[f"disp_{m}"] = mu.std(axis=1).mean()
    L.append(f"{name} & {f(100 * mu.std(axis=1).mean())} & {f(100 * r2)} & {f(icm, 3)} & {f(ict)} & {const} of {len(mu)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_forecast.tex", "w").write("\n".join(L))

# ---------------- Table 3: bootstrap tests
T = pd.read_pickle(f"{OUT}/tests.pkl")
lab = {"plugin": "Plug-in", "fixed|val": "W-DRO-MV fixed", "ridge|val": "Ridge", "dro_mean|val": "W-DRO mean-only",
       "ent|val": "W-DRO-MV entropy", "vol|val": "W-DRO-MV volatility", "1/N": "1/N", "GMV": "GMV", "MinCVaR": "Min CVaR"}


def nm(x):
    if ":" in x:
        return lab[x.split(":")[1]]
    return lab[x]


L = [r"\begin{tabular}{llrrrr}", r"\toprule",
     r" & & \multicolumn{2}{c}{Net of 10bp} & \multicolumn{2}{c}{Net of 50bp} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
     r"Model & Comparison (A minus B) & $\Delta$SR & $p$-value & $\Delta$SR & $p$-value \\", r"\midrule"]
order = ["plugin~GMV", "plugin~1/N", "fixed|val~plugin", "dro_mean|val~ridge|val", "fixed|val~ridge|val",
         "ent|val~fixed|val", "ent|val~vol|val", "fixed|val~GMV"]
for m, name in MODELS:
    first = True
    for o in order:
        a, b = o.split("~")
        A = f"{m}:{a}"; B = b if b in ("GMV", "1/N") else f"{m}:{b}"
        s = T[(T.A == A) & (T.B == B)]
        r10 = s[s.cost == "net10"].iloc[0]; r50 = s[s.cost == "net50"].iloc[0]
        L.append(f"{name if first else ''} & {nm(A)} $-$ {nm(B)} & {f(r10.dSR)} & {f(r10.p)} & {f(r50.dSR)} & {f(r50.p)} \\\\")
        first = False
    L.append(r"\midrule")
for A, B in (("GMV", "1/N"), ("MinCVaR", "1/N")):
    s = T[(T.A == A) & (T.B == B)]
    r10 = s[s.cost == "net10"].iloc[0]; r50 = s[s.cost == "net50"].iloc[0]
    L.append(f"None & {nm(A)} $-$ {nm(B)} & {f(r10.dSR)} & {f(r10.p)} & {f(r50.dSR)} & {f(r50.p)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_tests.tex", "w").write("\n".join(L))
N["tests"] = T.to_dict("records")

# ---------------- Table 4: radius grid (fixed-radius DRO-MV vs ridge vs mean-only, net Sharpe & turnover)
EPS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r" & \multicolumn{8}{c}{Radius parameter $c$ (ridge: $\lambda = 3c$)} \\", r"\cmidrule(lr){2-9}",
     " & " + " & ".join(f"{e:g}" for e in EPS) + r" \\", r"\midrule"]
for m, name in MODELS:
    L.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{name}}}}} \\")
    for fam, fl in (("ridge", "Ridge"), ("dro_mean", "W-DRO mean-only"), ("fixed", "W-DRO-MV fixed"),
                    ("vol", "W-DRO-MV volatility"), ("ent", "W-DRO-MV entropy")):
        L.append(f"\\quad {fl}: net SR & " + " & ".join(f(M[f'{m}:{fam}|{e}']['Sharpe(net10)']) for e in EPS) + r" \\")
    for fam, fl in (("ridge", "Ridge"), ("fixed", "W-DRO-MV fixed"), ("ent", "W-DRO-MV entropy")):
        L.append(f"\\quad {fl}: turnover & " + " & ".join(f(M[f'{m}:{fam}|{e}']['Turnover']) for e in EPS) + r" \\")
    L.append(f"\\quad Plug-in: net SR {f(M[f'{m}:plugin']['Sharpe(net10)'])}, turnover {f(M[f'{m}:plugin']['Turnover'])} & \\multicolumn{{8}}{{l}}{{}} \\\\")
    if m != "gb":
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_grid.tex", "w").write("\n".join(L))
for m, _ in MODELS:
    for fam in ("ridge", "dro_mean", "fixed", "vol", "ent"):
        v = [M[f"{m}:{fam}|{e}"]["Sharpe(net10)"] for e in EPS]
        N[f"grid_sr_range_{m}_{fam}"] = (min(v), max(v))
    N[f"to_plugin_{m}"] = M[f"{m}:plugin"]["Turnover"]
    N[f"to_fixed02_{m}"] = M[f"{m}:fixed|0.2"]["Turnover"]
    N[f"to_ridge02_{m}"] = M[f"{m}:ridge|0.2"]["Turnover"]

# ---------------- Table 5: mechanism regressions
O = pd.read_pickle(f"{OUT}/mechanism_reg.pkl")
L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
     r"Model & Dependent variable & $\beta_H$ & $t(\beta_H)$ & $\beta_{\sigma}$ & $t(\beta_{\sigma})$ & Months \\", r"\midrule"]
ylab = {"ic": "Rank IC", "ls": "Long--short return", "mae": "Mean abs. error"}
for m, name in MODELS:
    first = True
    for y in ["ic", "ls", "mae"]:
        for spec in ["H", "H+vol"]:
            s = O[(O.model == m) & (O.y == y) & (O.spec == spec)].iloc[0]
            bv = "" if spec == "H" else f(s.bV, 4)
            tv = "" if spec == "H" else f(s.tV)
            L.append(f"{name if first else ''} & {ylab[y]} & {f(s.bH, 4)} & {f(s.tH)} & {bv} & {tv} & {s.n} \\\\")
            first = False
    if m != "gb":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_mech.tex", "w").write("\n".join(L))
D = mech.copy()
D["Hq"] = D.groupby("model")["H"].transform(lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"]))
g = D.groupby(["model", "Hq"], observed=True)["ic"].mean()
N["ic_terciles"] = {f"{a}_{b}": v for (a, b), v in g.items()}
N["corr_H_vol"] = float(mech[mech.model == "rf"][["H", "vol"]].corr().iloc[0, 1])

# ---------------- Table 6: regime-conditional net Sharpe
RT = pd.read_pickle(f"{OUT}/regime_table.pkl")
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r" & \multicolumn{3}{c}{HMM entropy tercile} & \multicolumn{3}{c}{Volatility tercile} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Strategy & Low & Mid & High & Low & Mid & High \\", r"\midrule"]
labs = {"1/N": "1/N", "GMV": "GMV"}
for k in RT.index:
    if ":" in k:
        m, fam = k.split(":")
        lb = dict(MODELS)[m] + ", " + {"plugin": "plug-in", "ridge|val": "ridge", "fixed|val": "fixed radius", "vol|val": "volatility radius",
                                        "ent|val": "entropy radius"}[fam]
    else:
        lb = labs[k]
    L.append(lb + " & " + " & ".join(f(x) for x in RT.loc[k].values) + r" \\")
    if k in ("GMV", "lasso:ent|val", "rf:ent|val"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_regime.tex", "w").write("\n".join(L))

# ---------------- Table 7: robustness (gamma, subperiods)
rob = {}
for g in ["2", "5", "10"]:
    try:
        Rg = load(g)
    except FileNotFoundError:
        continue
    for k in ["1/N", "GMV"] + [f"{m}:{x}" for m, _ in MODELS for x in ["plugin", "ridge|val", "fixed|val", "vol|val", "ent|val"]]:
        mm = metrics(Rg[k][0])
        rob[(g, k)] = (mm["Sharpe(net10)"], mm["Sharpe(net50)"], mm["Turnover"])
S = pd.read_pickle(f"{OUT}/subperiod_table.pkl")
gs = sorted({g for g, _ in rob}, key=float)
L = [r"\begin{tabular}{l" + "r" * (2 * len(gs) + 4) + "}", r"\toprule",
     " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\gamma={g}$}}" for g in gs) +
     r" & \multicolumn{2}{c}{1985--2005} & \multicolumn{2}{c}{2006--2026} \\",
     "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(gs) + 2)),
     "Strategy & " + " & ".join(["SR$_{10}$ & TO"] * (len(gs) + 2)) + r" \\", r"\midrule"]
for k in ["1/N", "GMV"] + [f"{m}:{x}" for m, _ in MODELS for x in ["plugin", "ridge|val", "fixed|val", "vol|val", "ent|val"]]:
    if ":" in k:
        m, fam = k.split(":")
        lb = {"lasso": "Lasso", "rf": "RF", "gb": "GB"}[m] + ", " + {"plugin": "plug-in", "ridge|val": "ridge", "fixed|val": "fixed",
                                                                  "vol|val": "volatility", "ent|val": "entropy"}[fam]
    else:
        lb = k
    cells = []
    for g in gs:
        a = rob[(g, k)]; cells += [f(a[0]), f(a[2])]
    for per in ["1985-2005", "2006-2026"]:
        cells += [f(S.loc[k, ("SR_net10", per)]) if k in S.index else "", f(S.loc[k, ("TO", per)]) if k in S.index else ""]
    L.append(lb + " & " + " & ".join(cells) + r" \\")
    if k in ("GMV", "lasso:ent|val", "rf:ent|val"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAP}/tab_robust.tex", "w").write("\n".join(L))
N["robust"] = {f"{g}|{k}": v for (g, k), v in rob.items()}

# ---------------- chosen radius stats
for m, _ in MODELS:
    c = R[f"{m}:__chosen"]
    N[f"chosen_{m}"] = {fam: c[fam].value_counts(normalize=True).to_dict() for fam in c.columns}

# ---------------- Figures
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "violet": "#4a3aa7", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": C["ink2"], "axes.linewidth": 0.6,
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.labelcolor": C["ink"],
                     "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: cumulative net wealth
fig, ax = plt.subplots(figsize=(6.5, 3.4))
series = [("1/N", "1/N", C["ink2"], "-"), ("GMV", "GMV", C["ink"], "-"), ("rf:plugin", "RF plug-in", C["orange"], "-"),
          ("rf:fixed|val", "RF W-DRO-MV fixed", C["blue"], "-"), ("rf:ent|val", "RF W-DRO-MV entropy", C["aqua"], "--")]
for k, lb, col, ls in series:
    w = (1 + net(R[k][0], 0.001)).cumprod()
    ax.plot(w.index, w.values, color=col, lw=1.4, ls=ls, label=lb)
ax.set_yscale("log")
ax.set_ylabel("Growth of \\$1 (log scale)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}"))
ax.grid(axis="y", color=C["grid"], lw=0.5)
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(f"{PAP}/fig_wealth.pdf"); fig.savefig(f"{PAP}/fig_wealth.png", dpi=200); plt.close(fig)

# Figure 2: radius grid small multiples
fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4), sharex=True)
fams = [("ridge", "Ridge", C["ink2"], "o"), ("dro_mean", "W-DRO mean-only", C["magenta"], "s"),
        ("fixed", "W-DRO-MV fixed", C["blue"], "D"), ("vol", "W-DRO-MV volatility", C["yellow"], "v"),
        ("ent", "W-DRO-MV entropy", C["aqua"], "^")]
for j, (m, name) in enumerate(MODELS):
    for fam, lb, col, mk in fams:
        axes[0, j].plot(EPS, [M[f"{m}:{fam}|{e}"]["Sharpe(net10)"] for e in EPS], color=col, lw=1.2, marker=mk, ms=3.5, label=lb)
        axes[1, j].plot(EPS, [M[f"{m}:{fam}|{e}"]["Turnover"] for e in EPS], color=col, lw=1.2, marker=mk, ms=3.5, label=lb)
    axes[0, j].axhline(M[f"{m}:plugin"]["Sharpe(net10)"], color=C["orange"], lw=1, ls=":", label="Plug-in")
    axes[0, j].axhline(M["GMV"]["Sharpe(net10)"], color=C["ink"], lw=1, ls="--", label="GMV")
    axes[1, j].axhline(M[f"{m}:plugin"]["Turnover"], color=C["orange"], lw=1, ls=":")
    axes[0, j].set_title(name, fontsize=9)
    for i in range(2):
        axes[i, j].set_xscale("log"); axes[i, j].grid(axis="y", color=C["grid"], lw=0.5)
    axes[1, j].set_xlabel("Radius parameter $c$")
axes[0, 0].set_ylabel("Net Sharpe ratio (10bp)")
axes[1, 0].set_ylabel("Mean monthly turnover")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{PAP}/fig_grid.pdf"); fig.savefig(f"{PAP}/fig_grid.png", dpi=200); plt.close(fig)

# Figure 3: entropy over time
sig = pd.read_pickle(f"{OUT}/signals.pkl").loc["1980-06":]
fig, ax = plt.subplots(figsize=(6.5, 2.4))
ax.plot(sig.index, sig["entropy"], color=C["blue"], lw=0.8)
ax.set_ylabel("Normalised entropy $H_t$")
ax.set_ylim(0, 1.02); ax.grid(axis="y", color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(f"{PAP}/fig_entropy.pdf"); fig.savefig(f"{PAP}/fig_entropy.png", dpi=200); plt.close(fig)

# key numbers for the text
for k in ["1/N", "GMV", "MinCVaR", "HistMean-MV"] + [f"{m}:{x}" for m, _ in MODELS for x, _ in FAMS]:
    N["M|" + k] = M[k]
N["eval"] = (str(R["1/N"][0].index[0].date()), str(R["1/N"][0].index[-1].date()), len(R["1/N"][0]))
N["H_mean"] = float(pd.read_pickle(f"{OUT}/signals.pkl")["entropy"].loc["1980-06":].mean())
json.dump(N, open(f"{PAP}/numbers.json", "w"), indent=1, default=str)
print("ok")
