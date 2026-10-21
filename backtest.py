from pathlib import Path
"""Out-of-sample backtest. Every optimised strategy faces identical constraints:
  long-only, w_i <= 10%, fully invested, one-rebalance turnover ||w - w_drift||_1 <= 0.5,
  proportional cost kappa=10bp charged inside the objective and on realised turnover.
Covariance: Ledoit-Wolf (2004) shrinkage on the trailing 120 months of excess returns, for every strategy.

Strategy families (per ML model m):
  plug-in            max mu'w - g w'Sw
  ridge(lam)         max mu'w - lam ||w||_2^2 - g w'Sw          (norm-constrained / ridge portfolio)
  dro_mean(eps)      max mu'w - eps ||w||_2 - g w'Sw            (v1 formulation: robust mean only)
  dro_mv(eps)        max mu'w - eps ||w||_2 - g (||S^1/2 w|| + eps ||w||_2)^2   (Blanchet-Chen-Zhou 2022)
  dro_mv radius rules: fixed eps=c ; vol eps_t = c*vol_t/mean(vol_<=t) ; entropy eps_t = c*H_t/mean(H_<=t)
Validated versions pick the grid value with the best trailing-60m net Sharpe of the grid strategies.
"""
import sys
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from data_io import load_all

OUT = str(Path(__file__).resolve().parent / "out")
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
KAPPA = 0.001
WMAX = 0.10
TAU = 0.5
COV_WIN = 120
VAL_WIN = 60
EPS_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
LAM_GRID = [3 * e for e in EPS_GRID]


class MVProblem:
    """Parametrised convex problem, re-solved each month by updating parameters."""

    def __init__(self, n, kind):
        self.kind = kind
        self.w = cp.Variable(n)
        self.mu = cp.Parameter(n)
        self.L = cp.Parameter((n, n))
        self.wp = cp.Parameter(n)
        self.pen = cp.Parameter(nonneg=True)
        w = self.w
        tc = KAPPA * cp.norm1(w - self.wp)
        if kind == "plugin":
            obj = self.mu @ w - GAMMA * cp.sum_squares(self.L.T @ w) - tc
        elif kind == "ridge":
            obj = self.mu @ w - self.pen * cp.sum_squares(w) - GAMMA * cp.sum_squares(self.L.T @ w) - tc
        elif kind == "dro_mean":
            obj = self.mu @ w - self.pen * cp.norm(w, 2) - GAMMA * cp.sum_squares(self.L.T @ w) - tc
        elif kind == "dro_mv":
            t = cp.Variable(nonneg=True)
            s = cp.Variable(nonneg=True)
            self.extra = [t >= cp.norm(w, 2), s >= cp.norm(self.L.T @ w, 2) + self.pen * t]
            obj = self.mu @ w - self.pen * t - GAMMA * cp.square(s) - tc
        elif kind == "gmv":
            obj = -cp.sum_squares(self.L.T @ w) - tc
        cons = [cp.sum(w) == 1, w >= 0, w <= WMAX, cp.norm1(w - self.wp) <= TAU]
        if kind == "dro_mv":
            cons += self.extra
        self.prob = cp.Problem(cp.Maximize(obj), cons)

    def solve(self, mu, L, wp, pen=0.0):
        self.mu.value = mu
        self.L.value = L
        self.wp.value = wp
        self.pen.value = pen
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000)
        w = np.clip(self.w.value, 0, None)
        return w / w.sum()


class CVaRProblem:
    def __init__(self, n, S, alpha=0.95):
        self.w = cp.Variable(n)
        self.R = cp.Parameter((S, n))
        self.wp = cp.Parameter(n)
        z = cp.Variable()
        u = cp.Variable(S, nonneg=True)
        loss = -self.R @ self.w
        obj = z + cp.sum(u) / ((1 - alpha) * S) + KAPPA * cp.norm1(self.w - self.wp)
        cons = [u >= loss - z, cp.sum(self.w) == 1, self.w >= 0, self.w <= WMAX,
                cp.norm1(self.w - self.wp) <= TAU]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve(self, R, wp):
        self.R.value = R
        self.wp.value = wp
        self.prob.solve(solver=cp.CLARABEL)
        w = np.clip(self.w.value, 0, None)
        return w / w.sum()


def drift(w, r):
    g = w * (1 + r)
    return g / g.sum()


class Book:
    """Tracks one strategy's weight path and returns."""

    def __init__(self, n):
        self.wd = np.ones(n) / n   # drifted weights carried into the next decision
        self.rows = []

    def step(self, date, w, r_next, rf_next):
        to = np.abs(w - self.wd).sum()
        gross = float(w @ r_next)
        self.rows.append((date, gross - rf_next, to, w.copy()))
        self.wd = drift(w, r_next)


def sharpe(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) if len(x) > 2 and x.std() > 0 else -np.inf


def run(model):
    ind, ex, ff = load_all()
    assets = list(ind.columns)
    n = len(assets)
    sig = pd.read_pickle(f"{OUT}/signals.pkl")
    mu_all = pd.read_pickle(f"{OUT}/mu_{model}.pkl") if model != "bench" else None
    dates = (mu_all.index if mu_all is not None else pd.read_pickle(f"{OUT}/mu_lasso.pkl").index)
    idx = ind.index

    # radius multipliers: signal / expanding mean of signal (information up to t only)
    H = sig["entropy"]; V = sig["vol12"]
    mH = H / H.expanding().mean(); mV = V / V.expanding().mean()

    books = {}
    probs = {}

    def P(kind):
        if kind not in probs:
            probs[kind] = MVProblem(n, kind)
        return probs[kind]

    cvar = CVaRProblem(n, COV_WIN)
    names_grid = []
    if model == "bench":
        names = ["1/N", "GMV", "MinCVaR", "HistMean-MV"]
    else:
        names = ["plugin"]
        for e in EPS_GRID:
            names_grid += [f"ridge|{e}", f"dro_mean|{e}", f"fixed|{e}", f"vol|{e}", f"ent|{e}"]
        names = names + names_grid + ["ridge|val", "dro_mean|val", "fixed|val", "vol|val", "ent|val"]
    for nm in names:
        books[nm] = Book(n)
    chosen = []

    for k, t in enumerate(dates):
        pos = idx.get_loc(t)
        win = ex.iloc[pos - COV_WIN + 1: pos + 1].values
        S = LedoitWolf().fit(win).covariance_
        L = np.linalg.cholesky(S + 1e-10 * np.eye(n))
        r_next = ind.iloc[pos + 1].values
        rf_next = float(ff["RF"].iloc[pos + 1])
        tn = idx[pos + 1]
        if model == "bench":
            books["1/N"].step(tn, np.ones(n) / n, r_next, rf_next)
            books["GMV"].step(tn, P("gmv").solve(np.zeros(n), L, books["GMV"].wd), r_next, rf_next)
            books["MinCVaR"].step(tn, cvar.solve(win, books["MinCVaR"].wd), r_next, rf_next)
            books["HistMean-MV"].step(tn, P("plugin").solve(win.mean(0), L, books["HistMean-MV"].wd), r_next, rf_next)
            continue
        mu = mu_all.loc[t].values
        books["plugin"].step(tn, P("plugin").solve(mu, L, books["plugin"].wd), r_next, rf_next)
        mult = {"fixed": 1.0, "vol": float(mV.loc[t]), "ent": float(mH.loc[t])}
        for e in EPS_GRID:
            b = books[f"ridge|{e}"]; b.step(tn, P("ridge").solve(mu, L, b.wd, 3 * e), r_next, rf_next)
            b = books[f"dro_mean|{e}"]; b.step(tn, P("dro_mean").solve(mu, L, b.wd, e), r_next, rf_next)
            for rule in ("fixed", "vol", "ent"):
                b = books[f"{rule}|{e}"]
                b.step(tn, P("dro_mv").solve(mu, L, b.wd, e * mult[rule]), r_next, rf_next)
        # validated strategies: choose grid value by trailing net Sharpe (realised up to t)
        row = {"date": tn}
        for fam, kind in (("ridge", "ridge"), ("dro_mean", "dro_mean"), ("fixed", "dro_mv"),
                          ("vol", "dro_mv"), ("ent", "dro_mv")):
            if k < VAL_WIN:
                e_star = EPS_GRID[len(EPS_GRID) // 2]
            else:
                scores = []
                for e in EPS_GRID:
                    rows = books[f"{fam}|{e}"].rows[-VAL_WIN - 1:-1]  # exclude month just booked (t+1)
                    net = [x[1] - KAPPA * x[2] for x in rows]
                    scores.append(sharpe(net))
                e_star = EPS_GRID[int(np.argmax(scores))]
            row[fam] = e_star
            pen = e_star * (3 if fam == "ridge" else 1) * (mult[fam] if fam in mult else 1.0)
            b = books[f"{fam}|val"]
            b.step(tn, P(kind).solve(mu, L, b.wd, pen), r_next, rf_next)
        chosen.append(row)
        if k % 60 == 0:
            print(model, t.date(), flush=True)

    res = {}
    for nm, b in books.items():
        d = pd.DataFrame([(x[0], x[1], x[2]) for x in b.rows], columns=["date", "ex", "to"]).set_index("date")
        W = pd.DataFrame([x[3] for x in b.rows], index=d.index, columns=assets)
        res[nm] = (d, W)
    pd.to_pickle({"res": res, "chosen": pd.DataFrame(chosen)}, f"{OUT}/bt_{model}_g{GAMMA:g}.pkl")
    print("done", model)


if __name__ == "__main__":
    run(sys.argv[1])
