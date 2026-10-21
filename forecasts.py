from pathlib import Path
"""ML return forecasts and regime/volatility signals. Strictly out-of-sample.

Timing convention: row t of every output uses information available at the END of month t
and is a forecast / signal for month t+1.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from hmmlearn.hmm import GaussianHMM
from data_io import load_all

warnings.filterwarnings("ignore")
OUT = str(Path(__file__).resolve().parent / "out")
TRAIN_WIN = 120      # months of feature/target pairs used for training
REFIT_EVERY = 12     # refit ML models annually (Gu, Kelly & Xiu 2020 practice)
FIRST_FCST = "1980-06"  # first forecast made end of June 1980 for July 1980


def build_features(ex):
    """Per-industry features at end of month t (all use returns up to and including t)."""
    f = {
        "lag1": ex,
        "mom3": ex.rolling(3).mean(),
        "mom6": ex.rolling(6).mean(),
        "mom12": ex.rolling(12).mean(),
        "vol12": ex.rolling(12).std(),
    }
    return f


def panel(feats, dates, assets):
    X = np.stack([feats[k].loc[dates, assets].values for k in feats], axis=-1)  # T x N x K
    return X


def make_model(name, seed=0):
    if name == "lasso":
        return LassoCV(cv=5, n_alphas=50, max_iter=5000)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, max_depth=5, max_features=3,
                                     min_samples_leaf=20, random_state=seed, n_jobs=2)
    if name == "gb":
        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                            subsample=0.8, random_state=seed, n_jobs=2, verbosity=0)
    raise ValueError(name)


def ml_forecasts(name, ex):
    feats = build_features(ex)
    assets = list(ex.columns)
    dates = ex.index
    valid = feats["mom12"].notna().all(axis=1) & feats["vol12"].notna().all(axis=1)
    fc_dates = dates[dates >= FIRST_FCST][:-1]  # need t+1 to exist for evaluation
    mu = pd.DataFrame(np.nan, index=fc_dates, columns=assets)
    model, scaler = None, None
    for k, t in enumerate(fc_dates):
        if k % REFIT_EVERY == 0:
            pos = dates.get_loc(t)
            # training pairs: features at s, target at s+1, with s+1 <= t
            feat_dates = [d for d in dates[max(0, pos - TRAIN_WIN):pos] if valid.loc[d]]
            tgt_dates = [dates[dates.get_loc(d) + 1] for d in feat_dates]
            X = panel(feats, feat_dates, assets).reshape(-1, len(feats))
            y = ex.loc[tgt_dates, assets].values.reshape(-1)
            scaler = StandardScaler().fit(X)
            model = make_model(name)
            model.fit(scaler.transform(X), y)
        Xt = panel(feats, [t], assets).reshape(-1, len(feats))
        mu.loc[t] = model.predict(scaler.transform(Xt))
    return mu


def hmm_entropy(mkt, min_obs=120):
    """Filtered 2-state Gaussian HMM probabilities on market excess returns.

    Parameters are re-estimated each June on an expanding window ending at t; the probability at t
    is the last row of the forward-backward posterior over data up to t, which equals the filtered
    (forward-only) probability because no observation after t is used.
    Returns normalised entropy in [0,1] and P(high-variance state).
    """
    x = mkt.values.reshape(-1, 1)
    idx = mkt.index
    H = pd.Series(np.nan, index=idx)
    Phi = pd.Series(np.nan, index=idx)
    model = None
    for i in range(min_obs, len(idx)):
        if model is None or idx[i].month == 6:
            best = None
            for seed in range(5):
                m = GaussianHMM(n_components=2, covariance_type="full", n_iter=500,
                                random_state=seed, tol=1e-6)
                m.fit(x[: i + 1])
                sc = m.score(x[: i + 1])
                if best is None or sc > best[0]:
                    best = (sc, m)
            model = best[1]
            hi = int(np.argmax(model.covars_.reshape(-1)))
        p = model.predict_proba(x[: i + 1])[-1]
        p = np.clip(p, 1e-12, 1)
        H.iloc[i] = float(-(p * np.log(p)).sum() / np.log(2))
        Phi.iloc[i] = float(p[hi])
    return H, Phi


if __name__ == "__main__":
    ind, ex, ff = load_all()
    what = sys.argv[1]
    if what in ("lasso", "rf", "gb"):
        mu = ml_forecasts(what, ex)
        mu.to_pickle(f"{OUT}/mu_{what}.pkl")
        print(what, mu.shape, mu.index[0].date(), mu.index[-1].date())
    elif what == "signals":
        from data_io import load_factors
        f = load_factors()  # market from 1963-07 for a longer HMM history
        mkt = f["Mkt-RF"]
        H, Phi = hmm_entropy(mkt)
        vol = mkt.rolling(12).std()
        sig = pd.DataFrame({"entropy": H, "p_high": Phi, "vol12": vol, "mkt": mkt})
        sig.to_pickle(f"{OUT}/signals.pkl")
        print(sig.dropna().describe())
