from pathlib import Path
"""Load Ken French 49 industry portfolios (value-weighted, monthly) and the FF 5-factor file (for RF and Mkt-RF).

Source files: Kenneth R. French Data Library, vintage built from the 202606 CRSP database.
All returns converted from percent to decimals. Missing codes (-99.99, -999) set to NaN.
"""
import io
import numpy as np
import pandas as pd

DATA = str(Path(__file__).resolve().parent / "data")


def _block(lines, start_marker, stop_pred):
    s = [i for i, l in enumerate(lines) if start_marker in l][0]
    out = []
    for l in lines[s + 1:]:
        if stop_pred(l):
            break
        out.append(l)
    return out


def load_industries():
    L = open(f"{DATA}/49_Industry_Portfolios.csv").read().splitlines()
    blk = _block(L, "Average Value Weighted Returns -- Monthly", lambda l: l.strip() == "" )
    df = pd.read_csv(io.StringIO("\n".join(blk)), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m")
    df = df.replace([-99.99, -999], np.nan) / 100.0
    return df


def load_factors():
    L = open(f"{DATA}/F-F_Research_Data_5_Factors_2x3.csv").read().splitlines()
    s = [i for i, l in enumerate(L) if l.startswith(",Mkt-RF")][0]
    rows = [L[s]]
    for l in L[s + 1:]:
        if l.strip() == "" or "Annual" in l:
            break
        rows.append(l)
    df = pd.read_csv(io.StringIO("\n".join(rows)), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m")
    return df / 100.0


def load_all(start="1969-07"):
    ind = load_industries().loc[start:]
    ff = load_factors()
    ind = ind.loc[ind.index.isin(ff.index)]
    ff = ff.loc[ind.index]
    assert ind.isna().sum().sum() == 0
    excess = ind.sub(ff["RF"], axis=0)
    return ind, excess, ff


if __name__ == "__main__":
    ind, ex, ff = load_all()
    print(ind.shape, ind.index[0].date(), ind.index[-1].date())
    print(ff.tail(3))
