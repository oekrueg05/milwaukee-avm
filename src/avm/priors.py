"""Time detrending and prior-sales features (Stage 6). LEAKAGE-CRITICAL.

Time effect: from the fold's training-years-only hedonic (Stage 4 formula),
  te(t, moy) = log-price contribution of (trend at t, month moy) minus that
  of (trend at the reference point, neutral month). The reference is the Dec
  (Y-1) trend level, so a detrended price lp - te(t, moy) is "this sale at the
  Dec Y-1 trend level, neutral season".

Cutoffs: a TRAINING row may use sales before the start of its own sale month;
a SCORED row (validation year Y or test 2025) only sales before Jan 1 Y,
because a Jan 1 valuation cannot see that year's sales.

kNN features (k = K nearest clean SF/duplex sales in the 24 months before the
cutoff, same parcel excluded): median detrended log price per sq ft, median
neighbor distance, count in the pool; and the parcel's neighborhood median
detrended log price per sq ft over the same window (>= 5 sales).

Own prior sale: the parcel's most recent clean sale more than 12 months
before the target sale and before the cutoff; its detrended log price,
months from it to the row's valuation point (own sale date for training,
Jan 1 Y for scored rows), and has_prior_sale.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from avm.hedonic import fit as fit_hedonic

K = 10
WINDOW_MONTHS = 24
OWN_MIN_MONTHS = 12
NBHD_MIN = 5


class TimeEffect:
    """Additive log time effect from a fitted hedonic, relative to t_ref with
    a neutral season (mean over the 12 months)."""

    def __init__(self, train_frame: pd.DataFrame, t_ref: float):
        self.model = fit_hedonic(train_frame)
        self.ref = train_frame.iloc[[0]]
        self.t_ref = t_ref
        base = np.mean([self._pred(np.array([t_ref]), np.array([m]))[0] for m in range(1, 13)])
        self.base = base

    def _pred(self, t: np.ndarray, moy: np.ndarray) -> np.ndarray:
        X = pd.concat([self.ref] * len(t), ignore_index=True).assign(t=t, moy=moy)
        return np.asarray(self.model.predict(X))

    def __call__(self, t, moy) -> np.ndarray:
        t, moy = np.asarray(t, float), np.asarray(moy, int)
        out = np.empty(len(t))
        for m in range(1, 13):  # vectorize per month (the frame copy is the cost)
            i = moy == m
            if i.any():
                ut, inv = np.unique(t[i], return_inverse=True)
                out[i] = self._pred(ut, np.full(len(ut), m))[inv]
        return out - self.base


def month_start(d: pd.Series) -> pd.Series:
    return d.dt.to_period("M").dt.to_timestamp()


def cutoffs(rows: pd.DataFrame, scored_year: int | None) -> pd.Series:
    """Per-row cutoff (exclusive): own sale month start for training rows,
    Jan 1 of scored_year for rows in that year."""
    c = month_start(rows.sale_date)
    if scored_year is not None:
        c = c.where(rows.sale_year != scored_year, pd.Timestamp(f"{scored_year}-01-01"))
    return c


def knn_features(rows: pd.DataFrame, pool: pd.DataFrame, cut: pd.Series) -> pd.DataFrame:
    """rows: tk, x_m, y_m, neighborhood. pool: tk, sale_date, x_m, y_m,
    neighborhood, lppsf_detr (only rows with valid xy and lppsf)."""
    out = pd.DataFrame(index=rows.index, columns=["knn_lppsf", "knn_dist_m", "knn_pool_n", "nbhd_lppsf_24m"],
                       dtype=float)
    pool = pool.dropna(subset=["x_m", "y_m", "lppsf_detr"])
    for c, idx in rows.groupby(cut).groups.items():
        lo = c - pd.DateOffset(months=WINDOW_MONTHS)
        p = pool[(pool.sale_date < c) & (pool.sale_date >= lo)]
        if len(p) <= K:
            continue
        r = rows.loc[idx]
        ok = r[["x_m", "y_m"]].notna().all(axis=1)
        r = r[ok]
        if r.empty:
            continue
        tree = cKDTree(p[["x_m", "y_m"]].to_numpy())
        d, j = tree.query(r[["x_m", "y_m"]].to_numpy(), k=K + 1)
        ptk = p.tk.to_numpy()
        lp = p.lppsf_detr.to_numpy()
        vals, dist = [], []
        for row_tk, dd, jj in zip(r.tk.to_numpy(), d, j):
            keep = ptk[jj] != row_tk  # never the parcel itself
            jj, dd = jj[keep][:K], dd[keep][:K]
            vals.append(np.median(lp[jj]))
            dist.append(np.median(dd))
        out.loc[r.index, "knn_lppsf"] = vals
        out.loc[r.index, "knn_dist_m"] = dist
        out.loc[r.index, "knn_pool_n"] = len(p)
        nb = p.groupby("neighborhood").lppsf_detr.agg(["median", "size"])
        nb = nb[nb["size"] >= NBHD_MIN]["median"]
        out.loc[r.index, "nbhd_lppsf_24m"] = r.neighborhood.map(nb).to_numpy()
    return out


def own_prior_sale(rows: pd.DataFrame, history: pd.DataFrame, cut: pd.Series,
                   valuation_point: pd.Series) -> pd.DataFrame:
    """rows: tk, sale_date. history: tk, sale_date, lp_detr, sale_id."""
    limit = (rows.sale_date - pd.DateOffset(months=OWN_MIN_MONTHS)).where(lambda x: x < cut, cut)
    left = pd.DataFrame({"_i": rows.index, "tk": rows.tk.to_numpy(), "limit": limit.to_numpy()})
    left = left.dropna(subset=["tk"]).sort_values("limit")
    right = history[["tk", "sale_date", "lp_detr"]].dropna(subset=["tk"]).sort_values("sale_date")
    left["tk"], right["tk"] = left.tk.astype(str), right.tk.astype(str)  # merge_asof needs one key dtype
    # Latest history sale of the same parcel strictly before the row's limit.
    m = pd.merge_asof(left, right.rename(columns={"sale_date": "prior_sale_date"}), left_on="limit",
                      right_on="prior_sale_date", by="tk", direction="backward", allow_exact_matches=False)
    m = m.set_index("_i").reindex(rows.index)
    out = pd.DataFrame(index=rows.index)
    out["prior_lp_detr"] = m.lp_detr
    out["prior_sale_date"] = m.prior_sale_date
    has = out.prior_sale_date.notna()
    out["months_since_prior"] = ((valuation_point - out.prior_sale_date).dt.days / 30.4375).where(has)
    out["has_prior_sale"] = has.astype(int)
    # Leakage guard: every prior sale is before the cutoff and > 12 months before the sale.
    bad = has & ((out.prior_sale_date >= cut) |
                 (out.prior_sale_date >= rows.sale_date - pd.DateOffset(months=OWN_MIN_MONTHS)))
    if bad.any():
        raise AssertionError(f"own prior sale leakage on {int(bad.sum())} rows")
    return out
