"""Comparable sales (shared by Stage 7 scoring and the Stage 9 pages).

Eligible pool for a valuation date V (Jan 1 of the scored year): clean
SF/duplex sales dated before V, excluding first sales of a flip, IAAO ratio
outliers (3 x IQR on the assessor ratio) and price/sq ft outliers.

Tiered search per subject, same property type only (SFR -> SFR, duplex ->
duplex), never the subject's own parcel (same tax key). Tiers are tried in order and the search stops at the first with at
least MIN_COMPS eligible sales:
    12 months before V: 0.5 mi, 1 mi, 1.25 mi
    24 months before V: 0.5 mi, 1 mi, 1.25 mi
Within the chosen tier, comps are ranked by SHAP-weighted standardized
distance on structural features (ties by distance) and the top K_COMPS kept.
The tier is recorded per subject (comp_radius_mi, comp_window_months) as a
confidence field. Subjects with no tier reaching MIN_COMPS get no comps.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K_COMPS, MIN_COMPS = 5, 3
MILE_M = 1609.344
RADII_MI = [0.5, 1.0, 1.25]
WINDOWS_MONTHS = [12, 24]
TIERS = [(w, r) for w in WINDOWS_MONTHS for r in RADII_MI]
STRUCT = ["l_area", "age", "beds", "full_baths", "half_baths", "stories_num", "l_lot"]
EXCLUDE_FLAGS = ["first_sale_of_flip", "ratio_outlier_iaao", "ppsf_outlier"]


def eligible_pool(rows: pd.DataFrame, sales: pd.DataFrame, valuation_date: str) -> pd.DataFrame:
    """Sales in `rows` (frame rows with features) dated in the 24 months
    before the valuation date, minus the excluded flags (read from the clean
    sales table by sale_id, so the frame need not carry them)."""
    v = pd.Timestamp(valuation_date)
    start = v - pd.DateOffset(months=max(WINDOWS_MONTHS))
    flags = sales.set_index("sale_id")[EXCLUDE_FLAGS].reindex(rows.sale_id)
    if flags.isna().any().any():
        raise ValueError("comp pool rows missing from the clean sales table")
    bad = flags.any(axis=1).to_numpy()
    keep = (rows.sale_date >= start).to_numpy() & (rows.sale_date < v).to_numpy() & ~bad
    out = rows[keep].dropna(subset=["x_m", "y_m"]).copy()
    out["months_before"] = (v.year - out.sale_date.dt.year) * 12 + (v.month - out.sale_date.dt.month)
    return out


def comps_for(subjects: pd.DataFrame, pool: pd.DataFrame, weights: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """Long table: sale_id (subject), comp_sale_id, rank, dist_m, struct_dist,
    comp_radius_mi, comp_window_months. Subjects and pool must carry the tax
    key `tk`; a subject's own parcel's sales are never its comps."""
    for name, df in (("subjects", subjects), ("pool", pool)):
        if "tk" not in df.columns or df.tk.isna().any():
            raise ValueError(f"comps_for needs a tax key (tk) on every {name} row")
    rows = []
    w = weights.reindex(STRUCT).fillna(0).to_numpy()
    subjects = subjects.dropna(subset=["x_m", "y_m"])
    for duplex in (0, 1):
        p = pool[pool.duplex == duplex]
        t = subjects[subjects.duplex == duplex]
        if p.empty or t.empty:
            continue
        tree = cKDTree(p[["x_m", "y_m"]].to_numpy())
        months = p.months_before.to_numpy()
        Z = (p[STRUCT] / sd).to_numpy()
        px, py, pid, ptk = p.x_m.to_numpy(), p.y_m.to_numpy(), p.sale_id.to_numpy(), p.tk.astype(str).to_numpy()
        far = tree.query_ball_point(t[["x_m", "y_m"]].to_numpy(), max(RADII_MI) * MILE_M)
        for sid, stk, x, y, z, cand in zip(t.sale_id, t.tk.astype(str), t.x_m, t.y_m, (t[STRUCT] / sd).to_numpy(),
                                           far):
            cand = np.asarray(cand, dtype=int)
            cand = cand[ptk[cand] != stk]
            geo_all = np.hypot(px[cand] - x, py[cand] - y)
            for win, rad in TIERS:
                sel = (months[cand] <= win) & (geo_all <= rad * MILE_M)
                if sel.sum() >= MIN_COMPS:
                    break
            else:
                continue
            idx, geo = cand[sel], geo_all[sel]
            d = np.sqrt(np.nansum(w * (Z[idx] - z) ** 2, axis=1))
            for rank, j in enumerate(np.lexsort((geo, d))[:K_COMPS], 1):
                rows.append({"sale_id": sid, "comp_sale_id": pid[idx[j]], "rank": rank, "dist_m": geo[j],
                             "struct_dist": d[j], "comp_radius_mi": rad, "comp_window_months": win})
    return pd.DataFrame(rows, columns=["sale_id", "comp_sale_id", "rank", "dist_m", "struct_dist",
                                       "comp_radius_mi", "comp_window_months"])
