"""Conformalized quantile regression (CQR) for 80% prediction intervals.

1. Fit 10% and 90% quantile LightGBM models on sales through CAL_YEAR - 1.
2. Predict CAL_YEAR sales (at the Dec CAL_YEAR-1 trend point, neutral season)
   and compute conformity scores E = max(q_lo - y, y - q_hi) on log target.
3. Q = the ceil((n+1)(1-alpha))/n empirical quantile of E (globally, or per
   neighborhood value tier for Mondrian calibration).
4. Final quantile models trained through the last training year give
   [q_lo - Q, q_hi + Q] for the test year.

Calibration uses only sales before the test year (CAL_YEAR = 2024 for a
2025 test).
"""

import numpy as np
import pandas as pd

ALPHA = 0.20  # 80% intervals


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(k, n - 1)])


def scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def calibrate(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, group: pd.Series | None = None) -> dict:
    """Returns {"global": Q} or {"global": Q, tier: Q_tier, ...} (Mondrian)."""
    e = scores(lo, hi, y)
    out = {"global": conformal_q(e)}
    if group is not None:
        g = np.asarray(group)
        for t in pd.unique(g):
            m = g == t
            if m.sum() >= 30:
                out[str(t)] = conformal_q(e[m])
    return out


def apply(lo: np.ndarray, hi: np.ndarray, q: dict, group: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    if group is None:
        adj = np.full(len(lo), q["global"])
    else:
        adj = np.array([q.get(str(t), q["global"]) for t in np.asarray(group)])
    return lo - adj, hi + adj


def contain_point(lo: np.ndarray, hi: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Widen an interval just enough to include the point prediction. The
    quantile models are fit separately from the point model, so at the
    extremes (0.13% of 2026 parcels, mostly the highest values) the point can
    fall outside. Widening only raises coverage."""
    return np.minimum(lo, point), np.maximum(hi, point)


def coverage_by(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, group: pd.Series) -> pd.DataFrame:
    inside = (y >= lo) & (y <= hi)
    width = np.exp(hi) - np.exp(lo)
    rel = width / np.exp((lo + hi) / 2)
    d = pd.DataFrame({"group": np.asarray(group), "inside": inside, "rel_width": rel})
    t = d.groupby("group").agg(n=("inside", "size"), coverage=("inside", "mean"),
                               median_rel_width=("rel_width", "median")).reset_index()
    tot = pd.DataFrame([{"group": "all", "n": len(d), "coverage": d.inside.mean(),
                         "median_rel_width": d.rel_width.median()}])
    return pd.concat([tot, t], ignore_index=True)
