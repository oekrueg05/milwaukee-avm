"""IAAO ratio-study statistics with bootstrap confidence intervals.

ratio r = value / (time-adjusted) sale price.

  median ratio   level.                         IAAO target 0.90-1.10
  COD            100 * mean|r - median| / median. target <= 15 (SFR)
  PRD            mean(r) / (sum(value) / sum(price)). target 0.98-1.03
  PRB            slope b of  (r - med)/med  on  log2(V),
                 V = 0.5 * (value/med + price)  (IAAO Standard on Ratio
                 Studies, 2013, Appendix B).    target -0.05 to +0.05
  MKI            Gini(value, ordered by price) / Gini(price) (Quintos 2020;
                 used by Cook County).           target 0.95-1.05

PRD and PRB computed against sale prices are biased toward showing
regressivity: noise in sale prices sits in the denominator. Every report
states this next to the numbers.
"""

import numpy as np
import pandas as pd

IAAO_TARGETS = {
    "median_ratio": (0.90, 1.10),
    "cod": (None, 15.0),
    "prd": (0.98, 1.03),
    "prb": (-0.05, 0.05),
    "mki": (0.95, 1.05),
}
N_BOOT = 2000
SEED = 20261018
IQR_K = 3.0


def _stats(value: np.ndarray, price: np.ndarray) -> dict:
    """value, price: 2-D arrays (n_rep, n). Returns 1-D arrays per statistic."""
    r = value / price
    med = np.median(r, axis=1, keepdims=True)
    cod = 100 * np.mean(np.abs(r - med), axis=1) / med[:, 0]
    prd = r.mean(axis=1) / (value.sum(axis=1) / price.sum(axis=1))
    y = (r - med) / med
    x = np.log2(0.5 * (value / med + price))
    xc = x - x.mean(axis=1, keepdims=True)
    prb = (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)
    order = np.argsort(price, axis=1)
    v_sorted = np.take_along_axis(value, order, axis=1)
    p_sorted = np.take_along_axis(price, order, axis=1)
    mki = _gini_sorted(v_sorted) / _gini_sorted(p_sorted)
    return {"median_ratio": med[:, 0], "cod": cod, "prd": prd, "prb": prb, "mki": mki}


def _gini_sorted(x: np.ndarray) -> np.ndarray:
    """Gini (concentration) coefficient of rows already ordered by price."""
    n = x.shape[1]
    i = np.arange(1, n + 1)
    return (2 * (x * i).sum(axis=1)) / (n * x.sum(axis=1)) - (n + 1) / n


def ratio_stats(value, price, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Point estimates and percentile bootstrap 95% CIs. One row per statistic."""
    v = np.asarray(value, dtype=float)
    p = np.asarray(price, dtype=float)
    ok = np.isfinite(v) & np.isfinite(p) & (v > 0) & (p > 0)
    v, p = v[ok], p[ok]
    if len(v) < 2:
        raise ValueError(f"ratio_stats needs >= 2 valid pairs, got {len(v)}")
    point = {k: float(a[0]) for k, a in _stats(v[None, :], p[None, :]).items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = _stats(v[idx], p[idx])
    rows = []
    for k in point:
        lo_t, hi_t = IAAO_TARGETS[k]
        est = point[k]
        rows.append({
            "statistic": k, "n": len(v), "estimate": est,
            # nanpercentile: degenerate resamples (tiny n, all one sale) give NaN PRB.
            "ci_low": float(np.nanpercentile(boot[k], 2.5)), "ci_high": float(np.nanpercentile(boot[k], 97.5)),
            "target": (f"<= {hi_t}" if lo_t is None else f"{lo_t} to {hi_t}"),
            "meets_target": bool((lo_t is None or est >= lo_t) and est <= hi_t),
        })
    return pd.DataFrame(rows)


def iqr_outliers(ratio: pd.Series, k: float = IQR_K) -> pd.Series:
    """IAAO-style trim flag: outside Q1 - k*IQR or Q3 + k*IQR."""
    q1, q3 = ratio.quantile(0.25), ratio.quantile(0.75)
    return (ratio < q1 - k * (q3 - q1)) | (ratio > q3 + k * (q3 - q1))
