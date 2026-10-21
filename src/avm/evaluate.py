"""Head-to-head scoring of value estimates against the shared target prices.

For each estimate (assessor roll Y, or a model's Jan 1 prediction), ratio =
estimate / target price (Stage 3 trend-only adjustment). Statistics:
ratio-study (median ratio, COD, PRD, PRB, MKI) and accuracy (MdAPE = median
|ratio - 1|, share within 10% and 20%). Note MdAPE and within-X% penalize
level as well as dispersion (the assessor's level is about 0.89); COD is the
level-free dispersion measure.

Differences (model - assessor) use a PAIRED bootstrap: each resample draws
the same sales for both, so the CI reflects their shared noise.
"""

import numpy as np
import pandas as pd

from avm.ratio_study import IAAO_TARGETS, N_BOOT, SEED, _stats, iqr_outliers

STATS = ["median_ratio", "cod", "prd", "prb", "mki", "mdape_ln", "mdape", "within_10", "within_20"]
LOWER_IS_BETTER = {"cod", "mdape", "mdape_ln"}


def _all_stats(v: np.ndarray, p: np.ndarray) -> dict:
    """Ratio-study stats plus accuracy. mdape_ln is LEVEL-NEUTRAL MdAPE: each
    method's ratios are divided by that method's own median ratio first, so
    a uniform level offset (e.g. the assessor's 0.89) does not count as
    error. Headline accuracy claims use mdape_ln (Owen, 2026-10-18)."""
    out = _stats(v, p)
    r = v / p
    med = np.median(r, axis=1, keepdims=True)
    out["mdape_ln"] = np.median(np.abs(r / med - 1), axis=1)
    out["mdape"] = np.median(np.abs(r - 1), axis=1)
    out["within_10"] = (np.abs(r - 1) <= 0.10).mean(axis=1)
    out["within_20"] = (np.abs(r - 1) <= 0.20).mean(axis=1)
    return out


def value_tiers(s: pd.DataFrame, rows: pd.DataFrame, year: int) -> pd.Series:
    """Neighborhood value tier for `rows` scored in `year`: terciles of the
    neighborhood median sale price over the three prior years (neighborhoods
    with >= 30 such sales). Uses only sales before `year`."""
    prior = s[s.sale_year.between(year - 3, year - 1)]
    med = prior.groupby("neighborhood").sale_price.agg(["median", "size"])
    med = med[med["size"] >= 30]["median"]
    cuts = med.quantile([1 / 3, 2 / 3]).to_numpy()
    t = pd.cut(rows.neighborhood.map(med), [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"])
    # astype(str) keeps NaN on a categorical, which silently dropped the
    # unrated stratum once; fill explicitly.
    return pd.Series(t.astype(object), index=rows.index).fillna("unrated").astype(str)


def tier_prb(pred: pd.Series, price: pd.Series, tier: pd.Series) -> dict:
    """PRB within each rated value tier, and the mean |PRB| across them (the
    compression measure used in model selection)."""
    out = {}
    for t in ["low", "mid", "high"]:
        m = (tier == t).to_numpy()
        if m.sum() >= 30:
            out[t] = float(_stats(pred.to_numpy(float)[m][None, :], price.to_numpy(float)[m][None, :])["prb"][0])
    out["mean_abs"] = float(np.mean([abs(v) for v in out.values()])) if out else float("nan")
    return out


def compare(estimates: dict[str, pd.Series], price: pd.Series, base: str,
            n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """estimates: name -> Series aligned with price. Rows: statistic x method,
    plus difference vs `base` with paired CIs."""
    names = list(estimates)
    p = price.to_numpy(float)
    V = {k: estimates[k].to_numpy(float) for k in names}
    n = len(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    point = {k: {s: float(a[0]) for s, a in _all_stats(V[k][None, :], p[None, :]).items()} for k in names}
    boot = {k: _all_stats(V[k][idx], p[idx]) for k in names}
    rows = []
    for st in STATS:
        for k in names:
            b = boot[k][st]
            row = {"statistic": st, "method": k, "n": n, "estimate": point[k][st],
                   "ci_low": float(np.nanpercentile(b, 2.5)), "ci_high": float(np.nanpercentile(b, 97.5))}
            if st in IAAO_TARGETS:
                lo, hi = IAAO_TARGETS[st]
                row["meets_target"] = bool((lo is None or row["estimate"] >= lo) and row["estimate"] <= hi)
            if k != base:
                d = b - boot[base][st]
                row |= {"diff_vs_" + base: point[k][st] - point[base][st],
                        "diff_ci_low": float(np.nanpercentile(d, 2.5)),
                        "diff_ci_high": float(np.nanpercentile(d, 97.5))}
            rows.append(row)
    return pd.DataFrame(rows)


def union_trim(estimates: dict[str, pd.Series], price: pd.Series) -> pd.Series:
    """True = keep. Drops a sale if it is a 3x IQR ratio outlier under ANY
    of the estimates (CLAUDE.md: union trim for the head-to-head)."""
    out = pd.Series(False, index=price.index)
    for v in estimates.values():
        out |= iqr_outliers(v / price)
    return ~out


def accuracy(pred: pd.Series, price: pd.Series) -> dict:
    r = pred / price
    return {"n": int(r.notna().sum()), "median_ratio": float(r.median()),
            "mdape_ln": float((r / r.median() - 1).abs().median()),
            "mdape": float((r - 1).abs().median()),
            "within_10": float(((r - 1).abs() <= 0.10).mean()),
            "within_20": float(((r - 1).abs() <= 0.20).mean()),
            "cod": float(100 * (r - r.median()).abs().mean() / r.median())}
