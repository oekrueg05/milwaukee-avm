"""Monthly price indexes for time-adjusting sale prices to a valuation date.

EVALUATION ONLY. These indexes adjust test-year (and context-year) sale
prices to Jan 1 for the ratio study. They never feed model training; the
model predicts at Jan 1 using data through the prior year (CLAUDE.md).

MAIN (Owen, 2026-10-18): trend-only. Month-of-year effects are estimated
once on 2014-2024 sales (hedonic with sale-year fixed effects); the
valuation-year trend is piecewise linear in months with a knot at the
valuation date, seasonal effects held fixed. Each price is moved to the
valuation-date trend level with a neutral season, which is where the model
predicts (Dec 2024 trend level, neutral month-of-year).

SENSITIVITY: seasonal monthly hedonic index. OLS of log(price) on sale-month
dummies + roll Y-1 characteristics + neighborhood fixed effects; January of
the target year is the reference (1.0). Adjusted price = price / index.

IAAO ratio-trend cross-check: OLS of ratio on months since the valuation
date; adjusted ratio = ratio - slope * months.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MIN_ROWS = 200


def _design(s: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=s.index)
    d["lp"] = np.log(s.sale_price)
    d["l_area"] = np.log(s.bldg_area.where(s.bldg_area > 0))
    d["l_lot"] = np.log(s.lot_area.where(s.lot_area > 0))
    age = s.sale_year - s.year_built
    d["age_bin"] = pd.cut(age, [-1, 20, 40, 60, 80, 100, 120, 1000]).astype(str)
    d["beds"] = s.bedrooms.clip(upper=8)
    d["duplex"] = (s.land_use_y1 == "8820").astype(int)
    d["nbhd"] = s.neighborhood.astype(str)
    d["month"] = s.sale_date.dt.to_period("M").astype(str)
    return d


def fit_hedonic_index(s: pd.DataFrame, ref_month: str) -> tuple[pd.DataFrame, dict]:
    """Returns (index table: month, index, ci_low, ci_high, n_sales) and fit info.
    Rows with a missing characteristic or a price/sq ft outlier are left out
    of the fit (counts reported)."""
    d = _design(s[~s.ppsf_outlier.fillna(False)])
    used = d.dropna()
    if len(used) < MIN_ROWS:
        raise ValueError(f"hedonic index: only {len(used)} usable rows")
    months = sorted(used.month.unique())
    if ref_month not in months:
        raise ValueError(f"reference month {ref_month} has no sales")
    f = (f"lp ~ C(month, Treatment('{ref_month}')) + l_area + l_lot + C(age_bin) + beds + duplex"
         " + C(nbhd)")
    m = smf.ols(f, data=used).fit()
    rows = []
    for mo in months:
        if mo == ref_month:
            b, se = 0.0, 0.0
        else:
            name = f"C(month, Treatment('{ref_month}'))[T.{mo}]"
            b, se = m.params[name], m.bse[name]
        rows.append({"month": mo, "index": np.exp(b), "ci_low": np.exp(b - 1.96 * se),
                     "ci_high": np.exp(b + 1.96 * se), "n_sales": int((used.month == mo).sum())})
    info = {"n_input": len(s), "n_used": len(used), "n_dropped_missing_or_outlier": len(s) - len(used),
            "r2": round(m.rsquared, 4), "ref_month": ref_month}
    return pd.DataFrame(rows), info


def apply_index(s: pd.DataFrame, index: pd.DataFrame) -> pd.Series:
    month = s.sale_date.dt.to_period("M").astype(str)
    f = month.map(index.set_index("month")["index"])
    if f.isna().any():
        raise ValueError(f"no index value for months {sorted(month[f.isna()].unique())}")
    return s.sale_price / f


MONTHS = list(range(1, 13))


def months_since(s: pd.DataFrame, valuation_date: str) -> pd.Series:
    """Fractional months from the valuation date. Month-only sale dates (the
    1st of the month) are placed mid-month."""
    d = s.sale_date + pd.to_timedelta(np.where(s.date_precision == "month", 14, 0), unit="D")
    return (d - pd.Timestamp(valuation_date)).dt.days / 30.4375


def seasonal_effects(s: pd.DataFrame) -> pd.Series:
    """Month-of-year effects on log price (sum to zero over the 12 months),
    from a hedonic with sale-year fixed effects. Index: month 1-12."""
    d = _design(s[~s.ppsf_outlier.fillna(False)])
    d["year"] = s.sale_year.astype(str)
    d["moy"] = s.sale_date.dt.month
    used = d.drop(columns=["month"]).dropna()
    m = smf.ols("lp ~ C(year) + C(moy, Sum) + l_area + l_lot + C(age_bin) + beds + duplex + C(nbhd)",
                data=used).fit()
    eff = {k: m.params[f"C(moy, Sum)[S.{k}]"] for k in MONTHS[:-1]}
    eff[12] = -sum(eff.values())
    return pd.Series(eff, name="log_effect").rename_axis("month")


def fit_trend(s: pd.DataFrame, valuation_date: str, seasonal: pd.Series) -> dict:
    """Piecewise-linear log-price trend with a knot at the valuation date,
    seasonal effects held fixed (as an offset). Uses the sales passed in
    (typically the valuation year and the year before)."""
    d = _design(s[~s.ppsf_outlier.fillna(False)]).drop(columns=["month"])
    t = months_since(s.loc[d.index], valuation_date)
    d["t"], d["t_post"] = t, t.clip(lower=0)
    d["seas"] = s.loc[d.index].sale_date.dt.month.map(seasonal)
    used = d.dropna()
    m = smf.ols("lp ~ t + t_post + l_area + l_lot + C(age_bin) + beds + duplex + C(nbhd)",
                data=used, offset=used.seas).fit()
    return {"valuation_date": valuation_date, "slope_pre": m.params["t"],
            "slope_post": m.params["t"] + m.params["t_post"],
            "slope_post_se": float(np.sqrt(m.cov_params().loc[["t", "t_post"], ["t", "t_post"]].to_numpy().sum())),
            "n_used": len(used), "r2": round(m.rsquared, 4)}


def trend_adjust(s: pd.DataFrame, trend: dict, seasonal: pd.Series, remove_seasonal: bool = True) -> pd.Series:
    """Price at the valuation-date trend level (and neutral season unless
    remove_seasonal is False)."""
    t = months_since(s, trend["valuation_date"])
    log_trend = trend["slope_pre"] * t.clip(upper=0) + trend["slope_post"] * t.clip(lower=0)
    log_adj = log_trend + (s.sale_date.dt.month.map(seasonal) if remove_seasonal else 0)
    return s.sale_price * np.exp(-log_adj)


def ratio_trend(ratio: pd.Series, sale_date: pd.Series, valuation_date: str) -> dict:
    """IAAO sales ratio trend: ratio = a + b * months since valuation date."""
    t = (sale_date - pd.Timestamp(valuation_date)).dt.days / 30.4375
    ok = ratio.notna() & t.notna()
    m = smf.ols("r ~ t", data=pd.DataFrame({"r": ratio[ok], "t": t[ok]})).fit()
    a, b = m.params["Intercept"], m.params["t"]
    return {"intercept": a, "slope_per_month": b, "slope_se": m.bse["t"],
            "implied_monthly_appreciation": -b / a, "n": int(ok.sum())}


def trend_adjusted_ratio(ratio: pd.Series, sale_date: pd.Series, valuation_date: str, slope: float) -> pd.Series:
    t = (sale_date - pd.Timestamp(valuation_date)).dt.days / 30.4375
    return ratio - slope * t
