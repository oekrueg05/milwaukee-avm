"""Baseline 2: hedonic regression with location fixed effects (rule 5).

log(price) = spline(t) + month-of-year (sum-to-zero) + characteristics
             + location fixed effect

The year trend (natural cubic spline in months, about one knot per training
year) is separate from month-of-year, so a prediction can be placed at a
trend point (e.g. Dec 2024) with a neutral season: the log prediction is
averaged over the 12 months, whose sum-to-zero effects then cancel.
Retransform: exp(log prediction), the conditional median.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

Z80 = 1.2815515655446004  # 80% two-sided normal interval


INDICATORS = ["lot_missing", "baths_missing"]


def formula(n_years: int, indicators: list[str]) -> str:
    df = max(3, n_years + 1)
    ind = "".join(f" + {c}" for c in indicators)
    return (f"lp ~ cr(t, df={df}, constraints='center') + C(moy, Sum) + l_area + l_lot + C(age_bin, Treatment('41-60'))"
            f" + beds + full_baths + half_baths{ind} + C(stories, Treatment('1'))"
            " + C(style) + duplex + C(loc)")


def fit(train: pd.DataFrame):
    """Indicators constant in the training window (e.g. baths_missing when
    training starts in 2020) are left out rather than fit as a zero column."""
    n_years = int(train.sale_year.nunique())
    ind = [c for c in INDICATORS if train[c].nunique() > 1]
    m = smf.ols(formula(n_years, ind), data=train).fit()
    rank, k = np.linalg.matrix_rank(m.model.exog), m.model.exog.shape[1]
    if rank < k:
        raise ValueError(f"hedonic design rank-deficient ({rank} < {k}); check collinear features")
    return m


def predict_neutral(model, X: pd.DataFrame, t_point: float) -> pd.DataFrame:
    """Prediction at trend point t_point with a neutral month-of-year effect.
    Returns pred (price), log_pred, and an 80% prediction interval."""
    logs, se_mean = [], None
    for mo in range(1, 13):
        Xm = X.assign(t=t_point, moy=mo)
        if mo == 6:
            sf = model.get_prediction(Xm).summary_frame()
            se_mean = sf["mean_se"].to_numpy()
            logs.append(sf["mean"].to_numpy())
        else:
            logs.append(np.asarray(model.predict(Xm)))
    log_pred = np.mean(logs, axis=0)
    half = Z80 * np.sqrt(model.scale + se_mean ** 2)
    return pd.DataFrame({"log_pred": log_pred, "pred": np.exp(log_pred),
                         "pi80_low": np.exp(log_pred - half), "pi80_high": np.exp(log_pred + half)},
                        index=X.index)
