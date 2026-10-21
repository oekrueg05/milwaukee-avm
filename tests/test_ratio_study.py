"""Ratio-study statistics against hand-computed values."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from avm.ratio_study import iqr_outliers, ratio_stats


def _est(t, k):
    return t.set_index("statistic").loc[k, "estimate"]


def test_cod_prd_hand_computed():
    # ratios 1, 2, 1 -> median 1, COD 100 * (0 + 1 + 0) / 3 / 1, PRD (4/3) / (600/500)
    t = ratio_stats([100, 200, 300], [100, 100, 300], n_boot=50)
    assert _est(t, "median_ratio") == pytest.approx(1.0)
    assert _est(t, "cod") == pytest.approx(100 / 3)
    assert _est(t, "prd") == pytest.approx((4 / 3) / 1.2)


def test_prb_matches_ols():
    rng = np.random.default_rng(0)
    p = rng.lognormal(12, 0.5, 400)
    v = p * rng.lognormal(0, 0.15, 400) * (p / p.mean()) ** -0.1  # mildly regressive
    r = v / p
    med = np.median(r)
    x = np.log2(0.5 * (v / med + p))
    y = (r - med) / med
    b = sm.OLS(y, sm.add_constant(x)).fit().params[1]
    assert _est(ratio_stats(v, p, n_boot=50), "prb") == pytest.approx(b)


def test_mki_is_one_when_values_proportional():
    p = np.array([50, 80, 120, 200, 400.0])
    assert _est(ratio_stats(0.9 * p, p, n_boot=50), "mki") == pytest.approx(1.0)


def test_bootstrap_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    p = rng.lognormal(12, 0.4, 300)
    v = p * rng.lognormal(0, 0.1, 300)
    a, b = ratio_stats(v, p, n_boot=200), ratio_stats(v, p, n_boot=200)
    pd.testing.assert_frame_equal(a, b)
    assert ((a.ci_low <= a.estimate) & (a.estimate <= a.ci_high)).all()


def test_iqr_outliers():
    r = pd.Series(np.linspace(0.8, 1.2, 21).tolist() + [0.05, 10.0])  # IQR 0.2: fences 0.3, 1.7
    assert iqr_outliers(r).tolist() == [False] * 21 + [True, True]
