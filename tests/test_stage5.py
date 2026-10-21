"""Stage 5 logic on synthetic data: CQR, level-neutral MdAPE, value tiers,
GBM determinism and neutral-season prediction."""

import numpy as np
import pandas as pd
import pytest

from avm.evaluate import compare, value_tiers
from avm.gbm import GBM
from avm.intervals import apply, calibrate, conformal_q, coverage_by


def test_conformal_q_finite_sample_quantile():
    # n = 9, alpha 0.2: ceil(10 * 0.8) = 8th smallest
    assert conformal_q(np.arange(1, 10), 0.2) == 8


def test_cqr_hits_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    lo_c, hi_c = np.full(2000, -0.5), np.full(2000, 0.5)  # deliberately too narrow
    q = calibrate(lo_c, hi_c, y_cal)
    lo, hi = apply(np.full(2000, -0.5), np.full(2000, 0.5), q)
    cov = ((y_test >= lo) & (y_test <= hi)).mean()
    assert 0.77 < cov < 0.83


def test_mondrian_calibration_per_group():
    rng = np.random.default_rng(1)
    g = pd.Series(["a"] * 1000 + ["b"] * 1000)
    y = np.r_[rng.normal(0, 0.5, 1000), rng.normal(0, 2.0, 1000)]
    q = calibrate(np.zeros(2000), np.zeros(2000), y, g)
    assert q["b"] > 3 * q["a"]
    lo, hi = apply(np.zeros(2000), np.zeros(2000), q, g)
    c = coverage_by(y, lo, hi, g).set_index("group").coverage
    assert abs(c["a"] - c["b"]) < 0.05


def test_level_neutral_mdape_ignores_uniform_level():
    rng = np.random.default_rng(2)
    p = pd.Series(rng.lognormal(12, 0.4, 1000))
    v = p * rng.lognormal(0, 0.1, 1000)
    t = compare({"a": v, "b": 0.85 * v}, p, base="a", n_boot=50).set_index(["statistic", "method"])
    assert t.loc[("mdape_ln", "b"), "estimate"] == pytest.approx(t.loc[("mdape_ln", "a"), "estimate"])
    assert t.loc[("mdape", "b"), "estimate"] > t.loc[("mdape", "a"), "estimate"]


def test_value_tiers_use_prior_years_only():
    s = pd.DataFrame({"neighborhood": ["A"] * 40 + ["B"] * 40 + ["C"] * 40 + ["A"] * 40,
                      "sale_price": [100] * 40 + [200] * 40 + [300] * 40 + [10_000] * 40,
                      "sale_year": [2022] * 120 + [2025] * 40})
    rows = pd.DataFrame({"neighborhood": ["A", "B", "C", "D"]})
    t = value_tiers(s, rows, 2025)
    assert t.tolist() == ["low", "mid", "high", "unrated"]  # 2025 sales of A never count


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"l_area": rng.normal(7.2, 0.3, n), "l_lot": rng.normal(8.5, 0.3, n),
                      "lot_missing": 0, "age": rng.integers(5, 120, n).astype(float),
                      "beds": rng.integers(2, 5, n).astype(float), "full_baths": rng.integers(1, 3, n).astype(float),
                      "half_baths": 0.0, "baths_missing": 0, "stories_num": 1.0, "duplex": 0,
                      "t": rng.uniform(0, 120, n), "moy": rng.integers(1, 13, n),
                      "style": rng.choice(["1", "4"], n), "loc": rng.choice(["x", "y", "z"], n)})
    f["lp"] = 5 + 0.6 * f.l_area + 0.002 * f.t + 0.05 * np.sin(f.moy) + rng.normal(0, 0.1, n)
    return f


def test_gbm_deterministic_and_neutral_prediction():
    cfg = {"learning_rate": 0.05, "num_boost_round": 50, "num_leaves": 15, "min_data_in_leaf": 10,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 1.0, "cat_smooth": 10.0}
    f = _frame()
    a = GBM(cfg, "independent", monotone=True).fit(f).predict_log(f.head(20), 100.0)
    b = GBM(cfg, "independent", monotone=True).fit(f).predict_log(f.head(20), 100.0)
    assert np.array_equal(a, b)
    # Neutral prediction does not depend on the row's own sale month.
    g = f.head(20).assign(moy=1)
    assert np.allclose(GBM(cfg, "independent").fit(f).predict_log(g, 100.0),
                       GBM(cfg, "independent").fit(f).predict_log(f.head(20), 100.0))
