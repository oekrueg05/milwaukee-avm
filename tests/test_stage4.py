"""Stage 3-4 logic on synthetic data: time adjustment, folding, neutral
prediction, paired head-to-head."""

import numpy as np
import pandas as pd
import pytest

from avm.evaluate import compare, union_trim
from avm.location import fold_small_units
from avm.time_adjust import months_since, trend_adjust


def test_months_since_mid_month_for_month_precision():
    s = pd.DataFrame({"sale_date": pd.to_datetime(["2025-03-01", "2025-03-01"]),
                      "date_precision": ["day", "month"]})
    m = months_since(s, "2025-01-01")
    assert m.iloc[0] == pytest.approx(59 / 30.4375) and m.iloc[1] == pytest.approx(73 / 30.4375)


def test_trend_adjust_removes_trend_and_season():
    seasonal = pd.Series({k: 0.0 for k in range(1, 13)})
    seasonal[7] = 0.05
    trend = {"valuation_date": "2025-01-01", "slope_pre": 0.01, "slope_post": 0.002}
    s = pd.DataFrame({"sale_date": pd.to_datetime(["2025-01-01", "2025-07-01"]), "date_precision": "day",
                      "sale_price": [100.0, 100.0]})
    adj = trend_adjust(s, trend, seasonal)
    t = (pd.Timestamp("2025-07-01") - pd.Timestamp("2025-01-01")).days / 30.4375
    assert adj.iloc[0] == pytest.approx(100.0)
    assert adj.iloc[1] == pytest.approx(100 * np.exp(-(0.002 * t + 0.05)))
    kept = trend_adjust(s, trend, seasonal, remove_seasonal=False)
    assert kept.iloc[1] == pytest.approx(100 * np.exp(-0.002 * t))


def test_fold_small_units_uses_training_counts_only():
    unit = pd.Series(["A"] * 40 + ["B"] * 10 + ["C"] * 5)
    parent = pd.Series(["R1"] * 40 + ["R1"] * 10 + ["R2"] * 5)
    train = pd.Series([True] * 50 + [False] * 5)  # C appears only in held-out rows
    f = fold_small_units(unit, parent, train, min_sales=30)
    assert set(f[:40]) == {"A"}
    assert set(f[40:50]) == {"R:citywide"}   # B -> R:R1 has only 10 training rows -> citywide
    assert set(f[50:]) == {"R:citywide"}     # unseen in training: folded, never its own level


def test_paired_compare_identical_estimates_give_zero_diff():
    rng = np.random.default_rng(3)
    p = pd.Series(rng.lognormal(12, 0.4, 500))
    v = p * rng.lognormal(0, 0.15, 500)
    t = compare({"assessor": v, "model": v.copy()}, p, base="assessor", n_boot=200)
    d = t[t.method == "model"]
    assert np.allclose(d.diff_vs_assessor, 0) and np.allclose(d.diff_ci_low, 0) and np.allclose(d.diff_ci_high, 0)


def test_union_trim_drops_outlier_under_either_estimate():
    p = pd.Series(np.full(30, 100.0))
    a = pd.Series(np.linspace(90, 110, 30)); a.iloc[0] = 1000
    m = pd.Series(np.linspace(90, 110, 30)); m.iloc[1] = 5
    keep = union_trim({"a": a, "m": m}, p)
    assert not keep.iloc[0] and not keep.iloc[1] and keep.iloc[2:].all()
