"""Stage 6 leakage tests: kNN pool and own prior sale never see the future."""

import numpy as np
import pandas as pd

from avm.priors import cutoffs, knn_features, own_prior_sale


def test_cutoffs_training_vs_scored():
    rows = pd.DataFrame({"sale_date": pd.to_datetime(["2023-05-20", "2025-07-09"]), "sale_year": [2023, 2025]})
    c = cutoffs(rows, 2025)
    assert c.tolist() == [pd.Timestamp("2023-05-01"), pd.Timestamp("2025-01-01")]


def test_knn_uses_only_sales_before_cutoff_and_not_self():
    rng = np.random.default_rng(0)
    n = 60
    pool = pd.DataFrame({"tk": [f"p{i}" for i in range(n)], "x_m": rng.uniform(0, 1000, n),
                         "y_m": rng.uniform(0, 1000, n), "neighborhood": "A",
                         "sale_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, n), "D"),
                         "lppsf_detr": 5.0})
    future = pool.sale_date >= pd.Timestamp("2025-01-01")
    pool.loc[future, "lppsf_detr"] = 99.0  # would show up if the future leaked in
    rows = pd.DataFrame({"tk": ["p0", "q"], "x_m": [500, 500], "y_m": [500, 500], "neighborhood": "A",
                         "sale_date": pd.to_datetime(["2025-03-01", "2025-06-01"]), "sale_year": 2025})
    pool.loc[0, "lppsf_detr"] = 77.0  # the row's own parcel must be excluded
    f = knn_features(rows, pool, cutoffs(rows, 2025))
    assert (f.knn_lppsf == 5.0).all() and (f.nbhd_lppsf_24m == 5.0).all()


def test_own_prior_sale_rules():
    hist = pd.DataFrame({"tk": ["A", "A", "A", "B"],
                         "sale_date": pd.to_datetime(["2019-06-01", "2024-03-01", "2024-12-01", "2024-01-01"]),
                         "lp_detr": [11.0, 12.0, 13.0, 10.0], "sale_id": ["1", "2", "3", "4"]})
    rows = pd.DataFrame({"tk": ["A", "B"], "sale_date": pd.to_datetime(["2025-02-01", "2024-06-01"]),
                         "sale_year": [2025, 2024]})
    cut = cutoffs(rows, 2025)
    o = own_prior_sale(rows, hist, cut, cut)
    # A: 2024-03 and 2024-12 are within 12 months of 2025-02 -> use 2019-06.
    assert o.loc[0, "prior_lp_detr"] == 11.0 and o.loc[0, "has_prior_sale"] == 1
    # B: its only sale is 5 months before -> none.
    assert o.loc[1, "has_prior_sale"] == 0
