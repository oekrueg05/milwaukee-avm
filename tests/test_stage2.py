"""Unit tests for Stage 2 logic on synthetic data (no raw files needed)."""

import pandas as pd
import pytest

from avm.clean_sales import multi_parcel_groups
from avm.io import parse_dates
from avm.join_rolls import join_rolls
from avm.mprop import CROSSWALK
from avm.permits import normalize_address
from avm.sales import parse_sale_dates, repeat_sale_flags


def _sales(rows):
    df = pd.DataFrame(rows, columns=["tk", "sale_date", "date_precision"])
    df["sale_date"] = pd.to_datetime(df.sale_date)
    df["sale_id"] = [f"s{i:03d}" for i in range(len(df))]
    return df


def test_flip_day_precision():
    s = repeat_sale_flags(_sales([
        ("A", "2024-01-10", "day"), ("A", "2024-11-01", "day"),   # 296 days: flip
        ("B", "2023-01-10", "day"), ("B", "2024-03-01", "day"),   # 416 days: not
    ]))
    assert s.resale_lt_12m.tolist() == [False, True, False, False]
    assert s.first_sale_of_flip.tolist() == [True, False, False, False]


def test_flip_month_precision_and_ambiguous():
    s = repeat_sale_flags(_sales([
        ("A", "2015-01-01", "month"), ("A", "2015-12-01", "month"),  # 11 months: flip
        ("B", "2015-01-01", "month"), ("B", "2016-01-01", "month"),  # 12 months: ambiguous
    ]))
    assert s.resale_lt_12m.tolist() == [False, True, False, False]
    assert s.resale_12m_ambiguous.tolist() == [False, False, False, True]


def test_same_period_pair_flags_both():
    s = repeat_sale_flags(_sales([("A", "2024-05-01", "day"), ("A", "2024-05-01", "day"),
                                  ("A", "2025-08-01", "day")]))
    assert s.same_period_pair.tolist() == [True, True, False]
    assert not s.resale_lt_12m.iloc[1]  # a same-day repeat is not counted as a flip


def test_parse_sale_dates_formats_and_failure():
    d, p = parse_sale_dates(pd.Series(["2015-01", "10/15/2025", "2022-04-01"], dtype="string"))
    assert p.tolist() == ["month", "day", "day"]
    assert d.tolist() == [pd.Timestamp("2015-01-01"), pd.Timestamp("2025-10-15"), pd.Timestamp("2022-04-01")]
    with pytest.raises(ValueError):
        parse_sale_dates(pd.Series(["15.10.2025"], dtype="string"))


def test_parse_dates_refuses_corrupt_values():
    d, c = parse_dates(pd.Series(["00:00.0", "4/8/2015", "", "1899-12-30 00:00:00.0000000"]))
    assert d.isna().tolist() == [True, False, True, True]
    assert (c["n_corrupt"], c["n_sentinel"], c["n_parsed"]) == (1, 1, 1)


def test_to_number_strips_currency_and_fails_loudly():
    from avm.io import to_number
    from avm.sales import parse_price
    assert parse_price(pd.Series(["$357,000", "950000.00", "NULL", ""]), "t").tolist()[:2] == [357000, 950000]
    assert parse_price(pd.Series(["NULL", ""]), "t").isna().all()
    with pytest.raises(ValueError):
        to_number(pd.Series(["12 acres"]), "t")


def test_normalize_address():
    a = normalize_address(pd.Series(["123 N MAIN ST, Unit 2", "456  s main st"]))
    assert a.tolist() == ["123 N MAIN ST", "456 S MAIN ST"]


def test_multi_parcel_rules():
    s = pd.DataFrame({
        "tk": ["A", "B", "C", "D", "E", "F", "G"],
        "sale_date": pd.to_datetime(["2024-03-01"] * 2 + ["2024-04-01"] * 2 + ["2024-05-01"] * 3),
        "date_precision": "day",
        "sale_price": [200000] * 2 + [150000] * 2 + [90000] * 3,
        "sale_year": 2024, "prop_type": "Residential",
    })
    cents = pd.DataFrame({"tk": list("ABCDEFG"),
                          "x_m": [0, 100, 0, 5000, 0, 9000, 20000], "y_m": [0] * 7})
    rows, groups = multi_parcel_groups(s, cents)
    # Only the 200 m rule drops; E+F+G (3 parcels far apart, round price) is kept, unflagged.
    assert rows.mp_drop.tolist() == [True, True, False, False, False, False, False]
    assert groups.rule_b_within_radius.sum() == 1   # A+B, 100 m apart
    assert groups.rule_c_3plus.sum() == 1
    assert not rows.mp_flag_3plus_not_1000.any()    # $90,000 is a $1,000 multiple


def test_multi_parcel_3plus_flag_at_odd_price():
    s = pd.DataFrame({"tk": list("ABC"), "sale_date": pd.to_datetime(["2024-05-01"] * 3),
                      "date_precision": "day", "sale_price": [91234] * 3, "sale_year": 2024,
                      "prop_type": "Residential"})
    cents = pd.DataFrame({"tk": list("ABC"), "x_m": [0, 5000, 9000], "y_m": [0] * 3})
    rows, _ = multi_parcel_groups(s, cents)
    assert not rows.mp_drop.any() and rows.mp_flag_3plus_not_1000.all()


def test_multi_parcel_month_only_uses_radius_rule_only():
    s = pd.DataFrame({
        "tk": list("ABCDEFGH"),
        "sale_date": pd.to_datetime(["2015-06-01"] * 8), "date_precision": "month",
        "sale_price": [100000] * 3 + [123456] * 3 + [87654] * 2, "sale_year": 2015,
        "prop_type": "Residential",
    })
    cents = pd.DataFrame({"tk": list("ABCDEFGH"),
                          "x_m": [0, 5000, 10000, 0, 5000, 10000, 0, 50], "y_m": [0] * 8})
    rows, groups = multi_parcel_groups(s, cents)
    # 3+ parcels far apart are kept at any price; the pair 50 m apart is dropped.
    assert rows.mp_drop.tolist() == [False] * 6 + [True] * 2


def test_baths_pre2019_raw_only_for_pre2019_feature_rolls():
    from avm.join_rolls import baths_pre2019_raw
    concepts = _concepts({2017: {"land_use": "8810"}, 2019: {"land_use": "8810", "full_baths": 2}})
    concepts[2017]["full_baths_raw_excluded"] = "3"
    concepts[2017]["half_baths_raw_excluded"] = "1"
    lookup = pd.DataFrame({"file_year": [2017, 2019], "assessment_year": [2017, 2019],
                           "status": ["confirmed"] * 2})
    sales = pd.DataFrame({"sale_id": ["a", "b"], "tk": ["T", "T"], "sale_year": [2018, 2020]})
    o = baths_pre2019_raw(sales, concepts, lookup).set_index("sale_id")
    assert o.loc["a", "baths_era"] == "pre2019" and o.loc["a", "full_baths_pre2019_raw"] == 3
    assert o.loc["b", "baths_era"] == "2019plus" and pd.isna(o.loc["b", "full_baths_pre2019_raw"])


def _concepts(values: dict[int, dict]):
    """values: file_year -> {concept: value} for a single parcel 'T'."""
    out = {}
    for y, v in values.items():
        row = {c: "" for c in CROSSWALK} | {k: str(x) for k, x in v.items()}
        df = pd.DataFrame([row]).astype("string")
        df["tk"] = "T"
        out[y] = df
    return out


def test_join_rolls_fallback_only_from_earlier_rolls():
    base = {"assessed_total": 100, "assessed_land": 20, "land_use": "8810", "year_built": 1950}
    concepts = _concepts({
        2021: base | {"bldg_area": 1200},
        2022: base,                              # bldg_area missing (excluded year)
        2023: base,                              # missing again in roll Y-1
        2024: base | {"bldg_area": 9999},        # later roll: must never be used
    })
    lookup = pd.DataFrame({"file_year": [2021, 2022, 2023, 2024], "assessment_year": [2021, 2022, 2023, 2024],
                           "status": ["confirmed"] * 4})
    j = join_rolls(pd.DataFrame({"sale_id": ["x"], "tk": ["T"], "sale_year": [2024]}), concepts, lookup)
    r = j.iloc[0]
    assert r.bldg_area == 1200 and r.bldg_area_src_roll == 2021
    assert r.prior_assessed_total == 100 and r.bench_roll == 2024
    assert r.baths_missing


def test_join_rolls_requires_confirmed_feature_roll():
    concepts = _concepts({2023: {"assessed_total": 1, "land_use": "8810"}})
    lookup = pd.DataFrame({"file_year": [2023], "assessment_year": [2023], "status": ["inferred_between"]})
    with pytest.raises(ValueError):
        join_rolls(pd.DataFrame({"sale_id": ["x"], "tk": ["T"], "sale_year": [2024]}), concepts, lookup)
