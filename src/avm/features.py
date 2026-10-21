"""Model features from roll Y-1 (via the Stage 2 join), shared by Stages 4-5.

Every feature is listed in FEATURE_DOC with why it is included and whether
it could act as a proxy for a protected class (rule 7). No demographics.

Time: t = months since Jan 2014 (mid-month for month-only dates) for the
trend, and month of year for seasonality. Models predict at a chosen trend
point with a NEUTRAL month-of-year effect (CLAUDE.md).
"""

import numpy as np
import pandas as pd

from avm.location import attach_locations, fold_small_units

T0 = pd.Timestamp("2014-01-01")
MIN_LEVEL_SALES = 30

FEATURE_DOC = {
    "l_area": ("log finished sq ft (roll Y-1)", "size; no proxy concern"),
    "l_lot": ("log lot sq ft; lot_missing flags a filled median", "size; no proxy concern"),
    "age_bin": ("sale year - year built, binned (roll Y-1)", "building vintage; correlates with "
                "neighborhood history, monitored in Stage 8"),
    "beds": ("bedrooms, clipped 1-7 (MPROP only, never the sales file)", "no proxy concern"),
    "full_baths": ("full baths (2019+ rolls; 0 + baths_missing before)", "no proxy concern"),
    "half_baths": ("half baths (as above)", "no proxy concern"),
    "stories": ("stories category (1, 1.5, 2, 2.5+, missing)", "no proxy concern"),
    "style": ("assessor building type code; codes with < 30 training sales -> other", "no proxy concern"),
    "duplex": ("roll Y-1 land use 8820", "property type; no proxy concern"),
    "loc": ("location fixed effect: assessor neighborhood or census tract, units with < 30 "
            "training sales folded into the assessor region", "LOCATION: allowed by rule 7 but can "
            "proxy for protected-class composition; equity audited in Stage 8"),
    "age": ("numeric age (GBM; the hedonic uses age_bin)", "as age_bin"),
    "stories_num": ("numeric stories, NaN if missing (GBM)", "no proxy concern"),
    "l_prior_total": ("log roll Y-1 total assessed value: ASSESSMENT-INFORMED variant only (rule 6)",
                      "inherits any bias in the assessor's values; equity audited in Stage 8"),
    "l_prior_land": ("log roll Y-1 land assessed value: ASSESSMENT-INFORMED variant only", "as l_prior_total"),
    "t": ("months since Jan 2014: market trend (spline in the hedonic)", "time; no proxy concern"),
    "moy": ("month of year: seasonality (sum-to-zero; neutral at prediction)", "time; no proxy concern"),
}


def months_since_t0(s: pd.DataFrame) -> pd.Series:
    d = s.sale_date + pd.to_timedelta(np.where(s.date_precision == "month", 14, 0), unit="D")
    return (d - T0).dt.days / 30.4375


def build_frame(s: pd.DataFrame, loc_unit: str, train_mask: pd.Series,
                baths_mode: str = "missing") -> pd.DataFrame:
    """Model frame for a set of sales. train_mask marks the rows whose counts
    define category folding (so held-out rows never shape the categories).

    baths_mode: "missing" (pre-2019 baths = 0 + baths_missing, the default)
    or "raw_era" (pre-2019 raw values + baths_pre2019 era flag; Stage 5
    sensitivity)."""
    s = attach_locations(s)
    f = pd.DataFrame(index=s.index)
    f["sale_id"] = s.sale_id
    f["lp"] = np.log(s.sale_price)
    f["l_area"] = np.log(s.bldg_area.where(s.bldg_area > 0))
    lot = np.log(s.lot_area.where(s.lot_area > 0))
    f["lot_missing"] = lot.isna().astype(int)
    f["l_lot"] = lot.fillna(lot[train_mask].median())
    age = (s.sale_year - s.year_built).clip(lower=0)
    f["age_bin"] = pd.cut(age, [-1, 20, 40, 60, 80, 100, 120, 1000],
                          labels=["0-20", "21-40", "41-60", "61-80", "81-100", "101-120", "120+"]).astype(str)
    f["age"] = age
    f["beds"] = s.bedrooms.clip(1, 7)
    if baths_mode == "missing":
        f["baths_missing"] = s.baths_missing.astype(int)
        f["full_baths"] = s.full_baths.fillna(0).clip(upper=5)
        f["half_baths"] = s.half_baths.fillna(0).clip(upper=3)
    elif baths_mode == "raw_era":
        f["baths_missing"] = (s.baths_era == "pre2019").astype(int)  # era flag
        f["full_baths"] = s.full_baths.fillna(s.full_baths_pre2019_raw).fillna(0).clip(upper=5)
        f["half_baths"] = s.half_baths.fillna(s.half_baths_pre2019_raw).fillna(0).clip(upper=3)
    else:
        raise ValueError(f"baths_mode {baths_mode!r}")
    st = s.nr_stories
    f["stories_num"] = st
    f["stories"] = np.select([st.isna(), st <= 1, st <= 1.5, st <= 2], ["missing", "1", "1.5", "2"], "2.5+")
    style = s.bldg_type.fillna("missing").astype(str)
    n_style = style[train_mask].value_counts()
    f["style"] = style.where(style.isin(n_style[n_style >= MIN_LEVEL_SALES].index), "other")
    f["duplex"] = (s.land_use_y1 == "8820").astype(int)
    unit = s.neighborhood if loc_unit == "neighborhood" else s.tract
    f["loc"] = fold_small_units(unit, s.region, train_mask)
    f["region"] = s.region
    f["t"] = months_since_t0(s)
    f["moy"] = s.sale_date.dt.month
    f["sale_year"] = s.sale_year
    # Assessment-informed variant only (rule 6): roll Y-1 values, known before Y.
    f["l_prior_total"] = np.log(s.prior_assessed_total.where(s.prior_assessed_total > 0))
    f["l_prior_land"] = np.log(s.prior_assessed_land.where(s.prior_assessed_land > 0))
    return f


REQUIRED = ["lp", "l_area", "beds", "age_bin"]


def usable(f: pd.DataFrame) -> pd.Series:
    return f[REQUIRED].notna().all(axis=1) & (f.age_bin != "nan")


def usable_x(f: pd.DataFrame) -> pd.Series:
    """Features complete, price not required: rows to SCORE (e.g. parcels
    valued prospectively, which have no sale)."""
    return f[[c for c in REQUIRED if c != "lp"]].notna().all(axis=1) & (f.age_bin != "nan")
