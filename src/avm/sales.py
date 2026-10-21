"""City sales files (2002-2025) normalized to one schema.

Two raw layouts: 2002-2018 (Taxkey, Fin_sqft, Nr_of_rms, month-only
Sale_date "YYYY-MM") and 2019-2025 (PropertyID, taxkey, FinishedSqft, Rooms,
day-precision Sale_date "M/D/YYYY" or "YYYY-MM-DD").

Sales-file building characteristics are as of the file extract, not the sale
(Stage 1 finding), and are never model features. Bedrooms come from MPROP only.
"""

import re

import numpy as np
import pandas as pd

from avm.config import DATA_DIR
from avm.io import is_null, normalize_taxkey, read_raw_csv, to_number

SALES_CROSSWALK = {
    "taxkey": ["taxkey", "Taxkey"],
    "address": ["Address"],
    "prop_type": ["PropType"],
    "sale_date_raw": ["Sale_date"],
    "sale_price": ["Sale_price"],
    "units": ["Units"],
    "nbhd": ["nbhd", "Nbhd"],
    "style": ["Style"],
    "bldg_area": ["FinishedSqft", "Fin_sqft"],
    "year_built": ["Year_Built"],
}
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
DAY_FORMATS = ["%m/%d/%Y", "%Y-%m-%d"]  # both seen in 2019-2025 files


def parse_sale_dates(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Returns (date, precision). Month-only dates map to the 1st of the month
    with precision 'month'; callers must not treat them as exact days."""
    month = raw.str.match(MONTH_RE).fillna(False)
    d = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    d[month] = pd.to_datetime(raw[month], format="%Y-%m")
    for fmt in DAY_FORMATS:
        todo = ~month & d.isna()
        d[todo] = pd.to_datetime(raw[todo], format=fmt, errors="coerce")
    bad = ~month & d.isna() & ~is_null(raw)
    if bad.any():
        raise ValueError(f"unparseable sale dates: {raw[bad].head().tolist()}")
    return d, pd.Series(month.map({True: "month", False: "day"}), index=raw.index)


def parse_price(raw: pd.Series, label: str) -> pd.Series:
    """Prices come as 675000, 950000.00 or "$357,000" (2020-2021 files).
    Strip currency formatting explicitly; any non-blank value that still
    fails to parse raises (a silent coerce once blanked all 2020-2021 prices)."""
    return to_number(raw, f"{label} sale price")


def load_file(key: str, rel_path: str) -> pd.DataFrame:
    df = read_raw_csv(DATA_DIR / rel_path)
    blank = df.apply(is_null).all(axis=1)
    df = df[~blank]
    out = {}
    for concept, cands in SALES_CROSSWALK.items():
        hit = [c for c in cands if c in df.columns]
        if not hit:
            raise KeyError(f"{key}: no column for {concept!r} among {cands}")
        out[concept] = df[hit[0]]
    s = pd.DataFrame(out)
    s["tk"] = normalize_taxkey(s["taxkey"])
    s["sale_date"], s["date_precision"] = parse_sale_dates(s["sale_date_raw"])
    s["sale_price"] = parse_price(s["sale_price"], key)
    s["source_key"] = key
    s = s.reset_index(drop=True)
    # Stable row id: file + row position after blank rows are dropped.
    s["sale_id"] = key.replace("sales/", "") + "-" + s.index.astype(str).str.zfill(5)
    s.attrs["blank_rows"] = int(blank.sum())
    return s


FLIP_MONTHS = 12


def repeat_sale_flags(s: pd.DataFrame) -> pd.DataFrame:
    """Flag repeat sales of the same tax key. Input needs tk, sale_date,
    date_precision; it should already be free of exact duplicate rows.

    resale_lt_12m         previous sale of this parcel < 12 months earlier
                          (days when both dates are day-precision, else
                          calendar months < 12)
    resale_12m_ambiguous  month-only dates exactly 12 months apart: cannot
                          tell < from >= 12 months, so not guessed
    repeat_same_period    previous sale on the same day (or same month when
                          either date is month-only)
    same_period_pair      this sale or the next one is a repeat_same_period
    first_sale_of_flip    the next sale of this parcel is a resale_lt_12m
    """
    s = s.sort_values(["tk", "sale_date", "sale_id"]).copy()
    g = s.groupby("tk", sort=False)
    prev_date, prev_prec = g.sale_date.shift(), g.date_precision.shift()
    both_day = (s.date_precision == "day") & (prev_prec == "day")
    gap_days = (s.sale_date - prev_date).dt.days
    gap_months = ((s.sale_date.dt.year - prev_date.dt.year) * 12
                  + (s.sale_date.dt.month - prev_date.dt.month))
    has_prev = prev_date.notna()
    same = has_prev & np.where(both_day, gap_days == 0, gap_months == 0)
    lt12 = has_prev & np.where(both_day, gap_days < 365, gap_months < FLIP_MONTHS) & ~same
    s["prev_sale_gap_days"] = gap_days.where(both_day)
    s["prev_sale_gap_months"] = gap_months
    s["resale_lt_12m"] = lt12
    s["resale_12m_ambiguous"] = has_prev & ~both_day & (gap_months == FLIP_MONTHS)
    s["repeat_same_period"] = same
    g = s.groupby("tk", sort=False)
    s["first_sale_of_flip"] = g.resale_lt_12m.shift(-1, fill_value=False)
    # Both members of a same-period pair (the repeat and the sale before it).
    s["same_period_pair"] = same | g.repeat_same_period.shift(-1, fill_value=False)
    return s.sort_index()
