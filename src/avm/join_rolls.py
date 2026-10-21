"""Leakage-safe join of sales to MPROP rolls, through the assessment-year
lookup (confirmed rolls only).

For a sale in year Y:
  features         roll Y-1 (valuation date Jan 1 Y-1, published before Y)
  prior_assessed   roll Y-1 assessed value (assessment-informed variant)
  benchmark        roll Y assessed value (the assessor's value to compare)

Fallback: when a characteristic is missing or excluded in roll Y-1, take it
from the nearest EARLIER confirmed roll (Y-2, Y-3, ... up to MAX_LOOKBACK
rolls back) and record which roll it came from in <feature>_src_roll. Never
from a later roll. Assessed values and land use never fall back.
"""

import pandas as pd

from avm.io import to_number
from avm.rolls import file_year_for

# Characteristics read from roll Y-1, with optional fallback.
FEATURES_NUMERIC = ["year_built", "bldg_area", "bedrooms", "full_baths", "half_baths",
                    "rooms", "nr_stories", "nr_units", "lot_area"]
FEATURES_TEXT = ["bldg_type", "neighborhood"]
FALLBACK = FEATURES_NUMERIC + FEATURES_TEXT
MAX_LOOKBACK = 3


def _roll(concepts: dict, lookup: pd.DataFrame, assessment_year: int) -> pd.DataFrame | None:
    try:
        f = file_year_for(assessment_year, lookup)
    except (KeyError, ValueError):
        return None
    c = concepts[f].dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
    out = pd.DataFrame(index=c.index)
    for k in FEATURES_NUMERIC + ["assessed_total", "assessed_land"]:
        out[k] = to_number(c[k], f"roll {assessment_year} {k}")
    for k in FEATURES_TEXT + ["land_use"]:
        out[k] = c[k].where(c[k] != "")
    return out


def join_rolls(sales: pd.DataFrame, concepts: dict, lookup: pd.DataFrame) -> pd.DataFrame:
    """sales needs sale_id, tk, sale_year. Returns one row per sale_id with
    feature, benchmark and provenance columns. Raises if any feature source
    roll is not earlier than the sale year."""
    out = []
    for y, s in sales.groupby("sale_year"):
        y = int(y)
        r1 = _roll(concepts, lookup, y - 1)
        if r1 is None:
            raise ValueError(f"sales {y}: roll {y - 1} not confirmed in lookup; cannot build features")
        rb = _roll(concepts, lookup, y)
        tk = s.set_index("sale_id").tk
        o = pd.DataFrame(index=tk.index)
        o["in_roll_y1"] = tk.isin(r1.index).to_numpy()
        o["land_use_y1"] = r1.land_use.reindex(tk).to_numpy()
        o["prior_assessed_total"] = r1.assessed_total.reindex(tk).to_numpy()
        o["prior_assessed_land"] = r1.assessed_land.reindex(tk).to_numpy()
        for k in FALLBACK:
            o[k] = r1[k].reindex(tk).to_numpy()
            o[f"{k}_src_roll"] = pd.Series(y - 1, index=o.index).where(o[k].notna())
        earlier = [(a, _roll(concepts, lookup, a)) for a in range(y - 2, y - 2 - MAX_LOOKBACK, -1)]
        for a, ra in earlier:
            if ra is None:
                continue
            for k in FALLBACK:
                need = o[k].isna()
                if not need.any():
                    continue
                fill = ra[k].reindex(tk[need]).to_numpy()
                o.loc[need, k] = fill
                o.loc[need & o[k].notna(), f"{k}_src_roll"] = a
        o["in_roll_y"] = tk.isin(rb.index).to_numpy() if rb is not None else False
        o["land_use_y"] = rb.land_use.reindex(tk).to_numpy() if rb is not None else pd.NA
        o["bench_assessed_total"] = rb.assessed_total.reindex(tk).to_numpy() if rb is not None else pd.NA
        o["bench_assessed_land"] = rb.assessed_land.reindex(tk).to_numpy() if rb is not None else pd.NA
        o["bench_roll"] = y if rb is not None else pd.NA
        out.append(o)
    j = pd.concat(out)
    sale_year = sales.set_index("sale_id").sale_year.reindex(j.index)
    for k in FALLBACK:
        src = j[f"{k}_src_roll"]
        if (src.notna() & (src >= sale_year)).any():
            raise AssertionError(f"leakage: {k} taken from a roll not earlier than the sale year")
    # Pre-2019 baths are excluded (2019 data cleanup, not a mappable recode):
    # explicit indicator, plus the raw pre-2019 value and era for the Stage 5
    # sensitivity test (missing-only vs raw + era flag).
    j["baths_missing"] = j.full_baths.isna() | j.half_baths.isna()
    return j.reset_index().rename(columns={"index": "sale_id"})


BATHS_CLEANUP_YEAR = 2019


def baths_pre2019_raw(sales: pd.DataFrame, concepts: dict, lookup: pd.DataFrame) -> pd.DataFrame:
    """Raw roll Y-1 bath counts for sales whose feature roll predates the 2019
    cleanup, and the era of the feature roll. Never from any other roll."""
    out = []
    for y, s in sales.groupby("sale_year"):
        y = int(y)
        o = pd.DataFrame({"sale_id": s.sale_id.to_numpy()})
        o["baths_era"] = "pre2019" if y - 1 < BATHS_CLEANUP_YEAR else "2019plus"
        o["full_baths_pre2019_raw"] = float("nan")
        o["half_baths_pre2019_raw"] = float("nan")
        if y - 1 < BATHS_CLEANUP_YEAR:
            c = concepts[file_year_for(y - 1, lookup)].dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
            for k in ("full_baths", "half_baths"):
                o[f"{k}_pre2019_raw"] = to_number(c[f"{k}_raw_excluded"], f"roll {y - 1} {k} raw") \
                    .reindex(s.tk).to_numpy()
        out.append(o)
    return pd.concat(out, ignore_index=True)
