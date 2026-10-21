"""Stage 2: clean SF/duplex sales table with arm's-length filters, flags and a
leakage-safe join to MPROP rolls.

Usage:
    python -m avm.clean_sales --sample     # sales 2024-2025
    python -m avm.clean_sales              # sales 2014-2025 (train 2014-2024, test 2025)

Filters (rows removed, logged by sale year):
  F1 sale date outside its file's year
  F2 exact duplicate rows across files (tax key, date, price)
  F3 missing tax key, date or price
  F4 price < MIN_PRICE
  F5 property type not "Residential"
  F6 no roll Y-1 record for the tax key
  F7 roll Y-1 land use not SF (8810) or duplex (8820)
  F8 new construction or no building in roll Y-1
  F9 multi-parcel sale: same date and price on >1 parcel AND all parcels
     within MULTI_RADIUS_M (3+ parcel groups beyond that are flagged only)
  F10 resale < 12 months after the previous sale of the parcel (flip)
  F11 two sales of one parcel in the same day (month, if month-only)

Flags (kept, for later sensitivity checks):
  first_sale_of_flip, resale_12m_ambiguous, deed_* (from roll Y conveyance
  data, only where verifiable), permit_* (significant permit between roll
  Y-1's valuation date and the sale), ratio_outlier_iaao (benchmark ratio,
  3x IQR by year), ppsf_outlier (price per sq ft, 3x IQR within
  neighborhood-year; for training), land_use_changed (roll Y vs Y-1).
"""

import argparse
import re

import numpy as np
import pandas as pd

from avm.acquire import load_manifest
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.geo import max_pairwise_distance_m, parcel_centroids
from avm.io import parse_dates
from avm.join_rolls import baths_pre2019_raw, join_rolls
from avm.mprop import RES_LAND_USE, load_year
from avm.permits import full_coverage_start, load_permits, normalize_address
from avm.progress import log, step
from avm.rolls import file_year_for, load_lookup
from avm.sales import load_file, repeat_sale_flags

TRAIN_YEARS = list(range(2014, 2025))
TEST_YEAR = 2025
SAMPLE_YEARS = [2024, 2025]
MIN_PRICE = 10_000
MULTI_RADIUS_M = 200
MULTI_MIN_PARCELS = 3
ROUND_PRICE_STEP = 5_000  # 94% of 2024-2025 coincidence groups were multiples of $5,000
IQR_K = 3.0
PPSF_MIN_GROUP = 20
MARKET_DEEDS = {"WD", "SW", "D", "LC", "LD", "PR", "PD", "TD", "AD", "CD"}
STEP_TIMEOUT_S = 600
SALES_FILE_CHARS = ["units", "nbhd", "style", "bldg_area", "year_built"]


class FilterLog:
    """Rows in / removed / out per filter and sale year."""

    def __init__(self):
        self.rows = []
        self.removed = []  # per-row (sale_id, tk, sale_year, code): lets Stage 8 cut removal rates by group

    def apply(self, df: pd.DataFrame, drop: pd.Series, code: str, desc: str) -> pd.DataFrame:
        drop = drop.reindex(df.index).fillna(False).astype(bool)
        gone = df[drop]
        self.removed.append(pd.DataFrame({"sale_id": gone.sale_id.to_numpy(), "tk": gone.tk.to_numpy(),
                                          "sale_year": gone.sale_year.to_numpy(), "code": code}))
        for y, g in df.groupby("sale_year"):
            d = drop[g.index]
            self.rows.append({"code": code, "filter": desc, "sale_year": int(y),
                              "rows_in": len(g), "removed": int(d.sum()), "rows_out": int((~d).sum())})
        log(f"      {code} {desc}: removed {int(drop.sum()):,} of {len(df):,}")
        return df[~drop]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def iqr_flag(x: pd.Series, k: float = IQR_K) -> pd.Series:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (x < lo) | (x > hi)


# ------------------------------------------------------------- multi-parcel ---

def multi_parcel_groups(s: pd.DataFrame, cents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group sales that share a date and price across different tax keys.
    Month-only dates group by month. Returns (per-row columns, group table)."""
    period = np.where(s.date_precision == "day", s.sale_date.dt.strftime("%Y-%m-%d"),
                      s.sale_date.dt.strftime("%Y-%m"))
    key = pd.Series(period, index=s.index) + "|" + s.sale_price.astype("int64").astype(str)
    n_tk = s.groupby(key).tk.transform("nunique")
    cand = s[n_tk > 1].assign(mp_key=key[n_tk > 1])
    xy = cents.set_index("tk")[["x_m", "y_m"]]
    groups = []
    for k, g in cand.groupby("mp_key"):
        parcels = g.drop_duplicates("tk")
        dist = max_pairwise_distance_m(xy.reindex(parcels.tk))
        groups.append({
            "mp_key": k, "sale_year": int(g.sale_year.iloc[0]),
            "date_precision": g.date_precision.iloc[0], "n_parcels": len(parcels), "n_rows": len(g),
            "prop_types": "+".join(sorted(g.prop_type.fillna("?").unique())),
            "n_residential_rows": int((g.prop_type == "Residential").sum()),
            "max_dist_m": round(dist, 1) if dist == dist else None,
        })
    gt = pd.DataFrame(groups, columns=["mp_key", "sale_year", "date_precision", "n_parcels", "n_rows",
                                       "prop_types", "n_residential_rows", "max_dist_m"])
    gt["rule_a"] = True
    gt["rule_b_within_radius"] = gt.max_dist_m.notna() & (gt.max_dist_m <= MULTI_RADIUS_M)
    gt["rule_c_3plus"] = gt.n_parcels >= MULTI_MIN_PARCELS
    price = gt.mp_key.str.split("|").str[1].astype("int64")
    gt["round_price"] = price % ROUND_PRICE_STEP == 0
    gt["rule_c_3plus_nonround"] = gt.rule_c_3plus & ~gt.round_price
    gt["distance_unknown"] = gt.max_dist_m.isna()
    # Drop only (b), all years. 3+ parcel groups beyond 200 m look like
    # coincidences in every year (day-precision: 233 groups, median 11.7 km
    # apart, 1.7% within 1 km, 93% round prices), so (c) is a flag only:
    # 3+ parcels at a price that is not a $1,000 multiple (Owen, 2026-10-18).
    gt["flag_3plus_not_1000"] = gt.rule_c_3plus & (price % 1_000 != 0) & ~gt.rule_b_within_radius
    gt["drop"] = gt.rule_b_within_radius
    row = pd.DataFrame(index=s.index)
    row["mp_key"] = key.where(n_tk > 1)
    m = gt.set_index("mp_key")
    for c in ["n_parcels", "max_dist_m", "rule_b_within_radius", "rule_c_3plus", "flag_3plus_not_1000", "drop"]:
        row[f"mp_{c}"] = row.mp_key.map(m[c])
    row["mp_drop"] = row.mp_drop.fillna(False).astype(bool)
    row["mp_flag_3plus_not_1000"] = row.mp_flag_3plus_not_1000.fillna(False).astype(bool)
    return row, gt


def multi_parcel_summary(gt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in gt.groupby("sale_year"):
        for rule, mask in [("a_same_date_price", g.rule_a), ("b_within_200m", g.rule_b_within_radius),
                           ("c_3plus_parcels", g.rule_c_3plus),
                           ("c_3plus_nonround_price", g.rule_c_3plus_nonround),
                           ("flag_3plus_not_1000_kept", g.flag_3plus_not_1000),
                           ("dropped", g["drop"]),
                           ("a_only_kept_as_coincidence", ~g["drop"] & ~g.distance_unknown),
                           ("a_distance_unknown_not_3plus", ~g["drop"] & g.distance_unknown)]:
            gg = g[mask]
            rows.append({"sale_year": y, "date_precision": g.date_precision.iloc[0], "rule": rule,
                         "groups": len(gg), "rows": int(gg.n_rows.sum()),
                         "residential_rows": int(gg.n_residential_rows.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- flags ---

def deed_flags(s: pd.DataFrame, concepts: dict, lookup: pd.DataFrame) -> pd.DataFrame:
    """Deed type from roll Y where its conveyance year-month equals the sale's.
    Verifiable only where roll Y conveyance dates are not corrupt."""
    out = pd.DataFrame(index=s.index, columns=["deed_type", "deed_verifiable", "deed_non_market"])
    out["deed_verifiable"] = False
    out["deed_non_market"] = False
    for y, g in s.groupby("sale_year"):
        try:
            f = file_year_for(int(y), lookup)
        except (KeyError, ValueError):
            continue
        m = concepts[f].dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
        conv, _ = parse_dates(m["convey_date"])
        cd = pd.Series(conv.reindex(g.tk).to_numpy(), index=g.index)
        match = (cd.dt.year == g.sale_date.dt.year) & (cd.dt.month == g.sale_date.dt.month)
        dt = pd.Series(m["convey_type"].reindex(g.tk).to_numpy(), index=g.index).str.strip()
        out.loc[g.index, "deed_verifiable"] = match.to_numpy()
        out.loc[g.index, "deed_type"] = dt.where(match).to_numpy()
        out.loc[g.index, "deed_non_market"] = (match & ~dt.isin(MARKET_DEEDS)).to_numpy()
    return out


def permit_flags(s: pd.DataFrame, permits: pd.DataFrame) -> pd.DataFrame:
    """Significant permit issued after roll Y-1's valuation date (Jan 1 Y-1)
    and on or before the sale date (month-only dates: end of month)."""
    addr = normalize_address(s.address)
    end = s.sale_date.where(s.date_precision == "day", s.sale_date + pd.offsets.MonthEnd(0))
    start = pd.to_datetime((s.sale_year - 1).astype(str) + "-01-01")
    sales = pd.DataFrame({"addr": addr, "start": start, "end": end}, index=s.index).reset_index()
    j = sales.merge(permits, left_on="addr", right_on="address_norm", how="inner")
    j = j[(j.issued > j.start) & (j.issued <= j.end)]
    out = pd.DataFrame(index=s.index)
    out["permit_any_residential"] = s.index.isin(j[j.residential]["index"])
    out["permit_significant"] = s.index.isin(j[j.significant]["index"])
    out["permit_window_covered"] = (start >= full_coverage_start(permits)).to_numpy()
    return out


def ratio_and_ppsf_flags(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    ratio = s.bench_assessed_total / s.sale_price
    out["bench_ratio"] = ratio
    out["ratio_outlier_iaao"] = False
    for y, g in s.groupby("sale_year"):
        r = ratio[g.index].dropna()
        out.loc[r.index, "ratio_outlier_iaao"] = iqr_flag(r).to_numpy()
    lp = np.log(s.sale_price / s.bldg_area.where(s.bldg_area > 0))
    out["ln_ppsf"] = lp
    out["ppsf_outlier"] = False
    out["ppsf_group"] = ""
    for (y, nb), g in s.groupby(["sale_year", s.neighborhood.fillna("NA")]):
        x = lp[g.index].dropna()
        if len(x) >= PPSF_MIN_GROUP:
            out.loc[x.index, "ppsf_outlier"] = iqr_flag(x).to_numpy()
            out.loc[x.index, "ppsf_group"] = "neighborhood-year"
    for y, g in s.groupby("sale_year"):  # small neighborhoods: year-wide fences
        rest = g.index[out.loc[g.index, "ppsf_group"] == ""]
        x = lp[rest].dropna()
        allx = lp[g.index].dropna()
        if len(x):
            q1, q3 = allx.quantile(0.25), allx.quantile(0.75)
            lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
            out.loc[x.index, "ppsf_outlier"] = ((x < lo) | (x > hi)).to_numpy()
            out.loc[x.index, "ppsf_group"] = "year (neighborhood < 20 sales)"
    return out


# -------------------------------------------------------------------- main ---

def build(years: list[int], out_dir, out_parquet) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    lookup = load_lookup()
    paths = dict(zip(manifest.key, manifest.local_path))
    flog = FilterLog()

    with step("load MPROP rolls", STEP_TIMEOUT_S):
        need = {y for y in range(min(years) - 1 - 3, max(years) + 1)}
        concepts = {}
        for y in sorted(need):
            k = f"mprop_hist/{y}"
            if k in paths:
                concepts[y], _ = load_year(y, paths[k])

    with step("load sales (with one earlier year of history for flips)", STEP_TIMEOUT_S):
        hist_years = range(min(years) - 1, max(years) + 1)
        raw = []
        blank = {}
        for y in hist_years:
            s = load_file(f"sales/{y}", paths[f"sales/{y}"])
            # Sales-file characteristics are as of the extract, never features:
            # suffix them so they cannot collide with roll features.
            s = s.rename(columns={c: f"{c}_sales" for c in SALES_FILE_CHARS})
            s["sale_year"] = y  # file year
            blank[y] = s.attrs["blank_rows"]
            raw.append(s)
        s = pd.concat(raw, ignore_index=True)
        s.index = pd.RangeIndex(len(s))
        log(f"      loaded {len(s):,} rows; blank rows dropped at load: {blank}")

    in_years = s.sale_year.isin(years)
    flog.rows += [{"code": "F0", "filter": "loaded (blank rows already dropped)", "sale_year": y,
                   "rows_in": int((s.sale_year == y).sum()) + blank[y], "removed": blank[y],
                   "rows_out": int((s.sale_year == y).sum())} for y in years]

    # History-wide cleaning first, so flips and multi-parcel groups see all sales.
    s = flog.apply(s, s.sale_date.dt.year != s.sale_year, "F1", "sale date outside file year")
    s = flog.apply(s, s.duplicated(["tk", "sale_date", "sale_price"]), "F2", "exact duplicate row")
    s = flog.apply(s, s.tk.isna() | s.sale_date.isna() | s.sale_price.isna(), "F3",
                   "missing tax key, date or price")
    s = flog.apply(s, s.sale_price < MIN_PRICE, "F4", f"price < ${MIN_PRICE:,}")

    with step("repeat-sale and multi-parcel flags", STEP_TIMEOUT_S):
        s = repeat_sale_flags(s)
        cents = parcel_centroids()
        mp_rows, mp_groups = multi_parcel_groups(s, cents)
        s = s.join(mp_rows)

    # Restrict to the requested years and record flog only for those.
    s = s[s.sale_year.isin(years)].copy()
    flog.rows = [r for r in flog.rows if r["sale_year"] in years]
    mp_groups = mp_groups[mp_groups.sale_year.isin(years)]

    s = flog.apply(s, s.prop_type != "Residential", "F5", "property type not Residential")
    with step("join rolls (features Y-1, benchmark Y)", STEP_TIMEOUT_S):
        keys = s[["sale_id", "tk", "sale_year"]]
        j = join_rolls(keys, concepts, lookup).merge(baths_pre2019_raw(keys, concepts, lookup),
                                                     on="sale_id", validate="one_to_one")
        s = s.merge(j, on="sale_id", how="left", validate="one_to_one").set_index(s.index)
    s = flog.apply(s, ~s.in_roll_y1, "F6", "no roll Y-1 record")
    s = flog.apply(s, ~s.land_use_y1.isin(RES_LAND_USE), "F7", "roll Y-1 land use not SF/duplex")
    no_bldg = (s.year_built.fillna(0) <= 0) & (s.bldg_area.fillna(0) <= 0)
    new_build = s.year_built >= s.sale_year - 1
    s = flog.apply(s, no_bldg | new_build, "F8", "new construction or no building in roll Y-1")
    s = flog.apply(s, s.mp_drop, "F9", "multi-parcel sale (same date and price, all parcels within 200 m)")
    s = flog.apply(s, s.resale_lt_12m, "F10", "resale < 12 months (flip)")
    s = flog.apply(s, s.same_period_pair, "F11", "two sales of one parcel in the same day/month")

    with step("flags: deed, permits, ratio, price per sq ft", STEP_TIMEOUT_S):
        s = s.join(deed_flags(s, concepts, lookup))
        permits = load_permits()
        s = s.join(permit_flags(s, permits))
        s = s.join(ratio_and_ppsf_flags(s))
        s["land_use_changed"] = s.land_use_y.notna() & (s.land_use_y != s.land_use_y1)
        s["has_benchmark"] = s.bench_assessed_total.notna() & (s.bench_assessed_total > 0)
        s["split"] = np.where(s.sale_year == TEST_YEAR, "test",
                              np.where(s.sale_year.isin(TRAIN_YEARS), "train", "other"))

    keep = [c for c in s.columns if c not in {"taxkey", "sale_date_raw", "mp_key"}]
    s = s[keep]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(out_parquet, index=False)
    log(f"wrote {out_parquet} ({len(s):,} rows)")

    flog_df = flog.frame()
    removed = pd.concat(flog.removed, ignore_index=True)
    removed = removed[removed.sale_year.isin(years)]
    removed.to_parquet(out_parquet.with_name(out_parquet.stem + "_removed.parquet"), index=False)
    tables = {
        "filter_log": flog_df,
        "multi_parcel_groups": mp_groups,
        "multi_parcel_summary": multi_parcel_summary(mp_groups),
        "multi_parcel_by_prop_types": mp_groups.groupby(["prop_types"]).agg(
            groups=("mp_key", "size"), rows=("n_rows", "sum"), dropped_groups=("drop", "sum"),
            residential_rows=("n_residential_rows", "sum")).reset_index().sort_values("groups", ascending=False),
        "fallback_usage": fallback_usage(s),
        "flags_by_year": flags_by_year(s),
        "outlier_log": s[s.ratio_outlier_iaao | s.ppsf_outlier][
            ["sale_id", "sale_year", "sale_date", "sale_price", "bench_assessed_total", "bench_ratio",
             "bldg_area", "ln_ppsf", "neighborhood", "ppsf_group", "ratio_outlier_iaao", "ppsf_outlier"]],
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    write_summary(tables, s, out_dir, years)
    log(f"wrote tables and stage2_summary.md to {out_dir}")


def fallback_usage(s: pd.DataFrame) -> pd.DataFrame:
    from avm.join_rolls import FALLBACK
    rows = []
    for y, g in s.groupby("sale_year"):
        for k in FALLBACK:
            src = g[f"{k}_src_roll"]
            rows.append({"sale_year": y, "feature": k, "from_roll_y1": int((src == y - 1).sum()),
                         "from_earlier_roll": int((src < y - 1).sum()), "missing": int(src.isna().sum()),
                         "n": len(g)})
    return pd.DataFrame(rows)


def flags_by_year(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby("sale_year")
    return pd.DataFrame({
        "n_clean": g.size(),
        "first_sale_of_flip": g.first_sale_of_flip.sum(),
        "multi_parcel_3plus_flag": g.mp_flag_3plus_not_1000.sum(),
        "resale_12m_ambiguous": g.resale_12m_ambiguous.sum(),
        "deed_verifiable_share": g.deed_verifiable.mean().round(4),
        "deed_non_market": g.deed_non_market.sum(),
        "deed_non_market_share_of_verifiable": (g.deed_non_market.sum() / g.deed_verifiable.sum()).round(4),
        "permit_window_covered": g.permit_window_covered.mean().round(4),
        "permit_any_residential": g.permit_any_residential.sum(),
        "permit_significant": g.permit_significant.sum(),
        "has_benchmark": g.has_benchmark.sum(),
        "ratio_outlier_iaao": g.ratio_outlier_iaao.sum(),
        "ppsf_outlier": g.ppsf_outlier.sum(),
        "land_use_changed": g.land_use_changed.sum(),
        "baths_missing": g.baths_missing.sum(),
        "full_baths_pre2019_raw_present": g.full_baths_pre2019_raw.count(),
        "single_family": g.land_use_y1.agg(lambda x: int((x == "8810").sum())),
        "duplex": g.land_use_y1.agg(lambda x: int((x == "8820").sum())),
    }).reset_index()


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):  # itertuples keeps each column's dtype
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(t: dict, s: pd.DataFrame, out_dir, years) -> None:
    fl = t["filter_log"]
    piv = fl.pivot_table(index=["code", "filter"], columns="sale_year", values="removed", aggfunc="sum")
    order = sorted(piv.index, key=lambda k: int(re.sub(r"\D", "", k[0])))
    piv = piv.loc[order]
    final = fl[fl.code == fl.code.iloc[-1]].set_index("sale_year").rows_out
    loaded = fl[fl.code == "F0"].set_index("sale_year").rows_in
    piv.loc[("=", "rows loaded"), :] = loaded
    piv.loc[("=", "rows remaining"), :] = final
    piv = piv.fillna(0).astype(int)
    fb = t["fallback_usage"]
    fbp = fb.pivot_table(index="feature", columns="sale_year", values=["from_earlier_roll", "missing"])
    fbp.columns = [f"{a} {b}" for a, b in fbp.columns]
    fbp = fbp.astype(int)
    parts = [
        f"# Stage 2 summary (generated by src/avm/clean_sales.py), sales years {years[0]}-{years[-1]}",
        "## Filter log: rows removed", _md(piv.reset_index()),
        "## Multi-parcel groups by rule", _md(t["multi_parcel_summary"]),
        "## Multi-parcel groups by property-type mix", _md(t["multi_parcel_by_prop_types"].head(15)),
        "## Flags by year (kept rows)", _md(t["flags_by_year"]),
        "## Feature fallback (earlier roll used / still missing)", _md(fbp.reset_index()),
        "## Split", _md(s.groupby(["split", "sale_year"]).size().rename("rows").reset_index()),
    ]
    (out_dir / "stage2_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help=f"sales years {SAMPLE_YEARS} only")
    a = ap.parse_args()
    if a.sample:
        build(SAMPLE_YEARS, REPORTS_DIR / "stage2" / "sample", INTERIM_DIR / "sales_clean_sample.parquet")
    else:
        build(TRAIN_YEARS + [TEST_YEAR], REPORTS_DIR / "stage2", INTERIM_DIR / "sales_clean.parquet")


if __name__ == "__main__":
    main()
