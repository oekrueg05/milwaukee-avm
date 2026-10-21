"""Stage 9: divergence list, tax dollars, duplex yields (data for the page).

Usage:
    python -m avm.stage9_divergence --sample   # 500 random parcels
    python -m avm.stage9_divergence            # every SF/duplex parcel

Every SF/duplex parcel in roll 2024 is valued at Jan 1 2025 by the FROZEN
primary model (trained through 2024), with an 80% CQR range (pooled
2022-2024 calibration). Compared with roll 2025 (valuation date Jan 1 2025).
These are MODEL ESTIMATES, each labeled with its 80% range.

Relative assessment = (assessed / model) / citywide median of that ratio, so
the assessor's overall level is not divergence. A parcel is listed only if
the assessor-implied market value (assessed / citywide level) is outside the
model's 80% range AND the comparable sales (value (a), time-adjusted only)
point the same direction. Ranked by |log relative assessment|; top 1,000;
tax dollars are a column.

Tax, revenue-neutral within the SF/duplex class: fair assessment = model
value x (sum assessed / sum model); shift = (assessed - fair) x NET rate
(Owen, 2026-10-20: the school levy credit reduces each bill in proportion
to value, so the net rate is the marginal tax per assessed dollar): 20.80,
confirmed from the Milwaukee County comparative rate table. The lottery and
first dollar credits are excluded (capped value).
Group shifts are reported next to the
primary model's own Stage 8 CV bias for that group; a group is called over-
or under-assessed only if its shift exceeds that bias with CIs. Values are
never adjusted using demographics.

Duplex gross yield = 12 x 2 units x HUD FY2025 Small Area FMR (ZIP,
bedrooms per unit) / model value. FMR is a 40th-percentile rent, so yields
are conservative.
"""

import argparse
import json

import numpy as np
import pandas as pd

from avm import frames
from avm.acquire import load_manifest, sha256_of
from avm.config import DATA_DIR, INTERIM_DIR, PRIVATE_REPORTS_DIR, RAW_DIR, REPORTS_DIR
from avm.gbm import GBMX
from avm.intervals import calibrate, contain_point
from avm.io import normalize_taxkey
from avm.location import parcel_locations
from avm.progress import log, step
from avm.prospective import parcel_rows
from avm.ratio_study import N_BOOT, SEED
from avm.reporting import md
from avm.spatial import static_features
from avm.stage6_geo import fold_frame, roll_tables
from avm.comps import STRUCT, comps_for, eligible_pool
from avm.stage7_explain import shap_frame

OUT = REPORTS_DIR / "stage9"
SPEC_PATH = REPORTS_DIR / "frozen" / "model_spec.json"
YEAR = 2025
BASE_START = 2014
CAL_YEARS = [2022, 2023, 2024]
TOP_N = 1000
SAMPLE_N = 500
SOLD_2025: set = set()
SAFMR_DIR = RAW_DIR / "hud_safmr"
TAX_RATE = {
    "net_rate_per_1000": 20.80,
    "status": "CONFIRMED (Owen, 2026-10-20) from the Milwaukee County comparative rate table",
    "rate_used": "net",  # Owen, 2026-10-20: the school levy credit reduces each bill in proportion to value, so
    # the net rate is the marginal tax per assessed dollar; between-parcel shifts use it.
    "assessment_year": 2025, "tax_bill": "December 2025 (budget year 2026)",
    "gross_rate_per_1000": 22.42,
    "prior_year_net_rate_per_1000": 22.93,
    "prior_year_note": "22.93 is the City of Milwaukee's 2024 rate in the same county table, not a 2025 figure",
    "sources": [
        "Milwaukee County Treasurer's Office, Comparative Total Property Tax Rates for Milwaukee County "
        "Municipalities 2016-2025, 'dollar rate per $1,000.00 of assessed value to determine individual real estate "
        "taxes due per property', dated 2/18/2026 (franklinwi.gov copy; manifest key tax_rates_county_2016_2025): "
        "City of Milwaukee 2024 $22.93, 2025 $20.80",
        "City of Milwaukee Assessor, Tax Rate Chart (TAXRATECHART.pdf): assessment year 2025 gross 22.42 (read via "
        "a search-engine extract; the city site blocks automated access; the gross rate is shown for context only)"],
    "school_levy_credit": "included through the net rate: it reduces each bill in proportion to assessed value "
                          "(Wis. Stat. 79.10), so the net rate is the marginal tax per assessed dollar",
    "excluded": "the first dollar credit (every improved parcel) and the lottery and gaming credit (owner-occupied "
                "principal dwellings only); both are computed on value only up to a statutory cap (Wis. Stat. "
                "79.10(9)(bm), (9)(bn), (11)), so above the cap they do not grow with the assessment",
    "statute": "https://docs.legis.wisconsin.gov/statutes/statutes/79/ii/10"}
FEATURE_LABELS = {"l_area": "finished sq ft", "l_lot": "lot size", "age": "building age", "beds": "bedrooms",
                  "full_baths": "full baths", "half_baths": "half baths", "stories_num": "stories",
                  "style": "building style", "duplex": "duplex", "loc": "census tract", "x_m": "location (east-west)",
                  "y_m": "location (north-south)", "d_downtown_m": "distance to downtown",
                  "d_lake_m": "distance to Lake Michigan", "d_major_road_m": "distance to a major road",
                  "d_transit_m": "distance to transit", "d_park_m": "distance to a park",
                  "knn_lppsf": "nearby sale prices per sq ft", "knn_dist_m": "distance to nearby sales",
                  "nbhd_lppsf_24m": "neighborhood prices per sq ft", "prior_lp_detr": "this home's prior sale",
                  "months_since_prior": "time since prior sale", "has_prior_sale": "has a prior sale",
                  "n_city_200m": "city-owned parcels nearby", "n_vacant_200m": "vacant parcels nearby",
                  "lot_missing": "lot size missing", "baths_missing": "baths missing"}


def addresses() -> pd.DataFrame:
    """One display address and ZIP per tax key from the MAI (no owner names
    exist in the file). Duplexes with two numbers show a range."""
    m = load_manifest().set_index("key")
    a = pd.read_csv(DATA_DIR / m.loc["mai", "local_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    a["tk"] = normalize_taxkey(a.TAXKEY)
    a["hn"] = pd.to_numeric(a.HSE_NBR, errors="coerce")  # display only, not a model input
    a = a.dropna(subset=["tk", "hn"])
    street = (a.DIR.str.strip() + " " + a.STREET.str.strip() + " " + a.STTYPE.str.strip()).str.split().str.join(" ")
    a["street"] = street
    g = a.groupby(["tk", "street"]).agg(lo=("hn", "min"), hi=("hn", "max"), zip=("ZIP_CODE", "first")).reset_index()
    g = g.sort_values(["tk", "lo"]).drop_duplicates("tk")
    num = np.where(g.lo == g.hi, g.lo.astype(int).astype(str), g.lo.astype(int).astype(str) + "-" +
                   g.hi.astype(int).astype(str))
    return pd.DataFrame({"tk": g.tk, "address": num + " " + g.street, "zip5": g.zip.str[:5]})


def safmr() -> pd.DataFrame | None:
    files = sorted(SAFMR_DIR.glob("*.xls*"))
    if not files:
        return None
    x = pd.read_excel(files[0], dtype=str)
    x.columns = [str(c).strip() for c in x.columns]
    zipcol = next(c for c in x.columns if "zip" in c.lower())
    out = {"zip5": x[zipcol].str.zfill(5)}
    for br in range(0, 5):
        col = next((c for c in x.columns if f"SAFMR {br}BR" in c and "90%" not in c and "110%" not in c), None)
        if col is None:
            raise KeyError(f"SAFMR column for {br}BR not found in {files[0].name}: {list(x.columns)[:20]}")
        out[f"br{br}"] = pd.to_numeric(x[col].str.replace(r"[$,]", "", regex=True), errors="raise")
    return pd.DataFrame(out).drop_duplicates("zip5"), files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    out = OUT / "sample" if a.sample else OUT
    out.mkdir(parents=True, exist_ok=True)
    (OUT / "tax_rate.json").write_text(json.dumps(TAX_RATE, indent=2), encoding="utf-8")
    spec = json.loads(SPEC_PATH.read_text())
    cfg, cols = spec["models"]["primary"]["config"], spec["models"]["primary"]["features"]
    s = frames.sales()
    glob = frames.global_table(s)

    with step("parcel rows: SF/duplex in roll 2024, valued Jan 1 2025", 3600):
        rows = parcel_rows(s, YEAR)
        if a.sample:
            rows = rows.sample(SAMPLE_N, random_state=SEED)
        g2 = rows[["sale_id", "tk"]].merge(static_features(), on="tk", how="left").drop(columns="tk")
        g2 = g2.merge(roll_tables(rows), on="sale_id", how="left")
        s_ext = pd.concat([s, rows], ignore_index=True)
        glob_ext = pd.concat([glob, g2], ignore_index=True)
    with step("frame (train through 2024) and frozen primary predictions", 3600):
        fold = fold_frame(s_ext, YEAR, BASE_START, cfg["loc"], glob_ext)
        tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
        m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
        sc = fold["scored"]
        real = sc[~sc.sale_id.str.startswith("P")]
        frozen = pd.read_parquet(INTERIM_DIR / "frozen_primary_predictions_2025.parquet").set_index("sale_id").pred
        d = np.abs(np.exp(m.predict(real)) / frozen.reindex(real.sale_id).to_numpy() - 1).max()
        if not d < 1e-9:
            raise AssertionError(f"model differs from the frozen primary (max rel diff {d})")
        par = sc[sc.sale_id.str.startswith("P")].copy()
        par["model_value"] = np.exp(m.predict(par))
        log(f"      frozen model verified on 2025 sales; {len(par)} parcels scored of {len(rows)}")
        counts = {"parcels_sf_duplex_roll_2024": len(rows), "parcels_model_scored": len(par)}
    with step("80% ranges (CQR pooled 2022-2024)", 3600):
        cal = []
        for Y in CAL_YEARS:
            f = frames.fold(s, glob, Y, BASE_START, cfg["loc"])
            t = f["train"][f["train"].sale_year >= cfg["train_start"]]
            lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(t, t.y_detr).predict(f["scored"])
            hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(t, t.y_detr).predict(f["scored"])
            cal.append(pd.DataFrame({"lo": lo, "hi": hi, "y": np.log(f["scored"].target_price.to_numpy())}))
        cal = pd.concat(cal, ignore_index=True)
        q = calibrate(cal.lo.to_numpy(), cal.hi.to_numpy(), cal.y.to_numpy())["global"]
        lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(tr, tr.y_detr).predict(par) - q
        hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(tr, tr.y_detr).predict(par) + q
        lo, hi = contain_point(lo, hi, np.log(par.model_value.to_numpy()))
        par["range80_low"], par["range80_high"] = np.exp(lo), np.exp(hi)
    with step("comparable sales, value (a)", 3600):
        pool = eligible_pool(tr, s, f"{YEAR}-01-01")
        pool["adj_price"] = np.exp(pool.y_detr)
        fi = pd.read_csv(REPORTS_DIR / "stage7" / "feature_importance.csv").set_index("feature").mean_abs_shap
        w = fi.reindex(STRUCT).fillna(0)
        comps = comps_for(par, pool, w / w.sum(), tr[STRUCT].std())
        comps = comps.merge(pool[["sale_id", "adj_price"]].rename(columns={"sale_id": "comp_sale_id"}),
                            on="comp_sale_id")
        cv = comps.groupby("sale_id").agg(comp_value_a=("adj_price", "median"), n_comps=("comp_sale_id", "size"),
                                          comp_radius_mi=("comp_radius_mi", "first"),
                                          comp_window_months=("comp_window_months", "first"))
        par = par.merge(cv, left_on="sale_id", right_index=True, how="left")

    with step("divergence, tax, groups, yields", 3600):
        par = par.merge(addresses(), on="tk", how="left")
        par["address"] = par.address.fillna("address not in the city's address index")
        par["assessed_2025"] = par.bench_assessed_total
        ok = par.assessed_2025.notna() & (par.assessed_2025 > 0)
        n_no_assessment = int((~ok).sum())
        par = par[ok].copy()
        par = par.join(quality_flags(par))
        excluded = par.excluded_reason != ""
        exclusion_counts = par.excluded_reason.replace("", "(included)").value_counts().rename("parcels").reset_index()
        exclusion_counts = pd.concat([exclusion_counts, pd.DataFrame(
            [{"excluded_reason": NO_ASSESSMENT, "parcels": n_no_assessment}])], ignore_index=True)
        par_all = par
        par = par[~excluded].copy()  # list and tax pool: taxable parcels whose features match roll 2025
        level = float(np.median(par.assessed_2025 / par.model_value))
        par["relative_assessment"] = par.assessed_2025 / par.model_value / level
        par["implied_market"] = par.assessed_2025 / level
        outside = (par.implied_market < par.range80_low) | (par.implied_market > par.range80_high)
        direction = np.sign(par.implied_market - par.model_value)
        comps_dir = np.sign(par.implied_market - par.comp_value_a)
        par["listable"] = outside & par.comp_value_a.notna() & (direction == comps_dir)
        rate = TAX_RATE[f"{TAX_RATE['rate_used']}_rate_per_1000"] / 1000
        fair = par.model_value * par.assessed_2025.sum() / par.model_value.sum()
        par["fair_assessed"] = fair
        par["tax_shift_usd"] = (par.assessed_2025 - fair) * rate
        tg = pd.read_csv(REPORTS_DIR / "stage8" / "tract_groups.csv", dtype={"tract": str})
        par["tract"] = par.tk.map(parcel_locations().set_index("tk").tract)
        par = par.merge(tg[["tract", "race_group", "income_group"]], on="tract", how="left")
        par["context_flags"] = context_flags(par)
        par["special_reason"] = special_reasons(par)
        ranked = (par[par.listable].assign(abs_log=lambda d: np.abs(np.log(d.relative_assessment)))
                  .sort_values("abs_log", ascending=False))
        top = ranked[ranked.special_reason == ""].head(TOP_N)
        special = ranked[ranked.special_reason != ""]
        drivers = top_drivers(m, pd.concat([top, special]), cols)
        top = top.merge(drivers, on="sale_id", how="left")
        special = special.merge(drivers, on="sale_id", how="left")
        groups, by_year = group_shifts(par)
        yields, yinfo = duplex_yields(par)
        # Saved table: every scored parcel with its exclusion reason; list/tax columns only for included ones.
        par_all = par_all.merge(par[["sale_id", "relative_assessment", "implied_market", "listable", "fair_assessed",
                                     "tax_shift_usd", "race_group", "income_group", "tract", "context_flags",
                                     "special_reason"]],
                                on="sale_id",
                                how="left", suffixes=("_drop", ""))
        par_all = par_all[[c for c in par_all.columns if not c.endswith("_drop")]]
        par_all.to_parquet(INTERIM_DIR / ("stage9_parcels_sample.parquet" if a.sample else "stage9_parcels.parquet"),
                           index=False)
        exclusion_counts.to_csv(out / "exclusions.csv", index=False)

    keep = ["tk", "address", "zip5", "land_use_y1", "tract", "model_value", "range80_low", "range80_high",
            "assessed_2024", "assessed_2025", "relative_assessment", "implied_market", "comp_value_a", "n_comps",
            "comp_radius_mi", "comp_window_months",
            "tax_shift_usd", "context_flags", "driver_1", "driver_2", "driver_3"]
    # Parcel lists carry addresses and tax keys: written outside the repo (never committed).
    priv = PRIVATE_REPORTS_DIR / "stage9" / ("sample" if a.sample else "")
    priv.mkdir(parents=True, exist_ok=True)
    top[keep].to_csv(priv / "divergence_top.csv", index=False)
    special[keep + ["special_reason"]].to_csv(priv / "special_valuation_list.csv", index=False)
    stats = divergence_stats(par, top, special, level, q).assign(**counts)
    stats.to_csv(out / "divergence_stats.csv", index=False)
    groups.to_csv(out / "group_tax_shifts.csv", index=False)
    by_year.to_csv(out / "group_sales_check_by_year.csv", index=False)
    if yields is not None:
        yields.to_csv(out / "duplex_yield_summary.csv", index=False)
    tracts = par.groupby("tract").agg(parcels=("tk", "size"), median_relative=("relative_assessment", "median"),
                                      tax_shift_usd=("tax_shift_usd", "sum"),
                                      median_model_value=("model_value", "median")).reset_index()
    tracts.to_csv(out / "tract_summary.csv", index=False)
    write_summary(out, stats, top, special, groups, yields, yinfo, keep, a.sample)
    log(f"wrote {out / 'stage9_summary.md'}")


NO_ASSESSMENT = "no positive roll 2025 assessment"
QUALITY_RULES = {
    "exempt in roll 2025": "any exemption code in roll 2025 (pays no or partial tax)",
    "building value gone in roll 2025": "improvements < 10% of roll 2025 total while roll 2024 had a building "
                                        "(likely demolition, fire or condemnation; model features are stale)",
    "assessment collapsed": "roll 2025 total < 50% of roll 2024 total",
    "sq ft changed between rolls": "finished sq ft differs between rolls 2024 and 2025 (model features are stale)",
}


def quality_flags(par: pd.DataFrame) -> pd.DataFrame:
    """Data-quality exclusions for the divergence list and the tax pool, each
    named; the first rule that applies is the reason."""
    from avm.io import to_number
    from avm.mprop import load_year
    from avm.rolls import file_year_for, load_lookup
    m, lk = load_manifest().set_index("key"), load_lookup()
    roll = {}
    for a in (YEAR - 1, YEAR):
        f = file_year_for(a, lk)
        c = load_year(f, m.loc[f"mprop_hist/{f}", "local_path"])[0].dropna(subset=["tk"]).drop_duplicates("tk")
        roll[a] = c.set_index("tk")
    r24, r25 = roll[YEAR - 1].reindex(par.tk), roll[YEAR].reindex(par.tk)
    tot24 = to_number(r24.assessed_total, "roll 2024 total").to_numpy()
    tot25 = to_number(r25.assessed_total, "roll 2025 total").to_numpy()
    land25 = to_number(r25.assessed_land, "roll 2025 land").to_numpy()
    land24 = to_number(r24.assessed_land, "roll 2024 land").to_numpy()
    area24 = to_number(r24.bldg_area, "roll 2024 sq ft").to_numpy()
    area25 = to_number(r25.bldg_area, "roll 2025 sq ft").to_numpy()
    exempt = (r25.exempt_type.fillna("").str.strip() != "").to_numpy()
    gone = ((tot25 - land25) < 0.10 * tot25) & ((tot24 - land24) > 0.10 * tot24)  # roll 2024 had a building
    collapsed = tot25 < 0.5 * tot24
    area_chg = ~np.isnan(area24) & ~np.isnan(area25) & (area24 != area25)
    reason = np.select([exempt, gone, collapsed, area_chg], list(QUALITY_RULES), "")
    units = np.fmax(to_number(r24.nr_units, "roll 2024 units").to_numpy(),
                    to_number(r25.nr_units, "roll 2025 units").to_numpy())
    # Roll 2025 building type is excluded as a feature (renumbered), but its raw
    # value is kept as <concept>_raw_excluded and is fine for this data check.
    bt25 = r25["bldg_type_raw_excluded"] if "bldg_type_raw_excluded" in r25 else r25.bldg_type
    commercial_bt = (r24.bldg_type.fillna("").str.strip().str.match(r"^C") |
                     bt25.fillna("").str.strip().str.match(r"^C")).to_numpy()
    return pd.DataFrame({"excluded_reason": reason, "assessed_2024": tot24,
                         "land_use_2025": r25.land_use.to_numpy(), "units_max_2024_2025": units,
                         "commercial_bldg_type": commercial_bt}, index=par.index)


SPECIAL_RULES = {
    "assessment unchanged from 2024": "roll 2025 total equals roll 2024 total, while almost every assessment "
                                      "changed citywide (likely a condition-based or legally restricted value)",
    "land use changed in roll 2025": "roll 2025 land use differs from roll 2024, which the model used",
    "units inconsistent with land use": "residential units in roll 2024 or 2025 above 2 for single-family (8810) "
                                        "or above 4 for duplex (8820)",
    "commercial building type": "building type in roll 2024 or 2025 is a commercial code (starts with C), not one "
                                "of the documented residential style codes 01-22 (MPROP documentation, Appendix E)",
}
MAX_UNITS = {"8810": 2, "8820": 4}


def special_reasons(par: pd.DataFrame) -> pd.Series:
    """Likely special valuation or data issue: listed separately, never in
    the main ranking. All reasons that apply, joined with '; '. These parcels
    stay in the tax pool (they are taxable parcels)."""
    flat = (par.assessed_2025 == par.assessed_2024).to_numpy()
    lu = (par.land_use_2025.fillna("") != par.land_use_y1).to_numpy()
    units = (par.units_max_2024_2025 > par.land_use_y1.map(MAX_UNITS)).to_numpy()
    cbt = par.commercial_bldg_type.to_numpy(dtype=bool)
    names = list(SPECIAL_RULES)
    return pd.Series(["; ".join(n for n, hit in zip(names, h) if hit) for h in zip(flat, lu, units, cbt)],
                     index=par.index)


CLUSTER_N = 5
CAMPUSES = {"Marquette": "Marquette Univ", "UWM": "Univ of Wisconsin Milwaukee"}  # TIGER AREALM FULLNAME
STUDENT_AREA_M = 1207  # 0.75 mile
STUDENT_FLAG = "student-rental area (within 0.75 mi of the Marquette or UWM main campus)"
CONTEXT_FLAGS = {
    "N parcels in this tract have this exact assessment": f"{CLUSTER_N} or more included parcels in the same tract "
                                                          "share this exact roll 2025 total",
    STUDENT_FLAG: "parcel centroid within 0.75 mi of the main campus polygon (largest TIGER 2023 area landmark, "
                  "MTFCC K2540) of Marquette University or UW-Milwaukee. Income-producing student rentals carry "
                  "value the model's features cannot see, so divergence here may reflect the model (untested; comps there point the same way as the model)",
}


def campus_distance_m(par: pd.DataFrame) -> np.ndarray:
    import geopandas as gpd
    from avm.geo import METRIC_CRS
    m = load_manifest().set_index("key")
    g = gpd.read_file(f"zip://{DATA_DIR / m.loc['tiger_arealm_2023', 'local_path']}").to_crs(METRIC_CRS)
    g = g[g.MTFCC == "K2540"]
    mains = []
    for name in CAMPUSES.values():
        c = g[g.FULLNAME == name]
        if c.empty:
            raise KeyError(f"campus {name!r} not in the TIGER area landmark file")
        mains.append(c.loc[c.ALAND.idxmax(), "geometry"])
    pts = gpd.GeoSeries(gpd.points_from_xy(par.x_m, par.y_m), crs=METRIC_CRS, index=par.index)
    return np.minimum.reduce([pts.distance(geom).to_numpy() for geom in mains])


def context_flags(par: pd.DataFrame) -> pd.Series:
    """Patterns shown next to listed parcels; they do NOT move them: a group
    valuation (a development valued together) or a student-rental area."""
    same = par.groupby(["tract", "assessed_2025"]).tk.transform("size").to_numpy()
    near = campus_distance_m(par) <= STUDENT_AREA_M
    return pd.Series(["; ".join(([f"{n} parcels in this tract have this exact assessment"] if n >= CLUSTER_N else [])
                                + ([STUDENT_FLAG] if s else [])) for n, s in zip(same, near)], index=par.index)


def top_drivers(m, top: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sh = shap_frame(m, top)
    rows = []
    for sid, (_, r) in zip(top.sale_id, sh[cols].iterrows()):
        best = r.reindex(r.abs().sort_values(ascending=False).index[:3])
        rows.append({"sale_id": sid, **{f"driver_{i}": f"{FEATURE_LABELS.get(f, f)} {np.exp(v) - 1:+.0%}"
                                         for i, (f, v) in enumerate(best.items(), 1)}})
    return pd.DataFrame(rows)


CV_YEARS = [2021, 2022, 2023, 2024]
LAG_NOTE = ("consistent with assessments lagging faster appreciation in some areas, not a valuation pattern "
            "(does not survive tier-specific time adjustment)")


def _ci(d):
    return [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]


def check_a(c_lo: float, c_hi: float) -> bool:
    """(a): the bias-corrected shift's interval excludes zero."""
    return bool(c_lo > 0 or c_hi < 0)


def check_b(sales_rel: float, s_lo: float, s_hi: float, corrected: float) -> bool:
    """(b): the model-free sales check (a ratio to the city, so 1 = no gap)
    excludes one, in the same direction as the corrected shift."""
    return bool((s_lo > 1 or s_hi < 1) and np.sign(sales_rel - 1) == np.sign(corrected))


def guardrail_verdict(corrected: float, a: bool, b: bool, corrected_t: float, a_t: bool,
                      b_t: bool) -> tuple[bool, str]:
    """(c) and the verdict from the checks with citywide (a, b) and tercile
    (a_t, b_t) time adjustment. (c) also needs both corrected shifts to point
    the same way. Returns (check_c, verdict)."""
    c = bool(a_t and b_t and np.sign(corrected) == np.sign(corrected_t))
    side = "over-assessed vs market share" if corrected > 0 else "under-assessed vs market share"
    verdict = (side if a and b and c else LAG_NOTE if a and b else
               "no claim: " + ("corrected shift's CI includes zero" if not a else
                               "2025 sales check does not agree with a CI excluding one"))
    return c, verdict


def group_shifts(par: pd.DataFrame) -> pd.DataFrame:
    """Direction-aware guardrail (Owen, 2026-10-20). For each group:
    shift = median log(assessed / fair) over its homes; model bias = the frozen
    primary's mean log(pred / target) on CV 2021-2024 sales in the group minus
    its citywide mean (a model that under-values a group by b inflates its
    shift by about -b); corrected = shift + bias, with a bootstrap CI from
    independent draws of both. Sales check (model-free) = the assessor's
    median ratio on 2025 sales in the group / the city-tract median ratio.
    A claim requires ALL of: (a) the corrected CI excludes zero; (b) the sales
    check CI excludes one and points the same way; (c) (a) and (b) still hold
    when target prices use tier-specific (tercile) time adjustment. Passing
    (a) and (b) but not (c) is reported as LAG_NOTE."""
    from avm.stage3_assessor import tercile_target_prices
    from avm.time_adjust import seasonal_effects
    s = frames.sales()
    seasonal = seasonal_effects(s[s.sale_year.between(2014, 2024)])
    tt = tercile_target_prices(s, CV_YEARS + [YEAR], seasonal)
    tg = pd.read_csv(REPORTS_DIR / "stage8" / "tract_groups.csv", dtype={"tract": str})
    loc = parcel_locations().set_index("tk").tract
    p8 = pd.read_parquet(INTERIM_DIR / "stage8_predictions.parquet")
    base = s[["sale_id", "tk", "target_price"]].assign(target_tercile=s.sale_id.map(tt))

    def prep(method, years):
        d = p8[(p8.method == method) & p8.year.isin(years)].merge(base, on="sale_id")
        d["tract"] = d.tk.map(loc)
        return d.merge(tg[["tract", "race_group", "income_group"]], on="tract", how="left").reset_index(drop=True)

    pc = prep("primary", CV_YEARS)
    pa = prep("assessor", [YEAR])
    pa = pa[pa.tract.isin(set(tg.tract))].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx_c = rng.integers(0, len(pc), size=(N_BOOT, len(pc)))
    idx_a = rng.integers(0, len(pa), size=(N_BOOT, len(pa)))

    def bias_stats(gcol, g, tcol):
        y = np.log(pc.pred / pc[tcol]).to_numpy()
        m = (pc[gcol] == g).to_numpy()
        point = np.nanmean(y[m]) - np.nanmean(y)
        draws = np.array([np.nanmean(y[ix][m[ix]]) - np.nanmean(y[ix]) for ix in idx_c])
        return point, draws

    def sales_stats(gcol, g, tcol):
        r = (pa.pred / pa[tcol]).to_numpy()
        m = (pa[gcol] == g).to_numpy()
        point = np.nanmedian(r[m]) / np.nanmedian(r)
        draws = np.array([np.nanmedian(r[ix][m[ix]]) / np.nanmedian(r[ix]) for ix in idx_a])
        return point, draws, int(m.sum())

    global SOLD_2025
    SOLD_2025 = set(s[s.sale_year == YEAR].tk)
    rows = []
    for gcol in ["race_group", "income_group"]:
        for g, d in par.groupby(gcol):
            x = np.log(d.assessed_2025 / d.fair_assessed).to_numpy()
            sdraws = np.median(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))], axis=1)
            sh = float(np.median(x))
            res = {}
            for lab, tcol in [("", "target_price"), ("_tercile", "target_tercile")]:
                b, bd = bias_stats(gcol, g, tcol)
                c, cd = sh + b, sdraws + bd
                sr, sd, sn = sales_stats(gcol, g, tcol)
                clo, chi = _ci(cd)
                slo, shi = _ci(sd)
                ok_a = check_a(clo, chi)
                ok_b = check_b(sr, slo, shi, c)
                res |= {f"model_bias_rel_log{lab}": b, f"bias_lo{lab}": _ci(bd)[0], f"bias_hi{lab}": _ci(bd)[1],
                        f"corrected_log{lab}": c, f"corrected_lo{lab}": clo, f"corrected_hi{lab}": chi,
                        f"sales_rel{lab}": sr, f"sales_rel_lo{lab}": slo, f"sales_rel_hi{lab}": shi,
                        f"check_a{lab}": bool(ok_a), f"check_b{lab}": bool(ok_b)}
                if lab == "":
                    res["sales_n"] = sn
            c_, verdict = guardrail_verdict(res["corrected_log"], res["check_a"], res["check_b"],
                                            res["corrected_log_tercile"], res["check_a_tercile"],
                                            res["check_b_tercile"])
            sold = d[d.tk.isin(SOLD_2025)]
            xs = np.log(sold.assessed_2025 / sold.fair_assessed)
            rows.append({"grouping": gcol, "group": g, "parcels": len(d), "sold_2025_parcels": len(sold),
                         "median_shift_log_sold_2025": float(np.median(xs)) if len(xs) else np.nan,
                         "median_shift_log": sh, "shift_lo": _ci(sdraws)[0], "shift_hi": _ci(sdraws)[1],
                         "aggregate_shift_pct": float(d.assessed_2025.sum() / d.fair_assessed.sum() - 1),
                         "tax_shift_usd_total": float(d.tax_shift_usd.sum()),
                         "tax_shift_usd_median_parcel": float(d.tax_shift_usd.median()),
                         **res, "check_c": bool(c_), "verdict": verdict})
    city = float(np.nanmedian(pa.pred / pa.target_price))
    groups = pd.DataFrame(rows).assign(sales_citywide_ratio=city)

    # Persistence (Owen, 2026-10-20): the same model-free sales check on rolls 2021-2024, both time adjustments.
    # Its own seed, so the 2025 results above do not change.
    rng_y = np.random.default_rng(SEED + 1)
    tracts = set(tg.tract)
    by_year = []
    for Y in CV_YEARS + [YEAR]:
        py = prep("assessor", [Y])
        py = py[py.tract.isin(tracts)].reset_index(drop=True)
        idx = rng_y.integers(0, len(py), size=(N_BOOT, len(py)))
        for r in groups.itertuples():
            rec = {"year": Y, "grouping": r.grouping, "group": r.group}
            for lab, tcol in [("", "target_price"), ("_tercile", "target_tercile")]:
                v = (py.pred / py[tcol]).to_numpy()
                m = (py[r.grouping] == r.group).to_numpy()
                point = np.nanmedian(v[m]) / np.nanmedian(v)
                draws = np.array([np.nanmedian(v[ix][m[ix]]) / np.nanmedian(v[ix]) for ix in idx])
                lo, hi = _ci(draws)
                rec |= {f"sales_rel{lab}": point, f"sales_rel_lo{lab}": lo, f"sales_rel_hi{lab}": hi,
                        f"n{lab}": int(m.sum())}
            # Holds = tier-adjusted check excludes one, in the direction of the 2025 corrected shift.
            rec["holds"] = bool((rec["sales_rel_lo_tercile"] > 1 or rec["sales_rel_hi_tercile"] < 1) and
                                np.sign(rec["sales_rel_tercile"] - 1) == np.sign(r.corrected_log))
            by_year.append(rec)
    by_year = pd.DataFrame(by_year)
    earlier = by_year[by_year.year.isin(CV_YEARS)].groupby(["grouping", "group"]).holds.sum().rename("years_holding")
    groups = groups.merge(earlier.reset_index(), on=["grouping", "group"], how="left")
    groups["persistence"] = np.where(groups.years_holding > len(CV_YEARS) / 2,
                                     "persistent", "2025")  # most of the earlier rolls, else 2025 only

    # Dollars at the 2025 net rate (Owen): typical home = median assessment x corrected shift; group total;
    # revenue-neutral, so the opposite amount falls on the other homes in the pool.
    rate = TAX_RATE[f"{TAX_RATE['rate_used']}_rate_per_1000"] / 1000
    n_pool = len(par)
    dol = []
    for r in groups.itertuples():
        d = par[par[r.grouping] == r.group]
        frac = np.exp(r.corrected_log) - 1
        total = float(d.assessed_2025.sum()) * frac * rate
        dol.append({"median_assessed_2025": float(d.assessed_2025.median()),
                    "usd_typical_home": float(d.assessed_2025.median()) * frac * rate,
                    "usd_group_total": total, "other_homes": n_pool - len(d),
                    "usd_offset_per_other_home": -total / (n_pool - len(d))})
    groups = pd.concat([groups, pd.DataFrame(dol)], axis=1)
    return groups, by_year


def duplex_yields(par: pd.DataFrame):
    res = safmr()
    if res is None:
        where = f"<data>/{SAFMR_DIR.relative_to(DATA_DIR).as_posix()}"  # no machine paths in committed outputs
        log(f"      no SAFMR file in {where}: duplex yields skipped")
        return None, {"status": f"SKIPPED: place HUD FY2025 Small Area FMR .xlsx in {where}"}
    fmr, path = res
    d = par[par.land_use_y1 == "8820"].copy()
    d["br_per_unit"] = (d.beds / 2).round().clip(0, 4).fillna(2).astype(int)  # roll bedrooms split over 2 units
    d = d.merge(fmr, on="zip5", how="left")
    d["fmr_unit"] = [r[f"br{b}"] if pd.notna(r.get(f"br{b}")) else np.nan for b, (_, r) in
                     zip(d.br_per_unit, d.iterrows())]
    d["gross_yield"] = 12 * 2 * d.fmr_unit / d.model_value
    q = d.gross_yield.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    summ = [{"group": "all duplexes", "n": int(d.gross_yield.notna().sum()),
             **{f"p{int(k * 100)}": v for k, v in q.items()}}]
    for g, x in d.groupby("income_group"):
        qq = x.gross_yield.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        summ.append({"group": g, "n": int(x.gross_yield.notna().sum()), **{f"p{int(k * 100)}": v for k, v in qq.items()}})
    par.loc[d.index, "duplex_gross_yield"] = d.gross_yield.to_numpy()
    return pd.DataFrame(summ), {"status": "ok", "file": path.name, "sha256": sha256_of(path),
                                "unmatched_zip": int(d.fmr_unit.isna().sum())}


def divergence_stats(par, top, special, level, q) -> pd.DataFrame:
    by_reason = {f"special_{k.replace(' ', '_')}": int(special.special_reason.str.contains(k, regex=False).sum())
                 for k in SPECIAL_RULES}
    return pd.DataFrame([{
        "parcels_scored": len(par), "citywide_assessment_level": level, "cqr_q": q,
        "outside_80pct_range": int(((par.implied_market < par.range80_low) |
                                    (par.implied_market > par.range80_high)).sum()),
        "listable_outside_and_comps_agree": int(par.listable.sum()), "listed": len(top),
        "listed_over_assessed": int((top.relative_assessment > 1).sum()),
        "listed_under_assessed": int((top.relative_assessment < 1).sum()),
        "listed_rel_min": float(top.relative_assessment.min()) if len(top) else np.nan,
        "listed_rel_max": float(top.relative_assessment.max()) if len(top) else np.nan,
        "listed_tax_shift_usd_median_abs": float(top.tax_shift_usd.abs().median()) if len(top) else np.nan,
        "listed_with_any_context_flag": int((top.context_flags != "").sum()),
        "listed_student_area": int(top.context_flags.str.contains(STUDENT_FLAG, regex=False).sum()),
        "listed_student_area_above_model": int((top.context_flags.str.contains(STUDENT_FLAG, regex=False)
                                                & (top.relative_assessment > 1)).sum()),
        "listed_student_area_below_model": int((top.context_flags.str.contains(STUDENT_FLAG, regex=False)
                                                & (top.relative_assessment < 1)).sum()),
        "all_student_area": int(par.context_flags.str.contains(STUDENT_FLAG, regex=False).sum()),
        "special_listable": len(special),
        "special_below_model": int((special.relative_assessment < 1).sum()),
        "special_above_model": int((special.relative_assessment > 1).sum()),
        **by_reason,
        "all_change_2024_to_2025_median": float((par.assessed_2025 / par.assessed_2024).median()),
        "all_share_unchanged_from_2024": float((par.assessed_2025 == par.assessed_2024).mean()),
        "all_rel_p10": float(par.relative_assessment.quantile(0.1)),
        "all_rel_p90": float(par.relative_assessment.quantile(0.9)),
        "tax_shift_sum_check_usd": float(par.tax_shift_usd.sum())}])


PARCEL_IDS = {"tk", "address", "zip5"}  # never in committed summaries


def display(t: pd.DataFrame) -> pd.DataFrame:
    """Readable copy: dollars as $, ratios to 3 decimals."""
    t = t.copy()
    for c in ["model_value", "range80_low", "range80_high", "assessed_2024", "assessed_2025", "implied_market",
              "comp_value_a", "tax_shift_usd"]:
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
    t["relative_assessment"] = t.relative_assessment.round(3)
    return t


def write_summary(out, stats, top, special, groups, yields, yinfo, keep, sample):
    g = groups.copy()
    for c in ["median_shift_log", "shift_lo", "shift_hi", "model_bias_rel_log", "bias_lo", "bias_hi"]:
        g[c] = (np.exp(g[c]) - 1) * 100
    g["shift % [CI]"] = [f"{a:+.1f} [{b:+.1f}, {c:+.1f}]" for a, b, c in zip(g.median_shift_log, g.shift_lo, g.shift_hi)]
    g["model bias % [CI]"] = ["n/a" if pd.isna(a) else f"{a:+.1f} [{b:+.1f}, {c:+.1f}]"
                              for a, b, c in zip(g.model_bias_rel_log, g.bias_lo, g.bias_hi)]
    g["tax shift total"] = g.tax_shift_usd_total.map("${:,.0f}".format)
    g["shift % among 2025 sales"] = ((np.exp(g.median_shift_log_sold_2025) - 1) * 100).round(1)
    g["assessor ratio on 2025 sales vs citywide % [CI]"] = [
        "n/a" if pd.isna(a) else f"{(a - 1) * 100:+.1f} [{(b - 1) * 100:+.1f}, {(c - 1) * 100:+.1f}]"
        for a, b, c in zip(g.sales_rel, g.sales_rel_lo, g.sales_rel_hi)]
    g["corrected shift % [CI]"] = [f"{(np.exp(a) - 1) * 100:+.1f} [{(np.exp(b) - 1) * 100:+.1f}, {(np.exp(c) - 1) * 100:+.1f}]"
                                   for a, b, c in zip(g.corrected_log, g.corrected_lo, g.corrected_hi)]
    g["corrected shift %, tercile time adj. [CI]"] = [
        f"{(np.exp(a) - 1) * 100:+.1f} [{(np.exp(b) - 1) * 100:+.1f}, {(np.exp(c) - 1) * 100:+.1f}]"
        for a, b, c in zip(g.corrected_log_tercile, g.corrected_lo_tercile, g.corrected_hi_tercile)]
    g["sales check %, tercile [CI]"] = [f"{(a - 1) * 100:+.1f} [{(b - 1) * 100:+.1f}, {(c - 1) * 100:+.1f}]"
                                        for a, b, c in zip(g.sales_rel_tercile, g.sales_rel_lo_tercile,
                                                           g.sales_rel_hi_tercile)]
    parts = [f"# Stage 9: divergence, tax dollars, duplex yields{' (SAMPLE: 500 parcels)' if sample else ''} "
             "(generated by src/avm/stage9_divergence.py)",
             "Every figure per parcel is a MODEL ESTIMATE with its 80% range. Frozen primary model (public data), "
             "valued at Jan 1 2025, compared with roll 2025.",
             f"Tax rate: {TAX_RATE[TAX_RATE['rate_used'] + '_rate_per_1000']} per $1,000 ({TAX_RATE['rate_used']}, "
             f"{TAX_RATE['status']}; gross {TAX_RATE['gross_rate_per_1000']}). School levy credit: "
             f"{TAX_RATE['school_levy_credit']}. "
             f"Excluded: {TAX_RATE['excluded']}. Note: {TAX_RATE['prior_year_note']}.",
             "## Data-quality exclusions (list and tax pool)",
             md(pd.DataFrame([{"rule": k, "definition": v} for k, v in QUALITY_RULES.items()])),
             md(pd.read_csv(out / "exclusions.csv")),
             "## Divergence list", md(stats.round(4).T.reset_index().rename(columns={"index": "quantity", 0: "value"})),
             md(display(top[[c for c in keep if c not in PARCEL_IDS]].head(10))),
             "## Likely special valuation or data issue (listed separately, not in the main ranking)",
             "Listable parcels (outside the range, comps agree) meeting any rule below. They stay in the tax pool.",
             md(pd.DataFrame([{"rule": k, "definition": v} for k, v in SPECIAL_RULES.items()])),
             md(special.special_reason.value_counts().rename("parcels").reset_index()),
             md(display(special[[c for c in keep if c not in PARCEL_IDS] + ["special_reason"]].head(10))),
             "## Group tax shifts, with the direction-aware guardrail",
             "Shift = median of log(assessed / market-based fair share). Model bias = the frozen primary's mean log "
             "error on CV 2021-2024 sales in the group minus its citywide mean; corrected shift = shift + bias "
             "(bootstrap CI from independent draws). Sales check = assessor median ratio on 2025 city-tract sales in "
             "the group / the citywide median ratio (model-free). A claim needs (a) the corrected CI to exclude "
             "zero, (b) the sales check CI to exclude zero (as a percent) in the same direction, and (c) both to "
             "hold with tier-specific (tercile) time adjustment.",
             md(g[["grouping", "group", "parcels", "shift % [CI]", "model bias % [CI]", "corrected shift % [CI]",
                   "assessor ratio on 2025 sales vs citywide % [CI]", "corrected shift %, tercile time adj. [CI]",
                   "sales check %, tercile [CI]", "check_a", "check_b", "check_c", "verdict", "tax shift total"]]),
             f"Citywide assessor median ratio on the same 2025 city-tract sales: {g.sales_citywide_ratio.iloc[0]:.4f}.",
             "## Context flags on listed parcels (shown, not excluded)",
             md(pd.DataFrame([{"flag": k, "definition": v} for k, v in CONTEXT_FLAGS.items()])),
             "## Duplex gross yields (HUD Small Area FMR, a 40th-percentile rent: conservative)",
             md(yields.round(4)) if yields is not None else f"_{yinfo['status']}_", json.dumps(yinfo)]
    (out / "stage9_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
