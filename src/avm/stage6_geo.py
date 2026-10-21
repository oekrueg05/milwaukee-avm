"""Stage 6: geospatial and prior-sales features, level fix, calibration and
compression diagnostics, pooled-conformal intervals.

Usage:
    python -m avm.stage6_geo            # full run
    python -m avm.stage6_geo --sample   # folds 2024-2025 only

Baseline (Owen, 2026-10-18): GBM on a DETRENDED target (log price minus the
fold's training-years-only hedonic time effect), predicting directly at the
Dec Y-1 trend level with a neutral season. Hyperparameters fixed at the
Stage 5 selections (independent cfg 20, informed cfg 8, monotone on), so
lifts measure features, not tuning.

Feature groups, each added alone to the baseline, then all together:
  coords, distances, prior_sales_nearby (kNN), own_prior_sale, city_vacant.
Calibration diagnostic: slope of log target on log prediction (CV pooled
2021-2024 and 2025), overall and within value tiers, 95% HC1 CIs; the
spread-correction experiment runs only if CV slopes are clearly > 1.
Experiments: neighborhood-relative target; informed with prior assessed
value relative to the neighborhood median; spread correction (conditional).
Intervals: CQR with calibration scores pooled over 2022-2024; hedonic with
tier-wise (Mondrian) split-conformal residual quantiles pooled the same way.
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from avm.acquire import load_manifest
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.evaluate import accuracy, compare, tier_prb, value_tiers
from avm.features import build_frame, usable, usable_x
from avm.gbm import GBM, GBMX, sample_configs
from avm.intervals import calibrate, conformal_q, coverage_by
from avm.io import to_number
from avm.mprop import RES_LAND_USE, load_year
from avm.priors import TimeEffect, cutoffs, knn_features, own_prior_sale
from avm.progress import log, step
from avm.reporting import h2h_table, md
from avm.rolls import file_year_for, load_lookup
from avm.spatial import roll_features, static_features
from avm.stage4_hedonic import TEST_YEAR, fit_predict as hedonic_fit_predict, load, trend_point
from avm.stage5_gbm import frames as stage5_frames

OUT = REPORTS_DIR / "stage6"
CV_YEARS = [2021, 2022, 2023, 2024]
CAL_YEARS = [2022, 2023, 2024]
CFG_IDS = {"independent": 20, "assessment_informed": 8}   # Stage 5 selections
HEDONIC = {"loc": "neighborhood", "start": 2020}           # Stage 4 selection
BASE_COLS = ["l_area", "l_lot", "lot_missing", "age", "beds", "full_baths", "half_baths", "baths_missing",
             "stories_num", "duplex", "style", "loc"]
INFORMED_COLS = ["l_prior_total", "l_prior_land"]
INFORMED_REL = ["l_prior_total_rel", "l_prior_land_rel"]
GROUPS = {
    "coords": ["x_m", "y_m"],
    "distances": ["d_downtown_m", "d_lake_m", "d_major_road_m", "d_transit_m", "d_park_m"],
    "prior_sales_nearby": ["knn_lppsf", "knn_dist_m", "nbhd_lppsf_24m"],
    "own_prior_sale": ["prior_lp_detr", "months_since_prior", "has_prior_sale"],
    "city_vacant": ["n_city_200m", "n_vacant_200m"],
}
SLOPE_CLEAR = 1.05  # "clearly > 1": point >= 1.05 and 95% CI lower bound > 1

FEATURE_DOC_6 = {
    "x_m, y_m": ("parcel centroid (UTM m)", "LOCATION: allowed; fine-grained location can proxy for "
                 "neighborhood composition; audited in Stage 8"),
    "d_downtown_m, d_lake_m, d_major_road_m, d_transit_m, d_park_m": (
        "distances to amenities/disamenities", "location; low proxy risk; audited in Stage 8"),
    "knn_lppsf, knn_dist_m, nbhd_lppsf_24m": (
        "prior nearby sale prices per sq ft (before the cutoff, detrended)", "market evidence; inherits any "
        "historical price discrimination in the area; audited in Stage 8"),
    "prior_lp_detr, months_since_prior, has_prior_sale": (
        "the parcel's own prior sale (> 12 months before, before the cutoff)", "no proxy concern"),
    "n_city_200m, n_vacant_200m": (
        "city-owned (municipal exemption incl. tax-deed foreclosure) and vacant parcels within 200 m, roll Y-1",
        "STRONG PROXY RISK: disinvestment and foreclosure track racial composition in Milwaukee; flagged, "
        "audited in Stage 8"),
    "l_prior_total_rel, l_prior_land_rel": (
        "prior assessed value relative to its neighborhood's median (informed experiment)", "as l_prior_total"),
}


# ------------------------------------------------------------ global inputs ---

def roll_tables(s: pd.DataFrame) -> pd.DataFrame:
    """Per sale: roll Y-1 park distance, 200 m counts, and the roll Y-1
    neighborhood medians of assessed total/land for SF/duplex parcels."""
    m = load_manifest().set_index("key")
    lk = load_lookup()
    parts = []
    for y in sorted(s.sale_year.unique()):
        f = file_year_for(int(y) - 1, lk)
        c = load_year(f, m.loc[f"mprop_hist/{f}", "local_path"])[0]
        rows = s[s.sale_year == y]
        rf = roll_features(c, rows.tk)
        sf = c[c.land_use.isin(RES_LAND_USE)].copy()
        sf["tot"] = to_number(sf.assessed_total, f"roll {f} assessed_total")
        sf["land"] = to_number(sf.assessed_land, f"roll {f} assessed_land")
        med = sf.groupby("neighborhood")[["tot", "land"]].median()
        r = rows[["sale_id", "tk", "neighborhood"]].merge(rf, on="tk", how="left")
        r["nbhd_med_total"] = r.neighborhood.map(med.tot).to_numpy()
        r["nbhd_med_land"] = r.neighborhood.map(med.land).to_numpy()
        parts.append(r.drop(columns=["tk", "neighborhood"]))
        log(f"      roll features for {y} sales (roll {f})")
    return pd.concat(parts, ignore_index=True)


# -------------------------------------------------------------- fold frames ---

def fold_frame(s: pd.DataFrame, Y: int, start: int, loc: str, glob: pd.DataFrame,
               train_splits: tuple[str, ...] = ("train",)) -> dict:
    """Train rows: clean sales start..Y-1 in `train_splits` (("train", "test")
    for the prospective 2026 refit, which also learns from 2025). Scored rows:
    sale_year == Y; they need complete features but not a price (prospective
    parcels have none)."""
    train_rows = s.sale_year.between(start, Y - 1) & s.split.isin(train_splits) & ~s.ppsf_outlier
    scored_rows = s.sale_year == Y
    sub = s[train_rows | scored_rows | (s.sale_year < Y)]
    f = build_frame(sub, loc, train_rows[sub.index])
    f = f.merge(glob, on="sale_id", how="left", validate="one_to_one").set_index(sub.index)
    f["tk"] = sub.tk
    f["sale_date"] = sub.sale_date
    f["sale_year"] = sub.sale_year
    f["neighborhood"] = sub.neighborhood
    tr_mask = train_rows[sub.index] & usable(f)
    te = TimeEffect(f[tr_mask], trend_point(Y - 1))
    f["te"] = te(f.t.to_numpy(), f.moy.to_numpy())
    f["y_detr"] = f.lp - f.te
    f["lppsf_detr"] = f.y_detr - f.l_area
    # Pool / history: clean, usable, non-outlier sales (any year before the cutoffs).
    pool = f[usable(f) & ~s.loc[sub.index, "ppsf_outlier"]]
    cut = cutoffs(f, Y)
    knn = knn_features(f, pool[["tk", "sale_date", "x_m", "y_m", "neighborhood", "lppsf_detr"]], cut)
    val_point = f.sale_date.where(f.sale_year != Y, pd.Timestamp(f"{Y}-01-01"))
    own = own_prior_sale(f[["tk", "sale_date"]], pool[["tk", "sale_date"]].assign(
        lp_detr=pool.y_detr, sale_id=pool.sale_id), cut, val_point)
    f = f.join(knn).join(own.drop(columns=["prior_sale_date"]))
    f["l_prior_total_rel"] = f.l_prior_total - np.log(f.nbhd_med_total.where(f.nbhd_med_total > 0))
    f["l_prior_land_rel"] = f.l_prior_land - np.log(f.nbhd_med_land.where(f.nbhd_med_land > 0))
    ok_x = usable_x(f) & f.l_prior_total.notna() & f.l_prior_land.notna()
    ok = ok_x & usable(f)
    keep = ["sale_id", "target_price", "bench_assessed_total", "has_benchmark", "land_use_y1", "bldg_area",
            "sale_price"]
    tr = f[tr_mask & ok]
    sc_mask = scored_rows[sub.index] & ok_x
    sc = f[sc_mask].merge(s[keep], on="sale_id", how="left").set_index(f[sc_mask].index)
    sc["tier"] = value_tiers(s, sc, Y).to_numpy()
    return {"Y": Y, "train": tr, "scored": sc}


# ------------------------------------------------------------------ models ---

def cols_for(variant: str, groups: list[str], informed_rel: bool = False) -> list[str]:
    c = list(BASE_COLS)
    if variant == "assessment_informed":
        c += INFORMED_REL if informed_rel else INFORMED_COLS
    for g in groups:
        c += GROUPS[g]
    return c


def run_model(fold: dict, cfg: dict, cols: list[str], target: str = "detr") -> pd.DataFrame:
    tr, sc = fold["train"], fold["scored"]
    if target == "detr":
        y, anchor_tr, anchor_sc = tr.y_detr, 0.0, 0.0
    elif target == "nbhd_relative":
        anchor_tr = (tr.nbhd_lppsf_24m.fillna(tr.knn_lppsf) + tr.l_area)
        anchor_sc = (sc.nbhd_lppsf_24m.fillna(sc.knn_lppsf) + sc.l_area)
        okr = anchor_tr.notna()
        tr, anchor_tr = tr[okr], anchor_tr[okr]
        y = tr.y_detr - anchor_tr
        anchor_sc = anchor_sc.fillna(anchor_tr.median())
    else:
        raise ValueError(target)
    m = GBMX(cfg, cols, monotone=True).fit(tr, y)
    pred = np.exp(m.predict(sc) + np.asarray(anchor_sc))
    return sc[["sale_id", "target_price", "tier", "neighborhood"]].assign(pred=pred, year=fold["Y"])


def metrics(o: pd.DataFrame) -> dict:
    a = accuracy(o.pred, o.target_price)
    v, p = o.pred.to_numpy(float), o.target_price.to_numpy(float)
    from avm.ratio_study import _stats
    st = _stats(v[None, :], p[None, :])
    tp = tier_prb(o.pred, o.target_price, o.tier)
    return a | {"prd": float(st["prd"][0]), "prb": float(st["prb"][0]), "mki": float(st["mki"][0]),
                "prb_tier_abs": tp["mean_abs"], **{f"prb_{k}": tp.get(k, np.nan) for k in ["low", "mid", "high"]}}


def slopes(o: pd.DataFrame, label: str) -> list[dict]:
    """OLS of log target on log prediction, overall and within tiers, HC1 CIs."""
    rows = []
    for t in ["all", "low", "mid", "high"]:
        g = o if t == "all" else o[o.tier == t]
        if len(g) < 50:
            continue
        X = sm.add_constant(np.log(g.pred.to_numpy(float)))
        r = sm.OLS(np.log(g.target_price.to_numpy(float)), X).fit(cov_type="HC1")
        b, se = r.params[1], r.bse[1]
        rows.append({"method": label, "tier": t, "n": len(g), "slope": b, "ci_low": b - 1.96 * se,
                     "ci_high": b + 1.96 * se})
    return rows


def within_nbhd_slope(o: pd.DataFrame) -> float:
    """Slope of demeaned log target on demeaned log prediction within
    neighborhoods: the stretch factor for the spread correction."""
    lp, ly = np.log(o.pred), np.log(o.target_price)
    dp = lp - lp.groupby(o.neighborhood).transform("mean")
    dy = ly - ly.groupby(o.neighborhood).transform("mean")
    return float((dp * dy).sum() / (dp ** 2).sum())


def stretch(o: pd.DataFrame, b: float) -> pd.Series:
    lp = np.log(o.pred)
    m = lp.groupby(o.neighborhood).transform("mean")
    return np.exp(m + b * (lp - m))


# --------------------------------------------------------------- intervals ---

def cqr_pooled(folds: dict, cfg: dict, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    sc_all = []
    for c in CAL_YEARS:
        f = folds[c]
        lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(f["train"], f["train"].y_detr).predict(f["scored"])
        hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(f["train"], f["train"].y_detr).predict(f["scored"])
        sc_all.append(pd.DataFrame({"lo": lo, "hi": hi, "y": np.log(f["scored"].target_price.to_numpy()),
                                    "tier": f["scored"].tier.to_numpy()}))
    cal = pd.concat(sc_all, ignore_index=True)
    q = calibrate(cal.lo.to_numpy(), cal.hi.to_numpy(), cal.y.to_numpy(), cal.tier)
    f = folds[TEST_YEAR]
    lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(f["train"], f["train"].y_detr).predict(f["scored"])
    hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(f["train"], f["train"].y_detr).predict(f["scored"])
    y = np.log(f["scored"].target_price.to_numpy())
    out = []
    for mode, grp in [("global", None), ("mondrian (by tier)", f["scored"].tier)]:
        adj = (np.full(len(lo), q["global"]) if grp is None else
               np.array([q.get(str(t), q["global"]) for t in grp]))
        out.append(coverage_by(y, lo - adj, hi + adj, f["scored"].tier).assign(calibration=mode))
    return pd.concat(out, ignore_index=True), q


def hedonic_conformal(s: pd.DataFrame) -> pd.DataFrame:
    res = []
    for c in CAL_YEARS:
        _, o, _ = hedonic_fit_predict(s, HEDONIC["loc"], HEDONIC["start"], c)
        o["tier"] = value_tiers(s, o, c).to_numpy()
        res.append(pd.DataFrame({"e": np.abs(np.log(o.target_price) - np.log(o.pred)), "tier": o.tier}))
    cal = pd.concat(res, ignore_index=True)
    qg = conformal_q(cal.e.to_numpy())
    qt = {t: conformal_q(g.e.to_numpy()) for t, g in cal.groupby("tier") if len(g) >= 30}
    h = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet")
    h["tier"] = value_tiers(s, h, TEST_YEAR).to_numpy()
    y = np.log(h.target_price.to_numpy())
    lp = np.log(h.pred.to_numpy())
    out = []
    for mode, adj in [("split conformal, global", np.full(len(h), qg)),
                      ("split conformal, by tier", h.tier.map(qt).fillna(qg).to_numpy())]:
        out.append(coverage_by(y, lp - adj, lp + adj, h.tier).assign(calibration=mode))
    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    out = OUT / "sample" if a.sample else OUT
    out.mkdir(parents=True, exist_ok=True)
    s = load()
    all_cfg = {c["config_id"]: c for c in sample_configs(40)}
    cfgs = {v: all_cfg[i] for v, i in CFG_IDS.items()}
    years = [2024, TEST_YEAR] if a.sample else CV_YEARS + [TEST_YEAR]
    cv_years = [y for y in years if y != TEST_YEAR]

    with step("global spatial and roll features", 3600):
        st = static_features()
        glob = s[["sale_id", "tk"]].merge(st, on="tk", how="left").drop(columns="tk")
        glob = glob.merge(roll_tables(s), on="sale_id", how="left")

    folds = {}
    locs = {cfgs[v]["loc"] for v in cfgs}
    if len(locs) != 1:
        raise ValueError(f"variants use different location units {locs}; build frames per unit")
    loc, start = locs.pop(), min(c["train_start"] for c in cfgs.values())
    for Y in years:
        with step(f"fold frame {Y} (detrend, kNN, own prior sale)", 3600):
            folds[Y] = fold_frame(s, Y, start, loc, glob)
            fr = folds[Y]
            log(f"      {Y}: train {len(fr['train'])}, scored {len(fr['scored'])}, "
                f"scored with knn {fr['scored'].knn_lppsf.notna().mean():.3f}, own prior "
                f"{fr['scored'].has_prior_sale.mean():.3f}")

    # Lift table.
    configs = {"baseline": []} | {f"+{g}": [g] for g in GROUPS} | {"+all groups": list(GROUPS)}
    rows, preds = [], {}
    for v in cfgs:
        for name, groups in configs.items():
            with step(f"lift {v} {name}", 3600):
                cols = cols_for(v, groups)
                os_ = {Y: run_model(folds[Y], cfgs[v], cols) for Y in years}
                preds[(v, name)] = os_
                for Y, o in os_.items():
                    rows.append({"variant": v, "config": name, "year": Y, **metrics(o)})
    lift = pd.DataFrame(rows)
    lift.to_csv(out / "lift_by_year.csv", index=False)
    cvm = lift[lift.year.isin(cv_years)].groupby(["variant", "config"], sort=False).mean(numeric_only=True)
    tst = lift[lift.year == TEST_YEAR].set_index(["variant", "config"])
    keycols = ["mdape_ln", "cod", "median_ratio", "prd", "prb", "mki", "prb_tier_abs"]
    lift_tbl = pd.concat({"CV": cvm[keycols], "2025": tst[keycols]}, axis=1).round(4)
    lift_tbl.columns = [f"{a} {b}" for a, b in lift_tbl.columns]
    lift_tbl.to_csv(out / "lift_table.csv")
    # Paired 2025 lift CIs vs the variant's baseline.
    lift_ci = []
    for v in cfgs:
        base = preds[(v, "baseline")][TEST_YEAR].set_index("sale_id")
        est = {name: preds[(v, name)][TEST_YEAR].set_index("sale_id").pred for name in configs}
        common = pd.concat(est, axis=1).dropna().index
        t = compare({k: e.loc[common] for k, e in est.items()}, base.target_price.loc[common], base="baseline")
        lift_ci.append(t.assign(variant=v))
    lift_ci = pd.concat(lift_ci, ignore_index=True)
    lift_ci.to_csv(out / "lift_2025_paired.csv", index=False)

    # Calibration diagnostic.
    with step("calibration slopes for every method", 3600):
        sl = []
        ass = pd.concat([f["scored"][f["scored"].has_benchmark].assign(pred=f["scored"].bench_assessed_total)
                         for Y, f in folds.items() if Y in cv_years])
        sl += [r | {"split": "CV"} for r in slopes(ass, "assessor")]
        sl += [r | {"split": "2025"} for r in slopes(folds[TEST_YEAR]["scored"][folds[TEST_YEAR]["scored"].has_benchmark]
                                                    .assign(pred=lambda d: d.bench_assessed_total), "assessor")]
        hed_cv = []
        for Y in cv_years:
            _, o, _ = hedonic_fit_predict(s, HEDONIC["loc"], HEDONIC["start"], Y)
            o["tier"] = value_tiers(s, o, Y).to_numpy()
            hed_cv.append(o)
        sl += [r | {"split": "CV"} for r in slopes(pd.concat(hed_cv), "hedonic")]
        h25 = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet")
        h25["tier"] = value_tiers(s, h25, TEST_YEAR).to_numpy()
        sl += [r | {"split": "2025"} for r in slopes(h25, "hedonic")]
        for v in cfgs:  # Stage 5 GBMs (time features, no detrending)
            g5 = []
            for Y in cv_years:
                tr, pr = stage5_frames(s, cfgs[v]["loc"], cfgs[v]["train_start"], Y)
                m = GBM(cfgs[v], v, monotone=True).fit(tr)
                o = pr[["sale_id"]].assign(pred=np.exp(m.predict_log(pr, trend_point(Y - 1)))).merge(
                    s[["sale_id", "target_price", "neighborhood"]], on="sale_id")
                o["tier"] = value_tiers(s, o, Y).to_numpy()
                g5.append(o)
            sl += [r | {"split": "CV"} for r in slopes(pd.concat(g5), f"stage5 gbm {v}")]
            p25 = pd.read_parquet(INTERIM_DIR / f"gbm_{v}_predictions_2025.parquet")
            sl += [r | {"split": "2025"} for r in slopes(p25, f"stage5 gbm {v}")]
            for name in ["baseline", "+all groups"]:
                cv_o = pd.concat([preds[(v, name)][Y] for Y in cv_years])
                sl += [r | {"split": "CV"} for r in slopes(cv_o, f"stage6 {v} {name}")]
                sl += [r | {"split": "2025"} for r in slopes(preds[(v, name)][TEST_YEAR], f"stage6 {v} {name}")]
        slope_tbl = pd.DataFrame(sl)
        slope_tbl.to_csv(out / "calibration_slopes.csv", index=False)

    # Compression experiments on the all-groups model.
    exp_rows, exp_preds = [], {}
    cv6 = slope_tbl[(slope_tbl.split == "CV") & slope_tbl.method.str.startswith("stage6")]
    clearly = cv6[(cv6.slope >= SLOPE_CLEAR) & (cv6.ci_low > 1)]
    run_spread = len(clearly) > 0
    for v in cfgs:
        with step(f"experiments {v}", 3600):
            allg = list(GROUPS)
            exps = {"all groups (reference)": preds[(v, "+all groups")],
                    "nbhd-relative target": {Y: run_model(folds[Y], cfgs[v], cols_for(v, allg), "nbhd_relative")
                                             for Y in years}}
            if v == "assessment_informed":
                exps["prior value relative to nbhd median"] = {
                    Y: run_model(folds[Y], cfgs[v], cols_for(v, allg, informed_rel=True)) for Y in years}
            if run_spread:
                ref = preds[(v, "+all groups")]
                sp = {}
                for Y in years:  # stretch factor from the other CV years only (no peeking at Y)
                    others = pd.concat([ref[k] for k in cv_years if k != Y])
                    b = within_nbhd_slope(others)
                    sp[Y] = ref[Y].assign(pred=stretch(ref[Y], b).to_numpy(), stretch_b=b)
                exps["spread correction (within-nbhd stretch)"] = sp
            for name, os_ in exps.items():
                exp_preds[(v, name)] = os_
                for Y, o in os_.items():
                    exp_rows.append({"variant": v, "experiment": name, "year": Y, **metrics(o),
                                     "stretch_b": float(o.stretch_b.iloc[0]) if "stretch_b" in o else np.nan})
    exp = pd.DataFrame(exp_rows)
    exp.to_csv(out / "compression_experiments_by_year.csv", index=False)
    exp_tbl = pd.concat({"CV": exp[exp.year.isin(cv_years)].groupby(["variant", "experiment"], sort=False)
                         .mean(numeric_only=True)[keycols],
                         "2025": exp[exp.year == TEST_YEAR].set_index(["variant", "experiment"])[keycols]},
                        axis=1).round(4)
    exp_tbl.columns = [f"{a} {b}" for a, b in exp_tbl.columns]
    exp_tbl.to_csv(out / "compression_experiments.csv")

    # Intervals after the level fix.
    cover = []
    if not a.sample:
        with step("intervals: pooled CQR (2022-2024) and hedonic conformal", 3 * 3600):
            for v in cfgs:
                for name, groups in [("baseline", []), ("+all groups", list(GROUPS))]:
                    c, q = cqr_pooled(folds, cfgs[v], cols_for(v, groups))
                    cover.append(c.assign(method=f"stage6 {v} {name}"))
            cover.append(hedonic_conformal(s).assign(method="hedonic"))
        cover = pd.concat(cover, ignore_index=True)
        cover.to_csv(out / "interval_coverage.csv", index=False)

    # Final 2025 comparison on common sales.
    with step("2025 comparison (paired bootstrap)", 3600):
        t25 = folds[TEST_YEAR]["scored"]
        t25 = t25[t25.has_benchmark].set_index("sale_id")
        est = {"assessor": t25.bench_assessed_total}
        if not a.sample:
            est["hedonic"] = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet").set_index("sale_id").pred
            for v in cfgs:
                est[f"stage5 {v}"] = pd.read_parquet(INTERIM_DIR / f"gbm_{v}_predictions_2025.parquet").set_index(
                    "sale_id").pred
        for v in cfgs:
            for name in ["baseline", "+all groups"]:
                est[f"stage6 {v} {name}"] = preds[(v, name)][TEST_YEAR].set_index("sale_id").pred
        common = pd.concat(est, axis=1).dropna().index.intersection(t25.index)
        cmp = compare({k: e.loc[common] for k, e in est.items()}, t25.target_price.loc[common], base="assessor")
        cmp.to_csv(out / "comparison_2025.csv", index=False)
        tiers = t25.tier.loc[common]
        tprb = pd.DataFrame([{"method": k, **tier_prb(e.loc[common], t25.target_price.loc[common], tiers)}
                             for k, e in est.items()])
        tprb.to_csv(out / "within_tier_prb_2025.csv", index=False)

    write_summary(out, lift_tbl, lift_ci, slope_tbl, run_spread, exp_tbl, cover, cmp, tprb, list(est), folds)
    log(f"wrote {out / 'stage6_summary.md'}")


def write_summary(out, lift_tbl, lift_ci, slope_tbl, run_spread, exp_tbl, cover, cmp, tprb, methods, folds):
    lc = lift_ci[lift_ci.statistic.isin(["mdape_ln", "cod"]) & (lift_ci.method != "baseline")].copy()
    lc["diff [CI]"] = [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi in
                       zip(lc.diff_vs_baseline, lc.diff_ci_low, lc.diff_ci_high)]
    lc = lc.pivot_table(index=["variant", "method"], columns="statistic", values="diff [CI]", aggfunc="first")
    st = slope_tbl.copy()
    st["slope [CI]"] = [f"{b:.3f} [{lo:.3f}, {hi:.3f}]" for b, lo, hi in zip(st.slope, st.ci_low, st.ci_high)]
    st = st.pivot_table(index=["method", "split"], columns="tier", values="slope [CI]", aggfunc="first")
    st = st[[c for c in ["all", "low", "mid", "high"] if c in st.columns]].reset_index()
    cov = (cover.pivot_table(index=["method", "calibration"], columns="group", values="coverage").round(3).reset_index()
           if isinstance(cover, pd.DataFrame) and len(cover) else pd.DataFrame())
    fold_info = pd.DataFrame([{"year": Y, "n_train": len(f["train"]), "n_scored": len(f["scored"]),
                               "scored_with_knn": round(f["scored"].knn_lppsf.notna().mean(), 3),
                               "scored_with_own_prior": round(f["scored"].has_prior_sale.mean(), 3)}
                              for Y, f in folds.items()])
    parts = [
        "# Stage 6: geospatial features, level fix, calibration and compression (generated by "
        "src/avm/stage6_geo.py)",
        "Baseline = Stage 5 GBM hyperparameters on a detrended target (training-years-only hedonic time "
        "effect), predicting at the Dec Y-1 trend level, neutral season. Target prices from Stage 3. "
        "mdape_ln = level-neutral MdAPE (headline). **PRD/PRB against sale prices lean regressive.**",
        "## Folds", md(fold_info),
        "## Feature-group lift (CV = mean over 2021-2024; 2025 = test)", md(lift_tbl.reset_index()),
        "## 2025 lift vs baseline, paired bootstrap (negative = better)", md(lc.reset_index()),
        "## Calibration slopes: log target on log prediction (95% HC1 CI); > 1 means compressed predictions",
        md(st),
        f"Spread-correction experiment run: {run_spread} (needs a Stage 6 CV slope >= {SLOPE_CLEAR} with CI "
        "lower bound > 1).",
        "## Compression experiments (all-groups models)", md(exp_tbl.reset_index()),
    ]
    if len(cov):
        parts += ["## 80% interval coverage on 2025 after the level fix (pooled 2022-2024 calibration)", md(cov)]
    parts += ["## 2025 comparison (common sales; diffs vs assessor)",
              md(h2h_table(cmp.assign(stratum="all", trim="untrimmed"), ["mdape_ln", "cod", "median_ratio", "prd",
                                                                          "prb", "mki"], methods, "assessor")
                 .T.reset_index().rename(columns={"index": "column", 0: "value"})),
              "Within-tier PRB on 2025:", md(tprb.round(4)),
              "## New features (rule 7)", md(pd.DataFrame([{"features": k, "why": v[0], "proxy note": v[1]}
                                                           for k, v in FEATURE_DOC_6.items()]))]
    (out / "stage6_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
