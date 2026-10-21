"""Stage 8: equity audit. Do ERRORS differ by group?

Usage:
    python -m avm.stage8_equity

Groups (tract demographics, ACS 2019-2023, evaluation only): tract majority
race/ethnicity (groups with < 5 tracts folded into "mixed") and income
quintiles weighted by SF/duplex parcels. Methods: assessor, hedonic, frozen
primary, frozen secondary, and the primary refit without the city-owned/
vacant features. Splits: pooled out-of-fold CV 2021-2024 (decisions) and
2025 (reporting only; no selection uses 2025).

Primary measures are prediction-conditional or error-based, by group with
bootstrap CIs: median ratio (level), mean log residual (bias), MdAPE against
the method's OVERALL level (bias plus dispersion), COD, and the calibration
slope/intercept of log price on log prediction. Secondary: PRD, PRB, MKI.
Gaps: Black-majority minus White-majority, Q1 minus Q5 income, with paired
bootstrap CIs on method-minus-assessor gap differences.
"""

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from avm import frames
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.equity import VARS, fetch_acs, income_quintiles, majority_group, tract_table
from avm.gbm import GBMX
from avm.location import parcel_locations
from avm.mprop import RES_LAND_USE
from avm.progress import log, step
from avm.ratio_study import N_BOOT, SEED, _stats
from avm.reporting import md
from avm.stage4_hedonic import fit_predict as hedonic_fit_predict
from avm.stage6_geo import GROUPS

OUT = REPORTS_DIR / "stage8"
SPEC_PATH = REPORTS_DIR / "frozen" / "model_spec.json"
CV_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR = 2025
BASE_START = 2014
HEDONIC = {"loc": "neighborhood", "start": 2020}
METHODS = ["assessor", "hedonic", "primary", "secondary", "primary_no_city_vacant"]
CONTRASTS = {"race": ("Black majority", "White majority"), "income": ("Q1 (lowest income)", "Q5 (highest income)")}
SMALL_N = 200
GAP_TOL, MDAPE_COST = 0.01, 0.002
BLUE, INK, INK2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FRAMING = (
    "**What this audit tests.** Whether the ERRORS of each method differ by group (accuracy, bias, "
    "calibration), not whether the VALUES differ. The models reproduce market price levels by design: if "
    "homes sell for less in some areas, a correct model predicts less there, and that is not a model "
    "failure. A failure is a method that is systematically too high or too low, less accurate, or "
    "differently calibrated for one group than another, measured against the same time-adjusted sale "
    "prices for every method. Demographics are used only to form groups; none is a model input (rule 7, "
    "asserted below).")


# ------------------------------------------------------------- predictions ---

def predictions(s: pd.DataFrame, glob: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = []
    cfgs = {"primary": (spec["models"]["primary"], None),
            "secondary": (spec["models"]["secondary"], None),
            "primary_no_city_vacant": (spec["models"]["primary"], "city_vacant")}
    for Y in CV_YEARS + [TEST_YEAR]:
        for name, (m, drop_group) in cfgs.items():
            cfg = m["config"]
            cols = [c for c in m["features"] if not (drop_group and c in GROUPS[drop_group])]
            if name == "primary" and Y == TEST_YEAR:
                p = pd.read_parquet(INTERIM_DIR / "frozen_primary_predictions_2025.parquet")
            elif name == "secondary" and Y == TEST_YEAR:
                p = pd.read_parquet(INTERIM_DIR / "frozen_secondary_predictions_2025.parquet")
            else:
                f = frames.fold(s, glob, Y, BASE_START, cfg["loc"])
                tr = f["train"][f["train"].sale_year >= cfg["train_start"]]
                g = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
                p = f["scored"][["sale_id"]].assign(pred=np.exp(g.predict(f["scored"])))
            rows.append(p[["sale_id", "pred"]].assign(method=name, year=Y))
        if Y == TEST_YEAR:
            h = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet")[["sale_id", "pred"]]
        else:
            _, h, _ = hedonic_fit_predict(s, HEDONIC["loc"], HEDONIC["start"], Y)
            h = h[["sale_id", "pred"]]
        rows.append(h.assign(method="hedonic", year=Y))
        a = s[(s.sale_year == Y) & s.has_benchmark][["sale_id", "bench_assessed_total"]]
        rows.append(a.rename(columns={"bench_assessed_total": "pred"}).assign(method="assessor", year=Y))
        log(f"      predictions for {Y}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- measures ---

def _boot_stats(v: np.ndarray, p: np.ndarray, m0: float) -> dict:
    r = v / p
    med = np.median(r, axis=1)
    out = {"median_ratio": med, "bias_mean_log": np.mean(np.log(r), axis=1),
           "mdape_vs_overall_level": np.median(np.abs(r / m0 - 1), axis=1),
           "cod": 100 * np.mean(np.abs(r - med[:, None]), axis=1) / med}
    st = _stats(v, p)
    out |= {"prd": st["prd"], "prb": st["prb"], "mki": st["mki"]}
    return out


def calib(v: np.ndarray, p: np.ndarray) -> dict:
    X = sm.add_constant(np.log(v))
    r = sm.OLS(np.log(p), X).fit(cov_type="HC1")
    b, a = r.params[1], r.params[0]
    return {"slope": b, "slope_lo": b - 1.96 * r.bse[1], "slope_hi": b + 1.96 * r.bse[1],
            "intercept": a, "intercept_lo": a - 1.96 * r.bse[0], "intercept_hi": a + 1.96 * r.bse[0]}


def audit(w: pd.DataFrame, group_col: str, split: str) -> tuple[pd.DataFrame, dict]:
    """w: one row per sale with target_price and one column per method.
    Returns the group table and per-group bootstrap draws for gaps."""
    rng = np.random.default_rng(SEED)
    m0 = {m: float(np.median(w[m] / w.target_price)) for m in METHODS}
    rows, draws = [], {}
    for g, d in w.groupby(group_col):
        n = len(d)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        p = d.target_price.to_numpy(float)
        for m in METHODS:
            v = d[m].to_numpy(float)
            pt = {k: float(a[0]) for k, a in _boot_stats(v[None, :], p[None, :], m0[m]).items()}
            bt = _boot_stats(v[idx], p[idx], m0[m])
            draws[(g, m)] = bt
            row = {"split": split, "grouping": group_col, "group": g, "method": m, "n": n,
                   "small_n_flag": n < SMALL_N}
            for k in pt:
                row[k] = pt[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(bt[k], [2.5, 97.5])
            row |= calib(v, p)
            rows.append(row)
    return pd.DataFrame(rows), draws


def gaps(draws: dict, tbl: pd.DataFrame, grouping: str, split: str) -> pd.DataFrame:
    g1, g2 = CONTRASTS[grouping]

    def point(m, k):
        return (tbl[(tbl.group == g1) & (tbl.method == m)][k].iloc[0]
                - tbl[(tbl.group == g2) & (tbl.method == m)][k].iloc[0])

    rows = []
    for k in ["median_ratio", "bias_mean_log", "mdape_vs_overall_level", "cod", "prb"]:
        base_draw = draws[(g1, "assessor")][k] - draws[(g2, "assessor")][k]
        base_pt = point("assessor", k)
        for m in METHODS:
            gd = draws[(g1, m)][k] - draws[(g2, m)][k]
            pt = point(m, k)
            row = {"split": split, "contrast": f"{g1} minus {g2}", "measure": k, "method": m, "gap": pt,
                   "gap_lo": np.percentile(gd, 2.5), "gap_hi": np.percentile(gd, 97.5)}
            if m != "assessor":  # paired: same resampled sales for both methods
                dd = np.abs(gd) - np.abs(base_draw)
                row |= {"abs_gap_minus_assessor": abs(pt) - abs(base_pt), "diff_lo": np.percentile(dd, 2.5),
                        "diff_hi": np.percentile(dd, 97.5)}
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cleaning ---

def cleaning_rates(s: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Share of raw sales (2021-2025) removed by each Stage 2 filter, by group.
    F5 (not 'Residential') is scope, not cleaning, so cleaning is reported on
    rows that passed F1-F5 as well."""
    rem = pd.read_parquet(INTERIM_DIR / "sales_clean_removed.parquet")
    kept = s[["sale_id", "tk", "sale_year"]].assign(code="kept")
    allr = pd.concat([rem, kept], ignore_index=True)
    allr = allr[allr.sale_year.between(CV_YEARS[0], TEST_YEAR)]
    allr = allr.merge(groups, on="tk", how="left")
    out = []
    for grouping in ["race_group", "income_group"]:
        a = allr.assign(group=allr[grouping].fillna("no tract match"))
        tot = a.groupby("group").size()
        by = a.pivot_table(index="group", columns="code", values="sale_id", aggfunc="size", fill_value=0)
        scope = a[~a.code.isin(["F1", "F2", "F3", "F4", "F5"])]
        sc_tot = scope.groupby("group").size()
        clean_cols = [c for c in ["F6", "F7", "F8", "F9", "F10", "F11"] if c in by.columns]
        t = pd.DataFrame({"grouping": grouping, "raw_sales": tot,
                          "share_removed_any": 1 - by.get("kept", 0) / tot,
                          "share_not_residential_F5": by.get("F5", 0) / tot,
                          "residential_in_scope": sc_tot,
                          "cleaning_rate_F6_F11": by[clean_cols].sum(axis=1) / sc_tot})
        for c in clean_cols:
            t[f"share_{c}"] = by[c] / sc_tot
        out.append(t.reset_index())
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------- chart ---

def plot_gaps(g: pd.DataFrame, path) -> None:
    measures = ["bias_mean_log", "mdape_vs_overall_level", "cod"]
    titles = {"bias_mean_log": "bias (mean log residual)", "mdape_vs_overall_level": "MdAPE vs overall level",
              "cod": "COD"}
    contrasts = list(g.contrast.unique())
    fig, axes = plt.subplots(len(contrasts), len(measures), figsize=(13, 3.2 * len(contrasts)), facecolor=SURFACE,
                             squeeze=False)
    for i, c in enumerate(contrasts):
        for j, k in enumerate(measures):
            ax = axes[i, j]
            d = g[(g.contrast == c) & (g.measure == k)].set_index("method").reindex(METHODS[::-1])
            y = np.arange(len(d))
            ax.errorbar(d.gap, y, xerr=[d.gap - d.gap_lo, d.gap_hi - d.gap], fmt="o", color=BLUE, ms=6,
                        capsize=3, lw=1.5)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(y, d.index if j == 0 else [""] * len(d))
            ax.set_title(f"{titles[k]}\n{c}" if j == 0 else titles[k], loc="left", color=INK, fontsize=10)
            ax.set_facecolor(SURFACE)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=INK2)
    fig.suptitle("Between-group error gaps with 95% bootstrap CIs (CV 2021-2024 pooled); 0 = no gap", x=0.01,
                 ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- main ---

PRED_CACHE = INTERIM_DIR / "stage8_predictions.parquet"


def cached_predictions(s, glob, spec) -> pd.DataFrame:
    """Out-of-fold predictions, cached and keyed on the frozen spec and the
    clean-sales hash (they need no demographics)."""
    from avm.acquire import sha256_of
    key = sha256_of(SPEC_PATH) + sha256_of(INTERIM_DIR / "sales_clean.parquet")
    meta = PRED_CACHE.with_suffix(".key")
    if PRED_CACHE.exists() and meta.exists() and meta.read_text() == key:
        return pd.read_parquet(PRED_CACHE)
    p = predictions(s, glob, spec)
    p.to_parquet(PRED_CACHE, index=False)
    meta.write_text(key)
    return p


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-only", action="store_true", help="build and cache predictions, no ACS")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_PATH.read_text())
    if a.predictions_only:
        s = frames.sales()
        with step("predictions: CV 2021-2024 out-of-fold and 2025", 3 * 3600):
            p = cached_predictions(s, frames.global_table(s), spec)
        log(f"      {len(p)} prediction rows: {p.groupby(['year', 'method']).size().unstack().to_dict()}")
        return
    leak = {c for m in spec["models"].values() for c in m["features"]} & (set(VARS.values()) | {
        "share_nh_white", "share_nh_black", "share_nh_asian", "share_hispanic", "poverty_rate", "race_group",
        "income_group"})
    if leak:
        raise AssertionError(f"rule 7: demographic columns in model features: {leak}")
    with step("ACS tract demographics", 600):
        fetch_acs()
        t = tract_table()
    s = frames.sales()
    glob = frames.global_table(s)
    with step("groups", 600):
        loc = parcel_locations()
        c25 = pd.read_parquet(INTERIM_DIR / "mprop_concepts" / "2025.parquet")
        sf = c25[c25.land_use.isin(RES_LAND_USE)].dropna(subset=["tk"]).drop_duplicates("tk")
        per_tract = sf.merge(loc, on="tk").groupby("tract").size()
        # Milwaukee County includes suburbs: group (and fold, < 5 tracts) over
        # CITY tracts only, i.e. tracts holding city SF/duplex parcels.
        city = t.tract.isin(per_tract.index)
        t["race_group"] = "outside city (no SF/duplex parcels)"
        t["income_group"] = "outside city (no SF/duplex parcels)"
        t.loc[city, "race_group"] = majority_group(t[city]).to_numpy()
        t.loc[city, "income_group"] = income_quintiles(t[city], per_tract).to_numpy()
        tract_groups = t[["tract", "race_group", "income_group", "share_nh_black", "share_nh_white",
                          "share_hispanic", "median_hh_income"]]
        tract_groups.to_csv(OUT / "tract_groups.csv", index=False)
        groups = loc[["tk", "tract"]].merge(tract_groups[["tract", "race_group", "income_group"]], on="tract",
                                            how="left")
        tract_counts = pd.DataFrame({"race": t.race_group.value_counts(), "income": t.income_group.value_counts()})
    with step("predictions: CV 2021-2024 out-of-fold and 2025", 3 * 3600):
        p = cached_predictions(s, glob, spec)
    with step("audit tables and gaps", 3600):
        tables, gap_tbls, ns = [], [], []
        for split, years in [("CV 2021-2024", CV_YEARS), ("2025", [TEST_YEAR])]:
            w = p[p.year.isin(years)].pivot_table(index="sale_id", columns="method", values="pred")
            w = w.dropna().join(s.set_index("sale_id")[["target_price", "tk"]])
            w = w.merge(groups[["tk", "race_group", "income_group"]], left_on="tk", right_on="tk", how="left")
            w["race_group"] = w.race_group.fillna("no tract match")
            w["income_group"] = w.income_group.fillna("no tract match")
            ns.append({"split": split, "common_sales": len(w)})
            for gcol, short in [("race_group", "race"), ("income_group", "income")]:
                tb, dr = audit(w, gcol, split)
                tables.append(tb)
                gap_tbls.append(gaps(dr, tb, short, split))
        tab = pd.concat(tables, ignore_index=True)
        gp = pd.concat(gap_tbls, ignore_index=True)
        tab.to_csv(OUT / "group_tables.csv", index=False)
        gp.to_csv(OUT / "gaps.csv", index=False)
        plot_gaps(gp[gp.split == "CV 2021-2024"], OUT / "gaps_cv.png")
    with step("cleaning rates by group", 600):
        cr = cleaning_rates(s, groups)
        cr.to_csv(OUT / "cleaning_rates_by_group.csv", index=False)
    decision = city_vacant_decision(tab, gp, p, s)
    write_summary(tab, gp, cr, tract_counts, ns, decision)
    log(f"wrote {OUT / 'stage8_summary.md'}")


def city_vacant_decision(tab, gp, p, s) -> dict:
    """CV only. Drop city-owned/vacant features if removing them narrows the
    Black-White median-ratio gap by >= 0.01 at <= 0.2 pts CV MdAPE cost."""
    cv = gp[gp.split == "CV 2021-2024"]
    race = cv[cv.contrast.str.startswith("Black")]
    def g(meas, m):
        return float(race[(race.measure == meas) & (race.method == m)].gap.iloc[0])
    w = p[p.year.isin(CV_YEARS)].merge(s[["sale_id", "target_price"]], on="sale_id")
    md_ln = {}
    for m in ["primary", "primary_no_city_vacant"]:
        r = w[w.method == m]
        rr = r.pred / r.target_price
        md_ln[m] = float((rr / rr.median() - 1).abs().median())
    out = {"level_gap_primary": g("median_ratio", "primary"),
           "level_gap_without": g("median_ratio", "primary_no_city_vacant"),
           "mdape_gap_primary": g("mdape_vs_overall_level", "primary"),
           "mdape_gap_without": g("mdape_vs_overall_level", "primary_no_city_vacant"),
           "cod_gap_primary": g("cod", "primary"), "cod_gap_without": g("cod", "primary_no_city_vacant"),
           "cv_mdape_ln_primary": md_ln["primary"], "cv_mdape_ln_without": md_ln["primary_no_city_vacant"]}
    out["level_gap_narrowing"] = abs(out["level_gap_primary"]) - abs(out["level_gap_without"])
    out["mdape_cost"] = out["cv_mdape_ln_without"] - out["cv_mdape_ln_primary"]
    out["drop"] = bool(out["level_gap_narrowing"] >= GAP_TOL and out["mdape_cost"] <= MDAPE_COST)
    return out


def ci(x, lo, hi, f="{:.3f}"):
    return f"{f.format(x)} [{f.format(lo)}, {f.format(hi)}]"


def write_summary(tab, gp, cr, tract_counts, ns, d) -> None:
    def gtab(split, gcol):
        t = tab[(tab.split == split) & (tab.grouping == gcol)]
        rows = []
        for (g, m), r in t.set_index(["group", "method"]).iterrows():
            rows.append({"group": g, "method": m, "n": int(r.n) if not r.small_n_flag else f"{int(r.n)} (< 200)",
                         "median ratio": ci(r.median_ratio, r.median_ratio_lo, r.median_ratio_hi),
                         "bias (mean log resid)": ci(r.bias_mean_log, r.bias_mean_log_lo, r.bias_mean_log_hi),
                         "MdAPE vs overall level": ci(r.mdape_vs_overall_level, r.mdape_vs_overall_level_lo,
                                                      r.mdape_vs_overall_level_hi, "{:.1%}"),
                         "COD": ci(r.cod, r.cod_lo, r.cod_hi, "{:.1f}"),
                         "calibration slope": ci(r.slope, r.slope_lo, r.slope_hi)})
        return pd.DataFrame(rows)

    def side(split):
        t = tab[tab.split == split]
        return pd.DataFrame({"grouping": t.grouping, "group": t.group, "method": t.method,
                             "PRB": [ci(a, b, c) for a, b, c in zip(t.prb, t.prb_lo, t.prb_hi)],
                             "PRD": t.prd.round(3), "MKI": t.mki.round(3),
                             "calibration slope": [ci(a, b, c) for a, b, c in zip(t.slope, t.slope_lo, t.slope_hi)]})

    def gaptab(split):
        g = gp[gp.split == split].copy()
        g["gap [CI]"] = [ci(a, b, c) for a, b, c in zip(g.gap, g.gap_lo, g.gap_hi)]
        g["|gap| - |assessor gap| [CI]"] = [("" if pd.isna(a) else ci(a, b, c)) for a, b, c in
                                            zip(g.get("abs_gap_minus_assessor"), g.get("diff_lo"), g.get("diff_hi"))]
        return g[["contrast", "measure", "method", "gap [CI]", "|gap| - |assessor gap| [CI]"]]

    parts = [
        "# Stage 8: equity audit (generated by src/avm/stage8_equity.py)", FRAMING,
        "**Reading PRB and the calibration slope together.** PRB and PRD computed against sale prices are biased "
        "toward showing regressivity, because sale-price noise sits in the denominator. The calibration slope "
        "(log price regressed on log prediction) is biased below 1 by noise in the prediction (attenuation). "
        "They err in opposite directions, so both are shown side by side; prediction-conditional measures "
        "(bias, MdAPE against the overall level, calibration) lead.",
        "**Caveat on bias by group.** Target prices are moved to Jan 1 with a CITYWIDE trend. If some groups' "
        "prices rose faster than the city's, their time-adjusted targets stay higher and every method looks "
        "biased low there for that reason alone. The bias gaps below therefore mix method error with "
        "group-specific appreciation; the comparison BETWEEN methods (same targets) is unaffected.",
        "Methods: assessor (roll Y), hedonic (Stage 4), frozen primary (public data), frozen secondary "
        "(assessment-informed), primary without city-owned/vacant features. CV = pooled out-of-fold "
        "2021-2024; 2025 is reported only. Groups with n < 200 are flagged.",
        "## Sample sizes", md(pd.DataFrame(ns)), "Tracts per group:", md(tract_counts.fillna(0).astype(int).reset_index()),
        "## Gaps (CV 2021-2024 pooled)", "![gaps](gaps_cv.png)", md(gaptab("CV 2021-2024")),
        "## Gaps (2025)", md(gaptab("2025")),
        "## Group tables: race/ethnicity majority (CV)", md(gtab("CV 2021-2024", "race_group")),
        "## Group tables: income quintile (CV)", md(gtab("CV 2021-2024", "income_group")),
        "## Group tables: race/ethnicity majority (2025)", md(gtab("2025", "race_group")),
        "## Group tables: income quintile (2025)", md(gtab("2025", "income_group")),
        "## PRB vs calibration slope, side by side (CV)", md(side("CV 2021-2024")),
        "## PRB vs calibration slope, side by side (2025)", md(side("2025")),
        "## Cleaning rates by group (raw sales 2021-2025)",
        "Share of raw sales removed by Stage 2, by group. F5 (not 'Residential') is scope; the cleaning rate is "
        "F6-F11 on residential rows in scope.", md(cr.round(4)),
        "## City-owned/vacant features: decision (CV only)",
        md(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}]).T.reset_index()
           .rename(columns={"index": "quantity", 0: "value"})),
        f"Rule: drop if removing them narrows the Black-White median-ratio gap by >= {GAP_TOL} at a CV MdAPE cost "
        f"<= {MDAPE_COST:.1%}. **Decision: {'DROP' if d['drop'] else 'KEEP'} city-owned/vacant features.**",
    ]
    (OUT / "stage8_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
