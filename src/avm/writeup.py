"""Stage 10 write-up: renders README.md (and docs/methods.md) from templates.

Usage:
    python -m avm.writeup --sample   # Key results, Summary and Results only,
                                     # to reports/writeup/sample/README_sample.md
    python -m avm.writeup            # full README.md and docs/methods.md

Every number in the rendered text is a {{key}} placeholder filled from a
report file; reports/writeup/numbers.json records each value, its display
and its source. The build fails on a missing key, an unused key is listed,
and the lint refuses (a) any digit in template prose that is not a
placeholder or a whitelisted method constant or label, and (b) em or en
dashes. The headline head-to-head (identical 2025 sales, paired bootstrap)
is computed here from interim/stage8_predictions.parquet and the shared
target prices and written to reports/writeup/headline_2025.csv.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from avm import comps as C
from avm import frames
from avm.config import INTERIM_DIR, REPO_DIR, REPORTS_DIR
from avm.evaluate import compare
from avm.progress import log
from avm.ratio_study import IAAO_TARGETS, N_BOOT, SEED

OUT = REPORTS_DIR / "writeup"
CCAO_URL = "https://github.com/ccao-data/model-res-avm"
CCAO_READ = "2026-10-20"
TEMPLATES = REPO_DIR / "docs" / "templates"
METHODS = {"assessor": "City Assessor", "hedonic": "Hedonic baseline", "primary": "Model (public data)",
           "secondary": "Model (assessment-informed)"}
# Digits allowed in template prose without a placeholder: years and year
# ranges, stage numbers, quintile labels, the Jan 1 valuation date, and the
# method constants Owen whitelisted (80%, 95%, 0.75 mi, 3 to 5 comps).
WHITELIST = [r"\b(19|20)\d{2}\b", r"\bStages? \d+(\s*(-|to|and)\s*\d+)?\b", r"\bQ[1-5]\b", r"\bJan(uary)? 1\b",
             r"\b80%", r"\b95%", r"\b0\.75 mi\b", r"\b3 to 5\b", r"\bK2540\b", r"\b88[12]0\b",
             r"\b[Ww]ithin (10|20)% of price",  # the within-10%/20% accuracy thresholds
             r"\bsha256\b", r"\bY-1\b", r"\bL2\b", r"\bF\d{1,2}\b", r"\bv2\b"]  # labels, not quantities


class Numbers:
    def __init__(self):
        self.d: dict[str, dict] = {}

    def add(self, key: str, value, display: str, source: str) -> None:
        if key in self.d:
            raise KeyError(f"duplicate number key {key}")
        self.d[key] = {"value": None if value is None else (float(value) if not isinstance(value, str) else value),
                       "display": display, "source": source}


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def pts(v, d=1):
    return f"{v * 100:+.{d}f} pts"


def num(v, d=3):
    return f"{v:.{d}f}"


def signed(v, d=3):
    return f"{v:+.{d}f}"


def intc(v):
    return f"{int(round(v)):,}"


def headline() -> pd.DataFrame:
    """Assessor, hedonic and both frozen models on identical 2025 sales."""
    p = pd.read_parquet(INTERIM_DIR / "stage8_predictions.parquet")
    p = p[(p.year == 2025) & p.method.isin(METHODS)].pivot(index="sale_id", columns="method", values="pred")
    tp = frames.sales().set_index("sale_id").target_price
    p = p.dropna().join(tp, how="inner")
    est = {k: p[k] for k in METHODS}
    res = compare(est, p.target_price, base="assessor")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(p), size=(N_BOOT, len(p)))
    y = np.log(p.target_price.to_numpy())
    rows = []
    for k in METHODS:
        x = np.log(p[k].to_numpy())
        slope = lambda xx, yy: np.polyfit(xx, yy, 1)[0]  # noqa: E731
        b = np.array([slope(x[i], y[i]) for i in idx])
        rows.append({"statistic": "calibration_slope", "method": k, "n": len(p), "estimate": slope(x, y),
                     "ci_low": float(np.percentile(b, 2.5)), "ci_high": float(np.percentile(b, 97.5))})
    res = pd.concat([res, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "headline_2025.csv", index=False)
    return res


def collect(h: pd.DataFrame) -> Numbers:
    N = Numbers()
    src_h = "reports/writeup/headline_2025.csv"
    get = lambda st, m: h[(h.statistic == st) & (h.method == m)].iloc[0]  # noqa: E731
    N.add("n_test", get("mdape_ln", "primary").n, intc(get("mdape_ln", "primary").n), src_h)
    for m in METHODS:
        for st, f in [("mdape_ln", pct), ("mdape", pct), ("cod", lambda v: num(v, 2)), ("median_ratio", num),
                      ("prd", num), ("prb", signed), ("mki", num), ("within_10", lambda v: pct(v, 0)),
                      ("within_20", lambda v: pct(v, 0)), ("calibration_slope", lambda v: num(v, 2))]:
            r = get(st, m)
            N.add(f"{st}_{m}", r.estimate, f(r.estimate), src_h)
            N.add(f"{st}_{m}_ci", None, f"[{f(r.ci_low)}, {f(r.ci_high)}]", src_h)
            if st in IAAO_TARGETS:
                N.add(f"meets_{st}_{m}", bool(r.meets_target), "yes" if r.meets_target else "no", src_h)
            if m != "assessor" and st != "calibration_slope" and not pd.isna(r.get("diff_vs_assessor")):
                df = (lambda v: pts(v)) if f in (pct,) or st.startswith("within") else (
                    (lambda v: signed(v, 2)) if st == "cod" else signed)
                N.add(f"{st}_{m}_diff", r.diff_vs_assessor, df(r.diff_vs_assessor), src_h)
                dci = (lambda v: f"{v * 100:+.1f}") if df is not signed and st != "cod" else df
                N.add(f"{st}_{m}_diff_ci", None, f"[{dci(r.diff_ci_low)}, {dci(r.diff_ci_high)}]", src_h)
    for st, (lo, hi) in IAAO_TARGETS.items():
        N.add(f"target_{st}", None, (f"<= {hi:g}" if lo is None else f"{lo:.2f} to {hi:.2f}"),
              "avm.ratio_study.IAAO_TARGETS")
    # Assessor ratio study (Stage 3, IAAO-trimmed, the standards check)
    a3 = pd.read_csv(REPORTS_DIR / "stage3" / "assessor_ratio_study_2025.csv")
    a3 = a3[(a3.trim == "IAAO trimmed") & (a3.stratum == "all")].set_index("statistic")
    src3 = "reports/stage3/assessor_ratio_study_2025.csv (IAAO trimmed, all)"
    N.add("a3_n", a3.loc["median_ratio", "n"], intc(a3.loc["median_ratio", "n"]), src3)
    for st, f in [("median_ratio", num), ("cod", lambda v: num(v, 1)), ("prd", num), ("prb", signed), ("mki", num)]:
        N.add(f"a3_{st}", a3.loc[st, "estimate"], f(a3.loc[st, "estimate"]), src3)
        N.add(f"a3_{st}_ci", None, f"[{f(a3.loc[st, 'ci_low'])}, {f(a3.loc[st, 'ci_high'])}]", src3)
    # Training size (frozen primary refit for 2025: train rows 2014-2024)
    s = frames.sales()
    N.add("n_clean", len(s), intc(len(s)), "interim/sales_clean.parquet (rows)")
    N.add("n_train", int(s.sale_year.between(2014, 2024).sum()), intc(s.sale_year.between(2014, 2024).sum()),
          "interim/sales_clean.parquet, sale_year 2014-2024")
    # Stage 6 lift (CV 2021-2024, independent variant)
    lt = pd.read_csv(REPORTS_DIR / "stage6" / "lift_table.csv")
    lt = lt[lt.variant == "independent"].set_index("config")
    src6 = "reports/stage6/lift_table.csv (independent, CV mdape_ln)"
    for k, lab in [("baseline", "base"), ("+all groups", "all")]:
        N.add(f"lift_cv_{lab}", lt.loc[k, "CV mdape_ln"], pct(lt.loc[k, "CV mdape_ln"], 2), src6)
    single = lt.drop(index=["baseline", "+all groups"])["CV mdape_ln"]
    best = single.idxmin()
    N.add("lift_best_group", best, best.lstrip("+").replace("_", " "), src6)
    N.add("lift_best_gain", lt.loc["baseline", "CV mdape_ln"] - single.min(),
          f"{(lt.loc['baseline', 'CV mdape_ln'] - single.min()) * 100:.2f} pts", src6)
    # Intervals (frozen primary, 2025)
    iv = pd.read_csv(REPORTS_DIR / "stage7" / "interval_coverage_2025.csv").set_index("group")
    src7i = "reports/stage7/interval_coverage_2025.csv"
    for g in iv.index:
        N.add(f"cover_{g}", iv.loc[g, "coverage"], pct(iv.loc[g, "coverage"]), src7i)
        N.add(f"width_{g}", iv.loc[g, "median_rel_width"], pct(iv.loc[g, "median_rel_width"], 0), src7i)
    # Comparable sales (Stage 7, tiered comps)
    ca = pd.read_csv(REPORTS_DIR / "stage7" / "comp_accuracy_2025.csv")
    src7c = "reports/stage7/comp_accuracy_2025.csv"
    for m, lab in [("comps (a) time-adjusted", "a"), ("comps (b) model-adjusted", "b"), ("model", "model")]:
        r = ca[(ca.statistic == "mdape_ln") & (ca.method == m)].iloc[0]
        N.add(f"comp_{lab}_mdape", r.estimate, pct(r.estimate), src7c)
        if lab == "a":
            N.add("comp_n", r.n, intc(r.n), src7c)
    tiers = pd.read_csv(REPORTS_DIR / "stage7" / "comp_tiers_2025.csv")
    first = tiers.iloc[0]
    N.add("comp_first_tier_share", first.share, pct(first.share, 0), "reports/stage7/comp_tiers_2025.csv")
    N.add("comp_radii", None, ", ".join(f"{r:g}" for r in C.RADII_MI) + " mi", "avm.comps.RADII_MI")
    N.add("comp_windows", None, " then ".join(str(w) for w in C.WINDOWS_MONTHS) + " months",
          "avm.comps.WINDOWS_MONTHS")
    # Equity gaps (Stage 8, CV 2021-2024)
    g8 = pd.read_csv(REPORTS_DIR / "stage8" / "gaps.csv")
    g8 = g8[g8.split == "CV 2021-2024"]
    src8 = "reports/stage8/gaps.csv (CV 2021-2024)"
    for c, lab in [("Black majority minus White majority", "bw"), ("Q1 (lowest income) minus Q5 (highest income)", "q15")]:
        for meas, f in [("bias_mean_log", lambda v: f"{v * 100:+.1f} pts"),
                        ("mdape_vs_overall_level", lambda v: f"{v * 100:+.1f} pts")]:
            for m in ["assessor", "primary"]:
                r = g8[(g8.contrast == c) & (g8.measure == meas) & (g8.method == m)].iloc[0]
                N.add(f"gap_{lab}_{meas}_{m}", r.gap, f(r.gap), src8)
    # Stage 9
    st9 = pd.read_csv(REPORTS_DIR / "stage9" / "divergence_stats.csv").iloc[0]
    src9 = "reports/stage9/divergence_stats.csv"
    for k in ["parcels_scored", "outside_80pct_range", "listable_outside_and_comps_agree", "special_listable",
              "listed_student_area"]:
        N.add(f"s9_{k}", st9[k], intc(st9[k]), src9)
    gs = pd.read_csv(REPORTS_DIR / "stage9" / "group_tax_shifts.csv")
    is_claim = gs.verdict.str.endswith("vs market share")
    claims = int(is_claim.sum())
    N.add("s9_group_claims", claims, "no" if claims == 0 else str(claims), "reports/stage9/group_tax_shifts.csv")
    N.add("s9_group_result", None, group_result_sentence(gs), "reports/stage9/group_tax_shifts.csv (verdicts)")
    N.add("s9_group_claims_cap", claims, "No group passes" if claims == 0 else f"{claims} group(s) pass",
          "reports/stage9/group_tax_shifts.csv")
    tax = json.loads((REPORTS_DIR / "stage9" / "tax_rate.json").read_text())
    N.add("tax_gross", tax["gross_rate_per_1000"], f"${tax['gross_rate_per_1000']:.2f}", "reports/stage9/tax_rate.json")
    N.add("tax_status", tax["status"], "provisional" if not tax["status"].startswith("CONFIRMED") else "confirmed",
          "reports/stage9/tax_rate.json")
    # Prospective test
    pm = json.loads((REPORTS_DIR / "prospective" / "prospective_meta.json").read_text())
    srcp = "reports/prospective/prospective_meta.json"
    N.add("pros_rows", pm["rows"], intc(pm["rows"]), srcp)
    N.add("pros_sha", pm["predictions_sha256"], pm["predictions_sha256"][:12], srcp)
    N.add("pros_date", pm["created_at_utc"], pm["created_at_utc"][:10], srcp)
    collect_more(N, pm, st9, gs, g8)
    collect_review(N)
    collect_rebuild(N)
    collect_check2(N)
    collect_stage11(N)
    N.add("repo_slug", None, repo_slug(), "AVM_REPO_SLUG or the git origin remote")
    return N


def collect_stage11(N: Numbers) -> None:
    """Stage 11 (issue #4): within-tier PRB verdict, per-tier table and tradeoff.
    The verdict text is avm.stage11_prb.VERDICT_WORDING; the claims in it are
    asserted against the outputs so a changed result fails the build."""
    from avm import stage11_prb as S
    from avm.reporting import md
    d = REPORTS_DIR / "stage11_prb"
    src = "reports/stage11_prb/"
    v = json.loads((d / "verdict.json").read_text())["results"]
    ev = pd.read_csv(d / "within_tier_eval.csv").fillna({"robustness": ""})
    cv = ev[(ev.split == v["split"]) & (ev.robustness == "")].set_index(["tier", "method"])
    tc = pd.read_csv(d / "tradeoff.csv").set_index(["tier", "point"])
    rep = pd.read_csv(d / "proxy_representativeness.csv").set_index("subset")
    noise = pd.read_csv(d / "sale_noise.csv").set_index("tier")
    t4 = pd.read_csv(d / "tier_time_adjustment.csv")
    # The wording's claims, checked.
    if v["model"]["verdict"] != "mixed, not resolved":
        raise AssertionError(f"Stage 11 verdict is {v['model']['verdict']!r}; the README wording assumes 'mixed'")
    for m in ("model", "model_no_own_sale"):
        if (cv.xs(m, level="method").d_hi < 0).any():
            raise AssertionError(f"{m}: a tier has D below 0; 'no tier shows compression' no longer holds")
    e = cv.xs("model", level="method")
    if not (e.loc["mid", "e_hi"] < 0 and abs(e.loc["mid", "E"]) <= S.BAND
            and all(e.loc[t, "e_lo"] <= 0 <= e.loc[t, "e_hi"] for t in ("low", "high"))):
        raise AssertionError("E pattern changed: the wording names a small mid-tier excess only")
    for t in S.TIERS:
        a_, k1 = tc.loc[(t, "assessor")], tc.loc[(t, "step 0")]
        if not (a_.calibration_slope < k1.calibration_slope and a_.prb_obs > k1.prb_obs):
            raise AssertionError(f"{t}: the assessor no longer sits at the spread end of the tradeoff")
    w = "race: White majority"
    hi_inc = ["income: Q4", "income: Q5 (highest income)"]
    sub, all_ = rep.loc["with proxy sale"], rep.loc["all common sales"]
    if not (sub[w] > all_[w] and sub[hi_inc].sum() > all_[hi_inc].sum()):
        raise AssertionError("the prior-sale subset no longer over-represents White-majority and higher-income tracts")
    N.add("s11_verdict_text", v["model"]["verdict"],
          S.VERDICT_WORDING.format(verdict=v["model"]["verdict"], mid_e=f"{e.loc['mid', 'E']:.3f}"),
          src + "verdict.json, within_tier_eval.csv (wording: avm.stage11_prb.VERDICT_WORDING)")
    N.add("s11_mid_e", e.loc["mid", "E"], signed(e.loc["mid", "E"]), src + "within_tier_eval.csv")
    N.add("s11_mid_e_ci", None, f"[{signed(e.loc['mid', 'e_lo'])}, {signed(e.loc['mid', 'e_hi'])}]",
          src + "within_tier_eval.csv")
    N.add("s11_low_d", e.loc["low", "D"], signed(e.loc["low", "D"]), src + "within_tier_eval.csv")
    N.add("s11_low_d_ci", None, f"[{signed(e.loc['low', 'd_lo'])}, {signed(e.loc['low', 'd_hi'])}]",
          src + "within_tier_eval.csv")
    N.add("s11_pairs", noise.pairs.sum(), intc(noise.pairs.sum()), src + "sale_noise.csv")
    for t in S.TIERS:
        N.add(f"s11_sigma_{t}", noise.loc[t, "sigma_p"], num(noise.loc[t, "sigma_p"], 2), src + "sale_noise.csv")
    N.add("s11_proxy_share", sub.share_of_all, pct(sub.share_of_all, 0), src + "proxy_representativeness.csv")
    N.add("s11_proxy_months", S.PROXY_MIN_MONTHS, str(S.PROXY_MIN_MONTHS), "avm.stage11_prb.PROXY_MIN_MONTHS")
    N.add("s11_n_sim", S.N_SIM, intc(S.N_SIM), "avm.stage11_prb.N_SIM")
    N.add("s11_k_steps", S.N_K, str(S.N_K), "avm.stage11_prb.N_K")
    N.add("s11_white_proxy", sub[w], pct(sub[w], 0), src + "proxy_representativeness.csv")
    N.add("s11_white_all", all_[w], pct(all_[w], 0), src + "proxy_representativeness.csv")
    N.add("s11_hiinc_proxy", sub[hi_inc].sum(), pct(sub[hi_inc].sum(), 0), src + "proxy_representativeness.csv")
    N.add("s11_hiinc_all", all_[hi_inc].sum(), pct(all_[hi_inc].sum(), 0), src + "proxy_representativeness.csv")
    ch = (t4.prb_tercile_time_adj - t4.prb_citywide_time_adj).abs().max()
    N.add("s11_t4_max", ch, num(ch, 3), src + "tier_time_adjustment.csv")
    lo = tc.xs("low", level="tier")
    kx = lo.loc[f"step {S.N_K - 1}"]
    for key, r in [("cal", lo.loc["calibrated (k = b)"]), ("k1", lo.loc["step 0"]), ("kmax", kx),
                   ("ass", lo.loc["assessor"])]:
        N.add(f"s11_low_{key}_slope", r.calibration_slope, num(r.calibration_slope, 2), src + "tradeoff.csv")
        N.add(f"s11_low_{key}_prb", r.prb_obs, signed(r.prb_obs, 2), src + "tradeoff.csv")
        N.add(f"s11_low_{key}_mdape", r.mdape_ln, pct(r.mdape_ln), src + "tradeoff.csv")
    N.add("s11_low_kmax_k", kx.k_mean, num(kx.k_mean, 2), src + "tradeoff.csv")
    ass_sl = tc.xs("assessor", level="point").calibration_slope
    k1_sl = tc.xs("step 0", level="point").calibration_slope
    N.add("s11_ass_slope_range", None, f"{ass_sl.min():.2f} to {ass_sl.max():.2f}", src + "tradeoff.csv")
    N.add("s11_k1_slope_range", None, f"{k1_sl.min():.2f} to {k1_sl.max():.2f}", src + "tradeoff.csv")
    names = {"model": "Model", "model_no_own_sale": "Model without its own-sale feature", "assessor": "City Assessor"}
    rows = []
    for t in S.TIERS:
        for m in names:
            r = cv.loc[(t, m)]
            rows.append({"tier": t, "method": names[m], "PRB [95% CI]": f"{r.prb_obs:+.3f} [{r.prb_lo:+.3f}, {r.prb_hi:+.3f}]",
                         "calibrated-null PRB": f"{r.prb_null:+.3f}",
                         "E [95% CI]": f"{r.E:+.3f} [{r.e_lo:+.3f}, {r.e_hi:+.3f}]",
                         "D = Q minus Q_null [95% CI]": f"{r.D:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]"})
    N.add("s11_table", None, md(pd.DataFrame(rows)), src + "within_tier_eval.csv")
    trows = []
    for t in S.TIERS:
        g = tc.xs(t, level="tier")
        for label, key in [("rescaled to calibration (k = b)", "calibrated (k = b)"), ("frozen model (k = 1)", "step 0"),
                           ("spread (k = 1/b)", f"step {S.N_K - 1}"), ("City Assessor", "assessor")]:
            r = g.loc[key]
            trows.append({"tier": t, "point": label, "k": "" if key == "assessor" else f"{r.k_mean:.2f}",
                          "calibration slope": f"{r.calibration_slope:.2f}", "PRB": f"{r.prb_obs:+.3f}",
                          "level-neutral MdAPE": pct(r.mdape_ln)})
    N.add("s11_tradeoff_table", None, md(pd.DataFrame(trows)), src + "tradeoff.csv")


# ------------------------------------------------------------ hero figure ---

# Two themes for GitHub's <picture> switch; slots 1-2 of the dataviz default
# palette, validated against each background (all checks pass).
HERO_THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "model": "#2a78d6", "assessor": "#eb6834"},
    "dark": {"bg": "#0d1117", "ink": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d",
             "model": "#3987e5", "assessor": "#d95926"}}
HERO_GROUPS = {"all": "All 2025 sales", "low": "Low-value\nneighborhoods", "mid": "Mid-value\nneighborhoods",
               "high": "High-value\nneighborhoods"}
HERO_TITLE = "More accurate than the City Assessor on the same 2025 sales"


def hero() -> pd.DataFrame:
    """Level-neutral MdAPE, frozen model vs assessor, on the 2025 held-out
    sales: overall and by neighborhood value tier, with 95% paired-bootstrap
    intervals (evaluate.compare). The overall row must reproduce the
    headline exactly. Writes hero_mdape_2025.csv and a light and dark PNG."""
    p = pd.read_parquet(INTERIM_DIR / "stage8_predictions.parquet")
    p = p[(p.year == 2025) & p.method.isin(METHODS)].pivot(index="sale_id", columns="method", values="pred")
    p = p.dropna().join(frames.sales().set_index("sale_id").target_price, how="inner")
    tier = pd.read_parquet(INTERIM_DIR / "frozen_primary_predictions_2025.parquet").set_index("sale_id").tier
    p["tier"] = tier.reindex(p.index)
    if p.tier.isna().any():
        raise AssertionError("2025 sales without a value tier")
    rows = []
    for g_key in HERO_GROUPS:
        g = p if g_key == "all" else p[p.tier == g_key]
        r = compare({"model": g.primary, "assessor": g.assessor}, g.target_price, base="assessor")
        r = r[r.statistic == "mdape_ln"].set_index("method")
        rows.append({"group": g_key, "n": len(g),
                     **{f"{m}{x}": r.loc[m, c] for m in ("model", "assessor")
                        for x, c in (("", "estimate"), ("_lo", "ci_low"), ("_hi", "ci_high"))},
                     "diff": r.loc["model", "diff_vs_assessor"], "diff_lo": r.loc["model", "diff_ci_low"],
                     "diff_hi": r.loc["model", "diff_ci_high"]})
    t = pd.DataFrame(rows)
    h = pd.read_csv(OUT / "headline_2025.csv")
    hm = h[h.statistic == "mdape_ln"].set_index("method")
    a = t.iloc[0]
    want = [hm.loc["primary", "estimate"], hm.loc["assessor", "estimate"], hm.loc["primary", "diff_vs_assessor"],
            hm.loc["primary", "diff_ci_low"], hm.loc["primary", "diff_ci_high"], hm.loc["primary", "n"]]
    if not np.allclose([a.model, a.assessor, a["diff"], a.diff_lo, a.diff_hi, a.n], want, rtol=0, atol=1e-12):
        raise AssertionError("hero overall row does not reproduce the headline")
    t.to_csv(OUT / "hero_mdape_2025.csv", index=False)
    for theme in HERO_THEMES:
        hero_figure(t, theme, OUT / f"hero-mdape-2025-{theme}.png")
    return t


def hero_figure(t: pd.DataFrame, theme: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    c = HERO_THEMES[theme]
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200, facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    fig.subplots_adjust(left=0.24, right=0.74, top=0.70, bottom=0.12)
    ys = np.arange(len(t))[::-1].astype(float)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for y, r in zip(ys, t.itertuples()):
        for m, dy, mk in (("model", 0.15, "o"), ("assessor", -0.15, "s")):
            est, lo, hi = (getattr(r, m) * 100, getattr(r, f"{m}_lo") * 100, getattr(r, f"{m}_hi") * 100)
            ax.plot([lo, hi], [y + dy] * 2, color=c[m], lw=2.5, solid_capstyle="round", zorder=2)
            ax.plot(est, y + dy, mk, color=c[m], ms=10, mec=c["bg"], mew=1.5, zorder=3)
            if r.group == "all":
                ax.text(lo - 0.3, y + dy, "Model" if m == "model" else "Assessor", ha="right", va="center",
                        color=c["ink"], fontsize=10)
        ax.text(1.04, y, f"{r.diff * 100:+.1f} pts\n[{r.diff_lo * 100:+.1f}, {r.diff_hi * 100:+.1f}]",
                transform=trans, ha="left", va="center", color=c["ink"], fontsize=11, linespacing=1.3)
    ax.text(1.04, ys[0] + 0.62, "Model minus\nassessor", transform=trans, ha="left", va="bottom",
            color=c["muted"], fontsize=10, linespacing=1.2)
    ax.set_yticks(ys, [f"{HERO_GROUPS[g]}\nn = {n:,}" for g, n in zip(t.group, t.n)], fontsize=11, color=c["ink"])
    lo = min(t.model_lo.min(), t.assessor_lo.min()) * 100
    hi = max(t.model_hi.max(), t.assessor_hi.max()) * 100
    ax.set_xlim(np.floor(lo) - 2, np.ceil(hi) + 0.5)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    # Whole-number ticks only, so the integer labels are exact.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="x", colors=c["muted"], labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=c["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Level-neutral median absolute error", color=c["muted"], fontsize=10)
    fig.text(0.03, 0.955, HERO_TITLE, color=c["ink"], fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.885, f"Held-out 2025 sales (n = {t.n.iloc[0]:,}), lower is better. Lines are 95% "
             "paired-bootstrap intervals.", color=c["muted"], fontsize=10, ha="left", va="top")
    handles = [plt.Line2D([], [], color=c["model"], marker="o", lw=2.5, ms=9, label="Model (public data)"),
               plt.Line2D([], [], color=c["assessor"], marker="s", lw=2.5, ms=9, label="City Assessor")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.84), ncol=2, frameon=False,
                     fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(c["ink"])
    fig.savefig(path, facecolor=c["bg"], metadata={"Software": None})
    plt.close(fig)


def hero_caption(t: pd.DataFrame) -> str:
    """One honest sentence per group from the paired intervals."""
    names = {"all": "all 2025 sales", "low": "low-value neighborhoods", "mid": "mid-value neighborhoods",
             "high": "high-value neighborhoods"}
    parts = []
    for r in t.itertuples():
        verdict = ("the model is more accurate" if r.diff_hi < 0 else
                   "the assessor is more accurate" if r.diff_lo > 0 else "no clear difference")
        parts.append(f"{names[r.group]} {r.diff * 100:+.1f} pts [{r.diff_lo * 100:+.1f}, {r.diff_hi * 100:+.1f}], "
                     f"{verdict}")
    return ("Model minus assessor, in points of level-neutral median absolute error: " + "; ".join(parts) +
            ". Neighborhood value tiers are terciles of each neighborhood's median sale price over the three "
            "prior years; homes in neighborhoods with too few sales to rate are in the overall group only.")


def table(df: pd.DataFrame) -> str:
    from avm.reporting import md
    return md(df)


def collect_more(N: Numbers, pm: dict, st9: pd.Series, gs: pd.DataFrame, g8: pd.DataFrame) -> None:
    """Numbers and generated tables for the full README and docs/methods.md."""
    from avm.features import FEATURE_DOC
    from avm.stage6_geo import FEATURE_DOC_6
    from avm.stage9_divergence import CONTEXT_FLAGS, QUALITY_RULES, SPECIAL_RULES
    srcp, src9 = "reports/prospective/prospective_meta.json", "reports/stage9/divergence_stats.csv"
    N.add("pros_sha_full", pm["predictions_sha256"], pm["predictions_sha256"], srcp)
    N.add("pros_parcels", pm["parcels_sf_duplex_roll_2025"], intc(pm["parcels_sf_duplex_roll_2025"]), srcp)
    N.add("pros_train", pm["train_rows"], intc(pm["train_rows"]), srcp)
    # Stage 2 filter log (2014-2025)
    fl = pd.read_csv(REPORTS_DIR / "stage2" / "filter_log.csv")
    src2 = "reports/stage2/filter_log.csv"
    by = fl.groupby(["code", "filter"], sort=False).agg(rows_in=("rows_in", "sum"), removed=("removed", "sum"),
                                                       rows_out=("rows_out", "sum")).reset_index()
    by["code_n"] = by.code.str[1:].astype(int)
    by = by.sort_values("code_n")
    N.add("raw_sales", by.rows_in.iloc[0], intc(by.rows_in.iloc[0]), src2 + " (F0 rows_in, 2014-2025)")
    N.add("clean_sales", by.rows_out.iloc[-1], intc(by.rows_out.iloc[-1]), src2 + " (F11 rows_out, 2014-2025)")
    N.add("filter_table", None, table(pd.DataFrame({"filter": by.code + " " + by["filter"],
                                                    "removed": by.removed.map(intc),
                                                    "remaining": by.rows_out.map(intc)})), src2)
    for code in ["F5", "F10"]:
        r = by[by.code == code].iloc[0]
        N.add(f"removed_{code}", r.removed, intc(r.removed), src2)
    # Stage 8 cleaning rates
    cr = pd.read_csv(REPORTS_DIR / "stage8" / "cleaning_rates_by_group.csv").set_index("group")
    src8c = "reports/stage8/cleaning_rates_by_group.csv"
    for g, lab in [("Black majority", "black"), ("White majority", "white"), ("Q1 (lowest income)", "q1"),
                   ("Q5 (highest income)", "q5")]:
        N.add(f"clean_rate_{lab}", cr.loc[g, "cleaning_rate_F6_F11"], pct(cr.loc[g, "cleaning_rate_F6_F11"]), src8c)
        N.add(f"flip_rate_{lab}", cr.loc[g, "share_F10"], pct(cr.loc[g, "share_F10"]), src8c)
    for c, lab in [("Black majority minus White majority", "bw"), ("Q1 (lowest income) minus Q5 (highest income)", "q15")]:
        for m in ["assessor", "primary"]:
            r = g8[(g8.contrast == c) & (g8.measure == "cod") & (g8.method == m)].iloc[0]
            N.add(f"gap_{lab}_cod_{m}", r.gap, f"{r.gap:+.1f}", "reports/stage8/gaps.csv (CV 2021-2024)")
    # Frozen spec
    spec = json.loads((REPORTS_DIR / "frozen" / "model_spec.json").read_text())
    p = spec["models"]["primary"]
    srcs = "reports/frozen/model_spec.json"
    N.add("n_features", len(p["features"]), str(len(p["features"])), srcs)
    for k in ["num_leaves", "learning_rate", "num_boost_round", "min_data_in_leaf", "feature_fraction",
              "bagging_fraction", "lambda_l2"]:
        v = p["config"][k]
        N.add(f"cfg_{k}", v, f"{v:g}", srcs)
    N.add("seed", spec["lightgbm_seed"], str(spec["lightgbm_seed"]), srcs)
    N.add("spec_sha", None, sha_of(REPORTS_DIR / "frozen" / "model_spec.json")[:12], "sha256 of " + srcs)
    feats = {**{k: v for k, v in FEATURE_DOC.items() if k in p["features"]},
             **{k: v for k, v in FEATURE_DOC_6.items()
                if any(t.strip() in p["features"] for t in k.split(","))}}
    allnotes = {**FEATURE_DOC, **FEATURE_DOC_6}
    note = lambda v: allnotes[v[3:]][1] if v.startswith("as ") and v[3:] in allnotes else v  # noqa: E731
    N.add("feature_table", None, table(pd.DataFrame([{"feature": k, "what": v[0], "proxy note": note(v[1])}
                                                     for k, v in feats.items()])),
          "avm.features.FEATURE_DOC and avm.stage6_geo.FEATURE_DOC_6, filtered to the frozen primary's features")
    iv = pd.read_csv(REPORTS_DIR / "stage7" / "interval_coverage_2025.csv").set_index("group")
    N.add("cqr_q", iv.loc["all", "q_global"], num(iv.loc["all", "q_global"], 4), "reports/stage7/interval_coverage_2025.csv")
    # Stage 9
    for k in ["listed", "listed_over_assessed", "listed_under_assessed", "special_below_model", "special_above_model",
              "listed_student_area_above_model", "listed_student_area_below_model", "all_student_area"]:
        N.add(f"s9_{k}", st9[k], intc(st9[k]), src9)
    N.add("s9_level", st9.citywide_assessment_level, num(st9.citywide_assessment_level), src9)
    N.add("s9_median_shift", st9.listed_tax_shift_usd_median_abs, f"${st9.listed_tax_shift_usd_median_abs:,.0f}", src9)
    N.add("s9_change", st9.all_change_2024_to_2025_median, pct(st9.all_change_2024_to_2025_median - 1),
          src9 + " (median 2025/2024 total, minus one)")
    N.add("s9_flat_share", st9.all_share_unchanged_from_2024, pct(st9.all_share_unchanged_from_2024), src9)
    N.add("s9_rules_table", None, table(pd.DataFrame([{"rule": k, "definition": v,
                                                       "parcels": intc(st9[f"special_{k.replace(' ', '_')}"])}
                                                      for k, v in SPECIAL_RULES.items()])),
          "avm.stage9_divergence.SPECIAL_RULES and " + src9)
    ex = pd.read_csv(REPORTS_DIR / "stage9" / "exclusions.csv")
    from avm.stage9_divergence import NO_ASSESSMENT
    rules = {NO_ASSESSMENT: "roll 2025 total missing or zero (dropped before the rules below)", **QUALITY_RULES}
    N.add("s9_excl_table", None, table(pd.DataFrame([{"rule": k, "definition": v,
                                                      "parcels": intc(ex.set_index("excluded_reason").parcels.get(k, 0))}
                                                     for k, v in rules.items()])),
          "avm.stage9_divergence.QUALITY_RULES and reports/stage9/exclusions.csv")
    N.add("s9_flags_table", None, table(pd.DataFrame([{"flag": k, "definition": v} for k, v in CONTEXT_FLAGS.items()])),
          "avm.stage9_divergence.CONTEXT_FLAGS")
    pc = lambda v: "n/a" if pd.isna(v) else f"{(np.exp(v) - 1) * 100:+.1f}%"  # noqa: E731
    ci = lambda a, b, c: f"{pc(a)} [{pc(b)}, {pc(c)}]"  # noqa: E731
    rc = lambda a, b, c: "n/a" if pd.isna(a) else f"{(a - 1) * 100:+.1f}% [{(b - 1) * 100:+.1f}, {(c - 1) * 100:+.1f}]"  # noqa: E731
    yn = lambda v: "pass" if v else "fail"  # noqa: E731
    gt = pd.DataFrame({"group": gs.group, "homes": gs.parcels.map(intc),
                       "shift [95% CI]": [ci(*t) for t in zip(gs.median_shift_log, gs.shift_lo, gs.shift_hi)],
                       "model bias [95% CI]": [ci(*t) for t in zip(gs.model_bias_rel_log, gs.bias_lo, gs.bias_hi)],
                       "corrected shift [95% CI]": [ci(*t) for t in zip(gs.corrected_log, gs.corrected_lo,
                                                                        gs.corrected_hi)],
                       "assessor ratio on 2025 sales vs city": [rc(*t) for t in zip(gs.sales_rel, gs.sales_rel_lo,
                                                                                     gs.sales_rel_hi)],
                       "corrected, tercile time adj.": [ci(*t) for t in zip(gs.corrected_log_tercile,
                                                                           gs.corrected_lo_tercile,
                                                                           gs.corrected_hi_tercile)],
                       "sales check, tercile": [rc(*t) for t in zip(gs.sales_rel_tercile, gs.sales_rel_lo_tercile,
                                                                    gs.sales_rel_hi_tercile)],
                       "(a) / (b) / (c)": [f"{yn(a)} / {yn(b)} / {yn(c)}" for a, b, c in
                                           zip(gs.check_a, gs.check_b, gs.check_c)],
                       "result": gs.verdict})
    N.add("s9_group_table", None, table(gt), "reports/stage9/group_tax_shifts.csv")
    N.add("s9_by_year_table", None, by_year_table(), "reports/stage9/group_sales_check_by_year.csv")
    tax = json.loads((REPORTS_DIR / "stage9" / "tax_rate.json").read_text())
    N.add("tax_net", tax["net_rate_per_1000"], f"${tax['net_rate_per_1000']:.2f}", "reports/stage9/tax_rate.json")
    N.add("tax_prior", tax["prior_year_net_rate_per_1000"], f"${tax['prior_year_net_rate_per_1000']:.2f}",
          "reports/stage9/tax_rate.json (City of Milwaukee 2024 rate, county table)")
    N.add("tax_excluded", None, tax["excluded"], "reports/stage9/tax_rate.json")
    N.add("ccao_read", None, CCAO_READ, "date the Cook County README was read for the comparison")
    from avm import clean_sales, freeze, priors, sales
    from avm import equity as eq
    for key, v, disp, src in [
        ("flip_months", sales.FLIP_MONTHS, f"{sales.FLIP_MONTHS} months", "avm.sales.FLIP_MONTHS"),
        ("multi_radius", clean_sales.MULTI_RADIUS_M, f"{clean_sales.MULTI_RADIUS_M} m", "avm.clean_sales.MULTI_RADIUS_M"),
        ("iqr_k", clean_sales.IQR_K, f"{clean_sales.IQR_K:g}", "avm.clean_sales.IQR_K"),
        ("knn_k", priors.K, str(priors.K), "avm.priors.K"),
        ("knn_window", priors.WINDOW_MONTHS, f"{priors.WINDOW_MONTHS} months", "avm.priors.WINDOW_MONTHS"),
        ("own_min", priors.OWN_MIN_MONTHS, f"{priors.OWN_MIN_MONTHS} months", "avm.priors.OWN_MIN_MONTHS"),
        ("n_configs", freeze.N_CONFIGS, str(freeze.N_CONFIGS), "avm.freeze.N_CONFIGS"),
        ("mdape_tol", freeze.MDAPE_TOL, f"{freeze.MDAPE_TOL * 100:g} pts", "avm.freeze.MDAPE_TOL"),
        ("n_boot", N_BOOT, intc(N_BOOT), "avm.ratio_study.N_BOOT"),
        ("min_tracts_group", eq.MIN_TRACTS_OWN_GROUP, str(eq.MIN_TRACTS_OWN_GROUP), "avm.equity.MIN_TRACTS_OWN_GROUP"),
    ]:
        N.add(key, v, disp, src)
    # Comps before/after (Stage 7)
    b = pd.read_csv(REPORTS_DIR / "stage7" / "comp_accuracy_2025_before_tiered.csv")
    r = b[(b.statistic == "mdape_ln") & (b.method == "comps (a) time-adjusted")].iloc[0]
    N.add("comp_a_mdape_before", r.estimate, pct(r.estimate), "reports/stage7/comp_accuracy_2025_before_tiered.csv")


def collect_review(N: Numbers) -> None:
    """Numbers added after the independent claim check (2026-10-20)."""
    g8 = pd.read_csv(REPORTS_DIR / "stage8" / "gaps.csv")
    src8 = "reports/stage8/gaps.csv"
    for split, sl in [("CV 2021-2024", "cv"), ("2025", "y25")]:
        for c, lab in [("Black majority minus White majority", "bw"), ("Q1 (lowest income) minus Q5 (highest income)", "q15")]:
            for m in ["assessor", "primary"]:
                r = g8[(g8.split == split) & (g8.contrast == c) & (g8.measure == "bias_mean_log") & (g8.method == m)].iloc[0]
                N.add(f"lvl_{lab}_{m}_{sl}", r.gap, f"{r.gap * 100:+.1f}", f"{src8} ({split}, mean log ratio x 100)")
                if m == "primary":
                    N.add(f"lvl_{lab}_diff_{sl}", r.abs_gap_minus_assessor,
                          f"{r.abs_gap_minus_assessor * 100:+.1f} [{r.diff_lo * 100:+.1f}, {r.diff_hi * 100:+.1f}]",
                          f"{src8} ({split}, |gap| minus assessor |gap|)")
    t = pd.read_csv(REPORTS_DIR / "stage8" / "group_tables.csv")
    t = t[t.split == "CV 2021-2024"]
    for g, lab in [("Black majority", "black"), ("White majority", "white"), ("Q1 (lowest income)", "q1"),
                   ("Q5 (highest income)", "q5")]:
        for m in ["assessor", "primary"]:
            r = t[(t.group == g) & (t.method == m)].iloc[0]
            N.add(f"grp_{lab}_{m}", r.bias_mean_log, f"{r.bias_mean_log:+.3f}",
                  "reports/stage8/group_tables.csv (CV 2021-2024, mean log ratio)")
    spec = json.loads((REPORTS_DIR / "frozen" / "model_spec.json").read_text())
    cv = spec["models"]["primary"]["cv_2021_2024"]
    for k, f in [("prd", num), ("prb", signed)]:
        lo, hi = IAAO_TARGETS[k]
        N.add(f"cv_{k}_primary", cv[k], f(cv[k]), "reports/frozen/model_spec.json (cv_2021_2024)")
        N.add(f"cv_meets_{k}_primary", None, "yes" if lo <= cv[k] <= hi else "no", "reports/frozen/model_spec.json")
    fe = pd.read_csv(REPORTS_DIR / "frozen" / "frozen_eval_2025.csv").set_index("role")
    wt = pd.read_csv(REPORTS_DIR / "stage6" / "within_tier_prb_2025.csv").set_index("method")
    for tier in ["low", "mid", "high"]:
        N.add(f"tier_prb_{tier}_primary", fe.loc["primary", f"prb_{tier}"], signed(fe.loc["primary", f"prb_{tier}"]),
              "reports/frozen/frozen_eval_2025.csv")
        N.add(f"tier_prb_{tier}_assessor", wt.loc["assessor", tier], signed(wt.loc["assessor", tier]),
              "reports/stage6/within_tier_prb_2025.csv")
    from avm.stage8_equity import HEDONIC
    N.add("hedonic_loc", None, {"neighborhood": "assessor-neighborhood", "tract": "census-tract"}[HEDONIC["loc"]],
          "avm.stage8_equity.HEDONIC")
    N.add("hedonic_start", HEDONIC["start"], str(HEDONIC["start"]), "avm.stage8_equity.HEDONIC")
    iv = pd.read_csv(REPORTS_DIR / "stage7" / "interval_coverage_2025.csv").set_index("group")
    for g in ["all", "mid", "high"]:
        d = iv.loc[g, "coverage"] - iv.loc[g, "nominal"]
        N.add(f"cover_excess_{g}", d, f"{d * 100:.1f} pts", "reports/stage7/interval_coverage_2025.csv (coverage - nominal)")
    tax = json.loads((REPORTS_DIR / "stage9" / "tax_rate.json").read_text())
    used = tax[f"{tax['rate_used']}_rate_per_1000"]
    N.add("tax_used", used, f"${used:.2f}", f"reports/stage9/tax_rate.json ({tax['rate_used']} rate)")
    N.add("tax_used_kind", tax["rate_used"], tax["rate_used"], "reports/stage9/tax_rate.json")

    N.add("tax_levy_credit", None, tax["school_levy_credit"], "reports/stage9/tax_rate.json")
    st9 = pd.read_csv(REPORTS_DIR / "stage9" / "divergence_stats.csv").iloc[0]
    N.add("s9_student_share", st9.all_student_area / st9.parcels_scored, pct(st9.all_student_area / st9.parcels_scored),
          "reports/stage9/divergence_stats.csv (all_student_area / parcels_scored)")
    cr = pd.read_csv(REPORTS_DIR / "stage8" / "cleaning_rates_by_group.csv").set_index("group")
    for g, lab in [("Black majority", "black"), ("White majority", "white")]:
        N.add(f"scope_rate_{lab}", cr.loc[g, "share_F7"], pct(cr.loc[g, "share_F7"]), "reports/stage8/cleaning_rates_by_group.csv")


def collect_check2(N: Numbers) -> None:
    """Numbers added after the second independent claim check (2026-10-20)."""
    g8 = pd.read_csv(REPORTS_DIR / "stage8" / "gaps.csv")
    for split, sl in [("CV 2021-2024", "cv"), ("2025", "y25")]:
        for c, lab in [("Black majority minus White majority", "bw"), ("Q1 (lowest income) minus Q5 (highest income)", "q15")]:
            for m in ["assessor", "primary"]:
                r = g8[(g8.split == split) & (g8.contrast == c) & (g8.measure == "median_ratio") & (g8.method == m)].iloc[0]
                N.add(f"mr_{lab}_{m}_{sl}", r.gap, f"{r.gap * 100:+.1f}", f"reports/stage8/gaps.csv ({split}, median ratio)")
                if m == "primary":
                    N.add(f"mr_{lab}_diff_{sl}", r.abs_gap_minus_assessor,
                          f"{r.abs_gap_minus_assessor * 100:+.1f} [{r.diff_lo * 100:+.1f}, {r.diff_hi * 100:+.1f}]",
                          f"reports/stage8/gaps.csv ({split}, median ratio, |gap| minus assessor |gap|)")
    st9 = pd.read_csv(REPORTS_DIR / "stage9" / "divergence_stats.csv").iloc[0]
    for k in ["parcels_sf_duplex_roll_2024", "parcels_model_scored"]:
        N.add(f"s9_{k}", st9[k], intc(st9[k]), "reports/stage9/divergence_stats.csv")
    spec = json.loads((REPORTS_DIR / "frozen" / "model_spec.json").read_text())
    v = spec["models"]["primary"]["cv_2021_2024"]["mdape_ln"]
    N.add("cv_mdape_frozen", v, pct(v, 2), "reports/frozen/model_spec.json (primary cv_2021_2024 mdape_ln)")
    tax = json.loads((REPORTS_DIR / "stage9" / "tax_rate.json").read_text())
    N.add("tax_statute", None, tax["statute"], "reports/stage9/tax_rate.json")


def repo_slug() -> str:
    """GitHub owner/name for links and the CI badge: AVM_REPO_SLUG, else the
    origin remote, else the public repo."""
    import os
    import subprocess
    if os.environ.get("AVM_REPO_SLUG"):
        return os.environ["AVM_REPO_SLUG"]
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO_DIR, capture_output=True,
                         text=True).stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
    return m.group(1) if m else "oekrueg05/milwaukee-avm"


def collect_rebuild(N: Numbers) -> None:
    """Outcome of the last full rebuild, read from its saved log."""
    log_path = REPORTS_DIR / "rebuild" / "rebuild_log.txt"
    src = "reports/rebuild/rebuild_log.txt"
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    ok = all(k in text for k in ["frozen spec reproduced exactly", "frozen predictions reproduced",
                                 "prospective predictions verified", "pipeline: done"]) and \
        text.rstrip().endswith("exit 0")
    N.add("rebuild_status", ok, "completed without error, reproduced the frozen spec and the frozen model's 2025 "
          "predictions exactly, and verified the prospective file's hash" if ok else "has not completed",
          src)
    rc = REPORTS_DIR / "rebuild" / "stage9_recheck.txt"
    rtext = rc.read_text(encoding="utf-8") if rc.exists() else ""
    N.add("stage9_recheck", None,
          "After later Stage 9 code fixes the same day, Stage 9 was re-run in the rebuild folder and every table "
          "matched the live run" if rtext.rstrip().endswith("ALL MATCH") else
          "Stage 9 has changed since the rebuild and has not been re-checked",
          "reports/rebuild/stage9_recheck.txt")


def group_result_sentence(gs: pd.DataFrame) -> str:
    """Plain statement of the guardrail outcome, built from the verdicts."""
    name = lambda g: {"Black majority": "Black-majority tracts", "Hispanic majority": "Hispanic-majority tracts",  # noqa: E731
                      "White majority": "White-majority tracts", "no majority (mixed)": "mixed tracts",
                      "Q1 (lowest income)": "the lowest income quintile",
                      "Q5 (highest income)": "the highest income quintile"}.get(g, f"income quintile {g}")
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]  # noqa: E731
    parts = []
    for side, tax in [("under-assessed", "below"), ("over-assessed", "above")]:
        g = [name(x) for x in gs[gs.verdict == f"{side} vs market share"].group]
        if g:
            parts.append(f"{join(g)} {'is' if len(g) == 1 else 'are'} {side}, assessed and taxed {tax} "
                         "their market-value share")
    lag = [name(x) for x in gs[gs.verdict.str.startswith("consistent with")].group]
    out = ("Passing all three: " + "; ".join(parts) + ".") if parts else "No group passes all three checks."
    n_early = 4  # rolls 2021-2024
    for r in gs[gs.verdict.str.endswith("vs market share")].itertuples():
        less = r.usd_typical_home < 0
        label = "persistent" if r.persistence == "persistent" else "a 2025 result"
        out += (f" {name(r.group)[0].upper() + name(r.group)[1:]}: {label} (the tier-adjusted sales check holds in "
                f"{int(r.years_holding)} of {n_early} earlier rolls); a typical home (assessed "
                f"${r.median_assessed_2025:,.0f}) pays about ${abs(r.usd_typical_home):,.0f} a year "
                f"{'less' if less else 'more'}, ${abs(r.usd_group_total) / 1e6:,.1f} million across "
                f"{r.parcels:,} homes, and the fixed levy puts about ${abs(r.usd_offset_per_other_home):,.0f} a year "
                f"{'more' if less else 'less'} on each of the other {r.other_homes:,}.")
    if (gs.verdict.str.endswith("vs market share")).sum() > 1:
        out += " The groups overlap, so the amounts do not add."
    if lag:
        out += (f" {join(lag)} fail only the tier-specific time adjustment: consistent with assessments lagging "
                "faster appreciation in some areas, not a valuation pattern.")
    for r in conflicts(gs).itertuples():
        pc = lambda v: f"{(np.exp(v) - 1) * 100:+.1f}%"  # noqa: E731
        rc = lambda v: f"{(v - 1) * 100:+.1f}%"  # noqa: E731
        out += (f" {name(r.group)[0].upper() + name(r.group)[1:]}: the two checks conflict (the bias-corrected "
                f"shift, {pc(r.corrected_log)} [{pc(r.corrected_lo)}, {pc(r.corrected_hi)}], says "
                f"{'over' if r.corrected_log > 0 else 'under'}-assessed; the 2025 sales check, {rc(r.sales_rel)} "
                f"[{rc(r.sales_rel_lo)}, {rc(r.sales_rel_hi)}], says {'over' if r.sales_rel > 1 else 'under'}-assessed; "
                "both intervals exclude zero), so no claim is made.")
    return out


def conflicts(gs: pd.DataFrame) -> pd.DataFrame:
    """Groups whose corrected shift and 2025 sales check both exclude zero but point opposite ways."""
    excl_c = (gs.corrected_lo > 0) | (gs.corrected_hi < 0)
    excl_s = (gs.sales_rel_lo > 1) | (gs.sales_rel_hi < 1)
    return gs[excl_c & excl_s & (np.sign(gs.corrected_log) != np.sign(gs.sales_rel - 1))]


def by_year_table() -> str:
    """Tier-adjusted model-free sales check by group and roll year (percent vs city, 95% CI)."""
    from avm.reporting import md
    by = pd.read_csv(REPORTS_DIR / "stage9" / "group_sales_check_by_year.csv")
    by["cell"] = [f"{(a - 1) * 100:+.1f} [{(b - 1) * 100:+.1f}, {(c - 1) * 100:+.1f}]{' *' if h else ''}"
                  for a, b, c, h in zip(by.sales_rel_tercile, by.sales_rel_lo_tercile, by.sales_rel_hi_tercile,
                                        by.holds)]
    t = by.pivot_table(index=["grouping", "group"], columns="year", values="cell", aggfunc="first", sort=False)
    t.columns = [f"roll {c}" for c in t.columns]
    return md(t.reset_index().drop(columns="grouping"))


def sha_of(path) -> str:
    from avm.acquire import sha256_of
    return sha256_of(path)


def lint(template: str, name: str) -> None:
    if re.search("[–—]", template):
        raise ValueError(f"{name}: em or en dash in template")
    prose = re.sub(r"\{\{[a-z0-9_]+\}\}", "", template)
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)    # fenced code
    prose = re.sub(r"`[^`]*`", "", prose)                  # code spans
    prose = re.sub(r"\]\([^)]*\)", "]", prose)             # link targets
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
    for w in WHITELIST:
        prose = re.sub(w, "", prose)
    bad = [(i + 1, line.strip()) for i, line in enumerate(prose.splitlines()) if re.search(r"\d", line)]
    if bad:
        raise ValueError(f"{name}: digits outside placeholders/whitelist:\n" +
                         "\n".join(f"  line {i}: {t[:120]}" for i, t in bad[:20]))


def render(template: str, N: Numbers, name: str) -> tuple[str, set]:
    lint(template, name)
    used = set(re.findall(r"\{\{([a-z0-9_]+)\}\}", template))
    missing = used - set(N.d)
    if missing:
        raise KeyError(f"{name}: placeholders without a number: {sorted(missing)}")
    out = re.sub(r"\{\{([a-z0-9_]+)\}\}", lambda mm: N.d[mm.group(1)]["display"], template)
    if re.search("[–—]", out):
        raise ValueError(f"{name}: em or en dash in rendered output")
    return out, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    h = headline()
    N = collect(h)
    if not a.sample:
        t = hero()
        N.add("hero_caption", None, hero_caption(t), "reports/writeup/hero_mdape_2025.csv")
        N.add("hero_title", None, HERO_TITLE, "avm.writeup.HERO_TITLE")
    if a.sample:
        tpl = (TEMPLATES / "README_sample.template.md").read_text(encoding="utf-8")
        text, used = render(tpl, N, "README_sample")
        dest = OUT / "sample" / "README_sample.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        outs = {"README_sample": used}
    else:
        outs = {}
        for name, dest in [("README", REPO_DIR / "README.md"), ("methods", REPO_DIR / "docs" / "methods.md")]:
            text, used = render((TEMPLATES / f"{name}.template.md").read_text(encoding="utf-8"), N, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            outs[name] = used
    used_all = set().union(*outs.values())
    (OUT / "numbers.json").write_text(json.dumps({k: v | {"used_in": [n for n, u in outs.items() if k in u]}
                                                  for k, v in N.d.items() if k in used_all}, indent=1),
                                      encoding="utf-8")
    words = sum(len(re.findall(r"[A-Za-z][A-Za-z'-]*", t.read_text(encoding="utf-8")))
                for t in [OUT / "sample" / "README_sample.md"] if a.sample) if a.sample else None
    log(f"rendered {list(outs)}; {len(used_all)} numbers traced to sources in {OUT / 'numbers.json'}"
        + (f"; ~{words} words" if words else ""))


if __name__ == "__main__":
    main()
