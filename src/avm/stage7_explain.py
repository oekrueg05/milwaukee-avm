"""Stage 7: explainability for the frozen primary model (public data only).

Usage:
    python -m avm.stage7_explain

SHAP: LightGBM's exact TreeSHAP (pred_contrib) on 2025 test sales, in log
price. Global importance LEADS WITH FEATURE GROUPS (sum of SHAP within a
group, then mean |.|): coordinates, distances and nearby-sales features are
correlated and split credit, so single-feature ranks mislead.

Comparable sales for each 2025 test sale: avm.comps (same property type;
tiered 0.5 / 1 / 1.25 mi within 12 months, then 24; flips, IAAO and
price/sq ft outliers excluded; SHAP-weighted structural ranking).
  comp value (a): median of comp prices moved to the Jan 1 2025 trend level
                  (training-only time effect). No model adjustment.
  comp value (b): (a) per comp x model(subject) / model(comp). Circular (the
                  model adjusts its own check), so only (a) drives the
                  model-vs-comps disagreement flags.
Both are scored on 2025 against the shared target prices.

Integrity: the frozen primary model is refit through 2024 and its 2025
predictions must equal frozen_primary_predictions_2025.parquet.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from avm import frames
from avm.comps import STRUCT, TIERS, comps_for, eligible_pool
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.evaluate import accuracy, compare
from avm.gbm import GBMX
from avm.intervals import calibrate, contain_point
from avm.progress import log, step
from avm.reporting import md
from avm.stage6_geo import BASE_COLS, GROUPS

OUT = REPORTS_DIR / "stage7"
SPEC_PATH = REPORTS_DIR / "frozen" / "model_spec.json"
TEST_YEAR = 2025
BASE_START = 2014
CAL_YEARS = [2022, 2023, 2024]
DISAGREE = 0.25
BLUE, INK, INK2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SHAP_GROUPS = {"structure": [c for c in BASE_COLS if c != "loc"], "location effect (tract)": ["loc"], **GROUPS}

# Expected direction of SHAP vs feature value (+ rises with value, - falls,
# None = no prior expectation). Evidence: hedonic coefficients (Stage 4) and
# standard real-estate economics.
EXPECT = {"l_area": "+", "l_lot": "+", "full_baths": "+", "half_baths": "+", "age": "-", "beds": None,
          "stories_num": None, "d_lake_m": "-", "d_downtown_m": None, "d_major_road_m": "+", "d_transit_m": None,
          "d_park_m": "-", "knn_lppsf": "+", "nbhd_lppsf_24m": "+", "knn_dist_m": None, "prior_lp_detr": "+",
          "months_since_prior": None, "has_prior_sale": None, "n_city_200m": "-", "n_vacant_200m": "-",
          "x_m": None, "y_m": None}
UNITS = {"l_area": ("finished sq ft", np.exp), "l_lot": ("lot sq ft", np.exp), "age": ("age (years)", None),
         "d_lake_m": ("km to Lake Michigan", lambda v: v / 1000), "d_downtown_m": ("km to City Hall",
                                                                                  lambda v: v / 1000),
         "knn_lppsf": ("nearby $/sq ft (Jan 2025 level)", np.exp), "nbhd_lppsf_24m": ("nbhd $/sq ft", np.exp),
         "prior_lp_detr": ("own prior sale (Jan 2025 level, $)", np.exp)}


def refit(spec_role: dict, s, glob):
    cfg, cols = spec_role["config"], spec_role["features"]
    fold = frames.fold(s, glob, TEST_YEAR, BASE_START, cfg["loc"])
    tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
    m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
    return m, fold, tr, cfg, cols


def shap_frame(m: GBMX, X: pd.DataFrame) -> pd.DataFrame:
    contrib = m.booster.predict(m._X(X, False), pred_contrib=True)
    return pd.DataFrame(contrib, columns=m.cols + ["bias"], index=X.index)


def plot_groups(gi: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    g = gi.sort_values("mean_abs_shap")
    ax.barh(g.group, g.mean_abs_shap, color=BLUE, height=0.6)
    for y, v in enumerate(g.mean_abs_shap):
        ax.text(v, y, f"  {v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("mean |SHAP| (log price)", color=INK2)
    ax.set_title("Feature-group importance, frozen primary model, 2025 test sales", loc="left", color=INK,
                 fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dependence(X: pd.DataFrame, sh: pd.DataFrame, feats: list[str], path) -> None:
    n = len(feats)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7), facecolor=SURFACE)
    for ax, f in zip(np.ravel(axes), feats):
        label, tf = UNITS.get(f, (f, None))
        x = X[f].to_numpy(float)
        x = tf(x) if tf else x
        ax.scatter(x, np.exp(sh[f]) - 1, s=6, alpha=0.35, color=BLUE, linewidths=0)
        lo, hi = np.nanpercentile(x, [0, 99.5])  # one extreme lot would squash the axis
        ax.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel(label, color=INK2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.set_title(f, loc="left", color=INK, fontsize=10)
        _style(ax)
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    fig.suptitle("SHAP dependence: effect on value (%) vs feature, 2025 test sales", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_PATH.read_text())
    s = frames.sales()
    glob = frames.global_table(s)
    with step("refit frozen primary through 2024 and verify", 3600):
        m, fold, tr, cfg, cols = refit(spec["models"]["primary"], s, glob)
        sc = fold["scored"]
        pred = pd.Series(np.exp(m.predict(sc)), index=sc.index)
        frozen = pd.read_parquet(INTERIM_DIR / "frozen_primary_predictions_2025.parquet").set_index("sale_id").pred
        diff = np.abs(pred.to_numpy() / frozen.reindex(sc.sale_id).to_numpy() - 1).max()
        if not diff < 1e-9:
            raise AssertionError(f"refit differs from the frozen model's 2025 predictions (max rel diff {diff})")
        log(f"      frozen predictions reproduced (max rel diff {diff:.1e})")

    with step("SHAP (TreeSHAP) on 2025 test sales", 3600):
        sh = shap_frame(m, sc)
        fi = sh[cols].abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
        fi.columns = ["feature", "mean_abs_shap"]
        gsh = pd.DataFrame({g: sh[[c for c in fc if c in cols]].sum(axis=1) for g, fc in SHAP_GROUPS.items()})
        gi = gsh.abs().mean().rename("mean_abs_shap").reset_index().rename(columns={"index": "group"})
        gi["share"] = gi.mean_abs_shap / gi.mean_abs_shap.sum()
        gi = gi.sort_values("mean_abs_shap", ascending=False)
        gi.to_csv(OUT / "group_importance.csv", index=False)
        fi.to_csv(OUT / "feature_importance.csv", index=False)
        plot_groups(gi, OUT / "group_importance.png")
        numeric = [f for f in fi.feature if f not in ("loc", "style")]
        plot_dependence(sc, sh, numeric[:6], OUT / "dependence.png")
        san = []
        for f, exp in EXPECT.items():
            if f not in cols:
                continue
            rho = spearmanr(sc[f], sh[f], nan_policy="omit").statistic
            ok = (None if exp is None else (rho > 0.1 if exp == "+" else rho < -0.1))
            san.append({"feature": f, "expected": exp or "none", "spearman_value_vs_shap": round(float(rho), 3),
                        "mean_abs_shap": round(float(sh[f].abs().mean()), 4),
                        "verdict": "n/a" if ok is None else ("as expected" if ok else "CHECK")})
        san = pd.DataFrame(san)
        san.to_csv(OUT / "sanity.csv", index=False)

    with step("comparable sales and comp values (a) and (b)", 3600):
        pool = eligible_pool(tr, s, f"{TEST_YEAR}-01-01")
        pool["adj_price"] = np.exp(pool.y_detr)
        pool["model_pred"] = np.exp(m.predict(pool))
        test = sc.assign(model_pred=pred)
        w = fi.set_index("feature").mean_abs_shap.reindex(STRUCT).fillna(0)
        w = w / w.sum()
        sd = tr[STRUCT].std()
        comps = comps_for(test, pool, w, sd)
        comps = comps.merge(pool[["sale_id", "adj_price", "model_pred"]].rename(
            columns={"sale_id": "comp_sale_id", "model_pred": "comp_model_pred"}), on="comp_sale_id")
        comps = comps.merge(test[["sale_id", "model_pred"]], on="sale_id")
        comps["model_adj_price"] = comps.adj_price * comps.model_pred / comps.comp_model_pred
        cv = comps.groupby("sale_id").agg(comp_value_a=("adj_price", "median"),
                                          comp_value_b=("model_adj_price", "median"),
                                          n_comps=("comp_sale_id", "size"),
                                          comp_radius_mi=("comp_radius_mi", "first"),
                                          comp_window_months=("comp_window_months", "first"))
        comps.to_csv(OUT / "comps_2025.csv", index=False)
        t = test.set_index("sale_id").join(cv, how="inner")
        t = t[t.has_benchmark]
        est = {"assessor": t.bench_assessed_total, "model": t.model_pred, "comps (a) time-adjusted": t.comp_value_a,
               "comps (b) model-adjusted": t.comp_value_b}
        acc = compare(est, t.target_price, base="model")
        acc.to_csv(OUT / "comp_accuracy_2025.csv", index=False)
        gap = t.model_pred / t.comp_value_a - 1
        t["flag_model_vs_comps"] = gap.abs() > DISAGREE
        flag_eval = pd.DataFrame([{"group": k, "n": int(g.shape[0]), **{kk: v for kk, v in
                                   accuracy(g.model_pred, g.target_price).items() if kk != "n"}}
                                  for k, g in [("flagged (|model/comps(a) - 1| > 25%)", t[t.flag_model_vs_comps]),
                                               ("not flagged", t[~t.flag_model_vs_comps])]])
        flag_eval.to_csv(OUT / "disagreement_flags_2025.csv", index=False)
        coverage = {"test_sales": int(sc.has_benchmark.sum()), "with_comps": len(t)}
        tiers = (t.groupby(["comp_window_months", "comp_radius_mi"]).size().rename("subjects")
                 .reindex(pd.MultiIndex.from_tuples(TIERS, names=["comp_window_months", "comp_radius_mi"]),
                          fill_value=0).reset_index())
        tiers["share"] = tiers.subjects / len(t)
        tiers.to_csv(OUT / "comp_tiers_2025.csv", index=False)

    with step("intervals for examples (CQR pooled 2022-2024)", 3600):
        cal = []
        for Y in CAL_YEARS:
            f = frames.fold(s, glob, Y, BASE_START, cfg["loc"])
            tt = f["train"][f["train"].sale_year >= cfg["train_start"]]
            lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(tt, tt.y_detr).predict(f["scored"])
            hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(tt, tt.y_detr).predict(f["scored"])
            cal.append(pd.DataFrame({"lo": lo, "hi": hi, "y": np.log(f["scored"].target_price.to_numpy())}))
        cal = pd.concat(cal, ignore_index=True)
        q = calibrate(cal.lo.to_numpy(), cal.hi.to_numpy(), cal.y.to_numpy())["global"]
        lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(tr, tr.y_detr).predict(sc) - q
        hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(tr, tr.y_detr).predict(sc) + q
        lo, hi = contain_point(lo, hi, np.log(test.model_pred.to_numpy()))
        test["pi80_low"], test["pi80_high"] = np.exp(lo), np.exp(hi)
        b = test[test.has_benchmark]
        inside = (b.target_price >= b.pi80_low) & (b.target_price <= b.pi80_high)
        width = (b.pi80_high - b.pi80_low) / b.model_pred
        cover = pd.DataFrame([{"group": g, "n": int(m.sum()), "coverage": float(inside[m].mean()),
                               "median_rel_width": float(width[m].median())}
                              for g, m in [("all", pd.Series(True, index=b.index)),
                                           *[(k, b.tier == k) for k in ["low", "mid", "high"]]]])
        cover["nominal"], cover["q_global"] = 0.80, q
        cover.to_csv(OUT / "interval_coverage_2025.csv", index=False)

    examples = pick_examples(t)
    ex_md = example_markdown(examples, test, sh, cols, comps, s)
    write_summary(gi, fi, san, acc, flag_eval, coverage, tiers, ex_md)
    log(f"wrote {OUT / 'stage7_summary.md'}")


def pick_examples(t: pd.DataFrame) -> list[str]:
    """Deterministic: one SF sale per value tier, one duplex, and the largest
    model-vs-comps(a) disagreement (seeded choice within each group)."""
    rng = np.random.default_rng(20261018)
    out = []
    for tier in ["low", "mid", "high"]:
        g = t[(t.tier == tier) & (t.duplex == 0)]
        out.append(rng.choice(g.index.to_numpy()))
    out.append(rng.choice(t[t.duplex == 1].index.to_numpy()))
    gap = (t.model_pred / t.comp_value_a - 1).abs()
    out.append(gap.drop(out).idxmax())
    return out


def example_markdown(ids, test, sh, cols, comps, s) -> str:
    tt = test.set_index("sale_id")
    shi = sh.set_index(test.sale_id)
    info = s.set_index("sale_id")
    parts = []
    for i, sid in enumerate(ids, 1):
        r, x = tt.loc[sid], info.loc[sid]
        v = r.model_pred
        phi = shi.loc[sid, cols]
        top = phi.reindex(phi.abs().sort_values(ascending=False).index[:5])
        drivers = pd.DataFrame({"feature": top.index, "effect_pct": [f"{np.exp(p) - 1:+.1%}" for p in top],
                                "effect_usd": [f"{v - v * np.exp(-p):+,.0f}" for p in top]})
        c = comps[comps.sale_id == sid].sort_values("rank").merge(
            info[["address", "sale_date", "sale_price"]], left_on="comp_sale_id", right_index=True)
        c = c.merge(s[["sale_id", "bldg_area", "year_built"]].rename(columns={"sale_id": "comp_sale_id"}),
                    on="comp_sale_id")
        ctab = pd.DataFrame({"comp": [f"comp {k}" for k in c["rank"]], "sold": c.sale_date.dt.strftime("%Y-%m-%d"),
                             "price": c.sale_price.map("{:,.0f}".format),
                             "time-adjusted (a)": c.adj_price.map("{:,.0f}".format),
                             "model-adjusted (b)": c.model_adj_price.map("{:,.0f}".format),
                             "dist_m": c.dist_m.round(0).astype(int), "sq_ft": c.bldg_area.round(0),
                             "built": c.year_built})
        ca = c.adj_price.median()
        cb = c.model_adj_price.median()
        parts += [f"### Example {i} ({'duplex' if r.duplex else 'single-family'}, "
                  f"{r.tier} value tier)",
                  f"Sold {x.sale_date:%Y-%m-%d} for ${x.sale_price:,.0f}; at the Jan 1 2025 trend level "
                  f"${r.target_price:,.0f}. **Model ${v:,.0f}** (80% interval ${r.pi80_low:,.0f} to "
                  f"${r.pi80_high:,.0f}); assessor ${r.bench_assessed_total:,.0f}; comps (a) ${ca:,.0f}, "
                  f"comps (b) ${cb:,.0f}; model vs comps (a) {v / ca - 1:+.1%}.",
                  "Top drivers (SHAP; effect_usd = value minus value without that contribution):",
                  md(drivers), "Comparable sales:", md(ctab)]
    return "\n\n".join(parts)


def comp_change(acc: pd.DataFrame) -> pd.DataFrame:
    """Tiered comps vs the previous rule (1 km / 2 km, fixed 2023-2024 pool,
    no flag exclusions), from comp_accuracy_2025_before_tiered.csv."""
    before = pd.read_csv(OUT / "comp_accuracy_2025_before_tiered.csv")
    keys = ["statistic", "method"]
    m = acc.merge(before, on=keys, suffixes=("", "_before"))
    m = m[m.method.str.startswith("comps") & m.statistic.isin(["mdape_ln", "mdape", "cod", "median_ratio",
                                                                "within_10", "within_20"])]
    return pd.DataFrame({"statistic": m.statistic, "method": m.method, "n_before": m.n_before, "n_now": m.n,
                         "before": m.estimate_before.round(4), "now": m.estimate.round(4),
                         "change": (m.estimate - m.estimate_before).round(4)})


def write_summary(gi, fi, san, acc, flag_eval, coverage, tiers, ex_md) -> None:
    accw = acc[acc.statistic.isin(["mdape_ln", "cod", "median_ratio", "prb"])].copy()
    accw["value"] = [f"{e:.4f} [{lo:.4f}, {hi:.4f}]" for e, lo, hi in zip(accw.estimate, accw.ci_low, accw.ci_high)]
    accw["diff vs model"] = [("" if pd.isna(d) else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]") for d, lo, hi in
                             zip(accw.get("diff_vs_model"), accw.get("diff_ci_low"), accw.get("diff_ci_high"))]
    parts = [
        "# Stage 7: explainability (generated by src/avm/stage7_explain.py)",
        "Frozen primary model (public data only), refit through 2024 and verified against its frozen 2025 "
        "predictions. SHAP in log price (TreeSHAP). Group importance first: coordinates, distances and "
        "nearby-sales features are correlated and split credit.",
        "## Feature-group importance", "![groups](group_importance.png)", md(gi.round(4)),
        "## Top features", md(fi.head(15).round(4)),
        "## Dependence plots (top 6 numeric features)", "![dependence](dependence.png)",
        "## Sanity checks (Spearman of feature value vs its SHAP; |rho| > 0.1 in the expected direction)",
        md(san),
        f"## Comparable sales on 2025 (coverage: {coverage})",
        "Comp value (a) = median of comps moved to the Jan 1 2025 trend level, no model adjustment; (b) = "
        "model-adjusted (circular: the model adjusts its own check). Diffs vs the model, paired bootstrap.",
        md(accw[["statistic", "method", "n", "value", "diff vs model"]]),
        "Search tier used (confidence field: the first tier with at least 3 eligible comps):", md(tiers.round(4)),
        "Change vs the previous comp rule (1 km, else 2 km; fixed 2023-2024 pool; no flip or outlier "
        "exclusions). Point estimates only (no paired interval); compare n_before and n_now for coverage:",
        md(comp_change(acc)),
        "Disagreement flags use (a):", md(flag_eval.round(4)),
        "## Five example properties", ex_md,
    ]
    (OUT / "stage7_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
