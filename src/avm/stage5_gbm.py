"""Stage 5: LightGBM, independent and assessment-informed variants.

Usage:
    python -m avm.stage5_gbm            # full run
    python -m avm.stage5_gbm --sample   # 3 configs, independent variant only

Tuning (per variant): 40 random configs (fixed seed) x time-based folds
2021-2024 (train start..Y-1, predict Y at the Dec Y-1 trend level, neutral
season, score vs Y target prices). Selection (Owen, 2026-10-18): among
configs within 0.2 pts of the best mean level-neutral MdAPE, the smallest
mean |within-tier PRB|. Monotone constraints (sq ft, lot, baths) adopted if
they cost <= 0.2 pts of mean level-neutral MdAPE in CV.

Intervals: CQR, quantile models calibrated on 2024; coverage by value tier;
Mondrian (tier-wise) calibration if any rated tier is > 90% or < 75%.

Head-to-head: assessor, hedonic (Stage 4), GBM independent, GBM informed on
identical 2025 target prices; paired bootstrap; level-neutral MdAPE leads.
Sensitivity (independent variant): training-set variants scored on 2025.
"""

import argparse

import numpy as np
import pandas as pd

from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.evaluate import accuracy, compare, tier_prb, union_trim, value_tiers
from avm.features import build_frame, usable
from avm.gbm import GBM, sample_configs
from avm.intervals import apply, calibrate, coverage_by
from avm.progress import log, step
from avm.reporting import h2h_table, md
from avm.stage4_hedonic import CV_YEARS, TEST_YEAR, load, strata, trend_point

OUT = REPORTS_DIR / "stage5"
N_CONFIGS = 40
MDAPE_TOL = 0.002
MONO_TOL = 0.002
TIER_COVER_RANGE = (0.75, 0.90)
VARIANTS = ["independent", "assessment_informed"]
CAL_YEAR = TEST_YEAR - 1


def frames(s: pd.DataFrame, loc: str, start: int, pred_year: int, baths_mode: str = "missing",
           train_filter: pd.Series | None = None):
    train_rows = s.sale_year.between(start, pred_year - 1) & (s.split == "train") & ~s.ppsf_outlier
    if train_filter is not None:
        train_rows &= train_filter
    pred_rows = s.sale_year == pred_year
    sub = s[train_rows | pred_rows]
    f = build_frame(sub, loc, train_rows[sub.index], baths_mode)
    ok = usable(f) & f.l_prior_total.notna() & f.l_prior_land.notna()  # same rows for both variants
    return f[train_rows[sub.index] & ok], f[pred_rows[sub.index] & ok]


def scored(s: pd.DataFrame, pr: pd.DataFrame, log_pred: np.ndarray, year: int) -> pd.DataFrame:
    o = pr[["sale_id"]].assign(pred=np.exp(log_pred))
    o = o.merge(s[["sale_id", "target_price", "neighborhood", "bench_assessed_total", "has_benchmark",
                   "land_use_y1", "bldg_area", "sale_price"]], on="sale_id")
    o["tier"] = value_tiers(s, o, year).to_numpy()
    return o


def fold_metrics(o: pd.DataFrame) -> dict:
    a = accuracy(o.pred, o.target_price)
    tp = tier_prb(o.pred, o.target_price, o.tier)
    return a | {f"prb_tier_{k}": v for k, v in tp.items()}


def cv_eval(s: pd.DataFrame, cfg: dict, variant: str, monotone: bool = False) -> pd.DataFrame:
    rows = []
    for y in CV_YEARS:
        tr, pr = frames(s, cfg["loc"], cfg["train_start"], y)
        m = GBM(cfg, variant, monotone).fit(tr)
        o = scored(s, pr, m.predict_log(pr, trend_point(y - 1)), y)
        rows.append({"variant": variant, "config_id": cfg["config_id"], "monotone": monotone, "val_year": y,
                     "n_train": len(tr), **fold_metrics(o)})
    return pd.DataFrame(rows)


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    return cv.groupby(["variant", "config_id", "monotone"]).agg(
        mdape_ln=("mdape_ln", "mean"), cod=("cod", "mean"), median_ratio=("median_ratio", "mean"),
        prb_tier_abs=("prb_tier_mean_abs", "mean"), prb_low=("prb_tier_low", "mean"),
        prb_mid=("prb_tier_mid", "mean"), prb_high=("prb_tier_high", "mean")).reset_index()


def select(summ: pd.DataFrame) -> pd.Series:
    best = summ.mdape_ln.min()
    cand = summ[summ.mdape_ln <= best + MDAPE_TOL]
    return cand.sort_values(["prb_tier_abs", "mdape_ln"]).iloc[0]


def final_fit(s, cfg, variant, monotone, **kw):
    tr, pr = frames(s, cfg["loc"], cfg["train_start"], TEST_YEAR, **kw)
    m = GBM(cfg, variant, monotone).fit(tr)
    return m, scored(s, pr, m.predict_log(pr, trend_point(TEST_YEAR - 1)), TEST_YEAR), len(tr)


def cqr(s, cfg, variant, monotone, test: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (test with pi80 columns, calibration Qs, coverage table)."""
    qs = {}
    for a in (0.1, 0.9):
        tr, pr_cal = frames(s, cfg["loc"], cfg["train_start"], CAL_YEAR)
        # LightGBM forbids monotone constraints with the quantile objective, so
        # interval models are unconstrained; point predictions keep them.
        qm = GBM(cfg, variant, False, objective="quantile", alpha=a).fit(tr)
        qs[a] = qm.predict_log(pr_cal, trend_point(CAL_YEAR - 1))
    cal = scored(s, pr_cal, np.zeros(len(pr_cal)), CAL_YEAR)
    y_cal = np.log(cal.target_price.to_numpy())
    q = calibrate(qs[0.1], qs[0.9], y_cal, cal.tier)
    fin = {}
    for a in (0.1, 0.9):
        tr, pr = frames(s, cfg["loc"], cfg["train_start"], TEST_YEAR)
        fin[a] = GBM(cfg, variant, False, objective="quantile", alpha=a).fit(tr).predict_log(
            pr, trend_point(TEST_YEAR - 1))
    keyed = pr[["sale_id"]].assign(lo_raw=fin[0.1], hi_raw=fin[0.9])
    t = test.merge(keyed, on="sale_id", how="left", validate="one_to_one")
    y = np.log(t.target_price.to_numpy())
    rows = []
    for mode, grp in [("global", None), ("mondrian (by tier)", t.tier)]:
        lo, hi = apply(t.lo_raw.to_numpy(), t.hi_raw.to_numpy(), q, grp)
        c = coverage_by(y, lo, hi, t.tier).assign(calibration=mode)
        rows.append(c)
        t[f"pi80_low_{'g' if grp is None else 'm'}"] = np.exp(lo)
        t[f"pi80_high_{'g' if grp is None else 'm'}"] = np.exp(hi)
    return t, q, pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    out = OUT / "sample" if a.sample else OUT
    out.mkdir(parents=True, exist_ok=True)
    s = load()
    configs = sample_configs(3 if a.sample else N_CONFIGS)
    variants = VARIANTS[:1] if a.sample else VARIANTS

    cv_parts, chosen = [], {}
    for v in variants:
        with step(f"tuning {v}: {len(configs)} configs x {len(CV_YEARS)} folds", 6 * 3600):
            for cfg in configs:
                r = cv_eval(s, cfg, v)
                cv_parts.append(r)
                log(f"      {v} cfg {cfg['config_id']:>2}: MdAPE_ln {r.mdape_ln.mean():.4f} "
                    f"COD {r.cod.mean():.1f} |tierPRB| {r.prb_tier_mean_abs.mean():.3f}")
        cv = pd.concat(cv_parts, ignore_index=True)
        summ = summarize(cv[cv.variant == v])
        pick = select(summ)
        cfg = next(c for c in configs if c["config_id"] == pick.config_id)
        with step(f"monotone check {v}", 3600):
            mono = cv_eval(s, cfg, v, monotone=True)
            cv_parts.append(mono)
            cost = mono.mdape_ln.mean() - pick.mdape_ln
            adopt = bool(cost <= MONO_TOL)
            log(f"      monotone cost {cost:+.4f} -> adopt {adopt}")
        chosen[v] = {"cfg": cfg, "monotone": adopt, "mono_cost": cost, "pick": pick}
    cv = pd.concat(cv_parts, ignore_index=True)
    cv.to_csv(out / "cv_folds.csv", index=False)
    summ_all = summarize(cv)
    summ_all = summ_all.merge(pd.DataFrame(configs), on="config_id")
    summ_all.to_csv(out / "tuning_summary.csv", index=False)

    preds, cover_tbls, n_train = {}, [], {}
    for v in variants:
        c = chosen[v]
        with step(f"final fit + CQR {v}", 3600):
            m, o, n_train[v] = final_fit(s, c["cfg"], v, c["monotone"])
            o, q, cov = cqr(s, c["cfg"], v, c["monotone"], o)
            chosen[v]["q"] = q
            cover_tbls.append(cov.assign(method=f"gbm_{v}"))
            preds[v] = o
            o.to_parquet(INTERIM_DIR / f"gbm_{v}_predictions_2025.parquet", index=False)
    cover = pd.concat(cover_tbls, ignore_index=True)
    # UNEVEN = coverage differs across tiers (one above 90% while another is
    # below 75%, or a spread > 10 pts). A uniform shortfall is miscalibration,
    # not unevenness, and tier-wise calibration cannot fix it; reported apart.
    uneven = {}
    for v in variants:
        g = cover[(cover.method == f"gbm_{v}") & (cover.calibration == "global") &
                  cover.group.isin(["low", "mid", "high"])].coverage
        uneven[v] = bool(((g > TIER_COVER_RANGE[1]).any() and (g < TIER_COVER_RANGE[0]).any())
                         or (g.max() - g.min() > 0.10))
    if not a.sample:
        h = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet")
        h = h.merge(preds["independent"][["sale_id", "tier"]], on="sale_id", how="inner")
        y = np.log(h.target_price.to_numpy())
        cover = pd.concat([cover, coverage_by(y, np.log(h.pi80_low), np.log(h.pi80_high), h.tier)
                           .assign(method="hedonic", calibration="OLS normal")], ignore_index=True)
    cover.to_csv(out / "interval_coverage_by_tier.csv", index=False)

    tables = {}
    if not a.sample:
        with step("4-way head-to-head (paired bootstrap 2,000)", 3600):
            h = pd.read_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet")[["sale_id", "pred"]]
            base = preds["independent"].rename(columns={"pred": "gbm_independent"})
            base = base.merge(preds["assessment_informed"][["sale_id", "pred"]].rename(
                columns={"pred": "gbm_informed"}), on="sale_id").merge(h.rename(columns={"pred": "hedonic"}),
                                                                       on="sale_id")
            test = base[base.has_benchmark].set_index("sale_id")
            methods = ["assessor", "hedonic", "gbm_independent", "gbm_informed"]
            est = {"assessor": test.bench_assessed_total, "hedonic": test.hedonic,
                   "gbm_independent": test.gbm_independent, "gbm_informed": test.gbm_informed}
            keep = union_trim(est, test.target_price)
            parts = []
            for name, ix in strata(test, s).items():
                for trim, mask in [("untrimmed", pd.Series(True, index=test.index)), ("union trim", keep)]:
                    sub = test.loc[ix][mask[ix]]
                    t = compare({k: v_[sub.index] for k, v_ in est.items()}, sub.target_price, base="assessor")
                    parts.append(t.assign(stratum=name, trim=trim))
            h2h = pd.concat(parts, ignore_index=True)
            h2h.to_csv(out / "head_to_head_4way.csv", index=False)
            sub = test[keep]
            vs_hed = compare({k: est[k][sub.index] for k in ["hedonic", "gbm_independent", "gbm_informed"]},
                             sub.target_price, base="hedonic")
            vs_hed.to_csv(out / "gbm_vs_hedonic_union_trim.csv", index=False)
            tables |= {"h2h": h2h, "vs_hed": vs_hed, "methods": methods, "n_test": len(test),
                       "n_trim": int((~keep).sum())}

        with step("sensitivity (independent variant, scored on 2025)", 3 * 3600):
            c = chosen["independent"]
            cfg = c["cfg"]
            variants_sens = {
                "baseline": {},
                "exclude first sale of a flip": {"train_filter": ~s.first_sale_of_flip},
                "exclude non-market deeds (where verifiable)": {"train_filter": ~s.deed_non_market.astype(bool)},
                "baths: raw pre-2019 + era flag": {"baths_mode": "raw_era"},
                "exclude price < 50% of prior roll": {"train_filter": s.sale_price >= 0.5 * s.prior_assessed_total},
            }
            for st in [2014, 2017, 2020]:
                if st != cfg["train_start"]:
                    variants_sens[f"training start {st}"] = {"start": st}
            sens_preds, ntr = {}, {}
            for name, kw in variants_sens.items():
                cf = dict(cfg, train_start=kw.pop("start")) if "start" in kw else cfg
                _, o, ntr[name] = final_fit(s, cf, "independent", c["monotone"], **kw)
                sens_preds[name] = o.set_index("sale_id").pred
                log(f"      sensitivity {name}: n_train {ntr[name]}")
            common = pd.concat(sens_preds, axis=1).dropna().index
            tgt = s.set_index("sale_id").target_price.loc[common]
            tier = preds["independent"].set_index("sale_id").tier.reindex(common)
            sens = compare({k: v_.loc[common] for k, v_ in sens_preds.items()}, tgt, base="baseline")
            prbs = pd.DataFrame([{"method": k, "prb_tier_mean_abs": tier_prb(v_.loc[common], tgt, tier)["mean_abs"],
                                  "n_train": ntr[k]} for k, v_ in sens_preds.items()])
            sens.to_csv(out / "sensitivity.csv", index=False)
            tables |= {"sens": sens, "sens_prb": prbs}

    write_summary(out, summ_all, chosen, cover, uneven, tables, n_train, variants)
    log(f"wrote {out / 'stage5_summary.md'}")


def write_summary(out, summ, chosen, cover, uneven, t, n_train, variants) -> None:
    parts = ["# Stage 5: LightGBM (generated by src/avm/stage5_gbm.py)",
             "Target = sale price at the Jan 1 trend level, neutral season (Stage 3). GBMs predict at the "
             "Dec (Y-1) trend level, neutral month-of-year, from data through Y-1. mdape_ln = level-neutral "
             "MdAPE (each method's ratios divided by its own median ratio): the headline accuracy measure. "
             "**PRD/PRB against sale prices lean regressive** (price noise in the denominator)."]
    for v in variants:
        sv = summ[(summ.variant == v) & ~summ.monotone].sort_values("mdape_ln")
        c = chosen[v]
        parts += [f"## Tuning: {v}, top 5 by mean level-neutral MdAPE (CV 2021-2024)",
                  md(sv.head(5)[["config_id", "mdape_ln", "cod", "prb_tier_abs", "prb_low", "prb_mid", "prb_high",
                                 "train_start", "loc", "num_leaves", "learning_rate", "num_boost_round",
                                 "min_data_in_leaf"]].round(4)),
                  f"Selected config {int(c['pick'].config_id)} (within {MDAPE_TOL:.1%} of best, smallest mean "
                  f"|within-tier PRB| {c['pick'].prb_tier_abs:.3f}). Monotone constraints: CV cost "
                  f"{c['mono_cost']:+.4f} -> {'adopted' if c['monotone'] else 'not adopted'}. "
                  f"Final training rows: {n_train[v]}."]
    parts += ["## 80% interval coverage on 2025 by neighborhood value tier", md(cover.round(4)),
              "Uneven across tiers under global calibration (one tier > 90% while another < 75%, or a "
              "spread > 10 pts): " + ", ".join(f"{k}: {v}" for k, v in uneven.items()) +
              ". Mondrian rows show tier-wise calibration (adopted only where uneven). Overall coverage "
              "below 80% is a separate, uniform miscalibration."]
    if "h2h" in t:
        stats = ["mdape_ln", "cod", "median_ratio", "prd", "prb", "mki", "within_10", "within_20"]
        parts += [f"## 4-way head-to-head, all 2025 test sales (n = {t['n_test']}; union trim removes {t['n_trim']})",
                  md(h2h_table(t["h2h"][t["h2h"].stratum == "all"], stats, t["methods"], "assessor")),
                  "## By stratum (union trim): level-neutral MdAPE, COD, PRB",
                  md(h2h_table(t["h2h"][t["h2h"].trim == "union trim"], ["mdape_ln", "cod", "prb"], t["methods"],
                               "assessor")),
                  "## GBMs vs hedonic (union trim)",
                  md(h2h_table(t["vs_hed"].assign(stratum="all", trim="union trim"), ["mdape_ln", "cod", "prb"],
                               ["hedonic", "gbm_independent", "gbm_informed"], "hedonic")),
                  "## Sensitivity (independent variant; diffs vs baseline, paired bootstrap)",
                  md(h2h_table(t["sens"].assign(stratum="2025 test", trim="none"), ["mdape_ln", "cod", "median_ratio"],
                               list(t["sens"].method.unique()), "baseline").T.reset_index()
                     .rename(columns={"index": "column", 0: "value"})),
                  "Within-tier PRB (mean |PRB| over low/mid/high tiers):", md(t["sens_prb"].round(4))]
    (out / "stage5_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
