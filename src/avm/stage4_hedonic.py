"""Stage 4: Baseline 2, hedonic regression with location fixed effects.

Usage:
    python -m avm.stage4_hedonic

Time-based CV (rule 2): for each validation year Y in 2021-2024, train on
clean sales from `start` to Y-1, predict Y's sales at the Dec Y-1 trend level
with a neutral month-of-year, score against Y's target prices (Stage 3
trend-only adjustment to Jan 1 Y). Grid: location unit (neighborhood, tract)
x training start (2014, 2017, 2020). Selection: lowest mean level-neutral MdAPE.

Final: train start..2024, predict 2025 test sales at the Dec 2024 trend level.
Head-to-head vs the assessor (roll 2025) on identical target prices,
untrimmed and union-trimmed, paired bootstrap CIs on the differences.

Training rows exclude price/sq ft outliers (Stage 2 flag). Data through 2024
only: 2025 enters solely as the scoring target.
"""

import numpy as np
import pandas as pd

from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.evaluate import STATS, accuracy, compare, union_trim, value_tiers
from avm.features import FEATURE_DOC, T0, build_frame, usable
from avm.hedonic import fit, predict_neutral
from avm.progress import log, step
from avm.reporting import h2h_table, md

OUT = REPORTS_DIR / "stage4"
CV_YEARS = [2021, 2022, 2023, 2024]
STARTS = [2014, 2017, 2020]
LOCS = ["neighborhood", "tract"]
TEST_YEAR = 2025
STEP_TIMEOUT_S = 1800
CAT_COLS = ["loc", "style", "age_bin", "stories"]


def trend_point(year_before: int) -> float:
    """Mid-December of the last training year, in months since T0."""
    return (pd.Timestamp(f"{year_before}-12-15") - T0).days / 30.4375


def load() -> pd.DataFrame:
    s = pd.read_parquet(INTERIM_DIR / "sales_clean.parquet")
    t = pd.read_parquet(INTERIM_DIR / "target_prices.parquet")[["sale_id", "target_price"]]
    s = s.merge(t, on="sale_id", how="left", validate="one_to_one")
    if s.target_price.isna().any():
        raise ValueError(f"{int(s.target_price.isna().sum())} clean sales lack a target price; re-run Stage 3")
    return s


def fit_predict(s: pd.DataFrame, loc: str, start: int, pred_year: int):
    """Train on [start, pred_year-1], predict pred_year. Returns (model,
    predictions joined to targets, counts)."""
    train_rows = s.sale_year.between(start, pred_year - 1) & (s.split == "train") & ~s.ppsf_outlier
    pred_rows = s.sale_year == pred_year
    sub = s[train_rows | pred_rows]
    f = build_frame(sub, loc, train_rows[sub.index])
    tr = f[train_rows[sub.index] & usable(f)]
    pr = f[pred_rows[sub.index] & usable(f)]
    unseen = pd.Series(False, index=pr.index)
    for c in CAT_COLS:
        unseen |= ~pr[c].isin(set(tr[c]))
    counts = {"n_train": len(tr), "n_train_unusable": int(train_rows.sum()) - len(tr),
              "n_pred": int(pred_rows.sum()), "n_pred_unusable": int(pred_rows.sum()) - len(pr),
              "n_pred_unseen_level": int(unseen.sum())}
    pr = pr[~unseen]
    model = fit(tr)
    p = predict_neutral(model, pr, trend_point(pred_year - 1))
    out = pr[["sale_id"]].join(p).merge(s[["sale_id", "target_price", "bench_assessed_total", "has_benchmark",
                                           "land_use_y1", "neighborhood", "bldg_area", "sale_price"]],
                                        on="sale_id")
    return model, out, counts


def cv_table(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loc in LOCS:
        for start in STARTS:
            for y in CV_YEARS:
                if start > y - 1:
                    continue
                _, o, c = fit_predict(s, loc, start, y)
                a = accuracy(o.pred, o.target_price)
                rows.append({"loc": loc, "train_start": start, "val_year": y, **c, **a})
                log(f"      cv {loc:<12} start {start} val {y}: MdAPE {a['mdape']:.3f} COD {a['cod']:.1f}")
    return pd.DataFrame(rows)


def coefficient_table(model) -> pd.DataFrame:
    keep = [k for k in model.params.index if k in ("l_area", "l_lot", "lot_missing", "beds", "full_baths",
                                                   "half_baths", "baths_missing", "duplex")
            or k.startswith("C(age_bin") or k.startswith("C(stories")]
    t = pd.DataFrame({"term": keep, "coef": model.params[keep].to_numpy(), "se": model.bse[keep].to_numpy()})
    t["pct_effect"] = np.where(t.term.isin(["l_area", "l_lot"]), np.nan, np.exp(t.coef) - 1)
    t["note"] = np.where(t.term == "l_area", "elasticity: +10% sq ft -> " +
                         ((1.1 ** t.coef - 1) * 100).round(1).astype(str) + "% price",
                np.where(t.term == "l_lot", "elasticity: +10% lot -> " +
                         ((1.1 ** t.coef - 1) * 100).round(1).astype(str) + "% price", ""))
    return t


def strata(test: pd.DataFrame, s: pd.DataFrame) -> dict[str, pd.Index]:
    sqft = test.bldg_area
    bins = pd.cut(sqft, [0, 999, 1399, 1799, np.inf], labels=["<1,000", "1,000-1,399", "1,400-1,799", "1,800+"])
    tier = value_tiers(s, test, TEST_YEAR)
    out = {"all": test.index, "single-family": test.index[test.land_use_y1 == "8810"],
           "duplex": test.index[test.land_use_y1 == "8820"]}
    out |= {f"sq ft {b}": test.index[bins == b] for b in bins.cat.categories}
    out |= {f"nbhd value tier {t}": test.index[tier == t] for t in ["low", "mid", "high", "unrated"]
            if (tier == t).sum() >= 30}
    return out


def h2h_compact(t: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    return h2h_table(t, stats, ["assessor", "hedonic"], base="assessor")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()

    with step("time-based CV (location x training start x validation year)", STEP_TIMEOUT_S):
        cv = cv_table(s)
        cv.to_csv(OUT / "cv_results.csv", index=False)
        summ = cv.groupby(["loc", "train_start"]).agg(
            folds=("val_year", "size"), mdape_ln=("mdape_ln", "mean"), mdape=("mdape", "mean"), cod=("cod", "mean"),
            within_10=("within_10", "mean"), within_20=("within_20", "mean"),
            median_ratio=("median_ratio", "mean")).reset_index()
        # Compare configs on the folds every config has (2021-2024 requires start <= 2020).
        best = summ.sort_values(["mdape_ln", "cod"]).iloc[0]
        log(f"      best: {best.to_dict()}")

    with step("final fit through 2024, predict 2025", STEP_TIMEOUT_S):
        model, o, counts = fit_predict(s, best["loc"], int(best["train_start"]), TEST_YEAR)
        coefs = coefficient_table(model)
        coefs.to_csv(OUT / "coefficients.csv", index=False)
        o.to_parquet(INTERIM_DIR / "hedonic_predictions_2025.parquet", index=False)
        test = o[o.has_benchmark].set_index("sale_id")
        cover = ((test.target_price >= test.pi80_low) & (test.target_price <= test.pi80_high)).mean()
        width = ((test.pi80_high - test.pi80_low) / test.pred).median()
        acc = pd.DataFrame([accuracy(test.pred, test.target_price) |
                            {"pi80_coverage": cover, "pi80_median_rel_width": width, **counts,
                             "r2_train": model.rsquared, "n_params": len(model.params)}])
        acc.to_csv(OUT / "hedonic_test_metrics.csv", index=False)

    with step("head-to-head vs assessor (paired bootstrap 2,000)", STEP_TIMEOUT_S):
        est = {"assessor": test.bench_assessed_total, "hedonic": test.pred}
        keep = union_trim(est, test.target_price)
        parts = []
        for name, ix in strata(test, s).items():
            for trim, mask in [("untrimmed", pd.Series(True, index=test.index)), ("union trim", keep)]:
                sub = test.loc[ix][mask[ix]]
                t = compare({k: v[sub.index] for k, v in est.items()}, sub.target_price, base="assessor")
                parts.append(t.assign(stratum=name, trim=trim))
        h2h = pd.concat(parts, ignore_index=True)
        h2h.to_csv(OUT / "head_to_head.csv", index=False)
        q = pd.qcut(test.target_price, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        foot = []
        for k in q.cat.categories:
            sub = test[q == k]
            t = compare({kk: v[sub.index] for kk, v in est.items()}, sub.target_price, base="assessor")
            foot.append(t[t.statistic.isin(["median_ratio", "cod"])].assign(stratum=f"price {k}", trim="untrimmed"))
        foot = pd.concat(foot, ignore_index=True)
        foot.to_csv(OUT / "head_to_head_price_quartiles_footnote.csv", index=False)

    cvp = cv.pivot_table(index=["loc", "train_start"], columns="val_year", values="mdape_ln").round(4).reset_index()
    cvp = cvp.merge(summ.round(4), on=["loc", "train_start"])
    main_tbl = h2h_compact(h2h[h2h.stratum == "all"], STATS)
    strata_tbl = h2h_compact(h2h[h2h.trim == "union trim"], ["median_ratio", "cod", "mdape_ln", "prd", "prb"])
    parts = [
        "# Stage 4: hedonic baseline (generated by src/avm/stage4_hedonic.py)",
        "Target = 2025 sale price at the Jan 1 2025 trend level, neutral season (Stage 3, evaluation "
        "only). Hedonic predicts at the Dec 2024 trend level, neutral month-of-year, from data through "
        "2024. Cells: estimate [95% CI]; diffs are hedonic - assessor with paired bootstrap CIs (2,000 "
        "resamples). For COD and MdAPE a negative diff favors the hedonic. **PRD/PRB against sale "
        "prices lean regressive** (price noise in the denominator).",
        "## Time-based CV: level-neutral MdAPE by validation year, and means over folds", md(cvp),
        f"Selected: location = {best['loc']}, training start = {int(best['train_start'])} "
        f"(lowest mean level-neutral MdAPE {best['mdape_ln']:.4f}).",
        "## Coefficient sanity checks (final model)", md(coefs),
        "## Hedonic on 2025 test sales", md(acc.round(4)),
        "## Head-to-head, all test sales", md(main_tbl),
        "## Head-to-head by stratum (union trim)", md(strata_tbl),
        "## Footnote: by sale-price quartile (median ratio, COD only)",
        "Strata on the sale price select on the ratio's denominator, so both methods look regressive "
        "within quartiles by construction; shown for completeness only.",
        md(h2h_compact(foot, ["median_ratio", "cod"])),
        "## Features (rule 7)", md(pd.DataFrame([{"feature": k, "why": v[0], "proxy note": v[1]}
                                                  for k, v in FEATURE_DOC.items()])),
    ]
    (OUT / "stage4_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    log(f"wrote {OUT / 'stage4_summary.md'}")


if __name__ == "__main__":
    main()
