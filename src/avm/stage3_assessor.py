"""Stage 3: Baseline 1, the assessor's ratio study on the held-out year.

Usage:
    python -m avm.stage3_assessor

ratio = roll Y assessed value / sale price time-adjusted to Jan 1 Y.

MAIN time adjustment (Owen, 2026-10-18): trend-only. Month-of-year effects
from 2014-2024 sales; valuation-year trend piecewise linear with a knot at
Jan 1 Y (fit on Y-1 and Y sales); each price moved to the Jan 1 trend level
with a neutral season. Evaluation only; never a model input.

Sensitivities: trend-only without removing seasonality, seasonal monthly
hedonic index (Jan = 1), separate trend-only fits by tercile of the roll Y-1
assessed value (known before the sale), IAAO ratio trend, unadjusted.
Notice-date test: sales before vs after April 1 2025. Context: each sale
year 2014-2025 vs its own roll, own trend.

Writes interim/target_prices.parquet (sale_id, target_price) for every
clean sale with a benchmark: the shared target for the assessor and every
model, so all are scored on identical prices.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avm.acquire import load_manifest
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.io import to_number
from avm.mprop import load_year
from avm.progress import log, step
from avm.ratio_study import iqr_outliers, ratio_stats
from avm.reporting import compact, md
from avm.rolls import file_year_for, load_lookup
from avm.time_adjust import (apply_index, fit_hedonic_index, fit_trend, ratio_trend, seasonal_effects,
                             trend_adjust, trend_adjusted_ratio)

OUT = REPORTS_DIR / "stage3"
TARGETS_PATH = INTERIM_DIR / "target_prices.parquet"
TEST_YEAR = 2025
SEASONAL_YEARS = range(2014, 2025)
NOTICE_DATE = "2025-04-01"
STEP_TIMEOUT_S = 900
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}  # validated, light mode
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def stats_row(df: pd.DataFrame, value: pd.Series, price: pd.Series, **labels) -> pd.DataFrame:
    t = ratio_stats(value[df.index], price[df.index])
    for k, v in reversed(list(labels.items())):
        t.insert(0, k, v)
    return t


def trend_for_year(s: pd.DataFrame, y: int, seasonal: pd.Series) -> dict:
    years = [y - 1, y] if (s.sale_year == y - 1).any() else [y]
    return fit_trend(s[s.sale_year.isin(years)], f"{y}-01-01", seasonal)


def year_frame(s: pd.DataFrame, y: int, seasonal: pd.Series) -> tuple[pd.DataFrame, dict]:
    tr = trend_for_year(s, y, seasonal)
    t = s[(s.sale_year == y) & s.has_benchmark].copy()
    t["adj_price"] = trend_adjust(t, tr, seasonal)
    t["ratio"] = t.bench_assessed_total / t.adj_price
    t["ratio_unadj"] = t.bench_assessed_total / t.sale_price
    t["iaao_outlier"] = iqr_outliers(t.ratio)
    return t, tr


def tercile_adjusted(s: pd.DataFrame, test: pd.DataFrame, seasonal: pd.Series) -> pd.Series:
    est = s[s.sale_year.isin([TEST_YEAR - 1, TEST_YEAR])].copy()
    cuts = est.prior_assessed_total.quantile([1 / 3, 2 / 3]).to_numpy()
    lab = lambda v: pd.cut(v, [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"])
    est["tercile"], test = lab(est.prior_assessed_total), test.assign(tercile=lab(test.prior_assessed_total))
    adj = pd.Series(np.nan, index=test.index)
    for t in ["low", "mid", "high"]:
        tr = fit_trend(est[est.tercile == t], f"{TEST_YEAR}-01-01", seasonal)
        g = test[test.tercile == t]
        adj[g.index] = trend_adjust(g, tr, seasonal)
    return adj


def tercile_target_prices(s: pd.DataFrame, years: list[int], seasonal: pd.Series) -> pd.Series:
    """Sensitivity targets for every clean sale in `years`: the same trend-only
    adjustment, but fitted separately by tercile of the roll Y-1 assessed total
    (cuts from sales of Y-1 and Y), so low-, mid- and high-value homes each get
    their own appreciation. Sales without a prior assessed total get none."""
    out = {}
    for y in years:
        est = s[s.sale_year.isin([y - 1, y])]
        cuts = est.prior_assessed_total.quantile([1 / 3, 2 / 3]).to_numpy()
        lab = lambda v: pd.cut(v, [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"])  # noqa: E731
        e_t, ys = lab(est.prior_assessed_total), s[s.sale_year == y]
        y_t = lab(ys.prior_assessed_total)
        for t in ["low", "mid", "high"]:
            tr = fit_trend(est[e_t == t], f"{y}-01-01", seasonal)
            g = ys[y_t == t]
            out.update(zip(g.sale_id, trend_adjust(g, tr, seasonal).to_numpy()))
    return pd.Series(out, name="target_price_tercile")


def diagnose_2019(s: pd.DataFrame) -> pd.DataFrame:
    """Why does 2019 have so many IAAO ratio outliers? Compare each outlier's
    roll 2019 value with the same parcel in rolls 2018 and 2020."""
    m = load_manifest().set_index("key")
    lk = load_lookup()
    rolls = {}
    for a in (2018, 2019, 2020):
        f = file_year_for(a, lk)
        c = load_year(f, m.loc[f"mprop_hist/{f}", "local_path"])[0]
        c = c.dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
        rolls[a] = to_number(c.assessed_total, f"roll {a} assessed_total")
    o = s[(s.sale_year == 2019) & s.has_benchmark & s.ratio_outlier_iaao].copy()
    for a in rolls:
        o[f"roll_{a}"] = rolls[a].reindex(o.tk).to_numpy()
    o["ratio_2019"] = o.roll_2019 / o.sale_price
    o["jump_vs_2018"] = o.roll_2019 / o.roll_2018
    o["jump_vs_2020"] = o.roll_2019 / o.roll_2020
    # Rolls 2018 and 2020 are independent of roll 2019: if the price is low
    # against both, the sale price is the anomaly, not the 2019 value.
    spike = (o.jump_vs_2018 > 1.5) & (o.jump_vs_2020 > 1.5)
    low_price = (o.sale_price < 0.5 * o.roll_2018) & (o.sale_price < 0.5 * o.roll_2020)
    high_price = (o.sale_price > 2 * o.roll_2018) & (o.sale_price > 2 * o.roll_2020)
    o["diagnosis"] = np.select(
        [spike, low_price, high_price],
        ["roll 2019 value spike (>1.5x both 2018 and 2020 rolls)",
         "price < 50% of both 2018 and 2020 rolls (price anomaly)",
         "price > 2x both 2018 and 2020 rolls (price anomaly)"],
        "mixed / moderate")
    return o[["sale_id", "sale_date", "sale_price", "roll_2018", "roll_2019", "roll_2020",
              "ratio_2019", "jump_vs_2018", "jump_vs_2020", "diagnosis"]].sort_values("ratio_2019", ascending=False)


def low_price_share_by_year(s: pd.DataFrame) -> pd.DataFrame:
    """Share of clean sales priced far from the PRIOR roll (Y-1), which no
    year's own roll influences: a file-level check on price anomalies."""
    r = s.sale_price / s.prior_assessed_total
    g = pd.DataFrame({"sale_year": s.sale_year, "lt_half": r < 0.5, "lt_third": r < 1 / 3,
                      "gt_double": r > 2}).groupby("sale_year")
    return pd.DataFrame({"n": g.size(), "price_lt_50pct_prior_roll": g.lt_half.mean().round(4),
                         "price_lt_33pct_prior_roll": g.lt_third.mean().round(4),
                         "price_gt_2x_prior_roll": g.gt_double.mean().round(4)}).reset_index()


def plot_index(monthly: pd.DataFrame, trend: dict, seasonal: pd.Series, path) -> None:
    """Left: seasonal monthly index (sensitivity) and the trend-only level
    (main), both 1.0 at Jan 1 2025. Right: month-of-year effects, 2014-2024."""
    # rc_context, not rcParams.update: a global update leaked into later stages' figures when the
    # whole pipeline ran in one process (found by the full rebuild, 2026-10-20).
    with plt.rc_context({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "sans-serif"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1.6, 1]})
        ax = axes[0]
        x = pd.Series(pd.PeriodIndex(monthly.month, freq="M").to_timestamp()) + pd.Timedelta(days=14)
        ax.fill_between(x, monthly.ci_low, monthly.ci_high, color=SERIES["blue"], alpha=0.15, linewidth=0)
        ax.plot(x, monthly["index"], color=SERIES["blue"], linewidth=2, marker="o", markersize=3.5)
        ax.text(x.iloc[-1], monthly["index"].iloc[-1], "  monthly index\n  (sensitivity)", color=INK,
                va="center", fontsize=9)
        t0 = pd.Timestamp(f"{TEST_YEAR}-01-01")
        tt = pd.Series(pd.date_range(f"{TEST_YEAR - 1}-01-01", f"{TEST_YEAR}-12-31", freq="D"))
        m = (tt - t0).dt.days / 30.4375
        lvl = np.exp(trend["slope_pre"] * m.clip(upper=0) + trend["slope_post"] * m.clip(lower=0))
        ax.plot(tt, lvl, color=SERIES["orange"], linewidth=2)
        ax.text(tt.iloc[-1], lvl.iloc[-1], "  trend-only\n  (main)", color=INK, va="center", fontsize=9)
        ax.axhline(1.0, color=INK2, linewidth=0.8)
        ax.axvline(t0, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlim(x.iloc[0] - pd.Timedelta(days=30), x.iloc[-1] + pd.Timedelta(days=130))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_ylabel("Price level (Jan 1 2025 = 1.0)")
        ax.set_title("2024-2025 price level", loc="left", color=INK, fontsize=11)
        ax = axes[1]
        eff = np.exp(seasonal) - 1
        ax.bar(range(1, 13), eff, color=SERIES["blue"], width=0.7)
        ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.set_title("Month-of-year effect, 2014-2024 sales", loc="left", color=INK, fontsize=11)
        for ax in axes:
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Milwaukee SF/duplex sale prices: time adjustment (evaluation only)", x=0.01, ha="left",
                     color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(INTERIM_DIR / "sales_clean.parquet")

    with step("seasonal effects (2014-2024) and 2025 trend", STEP_TIMEOUT_S):
        seasonal = seasonal_effects(s[s.sale_year.isin(SEASONAL_YEARS)])
        seasonal.reset_index().assign(pct=lambda d: np.exp(d.log_effect) - 1).to_csv(
            OUT / "seasonal_effects_2014_2024.csv", index=False)
        test, trend = year_frame(s, TEST_YEAR, seasonal)
        log(f"      trend 2025: {trend}")

    with step("sensitivities: monthly index, terciles, IAAO trend", STEP_TIMEOUT_S):
        monthly, minfo = fit_hedonic_index(s[s.sale_year.isin([TEST_YEAR - 1, TEST_YEAR])], f"{TEST_YEAR}-01")
        monthly.to_csv(OUT / "time_index_2025_monthly.csv", index=False)
        test["adj_monthly"] = apply_index(test, monthly)
        test["adj_trend_keep_season"] = trend_adjust(test, trend, seasonal, remove_seasonal=False)
        test["adj_tercile"] = tercile_adjusted(s, test, seasonal)
        core = test[~test.iaao_outlier]
        iaao = ratio_trend(core.ratio_unadj, core.sale_date, f"{TEST_YEAR}-01-01")
        test["ratio_iaao_trend"] = trend_adjusted_ratio(test.ratio_unadj, test.sale_date,
                                                        f"{TEST_YEAR}-01-01", iaao["slope_per_month"])
        plot_index(monthly, trend, seasonal, OUT / "time_index_2025.png")

    with step("assessor ratio study 2025 (bootstrap 2,000)", STEP_TIMEOUT_S):
        v, p = test.bench_assessed_total, test.adj_price
        strata = {"all": test.index, "single-family": test.index[test.land_use_y1 == "8810"],
                  "duplex": test.index[test.land_use_y1 == "8820"]}
        rows = []
        for trim_label, keep in [("IAAO trimmed", ~test.iaao_outlier),
                                 ("untrimmed", pd.Series(True, index=test.index))]:
            for name, ix in strata.items():
                rows.append(stats_row(test.loc[ix][keep[ix]], v, p, trim=trim_label, stratum=name))
        study = pd.concat(rows, ignore_index=True)
        study.to_csv(OUT / "assessor_ratio_study_2025.csv", index=False)

        tr = test[~test.iaao_outlier]
        sens = pd.concat([
            stats_row(tr, v, test.adj_price, adjustment="trend-only, season removed (main)"),
            stats_row(tr, v, test.adj_trend_keep_season, adjustment="trend-only, season kept"),
            stats_row(tr, v, test.adj_monthly, adjustment="seasonal monthly index"),
            stats_row(tr, v, test.adj_tercile, adjustment="trend-only by tercile"),
            stats_row(tr, v, v / test.ratio_iaao_trend, adjustment="IAAO ratio trend"),
            stats_row(tr, v, test.sale_price, adjustment="unadjusted"),
        ], ignore_index=True)
        sens.to_csv(OUT / "assessor_sensitivity_time_adjustment.csv", index=False)

        notice = pd.concat([stats_row(tr[tr.sale_date < NOTICE_DATE], v, p, period=f"sold before {NOTICE_DATE}"),
                            stats_row(tr[tr.sale_date >= NOTICE_DATE], v, p, period=f"sold on/after {NOTICE_DATE}")],
                           ignore_index=True)
        notice.to_csv(OUT / "assessor_before_after_notice.csv", index=False)
        trim_log = test[test.iaao_outlier][["sale_id", "sale_date", "sale_price", "adj_price",
                                            "bench_assessed_total", "ratio", "land_use_y1"]].sort_values("ratio")
        trim_log.to_csv(OUT / "iaao_trim_log_2025.csv", index=False)

    with step("per-year context 2014-2025 and shared target prices", STEP_TIMEOUT_S):
        ctx, targets, trends = [], [], []
        for y in range(2014, TEST_YEAR + 1):
            t, tr_y = year_frame(s, y, seasonal)
            r = stats_row(t[~t.iaao_outlier], t.bench_assessed_total, t.adj_price, sale_year=y)
            r["n_trimmed"] = int(t.iaao_outlier.sum())
            ctx.append(r)
            trends.append({"sale_year": y, **{k: tr_y[k] for k in ("slope_pre", "slope_post", "n_used")}})
            all_y = s[s.sale_year == y]
            targets.append(pd.DataFrame({"sale_id": all_y.sale_id,
                                         "target_price": trend_adjust(all_y, tr_y, seasonal).to_numpy(),
                                         "sale_year": y}))
        context = pd.concat(ctx, ignore_index=True)
        context.to_csv(OUT / "assessor_context_by_year.csv", index=False)
        pd.DataFrame(trends).to_csv(OUT / "trend_by_year.csv", index=False)
        pd.concat(targets, ignore_index=True).to_parquet(TARGETS_PATH, index=False)
        log(f"      wrote {TARGETS_PATH}")

    with step("2019 outlier diagnosis", STEP_TIMEOUT_S):
        d19 = diagnose_2019(s)
        d19.to_csv(OUT / "outliers_2019_diagnosis.csv", index=False)
        lp = low_price_share_by_year(s)
        lp.to_csv(OUT / "price_vs_prior_roll_by_year.csv", index=False)

    ctx_w = compact(context, ["sale_year"]).merge(context.drop_duplicates("sale_year")[["sale_year", "n_trimmed"]],
                                                  on="sale_year")
    seas_tbl = seasonal.reset_index().assign(pct=lambda d: (np.exp(d.log_effect) - 1).round(4))
    parts = [
        "# Stage 3: assessor ratio study (generated by src/avm/stage3_assessor.py)",
        "Ratio = roll Y assessed value / sale price at the Jan 1 Y trend level, neutral season "
        "(trend-only adjustment; evaluation only). Cells: estimate [95% bootstrap CI, 2,000 resamples, "
        "seed fixed]; ' x' = misses the IAAO target. **PRD and PRB against sale prices are biased toward "
        "showing regressivity** (sale-price noise sits in the denominator).",
        f"## Time adjustment\n2025 trend: slope {trend['slope_pre']:.4f}/month before Jan 1 2025, "
        f"{trend['slope_post']:.4f}/month after (se {trend['slope_post_se']:.4f}); "
        f"n = {trend['n_used']}. IAAO ratio trend implies {100 * iaao['implied_monthly_appreciation']:.2f}%/month."
        "\n\n![time adjustment](time_index_2025.png)",
        "Month-of-year effects (2014-2024):", md(seas_tbl),
        "## 2025 assessor metrics vs IAAO targets", md(compact(study, ["trim", "stratum"])),
        f"## IAAO 3x IQR trim log: {int(test.iaao_outlier.sum())} of {len(test)} sales "
        "(full list in iaao_trim_log_2025.csv)", md(trim_log),
        "## Sensitivity: time adjustment (IAAO trimmed)", md(compact(sens, ["adjustment"])),
        "## Before vs after the assessment notice date (IAAO trimmed)", md(compact(notice, ["period"])),
        "## Context: each sale year vs its own roll (IAAO trimmed, own trend)", md(ctx_w),
        f"## 2019 outliers: {len(d19)} sales", md(d19.diagnosis.value_counts().rename("n").reset_index()),
        "### Price vs prior roll (Y-1), all clean sales by year", md(lp),
    ]
    (OUT / "stage3_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    log(f"wrote {OUT / 'stage3_summary.md'}")


if __name__ == "__main__":
    main()
