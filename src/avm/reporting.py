"""Markdown helpers shared by stage reports. Every number in a report comes
from a DataFrame produced by the stage's script (rule 9)."""

import pandas as pd

from avm.ratio_study import IAAO_TARGETS

FMT = {"median_ratio": "{:.3f}", "cod": "{:.1f}", "prd": "{:.3f}", "prb": "{:.3f}", "mki": "{:.3f}",
       "mdape": "{:.1%}", "mdape_ln": "{:.1%}", "within_10": "{:.1%}", "within_20": "{:.1%}"}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def cell(stat: str, est: float, lo: float, hi: float, miss: bool = False) -> str:
    f = FMT.get(stat, "{:.3f}")
    return f"{f.format(est)} [{f.format(lo)}, {f.format(hi)}]{' x' if miss else ''}"


def h2h_table(t: pd.DataFrame, stats: list[str], methods: list[str], base: str) -> pd.DataFrame:
    """evaluate.compare output (with stratum/trim columns) -> one row per
    stratum x trim; per statistic each method's 'est [CI]' and, for non-base
    methods, 'diff [CI]' vs base."""
    rows = []
    for (stratum, trim), g in t.groupby(["stratum", "trim"], sort=False):
        r = {"stratum": stratum, "trim": trim, "n": int(g.n.iloc[0])}
        for st in stats:
            for m in methods:
                x = g[(g.statistic == st) & (g.method == m)].iloc[0]
                r[f"{st} {m}"] = cell(st, x.estimate, x.ci_low, x.ci_high)
                if m != base:
                    r[f"{st} {m}-{base}"] = cell(st, x[f"diff_vs_{base}"], x.diff_ci_low, x.diff_ci_high)
        rows.append(r)
    return pd.DataFrame(rows)


def compact(t: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """ratio_stats output -> one row per group, 'estimate [CI]' per statistic."""
    t = t.copy()
    t["cell"] = [cell(s, e, lo, hi, not ok) for s, e, lo, hi, ok in
                 zip(t.statistic, t.estimate, t.ci_low, t.ci_high, t.meets_target)]
    w = t.pivot_table(index=by + ["n"], columns="statistic", values="cell", aggfunc="first").reset_index()
    return w[by + ["n"] + [c for c in IAAO_TARGETS if c in w.columns]]
