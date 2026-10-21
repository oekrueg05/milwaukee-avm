"""Stage 1 checks. Answers the open data questions with script-produced
tables in reports/stage1/ and a generated summary (stage1_summary.md).

Runs on whatever is in the manifest (sample or full).

Usage:
    python -m avm.stage1_checks
"""

import re

import numpy as np
import pandas as pd

from avm.acquire import load_manifest
from avm.config import REPORTS_DIR
from avm.io import normalize_taxkey, parse_dates, read_parcels
from avm.mprop import CROSSWALK, EXCLUDED, load_year
from avm.progress import log, step
from avm.rolls import LOOKUP_PATH, build_lookup
from avm.sales import load_file, repeat_sale_flags

OUT = REPORTS_DIR / "stage1"
RES_LAND_USE = {"8810": "single_family", "8820": "duplex"}
# Characteristics that should rarely change year to year on the same parcel.
STABLE_CONCEPTS = ["year_built", "bldg_area", "bedrooms", "full_baths", "half_baths", "rooms",
                   "nr_stories", "nr_units", "lot_area", "land_use", "bldg_type", "neighborhood"]
CONTINUITY_PASS = 0.985          # "~99%" same-parcel agreement
TOLERANT_CONCEPTS = {"lot_area"}  # measured value: GIS recalcs move it by a few sq ft


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def keys_by_prefix(manifest: pd.DataFrame, prefix: str) -> dict[int, str]:
    out = {}
    for k, p in zip(manifest["key"], manifest["local_path"]):
        m = re.fullmatch(rf"{prefix}/(\d{{4}})", k)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def by_tk(c: pd.DataFrame) -> pd.DataFrame:
    return c.dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")


def r4(x) -> float | None:
    return None if pd.isna(x) else round(float(x), 4)


def cod(ratios: pd.Series) -> float:
    r = ratios.dropna()
    med = r.median()
    return round(100 * (r - med).abs().mean() / med, 2) if len(r) else np.nan


# ---------------------------------------------------------------- MPROP ----

def file_stats(metas: dict) -> pd.DataFrame:
    return pd.DataFrame([{"year": y, "schema": m["schema"], **m["stats"]} for y, m in metas.items()])


def schema_by_year(metas: dict) -> pd.DataFrame:
    fields = sorted({c for m in metas.values() for c in m["columns"]})
    return pd.DataFrame({y: [int(f in m["columns"]) for f in fields] for y, m in metas.items()},
                        index=pd.Index(fields, name="field")).reset_index()


def crosswalk_table(metas: dict) -> pd.DataFrame:
    return pd.DataFrame({y: [m["crosswalk_used"][c] or "" for c in CROSSWALK]
                         for y, m in metas.items()},
                        index=pd.Index(list(CROSSWALK), name="concept")).reset_index()


def crosswalk_continuity(concepts: dict, metas: dict) -> pd.DataFrame:
    """Same-parcel agreement of each concept between consecutive files. A
    correct crosswalk shows ~99% agreement on stable characteristics across a
    schema change, just like within a schema. Reported for all parcels and for
    SF/duplex parcels (land use 8810/8820 in the earlier file), the scope that
    matters downstream. Concepts blanked by mprop.EXCLUDED are marked."""
    rows = []
    years = sorted(concepts)
    for a, b in zip(years, years[1:]):
        A, B = by_tk(concepts[a]), by_tk(concepts[b])
        common_all = A.index.intersection(B.index)
        sf = common_all[A.loc[common_all, "land_use"].isin(RES_LAND_USE).fillna(False).to_numpy()]
        for subset, common in [("all", common_all), ("sf_duplex", sf)]:
            for concept in STABLE_CONCEPTS:
                excl = [why for y in (a, b) for c, why in metas[y].get("excluded", {}).items()
                        if c == concept]
                va, vb = A.loc[common, concept], B.loc[common, concept]
                present = va.notna() & (va != "") & vb.notna() & (vb != "")
                na, nb = num(va[present]), num(vb[present])
                numeric = bool(present.any()) and na.notna().mean() > 0.99 and nb.notna().mean() > 0.99
                if numeric:
                    agree = na == nb
                    within5 = (na - nb).abs() <= 0.05 * na.abs().clip(lower=1)
                else:
                    agree = va[present].str.upper() == vb[present].str.upper()
                    within5 = agree
                rows.append({
                    "pair": f"{a}-{b}", "subset": subset, "concept": concept,
                    "col_a": metas[a]["crosswalk_used"][concept],
                    "col_b": metas[b]["crosswalk_used"][concept],
                    "excluded": "; ".join(excl) if excl else "",
                    "n_common_parcels": len(common), "n_both_present": int(present.sum()),
                    "present_rate_a": round((va.notna() & (va != "")).mean(), 4) if len(common) else None,
                    "present_rate_b": round((vb.notna() & (vb != "")).mean(), 4) if len(common) else None,
                    "compare_as": "number" if numeric else "text",
                    "exact_agree": r4(agree.mean()) if present.any() else None,
                    "within_5pct": r4(within5.mean()) if present.any() else None,
                })
    df = pd.DataFrame(rows)
    df["accept_metric"] = df.concept.map(lambda c: "within_5pct" if c in TOLERANT_CONCEPTS else "exact_agree")
    df["accept_value"] = [r[m] for (_, r), m in zip(df.iterrows(), df.accept_metric)]
    df["status"] = np.where(df.excluded.astype(bool), "excluded",
                            np.where(df.accept_value.isna(), "no data",
                                     np.where(df.accept_value >= CONTINUITY_PASS, "pass", "FAIL")))
    return df


def presence_by_year(concepts: dict) -> pd.DataFrame:
    """Share of SF/duplex parcels with a non-blank value, per concept and year.
    Continuity only compares parcels present in both years, so missingness
    (e.g. a partly blank file) shows up here, not there."""
    rows = []
    for y, c in concepts.items():
        sf = c[c.land_use.isin(RES_LAND_USE).fillna(False)]
        rows.append({"file_year": y, "n_sf_duplex": len(sf),
                     **{k: r4((sf[k].notna() & (sf[k] != "")).mean()) for k in STABLE_CONCEPTS}})
    return pd.DataFrame(rows)


def value_change_by_file(concepts: dict) -> pd.DataFrame:
    rows = []
    for y, c in concepts.items():
        cur, prev = num(c["assessed_total"]), num(c["prev_assessed_total"])
        ok = cur.notna() & prev.notna() & (prev > 0)
        rows.append({"file_year": y,
                     "share_current_eq_prev": round((cur[ok] == prev[ok]).mean(), 4),
                     "median_current_over_prev": round((cur[ok] / prev[ok]).median(), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sales ----

def sales_summary(sales: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, s in sales.items():
        res = s.prop_type == "Residential"
        u = num(s.units)
        rows.append({
            "file": key, "rows": len(s), "blank_rows_dropped": s.attrs.get("blank_rows", 0),
            "date_precision": "/".join(sorted(s.date_precision.unique())),
            "min_sale_date": s.sale_date.min().date(), "max_sale_date": s.sale_date.max().date(),
            "invalid_taxkey": int(s.tk.isna().sum()),
            "price_min": s.sale_price.min(), "price_le_1000": int((s.sale_price <= 1000).sum()),
            "price_lt_10000": int((s.sale_price < 10000).sum()),
            "price_median": s.sale_price.median(),
            "n_residential": int(res.sum()),
            "res_units_1": int((res & (u == 1)).sum()), "res_units_2": int((res & (u == 2)).sum()),
            "n_condo": int((s.prop_type == "Condominium").sum()),
        })
    return pd.DataFrame(rows)


def master_vs_yearly(master: pd.DataFrame, yearly: dict[int, pd.DataFrame]) -> pd.DataFrame:
    key = ["tk", "sale_date", "sale_price"]
    mk = master.assign(year=master.sale_date.dt.year)
    mset = set(map(tuple, master[key].astype(str).values))
    rows = []
    for y in range(int(mk.year.min()), int(mk.year.max()) + 1):
        s = yearly.get(y)
        rows.append({
            "year": y, "n_master": int((mk.year == y).sum()),
            "n_yearly_file": len(s) if s is not None else None,
            "n_yearly_rows_in_master": int(sum(t in mset for t in map(tuple, s[key].astype(str).values)))
                if s is not None else None,
        })
    return pd.DataFrame(rows)


def join_key_check(yearly, concepts, lookup, parcel_keys) -> pd.DataFrame:
    """Match rate of each sales year Y to the MPROP roll for Y-1 (the feature
    snapshot) and Y (the assessor benchmark), chosen via the lookup table."""
    fy = dict(zip(lookup.assessment_year, lookup.file_year))
    st = dict(zip(lookup.assessment_year, lookup.status))
    # Key sets as pd.Index, built once: Series.isin(set) rebuilt a 160K-element
    # array on every call and made this step take 15+ minutes.
    roll_keys = {f: pd.Index(c.tk.dropna().unique()) for f, c in concepts.items()}
    pkeys = {n: pd.Index(normalize_taxkey(s).dropna().unique()) for n, s in parcel_keys.items()}
    rows = []
    for y, s in yearly.items():
        targets = {}
        for label, a in [("roll_Y-1", y - 1), ("roll_Y", y)]:
            if a in fy:
                targets[label] = (roll_keys[fy[a]], f"file {fy[a]} ({st[a]})")
        targets |= {n: (ks, n) for n, ks in pkeys.items()}
        for subset, mask in [("all", pd.Series(True, index=s.index)),
                             ("residential", s.prop_type == "Residential")]:
            tk = s.tk[mask]
            for label, (keys, src) in targets.items():
                hit = pd.Series(keys.get_indexer(tk.fillna("")) >= 0, index=tk.index)
                rows.append({"sales_year": y, "subset": subset, "target": label, "source": src,
                             "n_sales": int(mask.sum()), "n_unmatched": int((~hit).sum()),
                             "match_rate": round(hit.mean(), 4)})
    return pd.DataFrame(rows)


def sales_chars_vs_mprop(yearly, concepts, lookup) -> pd.DataFrame:
    """Sales-file sq ft / year built vs MPROP roll Y-1, roll Y, and the latest file."""
    fy = dict(zip(lookup.assessment_year, lookup.file_year))
    latest = max(concepts)
    rows = []
    for y, s in yearly.items():
        r = s[s.prop_type == "Residential"].dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
        for label, f in [("roll_Y-1", fy.get(y - 1)), ("roll_Y", fy.get(y)), ("latest_file", latest)]:
            if f is None:
                continue
            m = by_tk(concepts[f]).reindex(r.index)
            for field in ["bldg_area", "year_built"]:
                a, b = num(r[field]), num(m[field])
                ok = a.notna() & b.notna() & (a > 0)
                rows.append({"sales_year": y, "compare": label, "file": f, "field": field,
                             "n": int(ok.sum()), "agree_rate": r4((a[ok] == b[ok]).mean())})
    return pd.DataFrame(rows)


def sales_chasing(yearly, concepts, lookup) -> pd.DataFrame:
    """Do parcels that sold in year Y get reassessed toward their price more
    than parcels that did not sell? Compares roll Y (valued Jan 1 Y, before
    the sale) with roll Y+1 (valued Jan 1 Y+1, after it).

    Universe: SF/duplex parcels (land use 8810/8820 in roll Y) present in both
    rolls with unchanged sq ft and year built (removes renovation and new
    construction). Sold = residential sale in Y. Unsold = no sale in Y-1 or Y.
    Ratios use unadjusted sale prices, so they mix level and timing."""
    fy = dict(zip(lookup.assessment_year, lookup.file_year))
    st = dict(zip(lookup.assessment_year, lookup.status))
    sold_by_year = {y: set(s.tk[s.prop_type == "Residential"].dropna()) for y, s in yearly.items()}
    rows = []
    for y, s in yearly.items():
        if y not in fy or y + 1 not in fy:
            continue
        A, B = by_tk(concepts[fy[y]]), by_tk(concepts[fy[y + 1]])
        res = A.index[A.land_use.isin(RES_LAND_USE)]
        common = res.intersection(B.index)
        stable = common[(num(A.loc[common, "bldg_area"]) == num(B.loc[common, "bldg_area"])) &
                        (num(A.loc[common, "year_built"]) == num(B.loc[common, "year_built"]))]
        before, after = num(A.loc[stable, "assessed_total"]), num(B.loc[stable, "assessed_total"])
        ok = before.notna() & after.notna() & (before > 0)
        before, after = before[ok], after[ok]
        sold = before.index.isin(pd.Index(list(sold_by_year[y])))
        unsold = ~before.index.isin(pd.Index(list(sold_by_year[y] | sold_by_year.get(y - 1, set()))))
        pct = after / before - 1
        r = s[(s.prop_type == "Residential")].dropna(subset=["tk"])
        r = r[~r.tk.duplicated(keep=False)].set_index("tk")  # one sale per parcel in Y
        rp = r.sale_price.reindex(before.index[sold]).dropna()
        rp = rp[rp > 0]
        rows.append({
            "sales_year": y, "roll_before": f"{y} (file {fy[y]}, {st[y]})",
            "roll_after": f"{y + 1} (file {fy[y + 1]}, {st[y + 1]})",
            "n_universe": int(ok.sum()), "n_excluded_chars_changed": len(common) - len(stable),
            "n_sold": int(sold.sum()), "n_unsold": int(unsold.sum()),
            "share_changed_sold": r4((pct[sold] != 0).mean()),
            "share_changed_unsold": r4((pct[unsold] != 0).mean()),
            "median_pct_change_sold": r4(pct[sold].median()),
            "median_pct_change_unsold": r4(pct[unsold].median()),
            "n_sold_single_sale": len(rp),
            "median_ratio_before": r4((before[rp.index] / rp).median()),
            "median_ratio_after": r4((after[rp.index] / rp).median()),
            "cod_before": cod(before[rp.index] / rp), "cod_after": cod(after[rp.index] / rp),
        })
    return pd.DataFrame(rows)


def _share_after(dates: pd.Series, cutoffs: pd.Series) -> float:
    """Mean over cutoffs of P(date > cutoff), using the empirical CDF of
    `dates` (NaT counts as no change). Matched baseline for unsold parcels."""
    d = np.sort(dates.dropna().values.astype("datetime64[ns]"))
    n = len(dates)
    if n == 0 or len(cutoffs) == 0:
        return np.nan
    n_after = len(d) - np.searchsorted(d, cutoffs.values.astype("datetime64[ns]"), side="right")
    return float(np.mean(n_after / n))


def sales_chasing_within_roll(yearly, concepts, lookup) -> pd.DataFrame:
    """The check that bears on the benchmark (year-Y sales vs roll Y): within
    roll Y, was a sold parcel's assessment changed AFTER its sale date more
    often than comparable unsold parcels'?

    Confirmed rolls only, and only where change dates survive. Sold = SF/
    duplex parcel (roll Y land use 8810/8820) with exactly one residential
    sale in Y. Month-only sale dates count as the month's last day, so
    "after" is conservative. Unsold = SF/duplex parcel with no sale in Y-1 or
    Y; its rate is matched to the sold parcels' sale dates (mean over sold
    sale dates of the share of unsold parcels changed after that date).

    Two measures: any change after the sale, and an individual change (change
    date after the sale and not the roll's mass date, i.e. not the citywide
    update every parcel received)."""
    lk = lookup[lookup.status == "confirmed"]
    fy = dict(zip(lk.assessment_year, lk.file_year))
    sold_by_year = {y: set(s.tk[s.prop_type == "Residential"].dropna()) for y, s in yearly.items()}
    rows = []
    for y, s in yearly.items():
        if y not in fy:
            continue
        m = by_tk(concepts[fy[y]])
        chg, _ = parse_dates(m["assessment_change_date"])
        if chg.notna().mean() < 0.5:
            rows.append({"roll_year": y, "note": "change dates not usable"})
            continue
        mass = chg.mode().iloc[0]
        sf = m.index[m.land_use.isin(RES_LAND_USE).fillna(False).to_numpy()]
        r = s[(s.prop_type == "Residential")].dropna(subset=["tk", "sale_date"])
        r = r[~r.tk.duplicated(keep=False) & r.tk.isin(sf)].set_index("tk")
        sale = r.sale_date.where(r.date_precision == "day",
                                 r.sale_date + pd.offsets.MonthEnd(0))
        unsold_idx = sf[~sf.isin(pd.Index(list(sold_by_year[y] | sold_by_year.get(y - 1, set()))))]
        c_sold, c_un = chg.reindex(sale.index), chg.reindex(unsold_idx)
        indiv_sold = c_sold.where(c_sold != mass)
        indiv_un = c_un.where(c_un != mass)
        any_after = (c_sold > sale)
        ind_after = (indiv_sold > sale)
        reasons = m["assessment_change_reason"].reindex(sale.index[ind_after.fillna(False)]) \
            .str.strip().value_counts().head(3)
        rows.append({
            "roll_year": y, "file": fy[y], "mass_change_date": mass.date(),
            "mass_date_share": r4((chg == mass).mean()),
            "sale_date_precision": "/".join(sorted(r.date_precision.unique())),
            "n_sold": len(sale), "n_unsold": len(unsold_idx),
            "n_sold_before_mass_date": int((sale < mass).sum()),
            "sold_any_change_after_sale": r4(any_after.mean()),
            "unsold_any_change_after_matched": r4(_share_after(c_un, sale)),
            "sold_individual_change_after_sale": r4(ind_after.mean()),
            "unsold_individual_change_after_matched": r4(_share_after(indiv_un, sale)),
            "n_sold_individual_change_after_sale": int(ind_after.sum()),
            "top_reasons_sold_individual_after": "; ".join(f"{k} ({v})" for k, v in reasons.items()),
            "note": "",
        })
    return pd.DataFrame(rows)


def quick_flips(yearly) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat sales of the same tax key across all yearly files. With
    month-only dates a 12-calendar-month gap cannot be resolved and is
    reported as ambiguous rather than guessed."""
    s = pd.concat(yearly.values(), ignore_index=True).dropna(subset=["tk", "sale_date"])
    n_raw = len(s)
    s = s.drop_duplicates(["tk", "sale_date", "sale_price"]).reset_index(drop=True)
    s = repeat_sale_flags(s)
    s["year"] = s.sale_date.dt.year

    # Same day and price across different parcels: candidate multi-parcel sale.
    day = s[s.date_precision == "day"]
    grp = day.groupby(["sale_date", "sale_price"]).tk.transform("nunique")
    s["multi_parcel_candidate"] = False
    s.loc[day.index, "multi_parcel_candidate"] = grp > 1

    res = s.prop_type == "Residential"
    by_year = s.groupby("year").agg(
        n_sales=("tk", "size"), date_precision=("date_precision", lambda x: "/".join(sorted(x.unique()))),
        n_resale_lt_12m=("resale_lt_12m", "sum"), n_resale_12m_ambiguous=("resale_12m_ambiguous", "sum"),
        n_first_sale_of_flip=("first_sale_of_flip", "sum"), n_repeat_same_period=("repeat_same_period", "sum"),
        n_multi_parcel_candidate=("multi_parcel_candidate", "sum"),
    ).reset_index()
    by_year_res = s[res].groupby("year").agg(
        res_sales=("tk", "size"), res_resale_lt_12m=("resale_lt_12m", "sum"),
        res_resale_12m_ambiguous=("resale_12m_ambiguous", "sum"),
    ).reset_index()
    out = by_year.merge(by_year_res, on="year", how="left")
    out.attrs["n_exact_duplicates_dropped"] = n_raw - len(s)
    return out, s


def sales_convey_types(yearly, concepts, lookup) -> pd.DataFrame:
    """Deed type of sold parcels, read from MPROP roll Y where its conveyance
    date (year-month) matches the sale. Only possible where MPROP dates are
    not corrupt."""
    fy = dict(zip(lookup.assessment_year, lookup.file_year))
    rows = []
    for y, s in yearly.items():
        if y not in fy:
            continue
        m = by_tk(concepts[fy[y]])
        conv, _ = parse_dates(m["convey_date"])
        r = s[s.prop_type == "Residential"].dropna(subset=["tk"])
        r = r[r.tk.isin(m.index)]
        cd = conv.reindex(r.tk.values).values
        cd = pd.Series(pd.to_datetime(cd), index=r.index)
        match = (cd.dt.year == r.sale_date.dt.year) & (cd.dt.month == r.sale_date.dt.month)
        types = m["convey_type"].reindex(r.tk[match].values).str.strip().value_counts()
        n = int(match.sum())
        row = {"sales_year": y, "mprop_file": fy[y], "n_res_sales": len(r), "n_month_match": n}
        for t in ["WD", "QC", "LC", "PR", "TD", "SD", "SW", "IR"]:
            row[f"share_{t}"] = round(types.get(t, 0) / n, 4) if n else None
        row["share_other"] = round(1 - sum(types.get(t, 0) for t in
                                           ["WD", "QC", "LC", "PR", "TD", "SD", "SW", "IR"]) / n, 4) if n else None
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- summary ----

def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(t: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> None:
    cont = t["mprop_crosswalk_continuity"]
    cont = cont[cont.subset == "sf_duplex"]
    pivot = cont.pivot(index="concept", columns="pair", values="accept_value").reindex(STABLE_CONCEPTS)
    status = cont.pivot(index="concept", columns="pair", values="status").reindex(STABLE_CONCEPTS)
    fails = cont[cont.status == "FAIL"][["pair", "concept", "col_a", "col_b", "accept_metric", "accept_value"]]
    jk = t["join_key_check"]
    jk_res = jk[jk.subset == "residential"].pivot(index="sales_year", columns="target", values="match_rate")
    parts = [
        "# Stage 1 summary (generated by src/avm/stage1_checks.py)",
        f"Files in manifest: {len(manifest)}. Downloaded {manifest.downloaded_at_utc.min()} "
        f"to {manifest.downloaded_at_utc.max()} UTC.",
        "## MPROP file-year -> assessment-year lookup", md(t["lookup"]),
        "## Assessed value vs own previous-year value, by file", md(t["mprop_value_change_by_file"]),
        "## MPROP file stats", md(t["mprop_file_stats"]),
        "## Crosswalk: raw column used per concept and file", md(t["mprop_crosswalk"]),
        "## Crosswalk continuity: same-parcel exact agreement, consecutive files, SF/duplex "
        "(blank = excluded, see mprop.EXCLUDED)",
        md(pivot.reset_index()),
        f"Pass threshold {CONTINUITY_PASS}; lot_area judged within 5%, others exact. "
        "Status per pair:", md(status.reset_index()),
        "### Failing pairs", md(fails) if len(fails) else "_None._",
        "### Excluded concept-years (mprop.EXCLUDED)",
        md(pd.DataFrame([{"concept": c, "file_year": y, "reason": why}
                         for (c, y), why in sorted(EXCLUDED.items())])),
        "## Presence of each concept on SF/duplex parcels, by file", md(t["mprop_presence_by_year"]),
        "## Sales files", md(t["sales_summary"]),
        "## Master file (2002-2018) vs yearly files", md(t["sales_master_vs_yearly"]),
        "## Tax key match rate, residential sales", md(jk_res.reset_index()),
        "## Sales chasing within roll Y (bears on the year-Y benchmark)",
        md(t["sales_chasing_within_roll"]),
        "## Sales chasing, roll Y -> Y+1 (write-up finding; does not touch the benchmark)",
        md(t["sales_chasing"]),
        f"## Repeat sales and quick flips (exact duplicate rows dropped: "
        f"{t['quick_flips_by_year'].attrs.get('n_exact_duplicates_dropped')})",
        md(t["quick_flips_by_year"]),
        "## Deed type of residential sales (from MPROP where conveyance month matches)",
        md(t["sales_convey_types"]),
        "## Sales-file characteristics vs MPROP",
        md(t["sales_chars_vs_mprop"].pivot_table(index=["sales_year", "field"], columns="compare",
                                                 values="agree_rate").reset_index()),
    ]
    (OUT / "stage1_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


FILE_TIMEOUT_S = 300   # one raw file load (an uncached MPROP year takes ~30s)
STEP_TIMEOUT_S = 600   # one check over all years (each takes well under a minute)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    concepts, metas = {}, {}
    mp = keys_by_prefix(manifest, "mprop_hist")
    for i, (y, p) in enumerate(mp.items(), 1):
        with step(f"[{i}/{len(mp)}] load MPROP {y}", FILE_TIMEOUT_S):
            concepts[y], metas[y] = load_year(y, p)

    sales_paths = dict(zip(manifest.key, manifest.local_path))
    sp = keys_by_prefix(manifest, "sales")
    yearly = {}
    with step(f"load {len(sp)} yearly sales files", FILE_TIMEOUT_S):
        for y, p in sp.items():
            yearly[y] = load_file(f"sales/{y}", p)
    master_key = "sales/2002-2018_master"
    master = None
    if master_key in sales_paths:
        with step("load sales master file", FILE_TIMEOUT_S):
            master = load_file(master_key, sales_paths[master_key])

    parcel_keys = {}
    for k in [k for k in manifest.key if k.startswith("parcels/")]:
        with step(f"load {k}", FILE_TIMEOUT_S):
            pdf = read_parcels(k)
            parcel_keys[k.replace("/", "_")] = pdf["Taxkey"] if "Taxkey" in pdf else pdf["FK_Tax"]

    with step("assessment-year lookup", STEP_TIMEOUT_S):
        lookup = build_lookup(concepts)
        lookup.to_csv(LOOKUP_PATH, index=False)

    all_sales = {**{f"sales/{y}": s for y, s in yearly.items()},
                 **({master_key: master} if master is not None else {})}
    flips_attrs = {}

    def flips():
        out, _ = quick_flips(yearly)
        flips_attrs.update(out.attrs)
        return out

    checks = {
        "mprop_value_change_by_file": lambda: value_change_by_file(concepts),
        "mprop_file_stats": lambda: file_stats(metas),
        "mprop_schema_by_year": lambda: schema_by_year(metas),
        "mprop_date_quality": lambda: pd.DataFrame([r for m in metas.values() for r in m["date_quality"]]),
        "mprop_crosswalk": lambda: crosswalk_table(metas),
        "mprop_crosswalk_continuity": lambda: crosswalk_continuity(concepts, metas),
        "mprop_presence_by_year": lambda: presence_by_year(concepts),
        "sales_summary": lambda: sales_summary(all_sales),
        "sales_master_vs_yearly": lambda: master_vs_yearly(master, yearly)
            if master is not None else pd.DataFrame(),
        "join_key_check": lambda: join_key_check(yearly, concepts, lookup, parcel_keys),
        "sales_chars_vs_mprop": lambda: sales_chars_vs_mprop(yearly, concepts, lookup),
        "sales_chasing": lambda: sales_chasing(yearly, concepts, lookup),
        "sales_chasing_within_roll": lambda: sales_chasing_within_roll(yearly, concepts, lookup),
        "quick_flips_by_year": flips,
        "sales_convey_types": lambda: sales_convey_types(yearly, concepts, lookup),
    }
    t = {"lookup": lookup}
    for i, (name, fn) in enumerate(checks.items(), 1):
        with step(f"[{i}/{len(checks)}] {name}", STEP_TIMEOUT_S):
            t[name] = fn()
            # Written immediately so a later failure keeps finished tables.
            t[name].to_csv(OUT / f"{name}.csv", index=False)
        log(f"      wrote {name}.csv ({len(t[name])} rows)")
    t["quick_flips_by_year"].attrs.update(flips_attrs)
    with step("write summary", STEP_TIMEOUT_S):
        write_summary(t, manifest)
    log(f"wrote {LOOKUP_PATH.name} and stage1_summary.md")


if __name__ == "__main__":
    main()
