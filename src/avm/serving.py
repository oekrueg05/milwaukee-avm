"""Precompute for the read-only value_parcel tool (MCP server, issue #6).

Usage:
    python -m avm.serving build [--sample]   # <data>/serving/value_parcel_2026[_sample].sqlite
    python -m avm.serving check [--sample]   # re-verify the hashes the server checks

The server only reads the database written here. Nothing here changes the
frozen model spec or the prospective file.

Recovering the model: no booster was saved when the 2026 predictions were
made, and SHAP needs one. The build refits the frozen primary spec exactly
as `prospective.predict` did (same parcel rows, frame, config and seed) and
ABORTS unless round(exp(prediction)) equals value_2026_01_01 for every
parcel in predictions_2026.parquet. The recovered booster is saved next to
the database with its sha256 in the meta table. Values and 80% ranges are
copied from the hashed file, never from the refit.

Drivers (hybrid grouping): TreeSHAP in log price, summed within groups for
features that are correlated and split credit (Stage 7's reason for leading
with groups): coordinates, distances and the tract effect as "location";
the nearby-sales features as "nearby sale prices"; the home's own prior sale
and the city-owned/vacant counts as their Stage 6 groups. Structure features
stay individual, because each is a fact about the home a reader can check.
Stage 9 lists single features; this differs so that "location" is not split
across eight correlated columns. Effects are multiplicative relative to the
model's average home: exp(sum of SHAP) - 1.

Comps: avm.comps for a Jan 1 2026 valuation (pool: clean sales in the 24
months before, same exclusions; never the parcel's own sales), with the
search tier per parcel.

Subject details: finished sq ft, bedrooms, baths, year built and lot size as
joined from roll 2025 for the model (a field that fell back to an earlier
roll is named in details_note). Subject last sale: the parcel's most recent
sale in the clean sales table (arm's-length, 2014-2025), recorded price. A
sale that Stage 2 cleaning removed and that is dated after it (or is the
only sale) is kept with its date, recorded price and filter, served as
context that the estimate does not use.
"""

import argparse
import json
import sqlite3
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from avm import frames
from avm.acquire import load_manifest, sha256_of
from avm.address import mai_rows, parcel_addresses
from avm.comps import STRUCT, comps_for, eligible_pool
from avm.config import DATA_DIR, INTERIM_DIR, REPORTS_DIR
from avm.gbm import GBMX
from avm.lookup import LABEL, db_path, verify
from avm.serving_db import SCHEMA, TABLE_COLS, _py, write_db
from avm.mprop import RES_LAND_USE, load_year
from avm.progress import log, step
from avm.prospective import BASE_START, PRED_PATH, SPEC_PATH, YEAR, check, parcel_rows
from avm.sales import load_file
from avm.ratio_study import SEED
from avm.rolls import file_year_for, load_lookup
from avm.spatial import static_features
from avm.stage6_geo import GROUPS, fold_frame, roll_tables
from avm.stage7_explain import shap_frame
from avm.stage9_divergence import FEATURE_LABELS

OUT_DIR = DATA_DIR / "serving"
SAMPLE_N = 500
DRIVER_GROUPS = {
    "location": GROUPS["coords"] + GROUPS["distances"] + ["loc"],
    "nearby sale prices": GROUPS["prior_sales_nearby"],
    "this home's prior sale": GROUPS["own_prior_sale"],
    "nearby vacant or city-owned lots": GROUPS["city_vacant"],
}
N_DRIVERS = 3
PROPERTY_TYPE = {8810: "single-family", 8820: "duplex"}
SUBJECT_FIELDS = {"bldg_area": "finished sq ft", "bedrooms": "bedrooms", "full_baths": "full baths",
                  "half_baths": "half baths", "year_built": "year built", "lot_area": "lot size"}
NOT_SALES = {"F1", "F2", "F3"}  # record errors (date outside file year, exact duplicate, missing fields), not sales


def build_commit() -> str:
    """HEAD, with '+dirty' if tracked files have uncommitted changes."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPORTS_DIR.parent).stdout  # noqa: E731
    return git("rev-parse", "HEAD").strip() + ("+dirty" if git("status", "--porcelain", "--untracked-files=no").strip() else "")


def booster_path(sample: bool):
    return OUT_DIR / f"frozen_primary_{YEAR}_booster{'_sample' if sample else ''}.txt"


def driver_table(sh: pd.DataFrame, cols: list[str], tks) -> pd.DataFrame:
    """Top N_DRIVERS hybrid drivers per parcel (log effects), long format."""
    grouped = {c for g in DRIVER_GROUPS.values() for c in g}
    single = [c for c in cols if c not in grouped]
    if not grouped <= set(cols) or len(grouped) != sum(len(g) for g in DRIVER_GROUPS.values()):
        raise ValueError("driver groups must cover model features exactly once")
    parts = {**{k: sh[g].sum(axis=1) for k, g in DRIVER_GROUPS.items()}, **{c: sh[c] for c in single}}
    d = pd.DataFrame(parts)
    order = np.argsort(-d.abs().to_numpy(), axis=1, kind="stable")[:, :N_DRIVERS]
    keys = np.array(d.columns)[order]
    vals = np.take_along_axis(d.to_numpy(), order, axis=1)
    out = pd.DataFrame({"tk": np.repeat(np.asarray(tks), N_DRIVERS), "rank": np.tile(np.arange(1, N_DRIVERS + 1), len(d)),
                        "key": keys.ravel(), "effect_log": vals.ravel()})
    out["label"] = out.key.map(lambda k: FEATURE_LABELS.get(k, k))
    return out


def own_prior_sales(sub: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """The sale behind each parcel's own-prior-sale features (tk, date, price).
    The frame keeps months_since_prior = days before Jan 1 YEAR / 30.4375, so
    the date is recovered exactly and the sale found in the clean sales table
    by tax key and date. Raises unless every parcel with a prior sale gets
    exactly one. The price shown is the recorded sale price."""
    has = sub[sub.has_prior_sale == 1]
    days = np.round(has.months_since_prior.to_numpy() * 30.4375).astype("int64")
    want = pd.DataFrame({"tk": has.tk.to_numpy(),
                         "sale_date": pd.Timestamp(f"{YEAR}-01-01") - pd.to_timedelta(days, unit="D")})
    cand = s[~s.ppsf_outlier][["tk", "sale_date", "sale_price", "date_precision"]]
    m = want.merge(cand, on=["tk", "sale_date"], how="left")
    n = m.groupby("tk").sale_price.count()
    if len(n) != len(want) or (n != 1).any():
        raise AssertionError(f"own prior sale not found uniquely for {int((n != 1).sum())} of {len(want)} parcels")
    day = m.date_precision.eq("day").to_numpy()
    return pd.DataFrame({"tk": m.tk, "prior_sale_date": np.where(day, m.sale_date.dt.strftime("%Y-%m-%d"),
                                                                 m.sale_date.dt.strftime("%Y-%m")),
                         "prior_sale_price": m.sale_price.round().astype("int64")}).set_index("tk")


def _date_str(d: pd.Series, precision: pd.Series) -> np.ndarray:
    return np.where(precision.eq("day").to_numpy(), d.dt.strftime("%Y-%m-%d"), d.dt.strftime("%Y-%m"))


def subject_details(rows: pd.DataFrame, tks) -> pd.DataFrame:
    """Roll YEAR-1 characteristics as joined for the model, indexed by tk;
    details_note names fields taken from an earlier roll."""
    if rows.tk.duplicated().any():
        raise AssertionError("parcel rows hold a tax key twice")
    r = rows.set_index("tk").reindex(tks)
    out = pd.DataFrame({c: r[c].astype("Float64").round().astype("Int64") for c in SUBJECT_FIELDS}, index=r.index)
    notes = pd.Series("", index=r.index)
    for c, label in SUBJECT_FIELDS.items():
        src = r[f"{c}_src_roll"]
        old = r[c].notna() & src.notna() & (src != YEAR - 1)
        notes[old] += label + " from roll " + src[old].astype("int64").astype(str) + "; "
    out["details_note"] = notes.str.rstrip("; ").replace("", None)
    return out


def last_sales(s: pd.DataFrame, tks) -> pd.DataFrame:
    """Most recent clean sale per parcel (date, recorded price) and the
    latest removed sale dated after it (date, filter), indexed by tk.
    Month-only dates count as later only in a later month."""
    tks = pd.Index(tks)
    c = s[s.tk.isin(tks)].sort_values(["tk", "sale_date", "sale_id"])
    last = c.groupby("tk").tail(1).set_index("tk")
    n_on_last = c.merge(last[["sale_date"]].reset_index(), on=["tk", "sale_date"]).groupby("tk").size()
    if (n_on_last > 1).any():
        raise AssertionError(f"{int((n_on_last > 1).sum())} parcels have two clean sales on their last sale date")
    out = pd.DataFrame(index=tks)
    out["last_sale_date"] = pd.Series(_date_str(last.sale_date, last.date_precision), index=last.index)
    out["last_sale_price"] = last.sale_price.round().astype("Int64")

    rm = pd.read_parquet(INTERIM_DIR / "sales_clean_removed.parquet")
    rm = rm[rm.tk.isin(tks) & ~rm.code.isin(NOT_SALES)]
    paths = load_manifest().set_index("key").local_path
    raw = pd.concat([load_file(f"sales/{y}", paths[f"sales/{y}"])[["sale_id", "sale_date", "date_precision",
                                                                    "sale_price"]]
                     for y in sorted(rm.sale_year.unique())], ignore_index=True)
    rm = rm.merge(raw, on="sale_id", how="left", validate="one_to_one")
    if rm[["sale_date", "sale_price"]].isna().any().any():
        raise AssertionError("removed sales without a date or price in the raw files")
    labels = pd.read_csv(REPORTS_DIR / "stage2" / "filter_log.csv").drop_duplicates("code").set_index("code")["filter"]
    if not set(rm.code) <= set(labels.index):
        raise KeyError(f"filter codes without a label: {set(rm.code) - set(labels.index)}")
    rm = rm.join(last[["sale_date", "date_precision"]].rename(columns=lambda x: "c_" + x), on="tk")
    month = (rm.date_precision.eq("month") | rm.c_date_precision.eq("month")).to_numpy()
    later = rm.c_sale_date.isna().to_numpy() | np.where(
        month, rm.sale_date.dt.to_period("M") > rm.c_sale_date.dt.to_period("M"), rm.sale_date > rm.c_sale_date)
    rm = rm[later].sort_values(["tk", "sale_date", "sale_id"]).groupby("tk").tail(1).set_index("tk")
    out["removed_sale_date"] = pd.Series(_date_str(rm.sale_date, rm.date_precision), index=rm.index)
    out["removed_sale_price"] = rm.sale_price.round().astype("Int64")
    out["removed_sale_reason"] = rm.code.map(labels)
    return out


def unscored_reasons(rows: pd.DataFrame, scored_tks: set) -> pd.DataFrame:
    """Why an SF/duplex parcel in roll 2025 has no model value (the model's
    required inputs, from the same roll values the frame used)."""
    u = rows[~rows.tk.isin(scored_tks)]
    checks = [("finished sq ft", ~(u.bldg_area > 0)), ("bedrooms", u.bedrooms.isna()),
              ("year built", u.year_built.isna()),
              ("prior assessed value", u.prior_assessed_total.isna() | u.prior_assessed_land.isna())]
    miss = [", ".join(name for name, m in checks if m.loc[i]) for i in u.index]
    reason = [f"missing in roll {YEAR - 1}: {m}" if m else "missing a required model input" for m in miss]
    return pd.DataFrame({"tk": u.tk.to_numpy(), "reason": reason}).drop_duplicates("tk")


def roll_land_use() -> pd.DataFrame:
    """Land use for every tax key in roll 2025 (for the not-covered message)."""
    m = load_manifest().set_index("key")
    f = file_year_for(YEAR - 1, load_lookup())
    roll = load_year(f, m.loc[f"mprop_hist/{f}", "local_path"])[0]
    return roll[["tk", "land_use"]].dropna(subset=["tk"]).drop_duplicates("tk")  # land_use: string code


def land_use_codes() -> pd.DataFrame:
    m = load_manifest().set_index("key")
    c = pd.read_csv(DATA_DIR / m.loc["landusecode", "local_path"], dtype=str)
    return pd.DataFrame({"code": pd.to_numeric(c.CODE, errors="raise"), "description": c.DESCRIPTION.str.strip()})


def table_bytes(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Bytes each table takes with its indexes: written alone to a temporary
    database (this SQLite build has no dbstat), minus an empty database."""
    stmts = [x.strip() for x in SCHEMA.split(";") if x.strip()]
    out = {}
    for name, cols in TABLE_COLS.items():
        f = OUT_DIR / f"_size_{name}.sqlite"
        f.unlink(missing_ok=True)
        con = sqlite3.connect(f)
        for st in stmts:
            if f"TABLE {name} (" in st or f" ON {name} (" in st:
                con.execute(st)
        con.commit()
        con.execute("VACUUM")
        base = f.stat().st_size
        con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(cols))})",
                        ([_py(v) for v in row] for row in tables[name].itertuples(index=False, name=None)))
        con.commit()
        con.execute("VACUUM")
        con.close()
        out[name] = f.stat().st_size - base
        f.unlink()
    return out


def build(sample: bool) -> None:
    pmeta = check()
    spec = json.loads(SPEC_PATH.read_text())
    prim = spec["models"]["primary"]
    cfg, cols = prim["config"], prim["features"]
    if sha256_of(SPEC_PATH) != pmeta["model_spec_sha256"] or cols != pmeta["features"]:
        raise ValueError("frozen spec differs from the one recorded with the prospective predictions")
    pred = pd.read_parquet(PRED_PATH).set_index("tk")
    timings = {}

    with step("2026 parcel rows and frame (mirrors prospective.predict)", 3600):
        t0 = time.perf_counter()
        s = frames.sales()
        glob = frames.global_table(s)
        rows = parcel_rows(s)
        g2 = rows[["sale_id", "tk"]].merge(static_features(), on="tk", how="left").drop(columns="tk")
        g2 = g2.merge(roll_tables(rows), on="sale_id", how="left")
        s_ext = pd.concat([s, rows], ignore_index=True)
        glob_ext = pd.concat([glob, g2], ignore_index=True)
        fold = fold_frame(s_ext, YEAR, BASE_START, cfg["loc"], glob_ext, train_splits=("train", "test"))
        tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
        sc = fold["scored"]
        timings["rows_and_frame_s"] = time.perf_counter() - t0
        log(f"      {len(rows)} SF/duplex parcels in roll {YEAR - 1}; scorable {len(sc)}; train {len(tr)}")

    with step("refit frozen primary; exact match on every parcel", 3600):
        t0 = time.perf_counter()
        m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
        got = pd.Series(np.round(np.exp(m.predict(sc))).astype("int64"), index=sc.tk.to_numpy()).sort_index()
        want = pred.value_2026_01_01.sort_index()
        if not got.index.equals(want.index):
            raise AssertionError(f"refit scores {len(got)} parcels, the frozen file has {len(want)}")
        n_bad = int((got != want).sum())
        if n_bad:
            raise AssertionError(f"refit differs from the frozen predictions on {n_bad} of {len(want)} parcels")
        if len(tr) != pmeta["train_rows"]:
            raise AssertionError(f"train rows {len(tr)} != {pmeta['train_rows']} recorded at predict time")
        bp = booster_path(sample)
        bp.parent.mkdir(parents=True, exist_ok=True)
        m.booster.save_model(str(bp))
        timings["refit_s"] = time.perf_counter() - t0
        log(f"      exact match: {len(want)} of {len(want)} values; booster sha256 {sha256_of(bp)[:16]}...")

    sub = sc.sample(SAMPLE_N, random_state=SEED) if sample else sc
    with step(f"TreeSHAP and hybrid drivers for {len(sub)} parcels", 7200):
        t0 = time.perf_counter()
        sh = shap_frame(m, sub)
        timings["shap_s"] = time.perf_counter() - t0
        gap = np.abs(sh.sum(axis=1).to_numpy() - m.predict(sub)).max()
        if not gap < 1e-6:
            raise AssertionError(f"SHAP values do not add up to the prediction (max gap {gap})")
        drivers = driver_table(sh[cols], cols, sub.tk.to_numpy())
        log(f"      {timings['shap_s'] / len(sub) * 1000:.1f} ms per parcel; additivity gap {gap:.1e}")

    with step("addresses, comps, land use", 3600):
        t0 = time.perf_counter()
        mrows = mai_rows()
        addr = parcel_addresses(mrows).set_index("tk")
        pool = eligible_pool(tr, s, f"{YEAR}-01-01")
        pool["adj_price"] = np.exp(pool.y_detr)
        fi = pd.read_csv(REPORTS_DIR / "stage7" / "feature_importance.csv").set_index("feature").mean_abs_shap
        w = fi.reindex(STRUCT).fillna(0)
        cp = comps_for(sub, pool, w / w.sum(), tr[STRUCT].std())
        sale = s.set_index("sale_id")
        c = cp.merge(pool[["sale_id", "adj_price"]].rename(columns={"sale_id": "comp_sale_id"}), on="comp_sale_id")
        info = sale.loc[c.comp_sale_id]
        tk_of = dict(zip(sub.sale_id, sub.tk))
        day = info.date_precision.eq("day").to_numpy()
        comps = pd.DataFrame({
            "tk": c.sale_id.map(tk_of).to_numpy(), "rank": c["rank"].to_numpy(), "comp_tk": info.tk.to_numpy(),
            "address": addr.address.reindex(info.tk).to_numpy(),
            "sale_date": np.where(day, info.sale_date.dt.strftime("%Y-%m-%d"), info.sale_date.dt.strftime("%Y-%m")),
            "sale_price": info.sale_price.round().astype("int64").to_numpy(), "adj_price": c.adj_price.to_numpy(),
            "dist_m": c.dist_m.to_numpy(), "bldg_area": info.bldg_area.to_numpy(), "bedrooms": info.bedrooms.to_numpy(),
            "full_baths": info.full_baths.to_numpy(), "half_baths": info.half_baths.to_numpy(),
            "year_built": info.year_built.to_numpy()})
        tier = cp.groupby("sale_id")[["comp_radius_mi", "comp_window_months"]].first()
        tier.index = tier.index.map(tk_of)
        p = pred.loc[sub.tk]
        parcels = pd.DataFrame({
            "tk": sub.tk.to_numpy(), "address": addr.address.reindex(sub.tk).to_numpy(),
            "zip5": addr.zip5.reindex(sub.tk).to_numpy(), "land_use": p.land_use.astype("int64").to_numpy(),
            "property_type": p.land_use.astype("int64").map(PROPERTY_TYPE).to_numpy(),
            "neighborhood": p.neighborhood.to_numpy(), "value": p.value_2026_01_01.to_numpy(),
            "pi80_low": p.pi80_low.to_numpy(), "pi80_high": p.pi80_high.to_numpy(),
            "comp_radius_mi": tier.comp_radius_mi.reindex(sub.tk).to_numpy(),
            "comp_window_months": tier.comp_window_months.reindex(sub.tk).to_numpy()})
        prior = own_prior_sales(sub, s)
        parcels["prior_sale_date"] = prior.prior_sale_date.reindex(sub.tk).to_numpy()
        parcels["prior_sale_price"] = prior.prior_sale_price.reindex(sub.tk).to_numpy()
        extra = subject_details(rows, sub.tk).join(last_sales(s, sub.tk))
        for col in extra.columns:
            parcels[col] = extra[col].reindex(sub.tk).to_numpy()
        parcels = parcels[TABLE_COLS["parcels"]]
        if parcels.bldg_area.isna().any():
            raise AssertionError("a valued parcel has no finished sq ft in the subject details")
        if parcels.property_type.isna().any():
            raise ValueError("a valued parcel is neither single-family nor duplex")
        lu = roll_land_use()
        uns = unscored_reasons(rows, set(sc.tk)).set_index("tk").reason
        status = np.where(lu.tk.isin(pred.index), "valued", np.where(lu.tk.isin(uns.index), "no_estimate",
                          np.where(lu.land_use.isin(RES_LAND_USE), "no_estimate", "not_covered")))
        pstatus = pd.DataFrame({"tk": lu.tk.to_numpy(), "land_use": pd.to_numeric(lu.land_use, errors="raise").to_numpy(),
                                "status": status,
                                "reason": uns.reindex(lu.tk).to_numpy()})
        if not set(pred.index) <= set(pstatus.tk[pstatus.status == "valued"]):
            raise AssertionError("a valued parcel is missing from roll 2025's land use table")
        timings["comps_addresses_s"] = time.perf_counter() - t0
        log(f"      own prior sale found for {len(prior)} of {int((sub.has_prior_sale == 1).sum())} parcels with one")
        log(f"      last clean sale for {int(parcels.last_sale_date.notna().sum())} parcels; a later or only sale "
            f"removed by cleaning for {int(parcels.removed_sale_date.notna().sum())} "
            f"({parcels.removed_sale_reason.value_counts().to_dict()}); details from an earlier roll for "
            f"{int(parcels.details_note.notna().sum())}")
        log(f"      comps for {comps.tk.nunique()} of {len(sub)} parcels; "
            f"statuses {pstatus.status.value_counts().to_dict()}")

    meta = {"label": LABEL, "valuation_date": f"{YEAR}-01-01", "features_roll": YEAR - 1, "sample": sample,
            "predictions_sha256": pmeta["predictions_sha256"], "model_spec_sha256": pmeta["model_spec_sha256"],
            "booster_sha256": sha256_of(booster_path(sample)), "booster_file": booster_path(sample).name,
            "refit_exact_match": f"{len(pred)} of {len(pred)}", "train_rows": len(tr),
            "parcels_in_db": len(parcels), "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "build_commit": build_commit(),
            "shap_ms_per_parcel": round(timings["shap_s"] / len(sub) * 1000, 2),
            "timings_s": {k: round(v, 1) for k, v in timings.items()}}
    tables = {"parcels": parcels, "drivers": drivers[TABLE_COLS["drivers"]], "comps": comps, "parcel_status": pstatus,
              "land_use_codes": land_use_codes(), "mai": mrows[TABLE_COLS["mai"]]}
    path = db_path(sample)
    with step(f"write {path.name}", 3600):
        write_db(path, meta, tables)
    size = path.stat().st_size
    tb = table_bytes(tables)
    log(f"wrote {path} ({size / 1e6:.1f} MB); table bytes {tb}")
    n_rows = {k: len(v) for k, v in tables.items()}
    (OUT_DIR / f"build_report{'_sample' if sample else ''}.json").write_text(json.dumps(
        {"meta": meta, "db_bytes": size, "table_bytes": tb, "rows": n_rows}, indent=2, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "check"])
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    if a.command == "build":
        build(a.sample)
    else:
        meta = verify(db_path(a.sample))
        print(f"serving database verified: {meta['parcels_in_db']} parcels, predictions sha256 "
              f"{meta['predictions_sha256'][:16]}..., booster sha256 {meta['booster_sha256'][:16]}...")


if __name__ == "__main__":
    main()
