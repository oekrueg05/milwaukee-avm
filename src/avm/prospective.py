"""Prospective test: Jan 1 2026 values for every SF/duplex parcel, predicted
BEFORE any 2026 sale is published, then scored later.

Usage:
    python -m avm.prospective predict   # now: write predictions + hash
    python -m avm.prospective check     # verify the file against its hash
    python -m avm.prospective score     # once 2026 sales AND roll 2026 exist

predict: the frozen primary model (reports/frozen/model_spec.json) is refit
on clean sales from its train_start through 2025 (time effect from those
years; reference = Dec 2025 trend level, neutral season). Each parcel with
SF/duplex land use in roll 2025 becomes a scoring row dated Jan 1 2026:
features from roll 2025 (earlier rolls only as fallback), prior sales before
Jan 1 2026, own prior sale more than 12 months before Jan 1 2026. 80%
intervals: CQR calibrated on pooled scored folds 2023-2025.

score: verifies the SHA-256 first (refuses to score a modified file), cleans
2026 sales with the Stage 2 rules, moves them to the Jan 1 2026 trend level
(trend-only, Stage 3 method), and compares the frozen predictions and roll
2026 on identical target prices.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from avm import frames
from avm.acquire import load_manifest, sha256_of
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.gbm import GBMX
from avm.intervals import calibrate, contain_point
from avm.join_rolls import baths_pre2019_raw, join_rolls
from avm.mprop import RES_LAND_USE, load_year
from avm.progress import log, step
from avm.rolls import file_year_for, load_lookup
from avm.spatial import static_features
from avm.stage6_geo import fold_frame, roll_tables

OUT = REPORTS_DIR / "prospective"
PRED_PATH = OUT / "predictions_2026.parquet"
META_PATH = OUT / "prospective_meta.json"
SPEC_PATH = REPORTS_DIR / "frozen" / "model_spec.json"
YEAR = 2026
CAL_YEARS = [2023, 2024, 2025]
BASE_START = 2014


def parcel_rows(s: pd.DataFrame, year: int = YEAR) -> pd.DataFrame:
    """One scoring row per SF/duplex parcel in roll year-1, joined like a sale
    dated Jan 1 of `year` (no price)."""
    m = load_manifest().set_index("key")
    lk = load_lookup()
    concepts = {}
    for a in range(year - 5, year + 1):
        try:  # roll `year` (the benchmark) exists for 2025, not yet for 2026
            f = file_year_for(a, lk)
        except KeyError:
            continue
        concepts[f] = load_year(f, m.loc[f"mprop_hist/{f}", "local_path"])[0]
    roll = concepts[file_year_for(year - 1, lk)]
    tks = roll[roll.land_use.isin(RES_LAND_USE)].tk.dropna().drop_duplicates()
    keys = pd.DataFrame({"sale_id": f"P{year}-" + tks.to_numpy(), "tk": tks.to_numpy(), "sale_year": year})
    j = join_rolls(keys, concepts, lk).merge(baths_pre2019_raw(keys, concepts, lk), on="sale_id")
    r = keys.merge(j, on="sale_id")
    r = r[r.in_roll_y1 & r.land_use_y1.isin(RES_LAND_USE)].copy()
    r["sale_date"] = pd.Timestamp(f"{year}-01-01")
    r["date_precision"] = "day"
    r["sale_price"] = np.nan
    r["target_price"] = np.nan
    r["split"] = "prospective"
    r["ppsf_outlier"] = False
    r["has_benchmark"] = False
    r["prop_type"] = "Residential"
    return r


def predict() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(SPEC_PATH.read_text())
    prim = spec["models"]["primary"]
    cfg, cols = prim["config"], prim["features"]
    s = frames.sales()
    glob = frames.global_table(s)
    with step("prospective rows: all SF/duplex parcels in roll 2025", 3600):
        rows = parcel_rows(s)
        st = static_features()
        g2 = rows[["sale_id", "tk"]].merge(st, on="tk", how="left").drop(columns="tk")
        g2 = g2.merge(roll_tables(rows), on="sale_id", how="left")
        s_ext = pd.concat([s, rows], ignore_index=True)
        glob_ext = pd.concat([glob, g2], ignore_index=True)
        log(f"      {len(rows)} parcels with SF/duplex land use in roll 2025")
    with step("2026 frame (train 2014-2025, detrend to Dec 2025)", 3600):
        fold = fold_frame(s_ext, YEAR, BASE_START, cfg["loc"], glob_ext, train_splits=("train", "test"))
        tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
        sc = fold["scored"]
        log(f"      train {len(tr)}; scorable parcels {len(sc)} of {len(rows)}")
    with step("frozen primary: fit through 2025, predict Jan 1 2026", 3600):
        m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
        log_pred = m.predict(sc)
    with step("CQR intervals calibrated on 2023-2025", 3600):
        cal = []
        for Y in CAL_YEARS:
            f = frames.fold(s, glob, Y, BASE_START, cfg["loc"])
            t = f["train"][f["train"].sale_year >= cfg["train_start"]]
            lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(t, t.y_detr).predict(f["scored"])
            hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(t, t.y_detr).predict(f["scored"])
            cal.append(pd.DataFrame({"lo": lo, "hi": hi, "y": np.log(f["scored"].target_price.to_numpy())}))
        cal = pd.concat(cal, ignore_index=True)
        q = calibrate(cal.lo.to_numpy(), cal.hi.to_numpy(), cal.y.to_numpy())["global"]
        lo = GBMX(cfg, cols, objective="quantile", alpha=0.1).fit(tr, tr.y_detr).predict(sc) - q
        hi = GBMX(cfg, cols, objective="quantile", alpha=0.9).fit(tr, tr.y_detr).predict(sc) + q
        n_outside = int(((log_pred < lo) | (log_pred > hi)).sum())
        lo, hi = contain_point(lo, hi, log_pred)
    out = pd.DataFrame({"tk": sc.tk.to_numpy(), "neighborhood": sc.neighborhood.to_numpy(),
                        "land_use": sc.land_use_y1.to_numpy(), "value_tier": sc.tier.to_numpy(),
                        "value_2026_01_01": np.round(np.exp(log_pred)).astype("int64"),
                        "pi80_low": np.round(np.exp(lo)).astype("int64"),
                        "pi80_high": np.round(np.exp(hi)).astype("int64")}).sort_values("tk", ignore_index=True)
    out.to_parquet(PRED_PATH, index=False)
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "predictions_sha256": sha256_of(PRED_PATH), "rows": len(out),
            "parcels_sf_duplex_roll_2025": len(rows), "train_rows": len(tr),
            "model_spec_sha256": sha256_of(SPEC_PATH), "config": cfg, "features": cols,
            "cqr_q_global": q, "cal_years": CAL_YEARS, "intervals_widened_to_contain_point": n_outside,
            "code_commit_at_predict": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                     text=True, cwd=REPORTS_DIR.parent).stdout.strip()}
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"wrote {PRED_PATH} ({len(out)} rows), sha256 {meta['predictions_sha256']}")


def check() -> dict:
    meta = json.loads(META_PATH.read_text())
    actual = sha256_of(PRED_PATH)
    if actual != meta["predictions_sha256"]:
        raise ValueError(f"predictions_2026.parquet hash {actual} != recorded {meta['predictions_sha256']}")
    p = pd.read_parquet(PRED_PATH)
    if len(p) != meta["rows"] or p.tk.duplicated().any():
        raise ValueError("prediction file rows or keys do not match the recorded metadata")
    log(f"prospective predictions verified: {len(p)} rows, sha256 {actual}")
    return meta


def score() -> None:
    """Run once 2026 sales and roll 2026 are published. Steps, each loud:
    1. check() the hash. 2. Add sales/2026 and mprop_hist/2026 to
    sources.SOURCES (extend SALES_YEARS / MPROP_HIST_YEARS), run
    `python -m avm.acquire` and `python -m avm.stage1_checks` so the lookup
    confirms roll 2026. 3. Clean 2026 sales with the Stage 2 rules. 4. Move
    them to the Jan 1 2026 trend level. 5. Compare frozen predictions and
    roll 2026 on identical prices."""
    from avm import clean_sales
    from avm.evaluate import compare, tier_prb, value_tiers
    from avm.time_adjust import fit_trend, seasonal_effects, trend_adjust

    check()
    keys = set(load_manifest().key)
    missing = [k for k in (f"sales/{YEAR}", f"mprop_hist/{YEAR}") if k not in keys]
    if missing:
        raise RuntimeError(f"not yet downloadable/added: {missing}; see the score() docstring")
    file_year_for(YEAR, load_lookup())  # raises unless roll 2026 is confirmed
    out = OUT / "score"
    clean_path = INTERIM_DIR / f"sales_clean_{YEAR}.parquet"
    clean_sales.build([YEAR], out / "stage2", clean_path)
    s26 = pd.read_parquet(clean_path)
    hist = pd.read_parquet(INTERIM_DIR / "sales_clean.parquet")
    seas = seasonal_effects(hist[hist.sale_year.between(2014, YEAR - 1)])
    both = pd.concat([hist[hist.sale_year == YEAR - 1], s26], ignore_index=True)
    tr = fit_trend(both, f"{YEAR}-01-01", seas)
    s26["target_price"] = trend_adjust(s26, tr, seas)
    p = pd.read_parquet(PRED_PATH)
    t = s26[s26.has_benchmark].merge(p, on="tk", how="inner").set_index("sale_id")
    t["tier"] = value_tiers(pd.concat([hist, s26]), t, YEAR).to_numpy()
    res = compare({"assessor": t.bench_assessed_total, "frozen_model": t.value_2026_01_01.astype(float)},
                  t.target_price, base="assessor")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "prospective_2026_comparison.csv", index=False)
    cover = ((t.target_price >= t.pi80_low) & (t.target_price <= t.pi80_high)).mean()
    tp = {k: tier_prb(v.astype(float), t.target_price, t.tier) for k, v in
          {"assessor": t.bench_assessed_total, "frozen_model": t.value_2026_01_01}.items()}
    pd.DataFrame([{"method": k, **v} for k, v in tp.items()]).to_csv(out / "within_tier_prb_2026.csv", index=False)
    log(f"scored {len(t)} 2026 sales; 80% interval coverage {cover:.3f}; results in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["predict", "check", "score"])
    a = ap.parse_args()
    {"predict": predict, "check": check, "score": score}[a.command]()


if __name__ == "__main__":
    main()
