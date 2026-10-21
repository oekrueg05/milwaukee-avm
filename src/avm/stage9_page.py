"""Stage 9 pages: builds two versions from reports/stage9 outputs.

Usage:
    python -m avm.stage9_page

Writes:
    <data>/reports_private/stage9/page/assessment_check_private.html  review copy: addresses and tax keys
                                     (outside the repo; never committed)
    reports/stage9/page/:
    assessment_check_portfolio.html  shareable: tract map, group table and
                                     methods; no parcel rows, addresses or
                                     tax keys in the page or its data

Inputs are the files written by avm.stage9_divergence (no model is run here)
plus TIGER 2023 tract outlines for the map. Neither version has owner names
(the MAI has none and none are added). Every parcel figure is labeled a model
estimate with its 80% range. Tracts with fewer than MIN_TRACT_PARCELS homes
carry no values in either version.
"""

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from avm.acquire import load_manifest
from avm.config import DATA_DIR, PRIVATE_REPORTS_DIR, REPORTS_DIR, REPO_DIR
from avm.geo import METRIC_CRS
from avm.progress import log
from avm.stage9_divergence import SPECIAL_RULES

IN = REPORTS_DIR / "stage9"
OUT = IN / "page"
PRIVATE_IN = PRIVATE_REPORTS_DIR / "stage9"           # parcel lists (outside the repo)
PRIVATE_OUT = PRIVATE_REPORTS_DIR / "stage9" / "page"  # the private review page (outside the repo)
TEMPLATE = REPO_DIR / "src" / "avm" / "templates" / "stage9_page.html"
MAP_W = 520  # SVG viewBox width
SIMPLIFY_M = 20
MIN_TRACT_PARCELS = 30  # fewer: drawn pale, no values shipped
VERSIONS = {"private": "assessment_check_private.html", "portfolio": "assessment_check_portfolio.html"}


def tract_paths(tracts: pd.DataFrame) -> tuple[list[dict], float]:
    m = load_manifest().set_index("key")
    g = gpd.read_file(f"zip://{DATA_DIR / m.loc['census_tracts_2023', 'local_path']}").to_crs(METRIC_CRS)
    g = g[g.GEOID.isin(set(tracts.tract))].copy()
    g["geometry"] = g.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    minx, miny, maxx, maxy = g.total_bounds
    k = MAP_W / (maxx - minx)

    def ring(coords):
        pts = [f"{(x - minx) * k:.1f},{(maxy - y) * k:.1f}" for x, y in coords]
        return "M" + "L".join(pts) + "Z"

    out = []
    t = tracts.set_index("tract")
    for _, r in g.iterrows():
        polys = [r.geometry] if r.geometry.geom_type == "Polygon" else list(r.geometry.geoms)
        d = "".join(ring(p.exterior.coords) + "".join(ring(i.coords) for i in p.interiors) for p in polys)
        row = t.loc[r.GEOID]
        rec = {"id": r.GEOID, "d": d, "n": int(row.parcels)}
        if row.parcels >= MIN_TRACT_PARCELS:
            rec.update(rel=round(float(row.median_relative), 4), usd=round(float(row.tax_shift_usd)),
                       mv=round(float(row.median_model_value), -2))
        out.append(rec)
    return out, round((maxy - miny) * k, 1)


def parcel_rows(t: pd.DataFrame) -> list[dict]:
    return [{"tk": r.tk, "addr": r.address, "zip": "" if pd.isna(r.zip5) else r.zip5,
             "type": "Duplex" if r.land_use_y1 == "8820" else "Single-family",
             "mv": round(r.model_value), "lo": round(r.range80_low), "hi": round(r.range80_high),
             "as": round(r.assessed_2025), "im": round(r.implied_market), "rel": round(r.relative_assessment, 4),
             "cv": None if pd.isna(r.comp_value_a) else round(r.comp_value_a), "nc": int(r.n_comps),
             "crad": float(r.comp_radius_mi), "cwin": int(r.comp_window_months),
             "usd": round(r.tax_shift_usd), "a24": None if pd.isna(r.assessed_2024) else round(r.assessed_2024),
             "flags": r.context_flags if isinstance(r.context_flags, str) else "",
             "reason": getattr(r, "special_reason", "") if isinstance(getattr(r, "special_reason", ""), str) else "",
             "drv": [d for d in (r.driver_1, r.driver_2, r.driver_3) if isinstance(d, str)]}
            for r in t.itertuples()]


def leaked_identifiers(html: str, identifiers) -> list[str]:
    """Every identifier (tax key or address) that appears anywhere in the page text."""
    return [str(v) for v in identifiers if str(v) in html]


def strip_blocks(html: str, drop: str, keep: str) -> str:
    """Remove <!--drop-->...<!--/drop--> blocks (in script, written as
    /*<!--drop-->*/ ... /*<!--/drop-->*/) and unwrap the keep markers."""
    html = re.sub(rf"(?:/\*)?<!--{drop}-->(?:\*/)?.*?(?:/\*)?<!--/{drop}-->(?:\*/)?\n?", "", html, flags=re.S)
    html = re.sub(rf"/\*<!--/?{keep}-->\*/\n?", "", html)
    return html.replace(f"<!--{keep}-->", "").replace(f"<!--/{keep}-->", "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    dt = {"tk": str, "zip5": str, "tract": str, "land_use_y1": str}
    top = pd.read_csv(PRIVATE_IN / "divergence_top.csv", dtype=dt)
    special = pd.read_csv(PRIVATE_IN / "special_valuation_list.csv", dtype=dt)
    stats = pd.read_csv(IN / "divergence_stats.csv").iloc[0]
    groups = pd.read_csv(IN / "group_tax_shifts.csv")
    tracts = pd.read_csv(IN / "tract_summary.csv", dtype={"tract": str})
    excl = pd.read_csv(IN / "exclusions.csv")
    tax = json.loads((IN / "tax_rate.json").read_text())
    ypath = IN / "duplex_yield_summary.csv"
    yields = pd.read_csv(ypath) if ypath.exists() else None

    paths, map_h = tract_paths(tracts)
    pct = lambda v: None if pd.isna(v) else round((np.exp(v) - 1) * 100, 2)  # noqa: E731
    grp = [{"grouping": r.grouping, "group": r.group, "n": int(r.parcels),
            "s": pct(r.median_shift_log), "slo": pct(r.shift_lo), "shi": pct(r.shift_hi),
            "b": pct(r.model_bias_rel_log), "blo": pct(r.bias_lo), "bhi": pct(r.bias_hi),
            "c": pct(r.corrected_log), "clo": pct(r.corrected_lo), "chi": pct(r.corrected_hi),
            "ct": pct(r.corrected_log_tercile), "ctlo": pct(r.corrected_lo_tercile), "cthi": pct(r.corrected_hi_tercile),
            "usd": round(r.tax_shift_usd_total), "verdict": r.verdict,
            "ok_a": bool(r.check_a), "ok_b": bool(r.check_b), "ok_c": bool(r.check_c),
            "persist": r.persistence, "yrs": int(r.years_holding), "usd_home": round(r.usd_typical_home),
            "usd_group": round(r.usd_group_total), "usd_other": round(r.usd_offset_per_other_home),
            "n_other": int(r.other_homes), "med_as": round(r.median_assessed_2025),
            "sn": int(r.sales_n), "city_ratio": float(r.sales_citywide_ratio),
            **{k: None if pd.isna(v) else round((v - 1) * 100, 2)
               for k, v in (("sr", r.sales_rel), ("srlo", r.sales_rel_lo), ("srhi", r.sales_rel_hi),
                            ("srt", r.sales_rel_tercile), ("srtlo", r.sales_rel_lo_tercile),
                            ("srthi", r.sales_rel_hi_tercile))}}
           for r in groups.itertuples()]
    fe = pd.read_csv(REPORTS_DIR / "frozen" / "frozen_eval_2025.csv").set_index("role")
    c6 = pd.read_csv(REPORTS_DIR / "stage6" / "comparison_2025.csv")
    a6 = c6[(c6.statistic == "mdape_ln") & (c6.method == "assessor")].iloc[0]
    if int(a6.n) != int(fe.loc["primary", "n"]):
        raise ValueError("assessor and frozen-model 2025 accuracy are on different sale counts")
    accuracy = {"model": float(fe.loc["primary", "mdape_ln"]), "assessor": float(a6.estimate), "n": int(a6.n)}
    st = {k: (None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating)) else int(v)))
          for k, v in stats.items()}
    rules = [{"rule": k, "definition": v, "n": st[f"special_{k.replace(' ', '_')}"]} for k, v in SPECIAL_RULES.items()]
    base = {"stats": st, "tracts": paths, "groups": grp, "accuracy": accuracy, "special_rules": rules,
            "map": {"w": MAP_W, "h": map_h, "min_parcels": MIN_TRACT_PARCELS, "tracts_drawn": len(paths),
                    "tracts_small": int((tracts.parcels < MIN_TRACT_PARCELS).sum())},
            "exclusions": excl.to_dict("records"), "tax": tax,
            "yields": None if yields is None else yields.round(4).to_dict("records")}
    template = TEMPLATE.read_text(encoding="utf-8")
    for version, fname in VERSIONS.items():
        data = dict(base)
        if version == "private":
            data.update(rows=parcel_rows(top), special=parcel_rows(special))
            html = strip_blocks(template, "portfolio", "private")
        else:
            html = strip_blocks(template, "private", "portfolio")
        html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
        if version == "portfolio":  # loud check: no parcel identifiers may reach the shareable copy
            leaked = leaked_identifiers(html, pd.concat([top.tk, special.tk, top.address, special.address]))
            if leaked:
                raise AssertionError(f"portfolio page contains parcel identifiers, e.g. {leaked[:3]}")
        dest = (PRIVATE_OUT if version == "private" else OUT) / fname
        dest.write_text(html, encoding="utf-8")
        log(f"wrote {dest} ({dest.stat().st_size / 1e6:.2f} MB; "
            f"{len(data.get('rows', []))} ranked + {len(data.get('special', []))} set-aside parcels, {len(paths)} tracts)")


if __name__ == "__main__":
    main()
