"""Tract demographics for the Stage 8 equity audit. EVALUATION ONLY (rule 7):
nothing here may enter a model; stage8 asserts that.

ACS 2019-2023 5-year estimates (vintage 2023), Milwaukee County tracts
(2020 geography, matching the TIGER 2023 tracts used for location):
  B03002 Hispanic or Latino origin by race: total, NH White, NH Black,
         NH Asian, Hispanic
  B19013 median household income
  B17001 poverty status (universe, below poverty)

The API key comes from CENSUS_API_KEY in the environment or the repo's
.env (git-ignored). It is never logged or written: the manifest records the
URL without it, and any error message is redacted.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from avm.acquire import MANIFEST_COLS, load_manifest, save_manifest, sha256_of
from avm.config import DATA_DIR, RAW_DIR, REPO_DIR

ACS_URL = "https://api.census.gov/data/2023/acs/acs5"
VARS = {"B03002_001E": "pop_total", "B03002_003E": "nh_white", "B03002_004E": "nh_black",
        "B03002_006E": "nh_asian", "B03002_012E": "hispanic", "B19013_001E": "median_hh_income",
        "B17001_001E": "pov_universe", "B17001_002E": "pov_below"}
KEY = "acs_2023_5yr_tracts"
DEST = RAW_DIR / KEY / "tracts.json"
MIN_TRACTS_OWN_GROUP = 5
MAJORITY = 0.5


def census_key() -> str:
    k = os.environ.get("CENSUS_API_KEY")
    env = REPO_DIR / ".env"
    if not k and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CENSUS_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        raise RuntimeError("CENSUS_API_KEY not found in the environment or in the repo's .env "
                           f"({env}); add a line CENSUS_API_KEY=<your key>")
    return k


def fetch_acs(force: bool = False) -> None:
    if DEST.exists() and not force:
        return
    key = census_key()
    params = {"get": "NAME," + ",".join(VARS), "for": "tract:*", "in": "state:55 county:079"}
    try:
        r = requests.get(ACS_URL, params={**params, "key": key}, timeout=(30, 120))
        r.raise_for_status()
        body = r.json()
    except Exception as e:  # never let the key reach a log or traceback
        raise RuntimeError(f"ACS request failed: {type(e).__name__}: {str(e).replace(key, '[REDACTED]')}") from None
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(body), encoding="utf-8")
    m = load_manifest().set_index("key", drop=False)
    url_no_key = requests.Request("GET", ACS_URL, params=params).prepare().url
    m.loc[KEY] = {"key": KEY, "package": "", "resource_name": "ACS 2019-2023 5-year, Milwaukee County tracts",
                  "resource_id": "", "url": url_no_key, "local_path": DEST.relative_to(DATA_DIR).as_posix(),
                  "bytes": str(DEST.stat().st_size), "sha256": sha256_of(DEST), "catalog_last_modified": "",
                  "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "daily_updated": "False"}
    save_manifest(m[MANIFEST_COLS])


def tract_table() -> pd.DataFrame:
    rows = json.loads(DEST.read_text())
    d = pd.DataFrame(rows[1:], columns=rows[0])
    d["tract"] = d.state + d.county + d.tract
    out = d[["tract"]].copy()
    for v, name in VARS.items():
        x = pd.to_numeric(d[v], errors="raise")
        out[name] = x.where(x >= 0)  # ACS uses large negative codes for suppressed values
    for g in ["nh_white", "nh_black", "nh_asian", "hispanic"]:
        out[f"share_{g}"] = out[g] / out.pop_total.where(out.pop_total > 0)
    out["poverty_rate"] = out.pov_below / out.pov_universe.where(out.pov_universe > 0)
    return out


def majority_group(t: pd.DataFrame) -> pd.Series:
    g = np.select([t.share_nh_black > MAJORITY, t.share_hispanic > MAJORITY, t.share_nh_white > MAJORITY,
                   t.share_nh_asian > MAJORITY],
                  ["Black majority", "Hispanic majority", "White majority", "Asian majority"], "no majority (mixed)")
    g = pd.Series(g, index=t.index)
    g[t.pop_total.isna() | (t.pop_total <= 0)] = "no population data"
    for grp, n in g.value_counts().items():
        if grp not in ("no majority (mixed)", "no population data") and n < MIN_TRACTS_OWN_GROUP:
            g[g == grp] = "no majority (mixed)"  # fold groups with < 5 tracts (Owen, 2026-10-18)
    return g


def income_quintiles(t: pd.DataFrame, parcels_per_tract: pd.Series) -> pd.Series:
    """Quintiles of tract median household income, weighted by SF/duplex
    parcels so each quintile holds about a fifth of the city's homes."""
    w = parcels_per_tract.reindex(t.tract).fillna(0).to_numpy()
    inc = t.median_hh_income.to_numpy()
    ok = ~np.isnan(inc) & (w > 0)
    order = np.argsort(inc[ok])
    cw = np.cumsum(w[ok][order]) / w[ok].sum()
    q = np.empty(ok.sum(), dtype=object)
    q[order] = [f"Q{min(5, int(c * 5 - 1e-9) + 1)}" for c in cw]
    out = pd.Series("income unavailable", index=t.index, dtype=object)
    out[ok] = q
    labels = {"Q1": "Q1 (lowest income)", "Q5": "Q5 (highest income)"}
    return out.replace(labels)
