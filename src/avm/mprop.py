"""MPROP yearly snapshots: one crosswalk from the three raw schemas to stable
concept names, plus a per-year cache of just those concepts.

Schemas seen in the raw files:
  old    2008-2021 and current MPROP: TAXKEY, C_A_TOTAL, YR_BUILT, ...
         (2017 cuts long old names to 10 chars: LAST_VALUE, CONVEY_DAT)
  new10  2022-2023: parcel-layer export, names cut to 10 chars (AssessValu, ...)
  new    2024-2025: parcel-layer export, full names (TotalAssessedValue, ...)

A name match is only a candidate. stage1_checks validates every concept by
value continuity on the same parcels across consecutive years.
"""

import json
import re

import pandas as pd

from avm.acquire import sha256_of
from avm.config import DATA_DIR, INTERIM_DIR
from avm.io import normalize_taxkey, parse_dates, read_raw_csv, to_number

CACHE_DIR = INTERIM_DIR / "mprop_concepts"
CACHE_VERSION = 7  # bump when CROSSWALK or extraction logic changes

# concept -> candidate raw columns, first present wins. Every name is listed
# explicitly, including 10-char truncations: automatic truncation once mapped
# CalculatedTotalSF -> "Calculated" (acres) and AsmtChangeReason ->
# "AsmtChange" (the date). Order: old, old 2017 (10-char), new, new10.
CROSSWALK: dict[str, list[str]] = {
    "taxkey": ["TAXKEY", "Taxkey"],
    "yr_assmt": ["YR_ASSMT"],
    "assessed_land": ["C_A_LAND", "TotalLandValue", "TotalLandV"],
    "assessed_total": ["C_A_TOTAL", "TotalAssessedValue", "AssessValu"],
    "prev_assessed_total": ["P_A_TOTAL", "TotaPrevNonExemptValue", "TotaPrevNo"],
    "assessment_change_date": ["LAST_VALUE_CHG", "LAST_VALUE", "AsmtChangeDate", "AsmtChange"],
    "assessment_change_reason": ["REASON_FOR_CHG", "REASON_FOR", "AsmtChangeReason", "AsmtChan_1"],
    "convey_date": ["CONVEY_DATE", "CONVEY_DAT", "SaleDate"],
    "convey_type": ["CONVEY_TYPE", "CONVEY_TYP", "Deed"],
    # New-layout LandUse is a different code (0% match); FK_LandUse matches
    # the old 4-digit LAND_USE on 99.0-99.8% of parcels (2021 vs 2022-2025).
    "land_use": ["LAND_USE", "FK_LandUse"],
    "land_use_gp": ["LAND_USE_GP", "LAND_USE_G"],
    # Assessor exemption code (Appendix B); municipal codes mark city-owned land.
    "exempt_type": ["C_A_EXM_TYPE", "C_A_EXM_TY", "CurrentYearExemptionType", "CurrentYea"],
    "neighborhood": ["NEIGHBORHOOD", "NEIGHBORHO", "Neighborhood", "Neighborho"],
    "bldg_type": ["BLDG_TYPE", "BuildingType", "BuildingTy"],
    "nr_stories": ["NR_STORIES"],
    "year_built": ["YR_BUILT", "YearBuilt"],
    "bldg_area": ["BLDG_AREA"],
    "rooms": ["NR_ROOMS", "NumberOfRooms", "NumberOfRo"],
    "bedrooms": ["BEDROOMS", "NumberOfBedrooms", "NumberOfBe"],
    "full_baths": ["BATHS", "NumberOfFullBaths", "NumberOfFu"],
    "half_baths": ["POWDER_ROOMS", "POWDER_ROO", "NumberOfHalfBaths", "NumberOfHa"],
    "nr_units": ["NR_UNITS", "ResidentialUnits", "Residentia"],
    # 2022-2023: "Calculated" is acres (CalculatedTotalAC); Calculat_1 is sq ft.
    "lot_area": ["LOT_AREA", "CalculatedTotalSF", "Calculat_1"],
}

# Concept values blanked for a file year after continuity validation showed
# the raw column measures something else. Each entry needs its evidence.
_BATHS_RECODE = (
    "Bath counts were recoded at 2018->2019: on SF/duplex parcels 3,421 go 2->1 full baths "
    "and ~4,300 lose a half bath; 2019-2025 agree with each other at >=99.6%. Kept the "
    "2019+ definition, which covers the test year's features (roll 2024).")
_STORIES_ROUNDED = (
    "Stories in 2019-2020 round 1.5 up to 2: 19,349 parcels go 15->2 at 2018->2019 and "
    "19,641 go 2->1.5 at 2020->2021, so 1.5-story homes are not identifiable.")
EXCLUDED: dict[tuple[str, int], str] = {
    ("bldg_area", 2022): (
        "2022 BLDG_AREA on SF/duplex parcels is a different measure: median 1.83x "
        "the 2021 value, 0.2% equal, while 2023 equals 2021 on 99.0%. Non-residential "
        "parcels are unaffected. Use another roll's sq ft (Stage 2 decision)."),
    ("nr_stories", 2019): _STORIES_ROUNDED,
    ("nr_stories", 2020): _STORIES_ROUNDED,
    ("bldg_type", 2025): (
        "Building type renumbered again in 2025 (10->17: 3,665; 1->10: 1,821; 5->8: 1,007; "
        "14->11: 879 on SF/duplex). Roll 2025 is the benchmark only, never a feature source."),
    **{(c, y): _BATHS_RECODE for c in ("full_baths", "half_baths") for y in range(2008, 2019)},
}

RES_LAND_USE = {"8810", "8820"}
STORIES_X10_YEARS = range(2008, 2019)    # 10 = 1 story, 15 = 1.5, 20 = 2
BLDG_TYPE_19_IS_8_YEARS = range(2020, 2025)


def _normalize(concepts: pd.DataFrame, year: int) -> pd.DataFrame:
    """Representation fixes: same meaning, different storage. Each rule is
    backed by the continuity table (mprop_crosswalk_continuity.csv)."""
    # Building type: "01" (new layout) and "1" (old) are the same code.
    bt = concepts["bldg_type"].str.strip()
    bt = bt.where(~bt.str.fullmatch(r"\d+").fillna(False), bt.str.lstrip("0"))
    # 2020-2024 code 19 on SF/duplex parcels: the documented 19 is a 7-9 unit
    # apartment (impossible there); 11,492 of 2019's code 8 ("Residence old
    # style") became 19 in 2020. Map back to 8.
    if year in BLDG_TYPE_19_IS_8_YEARS:
        res = concepts["land_use"].isin(RES_LAND_USE).fillna(False)
        bt = bt.mask(res & (bt == "19"), "8")
    concepts["bldg_type"] = bt
    # Stories stored x10 through 2018.
    if year in STORIES_X10_YEARS:
        st = to_number(concepts["nr_stories"], f"MPROP {year} nr_stories") / 10
        concepts["nr_stories"] = st.map(lambda v: "" if pd.isna(v) else f"{v:g}").astype("string")
    # Exemption codes: new layout writes "010", old "10".
    ex = concepts["exempt_type"].str.strip()
    concepts["exempt_type"] = ex.where(~ex.str.fullmatch(r"\d+").fillna(False), ex.str.lstrip("0"))
    # Lot area 0 is the missing-value code (2% of SF/duplex parcels flip to 0).
    la = concepts["lot_area"]
    concepts["lot_area"] = la.mask(to_number(la, f"MPROP {year} lot_area") == 0, pd.NA)
    return concepts
REQUIRED = {"taxkey", "assessed_total", "prev_assessed_total", "assessment_change_date"}
GEOMETRY_COLS = {"shape", "gdb_geomattr_data"}
DATE_FIELD_RE = re.compile(r"(DATE|_CHG$|Date$|^AsmtChange$|^SaleDate$|Dat$|^ActivatedD$)",
                           re.IGNORECASE)


def _schema(cols: list[str]) -> str:
    if "C_A_TOTAL" in cols:
        return "old"
    return "new10" if "AssessValu" in cols else "new"


def _extract(year: int, path) -> tuple[pd.DataFrame, dict]:
    df = read_raw_csv(path)
    cols = list(df.columns)
    char_len = df.apply(lambda s: s.str.len()).sum()
    total_chars = int(char_len.sum())
    geom_chars = int(sum(v for k, v in char_len.items() if k.lower() in GEOMETRY_COLS))

    used = {}
    for concept, cands in CROSSWALK.items():
        hit = [c for c in cands if c in cols]
        used[concept] = hit[0] if hit else None
    missing = REQUIRED - {k for k, v in used.items() if v}
    if missing:
        raise KeyError(f"MPROP {year}: no column for required concepts {sorted(missing)}")
    picked = [v for v in used.values() if v]
    dupes = {v for v in picked if picked.count(v) > 1}
    if dupes:
        raise ValueError(f"MPROP {year}: raw columns mapped to more than one concept: {dupes}")

    date_quality = []
    for c in cols:
        if DATE_FIELD_RE.search(c) and not c.upper().startswith("CONVEY_FEE"):
            _, cnt = parse_dates(df[c])
            date_quality.append({"file_year": year, "field": c, "n_rows": len(df), **cnt})

    concepts = pd.DataFrame({k: (df[v] if v else pd.Series(pd.NA, index=df.index, dtype="string"))
                             for k, v in used.items()}).astype("string")
    concepts["tk"] = normalize_taxkey(concepts["taxkey"])
    concepts = _normalize(concepts, year)
    excluded = {c: why for (c, y), why in EXCLUDED.items() if y == year}
    for c in excluded:
        # The raw value is kept apart (<concept>_raw_excluded) for explicit,
        # opt-in uses such as the Stage 5 baths sensitivity test; the main
        # column is blank so nothing uses it by accident.
        concepts[f"{c}_raw_excluded"] = concepts[c]
        concepts[c] = pd.Series(pd.NA, index=concepts.index, dtype="string")

    file_bytes = path.stat().st_size
    meta = {
        "year": year,
        "schema": _schema(cols),
        "columns": cols,
        "crosswalk_used": used,
        "date_quality": date_quality,
        "excluded": excluded,
        "stats": {
            "file_bytes": file_bytes,
            "rows": len(df),
            "cols": len(cols),
            "bytes_per_row": round(file_bytes / len(df)),
            "geometry_col_share_of_chars": round(geom_chars / total_chars, 3),
            "chars_per_row_excl_geometry": round((total_chars - geom_chars) / len(df)),
            "invalid_taxkeys": int(concepts["tk"].isna().sum()),
            "duplicate_taxkeys": int(concepts["tk"].dropna().duplicated().sum()),
        },
    }
    return concepts, meta


def load_year(year: int, rel_path: str) -> tuple[pd.DataFrame, dict]:
    """Concept frame + metadata for one file, cached in interim/ and keyed on
    the raw file's sha256 so a changed download invalidates the cache."""
    path = DATA_DIR / rel_path
    sha = sha256_of(path)
    pq, js = CACHE_DIR / f"{year}.parquet", CACHE_DIR / f"{year}_meta.json"
    if pq.exists() and js.exists():
        meta = json.loads(js.read_text())
        if meta.get("raw_sha256") == sha and meta.get("cache_version") == CACHE_VERSION:
            return pd.read_parquet(pq), meta
    concepts, meta = _extract(year, path)
    meta |= {"raw_sha256": sha, "cache_version": CACHE_VERSION, "raw_path": rel_path}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    concepts.to_parquet(pq, index=False)
    js.write_text(json.dumps(meta, indent=1, default=str))
    return concepts, meta
