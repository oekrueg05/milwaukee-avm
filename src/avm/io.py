"""Readers for raw files plus the two cleaning primitives every stage needs:
tax key normalization and date parsing that refuses corrupted values."""

import re
import zipfile
from pathlib import Path

import pandas as pd
import pyogrio

from avm.config import RAW_DIR

NULL_TOKENS = {"", "NULL", "NaN", "nan", "None"}

# Several exports replaced dates with Excel-style "mm:ss.f" fragments
# ("00:00.0", "45:32.0"). pandas would parse these as today's date, so they
# are detected and treated as corrupt, never parsed.
CORRUPT_DATE_RE = re.compile(r"^\d{1,2}:\d{2}(\.\d+)?$")
# 1899-12-30 is the Excel/SQL zero date, used as a placeholder.
SENTINEL_DATE_RE = re.compile(r"^1899-12-30")


def raw_file(key: str, suffix: str = ".csv") -> Path:
    files = sorted((RAW_DIR / key).glob(f"*{suffix}"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected 1 {suffix} file in {RAW_DIR / key}, found {len(files)}")
    return files[0]


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read every column as stripped text. No type guessing at this stage."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


def is_null(s: pd.Series) -> pd.Series:
    return s.isna() | s.isin(NULL_TOKENS)


def normalize_taxkey(s: pd.Series) -> pd.Series:
    """Return a 10-character zero-padded tax key, or NA if the value cannot be
    one. Some files store the key as a number, dropping leading zeros
    ("30171000") or adding ".0"; real keys have 7 to 10 significant digits."""
    t = s.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    ok = t.str.fullmatch(r"\d{7,10}").fillna(False)
    return t.where(ok).str.zfill(10)


def to_number(s: pd.Series, label: str) -> pd.Series:
    """Strict numeric parse for values that feed the model. Blank/NULL -> NA;
    "$1,234.00" -> 1234.0; anything else that fails raises with examples.
    Never use pd.to_numeric(errors="coerce") on model inputs: it once blanked
    every 2020-2021 sale price ("$357,000") without a sound."""
    t = s.astype("string").str.strip()
    t = t.mask(is_null(t)).str.replace(r"[$,]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    bad = t.notna() & out.isna()
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} unparseable numbers, e.g. {s[bad].head(5).tolist()}")
    return out


def parse_dates(s: pd.Series) -> tuple[pd.Series, dict]:
    """Parse a raw date column. Returns (datetimes, counts) where counts has
    n_null, n_corrupt, n_sentinel, n_unparsed, n_parsed."""
    t = s.astype("string").str.strip()
    null = is_null(t)
    corrupt = t.str.match(CORRUPT_DATE_RE).fillna(False) & ~null
    sentinel = t.str.match(SENTINEL_DATE_RE).fillna(False) & ~null
    usable = ~(null | corrupt | sentinel)
    # format="mixed" parses element by element; parse each distinct value once.
    codes, uniq = pd.factorize(t.where(usable))
    parsed_u = pd.DatetimeIndex(pd.to_datetime(pd.Series(uniq, dtype="string"), errors="coerce",
                                               format="mixed"))
    vals = parsed_u.take(codes, allow_fill=True, fill_value=pd.NaT) if len(uniq) else \
        pd.DatetimeIndex([pd.NaT] * len(t))
    parsed = pd.Series(vals, index=t.index)
    counts = {
        "n_null": int(null.sum()),
        "n_corrupt": int(corrupt.sum()),
        "n_sentinel": int(sentinel.sum()),
        "n_unparsed": int((usable & parsed.isna()).sum()),
        "n_parsed": int(parsed.notna().sum()),
    }
    return parsed, counts


def read_parcels(key: str, read_geometry: bool = False) -> pd.DataFrame:
    z = raw_file(key, ".zip")
    with zipfile.ZipFile(z) as zf:
        shps = [n for n in zf.namelist() if n.lower().endswith(".shp")]
    if len(shps) != 1:
        raise RuntimeError(f"{z}: expected 1 .shp member, found {shps}")
    return pyogrio.read_dataframe(f"zip://{z}!{shps[0]}", read_geometry=read_geometry)
