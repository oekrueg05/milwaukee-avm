"""Data dictionary: one CSV per downloaded tabular file, listing every field
with inferred type, null rate, distinct count, top values and examples.

Usage:
    python -m avm.profile_data      # profiles every CSV/ZIP in the manifest
"""

import re
import zipfile

import pandas as pd
import pyogrio

from avm.acquire import load_manifest
from avm.config import DATA_DIR, REPORTS_DIR
from avm.io import CORRUPT_DATE_RE, is_null, read_parcels, read_raw_csv
from avm.progress import log, step

PROFILE_TIMEOUT_S = 600  # one file; the largest MPROP CSV takes about a minute

OUT_DIR = REPORTS_DIR / "stage1" / "data_dictionary"
DATE_LIKE_RE = r"^(\d{1,4}[/-]\d{1,2}([/-]\d{1,4})?|[A-Z][a-z]{2} +\d{1,2} \d{4})"
MAX_EXAMPLE_LEN = 40
SEED = 0


def infer_type(vals: pd.Series) -> str:
    if vals.empty:
        return "empty"
    num = pd.to_numeric(vals, errors="coerce")
    if num.notna().mean() >= 0.99:
        return "integer" if (num.dropna() % 1 == 0).all() else "float"
    if (vals.str.match(DATE_LIKE_RE) | vals.str.match(CORRUPT_DATE_RE)).mean() >= 0.9:
        return "date"
    return "text"


def _short(v: str) -> str:
    return v if len(v) <= MAX_EXAMPLE_LEN else v[:MAX_EXAMPLE_LEN] + "..."


# Fields whose sample values would expose people or parcels: owner names and
# mailing addresses, street addresses, tax keys (the sales files' ID is a tax
# key), parcel geometry and legal descriptions. Their counts and null rates
# are kept; top values and examples are blanked.
SENSITIVE_FIELD_RE = re.compile(r"(?i)owner|mail|billing|address|addr|taxkey|tax_key|^fk_tax$|^tk$|^id$|shape|legal")
REDACTED = "(redacted)"


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col].astype("string")
        null = is_null(s)
        vals = s[~null]
        top = vals.value_counts().head(3)
        examples = vals.drop_duplicates().sample(min(3, vals.nunique()), random_state=SEED)
        rows.append({
            "field": col,
            "inferred_type": infer_type(vals),
            "n_rows": n,
            "null_rate": round(null.mean(), 4) if n else None,
            "n_unique": int(vals.nunique()),
            "corrupt_date_rate": round(vals.str.match(CORRUPT_DATE_RE).sum() / n, 4) if n else None,
            "top_values": REDACTED if SENSITIVE_FIELD_RE.search(col) else
            "; ".join(f"{_short(k)} ({v})" for k, v in top.items()),
            "examples": REDACTED if SENSITIVE_FIELD_RE.search(col) else "; ".join(_short(v) for v in examples),
        })
    return pd.DataFrame(rows)


def load_any(local_path: str, key: str) -> pd.DataFrame:
    if local_path.endswith(".csv"):
        return read_raw_csv(DATA_DIR / local_path)
    if local_path.endswith(".zip"):
        with zipfile.ZipFile(DATA_DIR / local_path) as zf:
            names = zf.namelist()
            if not any(n.lower().endswith(".shp") for n in names) and "stops.txt" in names:
                return pd.read_csv(zf.open("stops.txt"), dtype=str, keep_default_na=False)  # GTFS: stops used
        return read_parcels(key).astype("string")
    if local_path.endswith(".geojson"):
        return pyogrio.read_dataframe(DATA_DIR / local_path, read_geometry=False).astype("string")
    raise ValueError(f"no reader for {local_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    todo = manifest[manifest.local_path.str.endswith((".csv", ".zip", ".geojson"))]
    for i, (_, m) in enumerate(todo.iterrows(), 1):
        with step(f"[{i}/{len(todo)}] profile {m['key']}", PROFILE_TIMEOUT_S):
            df = load_any(m["local_path"], m["key"])
            dd = data_dictionary(df)
            out = OUT_DIR / f"{m['key'].replace('/', '_')}.csv"
            dd.to_csv(out, index=False)
        log(f"      {len(df):,} rows, {df.shape[1]} cols -> {out.name}")


if __name__ == "__main__":
    main()
