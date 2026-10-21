"""City building permits, matched to sales by street address.

The permit file has no tax key, only an address like "123 N MAIN ST".
Sales addresses use the same form, sometimes with ", Unit 2" appended, so
both are normalized to the text before the first comma, upper case, single
spaces. Coverage is partial: about 1,600-2,000 issued permits a year,
2016 (from mid-year) to 2026, far fewer than the city issues, so the flag
under-counts renovation.

"Significant" permit: residential new construction (any cost; half of all
residential permits carry a cost of 0), or a residential alteration with
construction cost >= SIGNIFICANT_COST.
"""

import pandas as pd

from avm.acquire import load_manifest
from avm.config import DATA_DIR
from avm.io import to_number

SIGNIFICANT_COST = 10_000


def normalize_address(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.split(",").str[0].str.upper()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def load_permits() -> pd.DataFrame:
    m = load_manifest().set_index("key")
    p = pd.read_csv(DATA_DIR / m.loc["permits", "local_path"], dtype=str, keep_default_na=False)
    p.columns = p.columns.str.strip()
    out = pd.DataFrame({
        "address_norm": normalize_address(p["Address"]),
        "issued": pd.to_datetime(p["Date Issued"], format="%Y-%m-%d %H:%M:%S"),
        "permit_type": p["Permit Type"].str.strip(),
        "cost": to_number(p["Construction Total Cost"], "permits construction cost"),
        "record_id": p["Record ID"].str.strip(),
    })
    res = out.permit_type.str.startswith("Residential")
    new = out.permit_type == "Residential New Construction Permit"
    out["significant"] = res & (new | (out.cost >= SIGNIFICANT_COST))
    out["residential"] = res
    if out.issued.isna().any():
        raise ValueError(f"{int(out.issued.isna().sum())} permits with unparseable Date Issued")
    return out


def full_coverage_start(permits: pd.DataFrame) -> pd.Timestamp:
    """Jan 1 of the first year with at least half the median annual permit
    count (2015 has 2 permits and 2016 starts mid-year)."""
    n = permits.issued.dt.year.value_counts().sort_index()
    full = n[n >= 0.5 * n.median()]
    return pd.Timestamp(year=int(full.index.min()), month=1, day=1)
