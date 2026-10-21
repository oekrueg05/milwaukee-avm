"""File-year -> assessment-year lookup for MPROP snapshots.

Every join to MPROP goes through this table, keyed by assessment year
(valuation date Jan 1 of that year), never by file name.

Evidence, strongest first:
  anchor     The modal assessment change date, when >= ANCHOR_SHARE of parcels
             share it. Milwaukee sets roll A during year A (spring values,
             December tax bills), so its year is the assessment year.
             Corrupt or blank in 2008-2013 and 2020.
  chain      File Y's previous-year value equals file Y-1's current value on
             >= LINK_THRESHOLD of parcels: the files hold consecutive rolls.
             Not usable when file Y is "post-rollover" (current == previous on
             >= ROLLOVER_SHARE of parcels: the extract was taken after the city
             copied the roll into the previous-year field for the next cycle).
  yr_assmt   Old-schema YR_ASSMT. Post-rollover files carry next year's
             label, so they vote YR_ASSMT - 1. Not trusted alone (2015 says
             2016 but its change dates and both chain links say 2015).
  between    An isolated file whose values differ from both neighbouring
             rolls holds the roll in between.

Linked files form groups in which assessment_year = file_year + k. Every
anchor in a group must agree on k, otherwise the group is a conflict.

Status:
  confirmed            group has a change-date anchor and >= 2 linked files
  anchor_only          own change-date anchor, no usable chain link
  chained_to_yr_assmt  group has no anchor; all yr_assmt votes agree
  inferred_between     isolated file, values distinct from both neighbours
  conflict / unresolved
"""

import pandas as pd

from avm.config import REPORTS_DIR
from avm.io import parse_dates

LOOKUP_PATH = REPORTS_DIR / "stage1" / "mprop_assessment_year_lookup.csv"
ANCHOR_SHARE = 0.50
LINK_THRESHOLD = 0.90
ROLLOVER_SHARE = 0.99
DISTINCT_MAX_EQUAL = 0.90


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _by_tk(c: pd.DataFrame) -> pd.DataFrame:
    return c.dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")


def _share_equal(a: pd.Series, b: pd.Series) -> float:
    a, b = _num(a), _num(b)
    ok = a.notna() & b.notna()
    return round(float((a[ok] == b[ok]).mean()), 4) if ok.any() else float("nan")


def build_lookup(concepts: dict[int, pd.DataFrame]) -> pd.DataFrame:
    years = sorted(concepts)
    info = {}
    for y in years:
        c = concepts[y]
        chg, _ = parse_dates(c["assessment_change_date"])
        mode = chg.mode().iloc[0] if chg.notna().any() else pd.NaT
        mode_share = float((chg == mode).mean()) if pd.notna(mode) else 0.0
        yr = c["yr_assmt"].dropna()
        yr = yr[yr != ""]
        own_eq = _share_equal(c["assessed_total"], c["prev_assessed_total"])
        info[y] = {
            "file_year": y,
            "change_date_mode": mode.date() if pd.notna(mode) else None,
            "change_date_mode_share": round(mode_share, 4),
            "anchor_year": mode.year if mode_share >= ANCHOR_SHARE else None,
            "yr_assmt_mode": int(yr.mode().iloc[0]) if len(yr) else None,
            "share_current_eq_own_prev": own_eq,
            "post_rollover": bool(own_eq >= ROLLOVER_SHARE),
        }
        r = info[y]
        r["yr_assmt_vote"] = (None if r["yr_assmt_mode"] is None
                              else r["yr_assmt_mode"] - (1 if r["post_rollover"] else 0))

    # Pairwise evidence between consecutive files.
    link = {}
    for a, b in zip(years, years[1:]):
        if b != a + 1:
            continue
        A, B = _by_tk(concepts[a]), _by_tk(concepts[b])
        common = A.index.intersection(B.index)
        prev_link = _share_equal(B.loc[common, "prev_assessed_total"], A.loc[common, "assessed_total"])
        cur_eq = _share_equal(B.loc[common, "assessed_total"], A.loc[common, "assessed_total"])
        usable = (not info[b]["post_rollover"]) and prev_link >= LINK_THRESHOLD
        link[(a, b)] = usable
        info[b]["link_from_prev_file"] = prev_link
        info[b]["current_eq_prev_file_current"] = cur_eq
        info[b]["chain_link_back"] = usable
        info[a]["chain_link_forward"] = usable

    # Groups of chain-linked files.
    groups, cur = [], [years[0]]
    for a, b in zip(years, years[1:]):
        if link.get((a, b)):
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)

    for g in groups:
        ks_anchor = {info[y]["anchor_year"] - y for y in g if info[y]["anchor_year"] is not None}
        ks_vote = {info[y]["yr_assmt_vote"] - y for y in g if info[y]["yr_assmt_vote"] is not None}
        gid = f"{g[0]}-{g[-1]}"
        if len(ks_anchor) > 1:
            status, k = "conflict: anchors disagree within chain", None
        elif len(ks_anchor) == 1:
            k = ks_anchor.pop()
            status = "confirmed" if len(g) >= 2 else "anchor_only"
        elif len(ks_vote) == 1 and len(g) >= 2:
            k, status = ks_vote.pop(), "chained_to_yr_assmt"
        else:
            k, status = None, "unresolved"
        for y in g:
            info[y] |= {"chain_group": gid, "status": status,
                        "assessment_year": y + k if k is not None else None}

    # Isolated unresolved files sitting between two resolved neighbours.
    for y in years:
        r = info[y]
        if r["status"] != "unresolved":
            continue
        lo, hi = info.get(y - 1, {}), info.get(y + 1, {})
        if lo.get("assessment_year") == y - 1 and hi.get("assessment_year") == y + 1:
            eq_lo = r.get("current_eq_prev_file_current", float("nan"))
            eq_hi = hi.get("current_eq_prev_file_current", float("nan"))
            if eq_lo < DISTINCT_MAX_EQUAL and eq_hi < DISTINCT_MAX_EQUAL:
                r["assessment_year"], r["status"] = y, "inferred_between"

    df = pd.DataFrame([info[y] for y in years])
    resolved = df["assessment_year"].notna()
    dup = df["assessment_year"].duplicated(keep=False) & resolved
    df.loc[dup, "status"] = "conflict: duplicate assessment year"
    df["valuation_date"] = df["assessment_year"].map(lambda a: f"{int(a)}-01-01" if pd.notna(a) else None)
    df["yr_assmt_agrees"] = df.apply(
        lambda r: None if pd.isna(r["yr_assmt_vote"]) or pd.isna(r["assessment_year"])
        else int(r["yr_assmt_vote"]) == int(r["assessment_year"]), axis=1)
    lead = ["file_year", "assessment_year", "valuation_date", "status", "chain_group"]
    return df[lead + [c for c in df.columns if c not in lead]]


def load_lookup() -> pd.DataFrame:
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(f"{LOOKUP_PATH} missing; run python -m avm.stage1_checks")
    return pd.read_csv(LOOKUP_PATH)


def file_year_for(assessment_year: int, lookup: pd.DataFrame | None = None,
                  allowed: tuple[str, ...] = ("confirmed",)) -> int:
    """MPROP file holding the roll for this assessment year. Raises unless
    exactly one file matches and its status is in `allowed`."""
    lk = load_lookup() if lookup is None else lookup
    hit = lk[lk["assessment_year"] == assessment_year]
    if len(hit) != 1:
        raise KeyError(f"assessment year {assessment_year}: {len(hit)} files in lookup")
    status = hit["status"].iloc[0]
    if status not in allowed:
        raise ValueError(f"assessment year {assessment_year}: status {status!r} not in {allowed}")
    return int(hit["file_year"].iloc[0])
