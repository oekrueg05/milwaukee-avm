"""Milwaukee addresses: the Master Address Index (MAI) loaders and the
parsing and matching used to look a parcel up by address.

The MAI has one row per house number per tax key (no owner names exist in
the file): HSE_NBR, SFX (letter suffix), DIR, STREET, STTYPE (abbreviated:
ST, AV, PL, ...), UNIT_NBR, ZIP_CODE (9 digits). A duplex usually has two
house numbers on one tax key, written as a range for display ("120-122").

`parse_address` and `match` are pure Python (no pandas), so the MCP server
can use them without the data stack; the loaders import pandas lazily.
"""

import re
from dataclasses import dataclass
from typing import Container, Iterable, NamedTuple

DIRECTIONS = {"N": "N", "NORTH": "N", "S": "S", "SOUTH": "S", "E": "E", "EAST": "E", "W": "W", "WEST": "W"}
STREET_TYPES = {
    "ST": "ST", "STREET": "ST", "AV": "AV", "AVE": "AV", "AVENUE": "AV", "PL": "PL", "PLACE": "PL",
    "DR": "DR", "DRIVE": "DR", "RD": "RD", "ROAD": "RD", "BL": "BL", "BLVD": "BL", "BOULEVARD": "BL",
    "CT": "CT", "COURT": "CT", "PK": "PK", "PKWY": "PK", "PARKWAY": "PK", "LA": "LA", "LN": "LA",
    "LANE": "LA", "CR": "CR", "CIR": "CR", "CIRCLE": "CR", "WA": "WA", "WAY": "WA", "TR": "TR",
    "TER": "TR", "TERRACE": "TR",
}
# Spellings people use -> the MAI's STREET value (checked against the MAI, 2026-10-21).
STREET_ALIASES = {
    "MARTIN LUTHER KING JR": "MARTIN L KING JR", "MARTIN LUTHER KING": "MARTIN L KING JR",
    "DR MARTIN LUTHER KING JR": "MARTIN L KING JR", "MLK": "MARTIN L KING JR", "MLK JR": "MARTIN L KING JR",
    "CESAR CHAVEZ": "CESAR E CHAVEZ", "SAINT PAUL": "ST PAUL", "SAINT CLAIR": "ST CLAIR",
}
CITY_WORDS = {"MILWAUKEE", "MKE", "WI", "WISCONSIN", "USA"}
UNIT_RE = re.compile(r"\b(?:UNIT|APT|APARTMENT|STE|SUITE)\b\s*#?\s*\w+|#\s*\w+|\b(?:UPPER|LOWER|UPR|LWR|REAR)\b")
NUMBER_RE = re.compile(r"^(\d{1,5})(?:-(\d{1,5}))?([A-Z]{1,3})?$")  # MAI suffixes: A, B, ..., ADJ, EXT, RA
ZIP_RE = re.compile(r"^\d{5}(?:-?\d{4})?$")
MAX_RANGE = 20  # wider ranges match only their two end numbers (a parcel spanning a block, or a typo)


class AddressError(ValueError):
    """Input that cannot be read as a Milwaukee street address."""


@dataclass(frozen=True)
class ParsedAddress:
    lo: int
    hi: int
    sfx: str          # "" if none
    dir: str | None   # None = not given (match any)
    street: str
    sttype: str | None
    zip5: str | None
    unit_ignored: bool

    @property
    def normalized(self) -> str:
        num = str(self.lo) if self.lo == self.hi else f"{self.lo}-{self.hi}"
        return " ".join(p for p in [num + self.sfx, self.dir, self.street, self.sttype] if p)

    @property
    def wide(self) -> bool:
        return self.hi - self.lo > MAX_RANGE


class MaiRow(NamedTuple):
    tk: str
    hn: int
    sfx: str
    dir: str
    street: str
    sttype: str
    zip5: str


def ordinal(n: int) -> str:
    suffix = "TH" if 10 <= n % 100 <= 13 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{n}{suffix}"


def _house_number(tok: str) -> tuple[int, int, str]:
    m = NUMBER_RE.match(tok)
    if not m:
        raise AddressError(f"expected a house number first, got '{tok}'")
    lo_s, hi_s, sfx = m.group(1), m.group(2), m.group(3) or ""
    lo = int(lo_s)
    hi = lo
    if hi_s:  # "2876-78": the short end borrows the leading digits of the long one
        hi = int(lo_s[: max(len(lo_s) - len(hi_s), 0)] + hi_s)
        if hi < lo:
            raise AddressError(f"house number range '{tok}' runs backwards")
    return lo, hi, sfx


def parse_address(text: str) -> ParsedAddress:
    """Read a free-text address: '2876-78 N. 40th St, Milwaukee WI 53210',
    '120 w north ave unit 2', '120A N Main'. Units are dropped (values are per
    parcel); direction and street type may be omitted (then any match)."""
    s = text.upper()
    unit_ignored = bool(UNIT_RE.search(s))
    s = UNIT_RE.sub(" ", s)
    s = re.sub(r"[.,;]", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)  # "2876 - 78" -> "2876-78"
    toks = s.split()
    zip5 = None
    while toks and (toks[-1] in CITY_WORDS or ZIP_RE.match(toks[-1])):
        t = toks.pop()
        if ZIP_RE.match(t):
            zip5 = t[:5]
    if len(toks) < 2:
        raise AddressError("expected a house number and a street name")
    lo, hi, sfx = _house_number(toks[0])
    rest = toks[1:]
    if rest[0] in ("1/2", "½") and not sfx:  # "124 1/2": kept as a suffix (the MAI has none, so no match)
        sfx, rest = "1/2", rest[1:]
        if not rest:
            raise AddressError("expected a street name")
    # Type first, then direction, each only if a street name remains: "W NORTH AVE" -> W / NORTH / AV,
    # "NORTH AVE" -> NORTH / AV. A lone one-letter direction keeps the type word as the street:
    # "N TERRACE" -> N / TERRACE (the MAI has streets named NORTH, COURT, PARK and TERRACE).
    t = None
    if len(rest) >= 2 and rest[-1] in STREET_TYPES and not (len(rest) == 2 and len(rest[0]) == 1
                                                            and rest[0] in DIRECTIONS):
        t, rest = STREET_TYPES[rest[-1]], rest[:-1]
    d = None
    if len(rest) >= 2 and rest[0] in DIRECTIONS:
        d, rest = DIRECTIONS[rest[0]], rest[1:]
    street = " ".join(rest)
    street = STREET_ALIASES.get(street, street)
    m = re.fullmatch(r"(\d{1,3})(?:ST|ND|RD|TH)?", street)
    if m:
        street = ordinal(int(m.group(1)))
    return ParsedAddress(lo, hi, sfx, d, street, t, zip5, unit_ignored)


def _on_street(p: ParsedAddress, rows: Iterable[MaiRow]) -> list[MaiRow]:
    return [r for r in rows if r.street == p.street and (p.dir is None or r.dir == p.dir)
            and (p.sttype is None or r.sttype == p.sttype)]


def _hits(p: ParsedAddress, rows: Iterable[MaiRow]) -> list[MaiRow]:
    """Tax keys whose MAI house numbers fall in the parsed range on that
    street, on the same side: odd and even numbers face each other, so
    "120-122" never matches 121 (unless the range's ends differ in parity).
    A range wider than MAX_RANGE matches only its two end numbers.
    A given suffix must match; with none given, plain and suffixed numbers
    all match (see `resolve`). A ZIP narrows the result only if it leaves a
    match."""
    same_side = (p.hi - p.lo) % 2 == 0
    if p.wide:
        hits = [r for r in _on_street(p, rows) if r.hn in (p.lo, p.hi)]
    else:
        hits = [r for r in _on_street(p, rows) if p.lo <= r.hn <= p.hi and (not same_side or (r.hn - p.lo) % 2 == 0)]
    if p.sfx:
        hits = [r for r in hits if r.sfx == p.sfx]
    if p.zip5 and any(r.zip5 == p.zip5 for r in hits):
        hits = [r for r in hits if r.zip5 == p.zip5]
    return hits


def match(p: ParsedAddress, rows: Iterable[MaiRow]) -> list[str]:
    return sorted({r.tk for r in _hits(p, rows)})


class Resolution(NamedTuple):
    status: str            # "ok", "multiple_matches" or "no_match"
    tax_keys: list[str]    # the parcel (ok) or the candidates (multiple_matches)
    note: str              # why several, or what was set aside
    nearby: list[str]      # no_match only: up to 5 addresses on that street


def resolve(p: ParsedAddress, rows: Iterable[MaiRow], valued: Container[str]) -> Resolution:
    """One parcel, a candidate list, or no match. Several tax keys resolve to
    one only if exactly one of them has a model value, and never when the
    same number exists with and without a letter suffix on different parcels
    (124 and 124A): then all are listed for the user to choose."""
    rows = list(rows)
    hits = _hits(p, rows)
    tks = sorted({r.tk for r in hits})
    if not tks:
        return Resolution("no_match", [], "", nearby_numbers(p, rows))
    if len(tks) == 1:
        return Resolution("ok", tks, "", [])
    if not p.sfx and any(r.sfx for r in hits) and any(not r.sfx for r in hits):
        return Resolution("multiple_matches", tks, "this number exists with and without a letter suffix", [])
    v = [t for t in tks if t in valued]
    if len(v) == 1:
        return Resolution("ok", v, f"{len(tks) - 1} other parcel(s) without a model value share this address", [])
    return Resolution("multiple_matches", tks, "several parcels share this address", [])


def nearby_numbers(p: ParsedAddress, rows: Iterable[MaiRow], k: int = 5) -> list[str]:
    """Up to k display addresses on the same street closest to the parsed
    house number, same side of the street first (for a no-match message).
    Empty if the street is unknown."""
    on = {(r.hn, r.sfx, r.dir, r.street, r.sttype) for r in _on_street(p, rows)}
    best = sorted(on, key=lambda r: ((r[0] - p.lo) % 2, abs(r[0] - p.lo), r[0], r[1]))[:k]  # same side first
    return [" ".join(x for x in [f"{hn}{sfx}", d, st, t] if x) for hn, sfx, d, st, t in best]


def _read_mai():
    import pandas as pd

    from avm.acquire import load_manifest
    from avm.config import DATA_DIR
    m = load_manifest().set_index("key")
    return pd.read_csv(DATA_DIR / m.loc["mai", "local_path"], dtype=str, keep_default_na=False, encoding="utf-8-sig")


def mai_rows(a=None):
    """The MAI as lookup rows (tk, hn, sfx, dir, street, sttype, zip5)."""
    import pandas as pd

    from avm.io import normalize_taxkey
    a = _read_mai() if a is None else a
    out = pd.DataFrame({"tk": normalize_taxkey(a.TAXKEY), "hn": pd.to_numeric(a.HSE_NBR, errors="raise"),
                        "sfx": a.SFX.str.strip(), "dir": a.DIR.str.strip(), "street": a.STREET.str.strip(),
                        "sttype": a.STTYPE.str.strip(), "zip5": a.ZIP_CODE.str.strip().str[:5]})
    return out.dropna(subset=["tk"]).astype({"hn": "int64"})


def parcel_addresses(r):
    """Display address per tax key for the value_parcel tool, from mai_rows():
    like display_addresses (the street with the lowest number; a range when a
    parcel has several numbers on it) but keeping letter suffixes, which
    display_addresses drops (124A shows as 124A, not 124)."""
    import pandas as pd
    r = r.assign(street_full=(r.dir + " " + r.street + " " + r.sttype).str.split().str.join(" "))
    r = r.sort_values(["tk", "hn", "sfx"], kind="stable")
    g = r.groupby(["tk", "street_full"], sort=False)
    lo, hi = g.head(1).set_index(["tk", "street_full"]), g.tail(1).set_index(["tk", "street_full"])
    out = pd.DataFrame({"lo": lo.hn, "lo_s": lo.hn.astype(str) + lo.sfx, "hi_s": hi.hn.astype(str) + hi.sfx,
                        "zip5": lo.zip5}).reset_index()
    out = out.sort_values(["tk", "lo"], kind="stable").drop_duplicates("tk")
    num = out.lo_s.where(out.lo_s == out.hi_s, out.lo_s + "-" + out.hi_s)
    return pd.DataFrame({"tk": out.tk.to_numpy(), "address": (num + " " + out.street_full).to_numpy(),
                         "zip5": out.zip5.to_numpy()})


def display_addresses(a=None):
    """One display address and ZIP per tax key (moved from Stage 9, same
    output). Duplexes with two numbers show a range."""
    import numpy as np
    import pandas as pd

    from avm.io import normalize_taxkey
    a = _read_mai() if a is None else a.copy()
    a["tk"] = normalize_taxkey(a.TAXKEY)
    a["hn"] = pd.to_numeric(a.HSE_NBR, errors="coerce")  # display only, not a model input
    a = a.dropna(subset=["tk", "hn"])
    street = (a.DIR.str.strip() + " " + a.STREET.str.strip() + " " + a.STTYPE.str.strip()).str.split().str.join(" ")
    a["street"] = street
    g = a.groupby(["tk", "street"]).agg(lo=("hn", "min"), hi=("hn", "max"), zip=("ZIP_CODE", "first")).reset_index()
    g = g.sort_values(["tk", "lo"]).drop_duplicates("tk")
    num = np.where(g.lo == g.hi, g.lo.astype(int).astype(str), g.lo.astype(int).astype(str) + "-" +
                   g.hi.astype(int).astype(str))
    return pd.DataFrame({"tk": g.tk, "address": num + " " + g.street, "zip5": g.zip.str[:5]})
