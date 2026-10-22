"""value_parcel: read-only lookups against the serving database built by
`python -m avm.serving build`. Standard library only (no pandas, no model),
so the MCP server starts fast and never trains or writes anything.

Startup checks (Lookup.__init__): the prediction file's sha256 must equal
prospective_meta.json and the database's meta, and the database must equal
its .sha256 sidecar. Any mismatch raises and the server does not start.

Responses carry a fixed label, no owner names (the database has none), and
no assessed values, ratios or over/under verdicts. Value and 80% range are
rounded to the nearest $1,000; exact values stay in the database. Subject
details (roll 2025 characteristics) and the subject's last sale are facts
about the home and are served as recorded. subject_last_sale is the latest
sale on record; when cleaning removed it, it carries EXCLUDED_NOTE with the
reason, and the latest arm's-length sale follows as latest_clean_sale.
"""

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

from avm.address import AddressError, MaiRow, parse_address, resolve
from avm.config import DATA_DIR, REPORTS_DIR

YEAR = 2026
LABEL = "model estimate, not an appraisal or assessment"
PRED_PATH = REPORTS_DIR / "prospective" / f"predictions_{YEAR}.parquet"
META_PATH = REPORTS_DIR / "prospective" / "prospective_meta.json"
TAX_KEY_RE = re.compile(r"\d{7,10}")
COVERED = "single-family (8810) and duplex (8820) homes"
OWN_PRIOR_SALE = "this home's prior sale"  # driver key (avm.serving.DRIVER_GROUPS)
LAST_SALE_SOURCE = "most recent arm's-length sale in the city's sales files, 2014 to 2025"
EXCLUDED_NOTE = "excluded from the model's training data: {reason}; shown for context, not used in the estimate"


def db_path(sample: bool = False) -> Path:
    return DATA_DIR / "serving" / f"value_parcel_{YEAR}{'_sample' if sample else ''}.sqlite"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, pred_path: Path = PRED_PATH, meta_path: Path = META_PATH) -> dict:
    """The startup checks; returns the database meta."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with python -m avm.serving build")
    want = json.loads(Path(meta_path).read_text())["predictions_sha256"]
    got = sha256_file(pred_path)
    if got != want:
        raise ValueError(f"{Path(pred_path).name} sha256 {got} != recorded {want}; refusing to serve a modified file")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or sha256_file(path) != side.read_text().strip():
        raise ValueError(f"{path.name} does not match its .sha256 sidecar; rebuild with python -m avm.serving build")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
    finally:
        con.close()
    if meta.get("predictions_sha256") != got:
        raise ValueError("serving database was built from a different prediction file")
    return meta


def round_k(x) -> int | None:
    """Nearest $1,000 (halves up)."""
    return None if x is None else int(math.floor(x / 1000 + 0.5) * 1000)


def _baths(full, half) -> str | None:
    return None if full is None else f"{int(full)}.{int(half or 0)}"


class Lookup:
    def __init__(self, path: Path | None = None, pred_path: Path = PRED_PATH, meta_path: Path = META_PATH):
        self.path = path or db_path()
        self.meta = verify(self.path, pred_path, meta_path)
        self.con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.con.row_factory = sqlite3.Row

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        return self.con.execute(sql, args).fetchall()

    def _status(self, tk: str) -> sqlite3.Row | None:
        r = self._q("SELECT s.tk, s.land_use, s.status, s.reason, c.description FROM parcel_status s "
                    "LEFT JOIN land_use_codes c ON c.code = s.land_use WHERE s.tk = ?", (tk,))
        return r[0] if r else None

    def _address_of(self, tk: str) -> str | None:
        r = self._q("SELECT hn, sfx, dir, street, sttype FROM mai WHERE tk = ? ORDER BY hn, sfx LIMIT 1", (tk,))
        return " ".join(x for x in [f"{r[0]['hn']}{r[0]['sfx']}", r[0]["dir"], r[0]["street"], r[0]["sttype"]] if x) \
            if r else None

    def value_parcel(self, address: str | None = None, tax_key: str | None = None) -> dict:
        if (address is None) == (tax_key is None) or (address is not None and not address.strip()):
            return {"status": "invalid_input", "label": LABEL,
                    "message": "Give exactly one of: a Milwaukee street address, or a 10-digit tax key."}
        notes = []
        if tax_key is not None:
            t = re.sub(r"[\s-]", "", tax_key)
            if not TAX_KEY_RE.fullmatch(t):
                return {"status": "invalid_input", "label": LABEL,
                        "message": f"'{tax_key}' is not a tax key (7 to 10 digits)."}
            tk, query = t.zfill(10), {"input": tax_key, "normalized": t.zfill(10), "matched_by": "tax_key"}
        else:
            try:
                p = parse_address(address)
            except AddressError as e:
                return {"status": "invalid_input", "label": LABEL, "message": f"Could not read the address: {e}."}
            query = {"input": address, "normalized": p.normalized, "matched_by": "address"}
            rows = [MaiRow(*r) for r in self.con.execute(
                "SELECT tk, hn, sfx, dir, street, sttype, zip5 FROM mai WHERE street = ?", (p.street,))]
            valued = {r["tk"] for r in self._q(
                "SELECT tk FROM parcel_status WHERE status = 'valued' AND tk IN (SELECT tk FROM mai WHERE street = ?)",
                (p.street,))}
            res = resolve(p, rows, valued)
            if res.status == "no_match":
                msg = f"No address in the city's Master Address Index matches '{p.normalized}'."
                return {"status": "no_match", "label": LABEL, "query": query, "message": msg,
                        "nearby_addresses": res.nearby,
                        "hint": "Check the number and street, or look the parcel up by tax key."}
            if res.status == "multiple_matches":
                cands = []
                for t in res.tax_keys:
                    st = self._status(t)
                    cands.append({"tax_key": t, "address": self._address_of(t),
                                  "land_use": None if st is None else st["land_use"],
                                  "land_use_description": None if st is None else st["description"],
                                  "has_model_value": bool(st and st["status"] == "valued")})
                return {"status": "multiple_matches", "label": LABEL, "query": query, "reason": res.note,
                        "candidates": cands, "hint": "Call again with the tax key of the parcel you mean."}
            tk = res.tax_keys[0]
            if res.note:
                notes.append(res.note)
            if p.unit_ignored:
                notes.append("unit ignored: values are per parcel")
        st = self._status(tk)
        if st is None:
            return {"status": "not_found", "label": LABEL, "query": query,
                    "message": f"Tax key {tk} is not in the {YEAR - 1} assessment roll."}
        if st["status"] == "not_covered":
            return {"status": "not_covered", "label": LABEL, "query": query, "tax_key": tk,
                    "land_use": st["land_use"], "land_use_description": st["description"],
                    "message": f"This parcel's land use is {st['land_use']} ({st['description']}); "
                               f"the model covers {COVERED} only."}
        if st["status"] == "no_estimate":
            return {"status": "no_estimate", "label": LABEL, "query": query, "tax_key": tk,
                    "message": f"Single-family or duplex in the {YEAR - 1} roll, but the model has no value for it "
                               f"({st['reason'] or 'missing a required model input'})."}
        par = self._q("SELECT * FROM parcels WHERE tk = ?", (tk,))
        if not par:  # valued, but not in a sample build
            return {"status": "no_estimate", "label": LABEL, "query": query, "tax_key": tk,
                    "message": "This parcel has a model value, but it is not in this sample database."}
        par = par[0]
        drivers = []
        for d in self._q("SELECT key, label, effect_log FROM drivers WHERE tk = ? ORDER BY rank", (tk,)):
            e = {"driver": d["label"], "effect_pct": round((math.exp(d["effect_log"]) - 1) * 100, 1)}
            if d["key"] == OWN_PRIOR_SALE and par["prior_sale_date"] is not None:
                e["prior_sale"] = {"sale_date": par["prior_sale_date"], "sale_price": par["prior_sale_price"]}
            drivers.append(e)
        details = {"finished_sqft": par["bldg_area"], "bedrooms": par["bedrooms"],
                   "baths": _baths(par["full_baths"], par["half_baths"]), "year_built": par["year_built"],
                   "lot_sqft": par["lot_area"], "source": f"roll {self.meta['features_roll']} (Master Property File)"}
        if par["details_note"]:
            details["note"] = par["details_note"]
        clean = None if par["last_sale_date"] is None else {
            "sale_date": par["last_sale_date"], "sale_price": par["last_sale_price"], "source": LAST_SALE_SOURCE}
        if par["removed_sale_date"] is None:
            last_sale = clean
        else:
            last_sale = {"sale_date": par["removed_sale_date"], "sale_price": par["removed_sale_price"],
                         "note": EXCLUDED_NOTE.format(reason=par["removed_sale_reason"]),
                         **({} if clean is None else {"latest_clean_sale": clean})}
        comps = [{"rank": c["rank"], "address": c["address"], "sale_date": c["sale_date"],
                  "sale_price": c["sale_price"], "price_at_dec_2025_level": round_k(c["adj_price"]),
                  "distance_mi": round(c["dist_m"] / 1609.344, 2), "finished_sqft": c["bldg_area"],
                  "bedrooms": c["bedrooms"], "baths": _baths(c["full_baths"], c["half_baths"]),
                  "year_built": c["year_built"]}
                 for c in self._q("SELECT * FROM comps WHERE tk = ? ORDER BY rank", (tk,))]
        out = {
            "status": "ok", "label": LABEL, "query": query,
            "parcel": {"tax_key": tk, "address": par["address"], "zip": par["zip5"],
                       "property_type": par["property_type"], "land_use": par["land_use"],
                       "neighborhood": par["neighborhood"]},
            "subject_details": details,
            **({} if last_sale is None else {"subject_last_sale": last_sale}),
            "estimate": {"valuation_date": self.meta["valuation_date"], "value": round_k(par["value"]),
                         "range_80": {"low": round_k(par["pi80_low"]), "high": round_k(par["pi80_high"])},
                         "rounding": "nearest $1,000",
                         "model": "frozen primary LightGBM (reports/frozen/model_spec.json), fit on sales through 2025",
                         "predictions_sha256": self.meta["predictions_sha256"]},
            "drivers": drivers,
            "drivers_note": "largest SHAP effects on this estimate, relative to the model's average home "
                            "(location and nearby sale prices are grouped features)",
            "comps": {"search_tier": None if par["comp_radius_mi"] is None else
                      {"radius_mi": par["comp_radius_mi"], "window_months": par["comp_window_months"]},
                      "sales": comps,
                      **({} if comps else {"note": "fewer than 3 eligible sales in every search tier"})},
            "provenance": {"features_roll": self.meta["features_roll"], "built_at_utc": self.meta["built_at_utc"],
                           "build_commit": self.meta["build_commit"]},
        }
        if notes:
            out["notes"] = notes
        return out
