"""value_parcel (avm.lookup + avm.mcp_server) on a synthetic serving
database: every status, hash refusal, the response key whitelist, and calls
through the MCP SDK's in-memory client. Addresses are joined at runtime and
tax keys built from numbers, so this file passes the privacy check."""

import json
import re
import sqlite3

import anyio
import pandas as pd
import pytest
from mcp import Client

from avm.ci_checks import OWNER_COL_RE
from avm.lookup import EXCLUDED_NOTE, LABEL, LAST_SALE_SOURCE, Lookup, round_k, sha256_file
from avm.mcp_server import DESCRIPTION, build_server
from avm.serving_db import TABLE_COLS, write_db


def A(*parts):
    return " ".join(parts)


def K(i):
    return str(10 ** 8 + i).zfill(10)  # a 10-digit tax key with a leading zero, built at runtime


# (tk, house number, suffix) on N MAIN ST; one parcel per status
MAI = [(K(1), 101, ""), (K(2), 120, ""), (K(2), 122, ""), (K(3), 130, ""), (K(4), 130, "A"), (K(5), 140, ""),
       (K(6), 140, ""), (K(7), 150, ""), (K(8), 160, ""), (K(9), 170, ""), (K(10), 180, "")]
VALUED = [1, 2, 3, 4, 5, 6, 9]          # in parcels
STATUS = {**{i: ("valued", 8810, None) for i in VALUED}, 2: ("valued", 8820, None), 7: ("not_covered", 8811, None),
          8: ("no_estimate", 8810, "missing in roll 2025: finished sq ft"), 10: ("valued", 8810, None)}  # 10: valued, not in db
ALLOWED_KEYS = {
    "status", "label", "message", "hint", "query", "input", "normalized", "matched_by", "parcel", "tax_key",
    "address", "zip", "property_type", "land_use", "neighborhood", "estimate", "valuation_date", "value",
    "range_80", "low", "high", "rounding", "model", "predictions_sha256", "drivers", "driver", "effect_pct",
    "prior_sale", "sale_date", "sale_price", "drivers_note", "comps", "search_tier", "radius_mi", "window_months",
    "sales", "rank", "price_at_dec_2025_level", "distance_mi", "finished_sqft", "bedrooms", "baths", "year_built",
    "note", "notes", "provenance", "features_roll", "built_at_utc", "build_commit", "nearby_addresses", "reason",
    "candidates", "land_use_description", "has_model_value", "subject_details", "lot_sqft", "source",
    "subject_last_sale", "latest_clean_sale"}
FORBIDDEN_KEY = re.compile(r"(?i)owner|assess|ratio|verdict|over|under|mail")


def _tables():
    addr = lambda i: A(str([h for t, h, _ in MAI if t == K(i)][0]), "N", "MAIN", "ST")  # noqa: E731
    parcels = pd.DataFrame([{
        "tk": K(i), "address": A("120-122", "N", "MAIN", "ST") if i == 2 else addr(i), "zip5": "53201",
        "land_use": STATUS[i][1], "property_type": "duplex" if i == 2 else "single-family", "neighborhood": "1",
        "value": 312400 + i, "pi80_low": 250500, "pi80_high": 380000,
        "comp_radius_mi": None if i == 9 else 0.5, "comp_window_months": None if i == 9 else 12,
        "prior_sale_date": "2023-05-02" if i == 1 else None, "prior_sale_price": 280000 if i == 1 else None,
        "bldg_area": 1400 + i, "bedrooms": 3, "full_baths": 1, "half_baths": 1 if i == 1 else 0, "year_built": 1925,
        "lot_area": 4800, "details_note": "lot size from roll 2024" if i == 2 else None,
        # 1: last sale = its prior sale; 2: an older sale plus a later flip resale removed; 3: only a removed sale
        "last_sale_date": {1: "2023-05-02", 2: "2017-04"}.get(i), "last_sale_price": {1: 280000, 2: 95000}.get(i),
        "removed_sale_date": {2: "2025-08-14", 3: "2024-02-09"}.get(i),
        "removed_sale_price": {2: 239900, 3: 5000}.get(i),
        "removed_sale_reason": {2: "resale < 12 months (flip)", 3: "price < $10,000"}.get(i)}
        for i in VALUED], columns=TABLE_COLS["parcels"])
    drivers = pd.DataFrame([{"tk": K(i), "rank": r, "key": k, "label": k, "effect_log": e}
                            for i in VALUED for r, (k, e) in enumerate(
                                [("location", -0.2), ("this home's prior sale" if i == 1 else "finished sq ft", 0.1),
                                 ("nearby sale prices", 0.05)], 1)], columns=TABLE_COLS["drivers"])
    comps = pd.DataFrame([{"tk": K(i), "rank": r, "comp_tk": K(50 + r), "address": A(str(200 + 2 * r), "N", "MAIN", "ST"),
                           "sale_date": "2025-06-01", "sale_price": 300000, "adj_price": 301500.0, "dist_m": 400.0,
                           "bldg_area": 1500, "bedrooms": 3, "full_baths": 1, "half_baths": 1, "year_built": 1950}
                          for i in VALUED if i != 9 for r in (1, 2, 3)], columns=TABLE_COLS["comps"])
    status = pd.DataFrame([{"tk": K(i), "land_use": lu, "status": st, "reason": why} for i, (st, lu, why) in STATUS.items()],
                          columns=TABLE_COLS["parcel_status"])
    codes = pd.DataFrame({"code": [8810, 8811, 8820], "description": ["SINGLE-PRIVATE HOUSEHOLDS", "CONDO",
                                                                      "TWO FAMILY RESIDENTIAL"]})
    mai = pd.DataFrame([{"tk": t, "hn": h, "sfx": s, "dir": "N", "street": "MAIN", "sttype": "ST", "zip5": "53201"}
                        for t, h, s in MAI], columns=TABLE_COLS["mai"])
    return {"parcels": parcels, "drivers": drivers, "comps": comps, "parcel_status": status,
            "land_use_codes": codes, "mai": mai}


@pytest.fixture
def env(tmp_path):
    pred = tmp_path / "predictions_2026.parquet"
    pred.write_bytes(b"synthetic stand-in for the hashed prediction file")
    meta_path = tmp_path / "prospective_meta.json"
    meta_path.write_text(json.dumps({"predictions_sha256": sha256_file(pred)}))
    db = tmp_path / "value_parcel_2026.sqlite"
    meta = {"label": LABEL, "valuation_date": "2026-01-01", "features_roll": 2025, "predictions_sha256": sha256_file(pred),
            "built_at_utc": "2026-10-22T00:00:00+00:00", "build_commit": "test", "parcels_in_db": len(VALUED)}
    write_db(db, meta, _tables())
    return {"db": db, "pred": pred, "meta": meta_path, "meta_dict": meta}


@pytest.fixture
def lk(env):
    return Lookup(env["db"], env["pred"], env["meta"])


def _keys(x):
    if isinstance(x, dict):
        return set(x) | set().union(*(_keys(v) for v in x.values()))
    if isinstance(x, list):
        return set().union(*(_keys(v) for v in x)) if x else set()
    return set()


def test_ok_single_family_typed_address(lk):
    r = lk.value_parcel(address=A("101", "north", "main", "street,", "Milwaukee,", "WI", "53201"))
    assert r["status"] == "ok" and r["label"] == LABEL and r["parcel"]["tax_key"] == K(1)
    assert r["estimate"]["value"] == 312000 and r["estimate"]["range_80"] == {"low": 251000, "high": 380000}
    assert [d["driver"] for d in r["drivers"]] == ["location", "this home's prior sale", "nearby sale prices"]
    assert r["drivers"][1]["prior_sale"] == {"sale_date": "2023-05-02", "sale_price": 280000}
    assert "prior_sale" not in r["drivers"][0] and r["drivers"][0]["effect_pct"] == -18.1
    assert r["comps"]["search_tier"] == {"radius_mi": 0.5, "window_months": 12} and len(r["comps"]["sales"]) == 3
    c = r["comps"]["sales"][0]
    assert c["price_at_dec_2025_level"] == 302000 and c["baths"] == "1.1" and c["distance_mi"] == 0.25
    assert r["subject_details"] == {"finished_sqft": 1401, "bedrooms": 3, "baths": "1.1", "year_built": 1925,
                                    "lot_sqft": 4800, "source": "roll 2025 (Master Property File)"}
    assert r["subject_last_sale"]["sale_date"] == "2023-05-02" and r["subject_last_sale"]["sale_price"] == 280000
    assert "arm's-length" in r["subject_last_sale"]["source"] and "notes" not in r


def test_subject_last_sale_removed_later_sale_and_absent(lk):
    r = lk.value_parcel(tax_key=K(2))  # latest sale removed by cleaning: shown, labelled, with the latest clean sale
    assert r["subject_last_sale"] == {
        "sale_date": "2025-08-14", "sale_price": 239900,
        "note": "excluded from the model's training data: resale < 12 months (flip); shown for context, "
                "not used in the estimate",
        "latest_clean_sale": {"sale_date": "2017-04", "sale_price": 95000, "source": LAST_SALE_SOURCE}}
    assert r["subject_details"]["note"] == "lot size from roll 2024" and r["subject_details"]["baths"] == "1.0"
    r = lk.value_parcel(tax_key=K(3))  # only a removed sale: labelled, no clean sale to add
    assert r["subject_last_sale"] == {"sale_date": "2024-02-09", "sale_price": 5000, "note": EXCLUDED_NOTE.format(
        reason="price < $10,000")} and "notes" not in r
    r = lk.value_parcel(tax_key=K(9))  # no sales at all
    assert "subject_last_sale" not in r and "notes" not in r and r["subject_details"]["finished_sqft"] == 1409


def test_ok_duplex_short_range_and_unit_note(lk):
    r = lk.value_parcel(address=A("120-22", "N", "Main", "St"))
    assert r["status"] == "ok" and r["parcel"]["property_type"] == "duplex" and r["parcel"]["tax_key"] == K(2)
    r = lk.value_parcel(address=A("122", "N", "Main", "St", "Upper"))
    assert r["status"] == "ok" and r["notes"] == ["unit ignored: values are per parcel"]


def test_ok_by_tax_key_without_comps(lk):
    r = lk.value_parcel(tax_key=K(9).lstrip("0"))  # leading zeros optional: padded back to 10 digits
    assert r["status"] == "ok" and r["parcel"]["tax_key"] == K(9)
    assert r["comps"]["sales"] == [] and r["comps"]["search_tier"] is None and "fewer than 3" in r["comps"]["note"]


def test_multiple_matches_suffix_variants_and_shared(lk):
    r = lk.value_parcel(address=A("130", "N", "Main", "St"))
    assert r["status"] == "multiple_matches" and "suffix" in r["reason"]
    assert [c["tax_key"] for c in r["candidates"]] == [K(3), K(4)]
    assert {c["address"] for c in r["candidates"]} == {A("130", "N", "MAIN", "ST"), A("130A", "N", "MAIN", "ST")}
    r = lk.value_parcel(address=A("140", "N", "Main", "St"))
    assert r["status"] == "multiple_matches" and r["reason"] == "several parcels share this address"
    assert all(c["has_model_value"] for c in r["candidates"])
    assert not any({"value", "estimate", "range_80"} & set(c) for c in r["candidates"])  # no values in a list


def test_no_match_with_at_most_five_nearby(lk):
    r = lk.value_parcel(address=A("190", "N", "Main", "St"))
    assert r["status"] == "no_match" and 0 < len(r["nearby_addresses"]) <= 5
    assert r["nearby_addresses"][0] == A("180", "N", "MAIN", "ST")
    r = lk.value_parcel(address=A("190", "N", "Elm", "St"))
    assert r["status"] == "no_match" and r["nearby_addresses"] == []


def test_not_covered_no_estimate_not_found(lk):
    r = lk.value_parcel(address=A("150", "N", "Main", "St"))
    assert r["status"] == "not_covered" and r["land_use"] == 8811 and r["land_use_description"] == "CONDO"
    assert "single-family (8810) and duplex (8820)" in r["message"] and "estimate" not in r
    r = lk.value_parcel(address=A("160", "N", "Main", "St"))
    assert r["status"] == "no_estimate" and "finished sq ft" in r["message"] and "estimate" not in r
    r = lk.value_parcel(tax_key=K(10))  # valued, but not in this (sample) database
    assert r["status"] == "no_estimate" and "sample" in r["message"]
    r = lk.value_parcel(tax_key=K(99))
    assert r["status"] == "not_found"


@pytest.mark.parametrize("kwargs", [{}, {"address": A("101", "N", "Main", "St"), "tax_key": K(1)}, {"tax_key": "12AB"},
                                    {"tax_key": "123"}, {"address": "Main St"}, {"address": "  "}])
def test_invalid_input(lk, kwargs):
    r = lk.value_parcel(**kwargs)
    assert r["status"] == "invalid_input" and r["label"] == LABEL and r["message"]


def test_rounding_halves_up():
    assert [round_k(v) for v in (312499, 312500, 999, 1500, None)] == [312000, 313000, 1000, 2000, None]


def test_hash_refusal(env):
    Lookup(env["db"], env["pred"], env["meta"])  # untouched: starts
    good = env["pred"].read_bytes()
    env["pred"].write_bytes(good + b"x")  # tampered predictions
    with pytest.raises(ValueError, match="refusing to serve a modified file"):
        Lookup(env["db"], env["pred"], env["meta"])
    env["pred"].write_bytes(good)
    with open(env["db"], "ab") as f:  # tampered database
        f.write(b"\0")
    with pytest.raises(ValueError, match="sidecar"):
        Lookup(env["db"], env["pred"], env["meta"])
    with pytest.raises(FileNotFoundError):
        Lookup(env["db"].with_name("missing.sqlite"), env["pred"], env["meta"])


def test_database_from_another_prediction_file_refused(env, tmp_path):
    other = tmp_path / "other.sqlite"
    write_db(other, {**env["meta_dict"], "predictions_sha256": "0" * 64}, _tables())
    with pytest.raises(ValueError, match="different prediction file"):
        Lookup(other, env["pred"], env["meta"])


def test_response_keys_whitelisted_and_schema_has_no_owner_fields(lk, env):
    calls = [{"address": A("101", "N", "Main", "St")}, {"address": A("122", "N", "Main", "St", "Upper")},
             {"tax_key": K(9)}, {"tax_key": K(3)}, {"address": A("130", "N", "Main", "St")}, {"address": A("190", "N", "Main", "St")},
             {"address": A("150", "N", "Main", "St")}, {"address": A("160", "N", "Main", "St")}, {"tax_key": K(99)}, {}]
    responses = [lk.value_parcel(**c) for c in calls]
    assert {r["status"] for r in responses} == {"ok", "multiple_matches", "no_match", "not_covered", "no_estimate",
                                                "not_found", "invalid_input"}
    keys = set().union(*(_keys(r) for r in responses))
    assert keys <= ALLOWED_KEYS, keys - ALLOWED_KEYS
    assert not [k for k in keys if FORBIDDEN_KEY.search(k)]
    con = sqlite3.connect(env["db"])
    cols = [c[1] for t in TABLE_COLS for c in con.execute(f"PRAGMA table_info({t})")]
    con.close()
    assert not [c for c in cols if OWNER_COL_RE.search(c) or FORBIDDEN_KEY.search(c)]


def test_through_mcp_in_memory_client(lk):
    async def run():
        async with Client(build_server(lk)) as c:
            tools = (await c.list_tools()).tools
            ok = await c.call_tool("value_parcel", {"address": A("101", "N", "Main", "St")})
            nc = await c.call_tool("value_parcel", {"tax_key": K(7)})
            return tools, ok, nc
    tools, ok, nc = anyio.run(run)
    assert [t.name for t in tools] == ["value_parcel"] and tools[0].annotations.read_only_hint is True
    d = tools[0].description
    for phrase in ["City of Milwaukee", "single-family and duplex homes only", "January 1, 2026",
                   "frozen model fit only on data through 2025", "80% range", "not an appraisal or an assessment",
                   "no over- or under-assessment verdicts", "street address", "tax key"]:
        assert phrase in d and phrase in DESCRIPTION, phrase
    assert set(tools[0].input_schema["properties"]) == {"address", "tax_key"}
    assert not ok.is_error and ok.structured_content == lk.value_parcel(address=A("101", "N", "Main", "St"))
    assert json.loads(ok.content[0].text)["estimate"]["value"] == 312000
    assert nc.structured_content["status"] == "not_covered"
