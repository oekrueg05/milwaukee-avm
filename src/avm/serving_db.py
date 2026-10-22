"""Serving database schema and writer for the value_parcel tool, kept apart
from avm.serving (which imports the model stack) so tests can build
synthetic databases with pandas and sqlite3 only."""

import json
import sqlite3

import numpy as np
import pandas as pd

from avm.lookup import sha256_file

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE parcels (tk TEXT PRIMARY KEY, address TEXT, zip5 TEXT, land_use INTEGER, property_type TEXT,
    neighborhood TEXT, value INTEGER NOT NULL, pi80_low INTEGER NOT NULL, pi80_high INTEGER NOT NULL,
    comp_radius_mi REAL, comp_window_months INTEGER, prior_sale_date TEXT, prior_sale_price INTEGER,
    bldg_area INTEGER, bedrooms INTEGER, full_baths INTEGER, half_baths INTEGER, year_built INTEGER, lot_area INTEGER,
    details_note TEXT, last_sale_date TEXT, last_sale_price INTEGER, removed_sale_date TEXT, removed_sale_price INTEGER,
    removed_sale_reason TEXT);
CREATE TABLE drivers (tk TEXT, rank INTEGER, key TEXT, label TEXT, effect_log REAL, PRIMARY KEY (tk, rank));
CREATE TABLE comps (tk TEXT, rank INTEGER, comp_tk TEXT, address TEXT, sale_date TEXT, sale_price INTEGER,
    adj_price REAL, dist_m REAL, bldg_area INTEGER, bedrooms INTEGER, full_baths INTEGER, half_baths INTEGER,
    year_built INTEGER, PRIMARY KEY (tk, rank));
CREATE TABLE parcel_status (tk TEXT PRIMARY KEY, land_use INTEGER, status TEXT NOT NULL, reason TEXT);
CREATE TABLE land_use_codes (code INTEGER PRIMARY KEY, description TEXT);
CREATE TABLE mai (tk TEXT, hn INTEGER, sfx TEXT, dir TEXT, street TEXT, sttype TEXT, zip5 TEXT);
CREATE INDEX mai_street ON mai (street);
CREATE INDEX mai_tk ON mai (tk);
"""
TABLE_COLS = {"parcels": ["tk", "address", "zip5", "land_use", "property_type", "neighborhood", "value", "pi80_low",
                          "pi80_high", "comp_radius_mi", "comp_window_months", "prior_sale_date", "prior_sale_price",
                          "bldg_area", "bedrooms", "full_baths", "half_baths", "year_built", "lot_area", "details_note",
                          "last_sale_date", "last_sale_price", "removed_sale_date", "removed_sale_price",
                          "removed_sale_reason"],
              "drivers": ["tk", "rank", "key", "label", "effect_log"],
              "comps": ["tk", "rank", "comp_tk", "address", "sale_date", "sale_price", "adj_price", "dist_m",
                        "bldg_area", "bedrooms", "full_baths", "half_baths", "year_built"],
              "parcel_status": ["tk", "land_use", "status", "reason"],
              "land_use_codes": ["code", "description"],
              "mai": ["tk", "hn", "sfx", "dir", "street", "sttype", "zip5"]}


def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    return v.item() if isinstance(v, np.generic) else v


def write_db(path, meta: dict, tables: dict[str, pd.DataFrame]) -> None:
    """Write the serving database (also used by tests with synthetic frames)
    and its .sha256 sidecar. Written to a temporary file and moved into place
    only when complete, so a failed build never leaves a partial database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    for name, cols in TABLE_COLS.items():
        t = tables[name]
        if list(t.columns) != cols:
            raise ValueError(f"{name}: columns {list(t.columns)} != {cols}")
        con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(cols))})",
                        ([_py(v) for v in row] for row in t.itertuples(index=False, name=None)))
    con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()
    con.execute("VACUUM")
    con.close()
    path.unlink(missing_ok=True)
    tmp.rename(path)
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")
