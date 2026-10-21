"""Parcel centroids, in meters (UTM 16N, EPSG:32616), keyed by tax key.

Source: the 2025 parcel layer, with the current layer filling tax keys
missing from 2025. Parcels change little year to year; centroids are used
for proximity (multi-parcel sale grouping now, geospatial features in
Stage 6), not for anything that depends on parcel boundaries at the sale
date. Cached in interim/ keyed on the source files' sha256.
"""

import json

import numpy as np
import pandas as pd

from avm.acquire import load_manifest
from avm.config import INTERIM_DIR
from avm.io import normalize_taxkey, read_parcels

CACHE = INTERIM_DIR / "parcel_centroids.parquet"
CACHE_META = INTERIM_DIR / "parcel_centroids_meta.json"
SOURCES = ["parcels/2025", "parcels/current"]  # priority order
METRIC_CRS = "EPSG:32616"


def _build() -> pd.DataFrame:
    parts = []
    for key in SOURCES:
        g = read_parcels(key, read_geometry=True).to_crs(METRIC_CRS)
        tk = normalize_taxkey(g["Taxkey"] if "Taxkey" in g else g["FK_Tax"])
        c = g.geometry.centroid
        df = pd.DataFrame({"tk": tk, "x_m": c.x, "y_m": c.y}).dropna(subset=["tk"])
        # A tax key with several polygons gets the mean of their centroids.
        df = df.groupby("tk", as_index=False)[["x_m", "y_m"]].mean()
        df["source"] = key
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    # Empty or invalid polygons give NaN centroids: drop them so a later
    # source can fill the tax key, and every consumer sees "no location".
    out = out[np.isfinite(out.x_m) & np.isfinite(out.y_m)]
    out = out.drop_duplicates("tk", keep="first")
    return out.reset_index(drop=True)


def parcel_centroids() -> pd.DataFrame:
    m = load_manifest().set_index("key")
    shas = {k: m.loc[k, "sha256"] for k in SOURCES}
    if CACHE.exists() and CACHE_META.exists() and json.loads(CACHE_META.read_text()) == shas:
        return pd.read_parquet(CACHE)
    out = _build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    CACHE_META.write_text(json.dumps(shas))
    return out


def max_pairwise_distance_m(xy: pd.DataFrame) -> float:
    """Largest distance between any two points; NaN if any point is missing."""
    if xy[["x_m", "y_m"]].isna().any().any():
        return float("nan")
    a = xy[["x_m", "y_m"]].to_numpy()
    d = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1) ** 0.5
    return float(d.max())
