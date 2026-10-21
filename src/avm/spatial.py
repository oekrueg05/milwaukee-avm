"""Geospatial features for parcels (Stage 6). Meters, UTM 16N.

Static (geometry that does not change in our window):
  x_m, y_m          parcel centroid
  d_downtown_m      to Milwaukee City Hall (43.04164, -87.90958)
  d_lake_m          to the Lake Michigan shoreline (TIGER 2023 coastline)
  d_major_road_m    to the nearest primary/secondary road (TIGER 2023)
  d_transit_m       to the nearest MCTS stop (current GTFS: stops treated as
                    static, a limitation)
Per roll year (from roll Y-1, so time-consistent with the other features):
  d_park_m          to the nearest park/parkway/playground parcel (land use
                    8870/8871/8860; 2025 parcel polygons); 0 inside one
  n_city_200m       parcels within 200 m in a municipal exemption code
                    (480-571: incl. 530 tax-deed foreclosures)
  n_vacant_200m     parcels within 200 m with vacant land use (8880, 8885)

PROXY NOTE (rule 7): n_city_200m and n_vacant_200m measure disinvestment and
city foreclosure. In Milwaukee these are strongly correlated with
neighborhood racial composition; they are location features allowed by rule
7 but flagged, and their effect is audited in Stage 8.
"""

import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from avm.acquire import load_manifest
from avm.config import DATA_DIR
from avm.geo import METRIC_CRS, parcel_centroids
from avm.io import read_parcels, normalize_taxkey

CITY_HALL_LONLAT = (-87.90958, 43.04164)
MUNICIPAL_EXEMPT = {"480", "490", "500", "510", "520", "530", "540", "570", "571"}
VACANT_LAND_USE = {"8880", "8885"}
PARK_LAND_USE = {"8860", "8870", "8871"}
RADIUS_M = 200


def _raw(key: str):
    m = load_manifest().set_index("key")
    return DATA_DIR / m.loc[key, "local_path"]


def static_features() -> pd.DataFrame:
    c = parcel_centroids()
    xy = c[["x_m", "y_m"]].to_numpy()
    pts = gpd.GeoSeries(gpd.points_from_xy(c.x_m, c.y_m), crs=METRIC_CRS)
    hall = gpd.GeoSeries([Point(*CITY_HALL_LONLAT)], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
    bbox = pts.total_bounds
    pad = 20_000
    window = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    coast = gpd.read_file(f"zip://{_raw('tiger_coastline_2023')}").to_crs(METRIC_CRS).cx[
        window[0]:window[2], window[1]:window[3]]
    roads = gpd.read_file(f"zip://{_raw('tiger_prisecroads_2023')}").to_crs(METRIC_CRS).cx[
        window[0]:window[2], window[1]:window[3]]
    with zipfile.ZipFile(_raw("gtfs_mcts")) as zf:
        stops = pd.read_csv(zf.open("stops.txt"))
    stops = gpd.GeoSeries(gpd.points_from_xy(stops.stop_lon, stops.stop_lat), crs="EPSG:4326").to_crs(METRIC_CRS)
    if coast.empty or roads.empty:
        raise ValueError("coastline or roads empty inside the city window")
    out = pd.DataFrame({"tk": c.tk, "x_m": c.x_m, "y_m": c.y_m})
    out["d_downtown_m"] = np.hypot(xy[:, 0] - hall.x, xy[:, 1] - hall.y)
    out["d_lake_m"] = pts.distance(coast.union_all()).to_numpy()
    out["d_major_road_m"] = pts.distance(roads.union_all()).to_numpy()
    sxy = np.c_[stops.x, stops.y]
    out["d_transit_m"] = cKDTree(sxy).query(xy)[0]
    return out


def roll_features(concepts: pd.DataFrame, tks: pd.Series) -> pd.DataFrame:
    """Park distance and 200 m counts for `tks`, from one roll's concepts."""
    c = parcel_centroids().set_index("tk")
    roll = concepts.dropna(subset=["tk"]).drop_duplicates("tk").set_index("tk")
    roll = roll[roll.index.isin(c.index)]
    xy_all = c.loc[roll.index, ["x_m", "y_m"]].to_numpy()
    tree = cKDTree(xy_all)
    city = roll.exempt_type.isin(MUNICIPAL_EXEMPT).to_numpy()
    vacant = roll.land_use.isin(VACANT_LAND_USE).to_numpy()
    q = pd.Index(tks.unique())
    q = q[q.isin(c.index)]
    qxy = c.loc[q, ["x_m", "y_m"]].to_numpy()
    nbrs = tree.query_ball_point(qxy, r=RADIUS_M)
    self_pos = pd.Series(np.arange(len(roll)), index=roll.index)
    n_city, n_vac = [], []
    for tk, idx in zip(q, nbrs):
        idx = [i for i in idx if i != self_pos.get(tk, -1)]
        n_city.append(int(city[idx].sum()))
        n_vac.append(int(vacant[idx].sum()))
    out = pd.DataFrame({"tk": q, "n_city_200m": n_city, "n_vacant_200m": n_vac})
    parks = roll.index[roll.land_use.isin(PARK_LAND_USE)]
    poly = _park_polygons(parks)
    pts = gpd.GeoSeries(gpd.points_from_xy(qxy[:, 0], qxy[:, 1]), crs=METRIC_CRS)
    out["d_park_m"] = pts.distance(poly.union_all()).to_numpy() if len(poly) else np.nan
    return out


_PARCEL_GEOM = None


def _park_polygons(park_tks: pd.Index) -> gpd.GeoSeries:
    global _PARCEL_GEOM
    if _PARCEL_GEOM is None:
        g = read_parcels("parcels/2025", read_geometry=True).to_crs(METRIC_CRS)
        g["tk"] = normalize_taxkey(g["Taxkey"] if "Taxkey" in g else g["FK_Tax"])
        _PARCEL_GEOM = g.dropna(subset=["tk"])[["tk", "geometry"]]
    return _PARCEL_GEOM[_PARCEL_GEOM.tk.isin(park_tks)].geometry
