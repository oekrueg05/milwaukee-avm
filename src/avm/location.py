"""Location units for fixed effects: assessor neighborhood, census tract, and
the parent region used to fold small units.

  neighborhood  MPROP roll Y-1 neighborhood code (already on each sale)
  tract         2023 census tract GEOID containing the parcel centroid
  region        the assessor's residential land region (RES_LAND_1, 18
                regions), from the assessment-neighborhood layer: by code for
                neighborhoods on the layer, by centroid for everything else

Tracts are a LOCATION unit only. No tract demographics are used as features
(rule 7); they enter only the Stage 8 equity audit.
"""

import json

import geopandas as gpd
import pandas as pd

from avm.acquire import load_manifest
from avm.config import DATA_DIR, INTERIM_DIR
from avm.geo import METRIC_CRS, parcel_centroids

CACHE = INTERIM_DIR / "parcel_locations.parquet"
CACHE_META = INTERIM_DIR / "parcel_locations_meta.json"
MIN_UNIT_SALES = 30


def _layers():
    m = load_manifest().set_index("key")
    nb = gpd.read_file(DATA_DIR / m.loc["assessment_nbhds", "local_path"]).to_crs(METRIC_CRS)
    tr = gpd.read_file(f"zip://{DATA_DIR / m.loc['census_tracts_2023', 'local_path']}").to_crs(METRIC_CRS)
    tr = tr[tr.COUNTYFP == "079"]  # Milwaukee County
    shas = {k: m.loc[k, "sha256"] for k in ["assessment_nbhds", "census_tracts_2023", "parcels/2025",
                                            "parcels/current"]}
    return nb, tr, shas


def parcel_locations() -> pd.DataFrame:
    """tk -> tract, region_by_point (cached; keyed on source sha256)."""
    nb, tr, shas = _layers()
    if CACHE.exists() and CACHE_META.exists() and json.loads(CACHE_META.read_text()) == shas:
        return pd.read_parquet(CACHE)
    c = parcel_centroids()
    pts = gpd.GeoDataFrame(c[["tk"]], geometry=gpd.points_from_xy(c.x_m, c.y_m), crs=METRIC_CRS)
    j = gpd.sjoin(pts, tr[["GEOID", "geometry"]], how="left", predicate="within").drop_duplicates("tk")
    j = gpd.sjoin(j.drop(columns="index_right"), nb[["RES_LAND_1", "geometry"]], how="left",
                  predicate="within").drop_duplicates("tk")
    out = pd.DataFrame({"tk": j.tk, "tract": j.GEOID, "region_by_point": j.RES_LAND_1}).reset_index(drop=True)
    out.to_parquet(CACHE, index=False)
    CACHE_META.write_text(json.dumps(shas))
    return out


def neighborhood_regions() -> pd.Series:
    nb, _, _ = _layers()
    return nb.drop_duplicates("NBHD").set_index("NBHD").RES_LAND_1


def attach_locations(s: pd.DataFrame) -> pd.DataFrame:
    """Adds tract and region to a sales frame (needs tk, neighborhood)."""
    loc = parcel_locations().set_index("tk")
    s = s.copy()
    s["tract"] = loc.tract.reindex(s.tk).to_numpy()
    by_code = s.neighborhood.map(neighborhood_regions())
    s["region"] = by_code.fillna(pd.Series(loc.region_by_point.reindex(s.tk).to_numpy(), index=s.index))
    s["region"] = s.region.fillna("unassigned")
    return s


def fold_small_units(unit: pd.Series, parent: pd.Series, train_mask: pd.Series,
                     min_sales: int = MIN_UNIT_SALES) -> pd.Series:
    """Units with < min_sales TRAINING sales (or unseen in training) become
    'R:<parent>'; parents that are themselves small become 'R:citywide'.
    Counts come from training rows only, so validation/test rows never
    influence the grouping."""
    u = unit.fillna("missing").astype(str)
    p = parent.fillna("unassigned").astype(str)
    n_unit = u[train_mask].value_counts()
    small = ~u.isin(n_unit[n_unit >= min_sales].index)
    folded = u.where(~small, "R:" + p)
    n_fold = folded[train_mask].value_counts()
    still_small = ~folded.isin(n_fold[n_fold >= min_sales].index)
    return folded.where(~still_small, "R:citywide")
