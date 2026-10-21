"""Registry of raw datasets. Resources are looked up by name in the CKAN
catalog at download time, so URLs are never hard-coded here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str            # unique id, also the folder under data/raw
    package: str        # CKAN package id on data.milwaukee.gov ("" for direct URLs)
    resource_name: str  # exact CKAN resource name ("" for direct URLs)
    sample: bool = False
    url: str = ""       # direct download, for sources outside the city catalog
    daily: bool = False  # city refreshes it daily: pinned once downloaded (rule 8)


MPROP_HIST_YEARS = range(2008, 2026)
SALES_YEARS = range(2002, 2026)
SAMPLE_MPROP_YEARS = {2015, 2021, 2025}
# 2015 sales pair with the 2015 snapshot to test whether sales-file building
# characteristics are as of the sale or as of the extract.
SAMPLE_SALES_YEARS = {2015, 2025}


def _mprop_hist_name(year: int) -> str:
    return f"Parcel Tax - MPROP {year}" if year >= 2022 else f"MPROP {year}"


SOURCES: list[Source] = [
    # Sales (ground truth)
    *[Source(f"sales/{y}", "property-sales-data", str(y), sample=(y in SAMPLE_SALES_YEARS))
      for y in SALES_YEARS],
    Source("sales/2002-2018_master", "property-sales-data", "2002-2018 Master File"),
    # MPROP yearly snapshots
    *[Source(f"mprop_hist/{y}", "historical-master-property-file", _mprop_hist_name(y),
             sample=(y in SAMPLE_MPROP_YEARS))
      for y in MPROP_HIST_YEARS],
    Source("mprop_current", "mprop", "CSV", daily=True),
    # Documentation
    Source("docs/mprop_ref_1975_2009", "historical-master-property-file",
           "MPROP Reference 1975-2009", sample=True),
    Source("docs/mprop_ref_2009_now", "historical-master-property-file",
           "MPROP Reference 2009-now", sample=True),
    Source("docs/mprop_fall_2024", "mprop", "Master Property File Documentation", sample=True),
    # Code tables
    Source("landusecode", "landusecode", "CSV", sample=True, daily=True),
    # Parcels
    Source("parcels/2025", "parcel-outlines", "Parcel - Tax 2025", sample=True),
    Source("parcels/current", "parcel-outlines", "ParcelPolygonTax", daily=True),
    Source("assessment_nbhds", "residential-assessment-neighborhoods",
           "Residential Assessment Neighborhoods"),
    # Census tract boundaries (US Census TIGER/Line 2023, Wisconsin). Used as a
    # LOCATION unit only (fixed effects); no demographics (rule 7).
    Source("census_tracts_2023", "", "",
           url="https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_55_tract.zip"),
    # Stage 6 geospatial sources.
    # MCTS transit schedule (GTFS): a CURRENT snapshot, so it is pinned like the
    # daily city files; stop locations are treated as static (limitation).
    Source("gtfs_mcts", "", "", url="https://kamino.mcts.org/gtfs/google_transit.zip", daily=True),
    # TIGER 2023 coastline (Lake Michigan shoreline) and WI primary/secondary roads.
    Source("tiger_coastline_2023", "", "",
           url="https://www2.census.gov/geo/tiger/TIGER2023/COASTLINE/tl_2023_us_coastline.zip"),
    Source("tiger_prisecroads_2023", "", "",
           url="https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/tl_2023_55_prisecroads.zip"),
    # Stage 9: TIGER 2023 area landmarks (campus polygons for the student-rental-area flag; display only).
    Source("tiger_arealm_2023", "", "",
           url="https://www2.census.gov/geo/tiger/TIGER2023/AREALM/tl_2023_55_arealm.zip"),
    # Stage 9: Milwaukee County comparative tax rates (Franklin, WI copy); City of Milwaukee net rate per year.
    Source("tax_rates_county_2016_2025", "", "",
           url="https://www.franklinwi.gov/Files/Finance/PropertyTaxInfo/"
               "Comparative-Total-Property-Tax-Rates-for-Milw-Co-Municipalities-2016-2025.pdf"),
    # Stage 9: Master Address Index (addresses and ZIP for every parcel; daily-updated, so pinned).
    Source("mai", "mai", "CSV", daily=True),
    # Optional: permits
    Source("permits", "buildingpermits", "CSV", daily=True),
]
