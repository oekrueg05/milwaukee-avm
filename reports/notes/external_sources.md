# External sources: availability findings (Stage 1, 2026-10-17)

## City "_valid" arm's-length sales files: how are they filtered?
- **No city documentation found.** The catalog entry
  (https://data.milwaukee.gov/dataset/property-sales-data) lists the author
  as the Assessor's Office and describes the data only as "Residential,
  Condominium, Commercial, Apartment properties and vacant land sales history
  data". Resource notes give revision dates only: 2020-2023 are marked
  "Final"; 2024 ("Updated 9/17/2025") and 2025 ("Updated 4/17/2026") are not,
  so they may still be revised.
- The Assessor site (https://city.milwaukee.gov/assessor) returns HTTP 403 to
  automated requests, so it could not be checked from here.
- Web searches found no Milwaukee sales-validation or ratio-study document.
  The governing standard is Wis. Stat. 70.32(1) and the Wisconsin Property
  Assessment Manual definition of an arm's-length sale (willing buyer and
  seller, no compulsion, adequate exposure).
- File names: 2020-2025 are `armslengthsales_YYYY_valid*.csv`; 2002-2019
  are `YYYY-property-sales-data.csv`. Whether the older files were filtered
  the same way is not documented.
- **What the data itself shows** is in the Stage 1 summary: price floor,
  deed types of sold parcels (where MPROP conveyance dates survive), repeat
  sales and multi-parcel candidates.
- **Action for Owen:** ask the Assessor's Office (or check the site in a
  browser) for the validation criteria behind "_valid".

## WI DOR Real Estate Transfer Return (eRETR)
- Bulk monthly historical files are offered at
  https://revenue.wi.gov/Pages/ERETR/data-home.aspx, which now redirects to
  https://tap.revenue.wi.gov/RETRHistoric. That page requires cookies (a
  Wisconsin TAP session), so it could not be read or downloaded from here.
  Not scraped.
- Per DOR's description (via search results): monthly zipped CSVs, about
  the last 5 years only; older years must be requested from the county
  register of deeds. A searchable export exists at
  https://propertyinfo.revenue.wi.gov/ (DOR advises against large queries).
  DOR launched a new RETR system on 2026-01-12; whether the bulk files
  changed is unknown.
- **Why it matters:** RETR includes a filer-reported arm's-length indicator
  and conveyance details. Johnson (2023) matched state transaction records to
  city parcels for 2018 to early 2023, describing the result as "self-reported
  arm's length sales", with about 1.5% unmatched. That is evidence the join is
  feasible; the method is not published.
  - Johnson, John. 2023. "Out-of-state landlords dramatically slowed
    acquisitions in the first quarter of 2023." *Marquette University Law
    School Faculty Blog*, April 25, 2023.
    https://law.marquette.edu/facultyblog/2023/04/out-of-state-landlords-dramatically-slowed-acquisitions-in-the-first-quarter-of-2023/
- **Action for Owen:** open the TAP page in a browser, check whether the
  bulk files are free and what years they cover, and download one month for
  a field/join-key check. Then decide whether eRETR is in scope for v1.

## Census (Stage 8, evaluation only)
- Tract boundaries: TIGER/Line 2023 Wisconsin tracts download without a key
  (https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_55_tract.zip,
  HTTP 200, 7.7 MB).
- ACS 5-year API (api.census.gov) now redirects to a "Missing Key" page:
  a free API key is required. Alternative: data.census.gov table downloads.
- **Action for Owen (Stage 8):** request a free Census API key.

## City permits (optional feature source)
- `buildingpermits.csv`: 16,685 issued permits (dates opened/issued,
  permit type, construction cost, use of building, dwelling-unit impact).
  No tax key column, address only, so a join needs address matching (the
  city's Master Address Index dataset, `mai`, is the likely bridge). 40% of
  construction cost values are 0. Deferred; not needed for v1.

## HUD Fair Market Rents (Stage 9)
- https://www.huduser.gov/portal/datasets/fmr.html is reachable. The HUD
  User API returns 401 without a token (free registration). FMR tables are
  also published as downloadable spreadsheets, which need no token.
