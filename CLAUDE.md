# CLAUDE.md: Milwaukee AVM

Automated valuation model for City of Milwaukee single-family and duplex homes,
built on public data, validated against real sale prices and benchmarked against
the City Assessor. Full context, stage plan, and stage log live in docs/decision-log.md.
Read it before starting any stage.

## Environment
- Data and the virtual environment live outside the repo. Data folder: `AVM_DATA_DIR` (default `~/data/milwaukee-avm`) with `raw/`, `interim/`, `processed/`; parcel-level outputs go to `<data>/reports_private/` and are never committed.
- Package installed editable (`pip install -e .`); run modules as `python -m avm.<module>`.
- Dependencies: `requirements.txt` (bounds), `requirements.lock` (exact versions).
- Source code: `src/avm/`. Tests: `tests/`. Paths live in `src/avm/config.py`. Outputs: `reports/`.
- The decision log is `docs/decision-log.md`.

## Non-negotiable rules
1. **No leakage.** For a sale in year Y, features come from the MPROP snapshot
   as of before the sale, never the current MPROP. If no historical snapshot
   exists for a year, exclude those sales and say so. Neighborhood price-trend
   features use only sales dated before the target sale.
2. **Time-based validation.** Train on older sales, test on the most recent full
   year. Never random-split. Report results on the held-out year only.
3. **Fair assessor comparison.** Compare each sale to the assessment in force at
   the time of sale (the Jan 1 roll for that year). Time-adjust sale prices to
   that valuation date for the ratio study. Train the model only on sales the
   assessor could also have seen (before the valuation date).
4. **Arm's-length sales only.** Remove family transfers, $1 or nominal sales,
   multi-parcel sales, foreclosure/sheriff and city in-rem resales, and quick
   flips (resale within 12 months). Log every filter with rows removed.
5. **Baselines first, and fair ones.** Measure (a) the assessor and (b) a hedonic
   regression with location fixed effects before any ML. ML counts only if it
   beats both.
6. **Two model variants.** "Independent AVM" (no prior assessed value) and
   "assessment-informed" (prior-year assessed value as a feature). Report both.
7. **Fair housing.** No protected-class demographics or obvious proxies as
   features. Document every feature and why it is included; flag possible
   proxies. Demographics are used ONLY for the Stage 8 equity audit.
8. **Reproducible.** One command rebuilds data -> features -> model -> report
   from raw downloads. Fixed random seeds. Raw data never edited in place.
9. **Traceable numbers.** Every number in a report comes from a script output.
   No hand-typed metrics.

## Data decisions (Stage 1)
- **MPROP joins go through the assessment-year lookup**
  (`reports/stage1/mprop_assessment_year_lookup.csv`, via
  `avm.rolls.file_year_for`), never by file name. Features for a sale in
  year Y come from roll Y-1; the assessor benchmark is roll Y (valuation
  date Jan 1 of Y). **Only `confirmed` rolls (2013-2025) are used** (Owen,
  2026-10-17): call `file_year_for(year)` with its default `allowed`.
- **Training sales 2014-2024, test 2025** (features from rolls 2013-2024;
  benchmark roll 2025). Roll 2025 is never a feature source.
- **MPROP fields are read through the crosswalk in `avm.mprop.CROSSWALK`.**
  It covers four raw layouts (old, old 10-char 2017, new parcel layer,
  new 10-char 2022-2023). Every raw name is listed explicitly; never
  auto-truncate names. A field is usable for a year only if same-parcel
  continuity on SF/duplex parcels passes (>= 0.985, lot area within 5%) on
  its pairs (`mprop_crosswalk_continuity.csv`), or it is listed in
  `mprop.EXCLUDED` with evidence. Representation fixes (stories x10 through
  2018, building type 19->8 in 2020-2024, lot area 0 = missing) live in
  `mprop._normalize`.
- **Known gaps for features:** 2022 sq ft excluded; stories 2019-2020
  excluded; baths excluded before 2019; rooms only 2014, 2017, 2019+; the 2016
  file is blank on characteristics for 16% of SF/duplex parcels
  (`mprop_presence_by_year.csv`). Any fallback to another roll must use an
  earlier roll, never a later one.
- **MPROP land use** comes from `FK_LandUse` in the new layout (the new
  `LandUse` field is a different code). SF = 8810, duplex = 8820.
- **Sales-file building characteristics are as of the file extract, not the
  sale.** Never use them as features. Bedrooms come from MPROP only, never
  the sales-file count.
- **MPROP date fields can be corrupt** (`00:00.0`). Use `avm.io.parse_dates`,
  which refuses them; never call `pd.to_datetime` on raw MPROP dates.
- **Sale dates are month-only for 2002-2018 files.** Treat them as months,
  not days, in flip detection, time trends and "before the sale" logic.
- **Quick flips** are detected from repeated tax keys across sales files, not
  from MPROP conveyance dates.
- **Shared target prices:** `interim/target_prices.parquet` (Stage 3) holds
  every clean sale's price at the Jan 1 trend level, neutral season. The
  assessor and every model are scored against it, never against raw prices.
  Models predict at the Dec (Y-1) trend level with a neutral month-of-year.
- **Headline accuracy = level-neutral MdAPE** (`mdape_ln`: each method's
  ratios divided by its own median ratio first). Raw MdAPE is reported but
  never headlined; it rewards level, not accuracy. COD remains the IAAO
  uniformity measure. Model selection uses mean `mdape_ln`; GBM ties (within
  0.2 pts) go to the smallest mean |within-tier PRB|.
- **GBM target is detrended (Stage 6 baseline):** log price minus the fold's
  training-years-only hedonic time effect (`avm.priors.TimeEffect`); the GBM
  has no time features and predicts directly at the Dec (Y-1) trend level,
  neutral season.
- **Prior-sales features (leakage-critical):** training rows use sales before
  their own sale month; scored rows (validation year Y, test 2025) only sales
  before Jan 1 Y. Own prior sale must be > 12 months before the sale and
  before the cutoff (`own_prior_sale` asserts both).
- **Prior assessed values in the informed variant must be level-free**
  (relative to the roll's neighborhood median): raw roll values carry each
  roll's assessment level into a detrended model (level 1.13 in 2025).
- **Compression:** calibration slopes (log target on log prediction) are < 1
  for every method, so predictions are not compressed given the prediction;
  no spread correction. Within-tier PRB is read with the sale-price-noise
  caveat.
- **Test-set discipline:** no selection decision uses 2025 results (from
  the freeze, 2026-10-18). Selection uses CV 2021-2024 only. The frozen
  models are in `reports/frozen/model_spec.json`; do not change them. The
  prospective 2026 file is hashed and tagged `prospective-2026`: never
  regenerate or edit it; score it with `python -m avm.prospective score`.
- **Daily-updated city files are pinned** (current MPROP, current parcels,
  permits, land-use codes): `acquire` skips them once downloaded unless run
  with `--refresh-current`; their sha256 and `daily_updated` flag are in the
  manifest. Refreshing them changes Stage 2 inputs, so re-run from Stage 2.
- **Head-to-head strata:** SF/duplex, sq ft bins, neighborhood value tier
  (prior-year neighborhood median price). Sale-price strata only as a
  footnote (selection on the denominator).
- **The 2025 time index is evaluation-only.** It time-adjusts 2025 sale prices
  to the Jan 1 2025 valuation date for the ratio study and never feeds model
  training. Stage 5 predicts at Jan 1 2025 using data through 2024 only.
- **Trims:** the IAAO 3x IQR trim on the assessor's ratio is for the
  assessor's standards check only. The Stage 5 head-to-head reports untrimmed
  plus a union trim (drop if an outlier under either the assessor's or the
  model's ratio).
- **Never `pd.to_numeric(errors="coerce")` on model inputs.** Use
  `avm.io.to_number`, which strips `$` and commas and raises on anything else.
  A silent coerce once blanked every 2020-2021 sale price ("$357,000").
- **Clean sales table** is `interim/sales_clean.parquet` from
  `python -m avm.clean_sales` (or `python -m avm.pipeline`): one row per
  arm's-length SF/duplex sale, features from roll Y-1 with `<f>_src_roll`
  provenance, benchmark from roll Y, flags kept for sensitivity checks.

- **Stage 9 rules (Owen, 2026-10-19; guardrail revised 2026-10-20):** a
  group is called over- or under-assessed only if ALL hold: (a) the bootstrap
  CI of the bias-corrected shift (shift + the model's CV bias for the group)
  excludes zero; (b) the model-free 2025 sales check agrees in direction with
  a CI excluding zero; (c) (a) and (b) survive tier-specific (tercile) time
  adjustment. Passing (a) and (b) but not (c) is reported as consistent with
  assessments lagging faster appreciation, not a valuation pattern. Parcels
  held flat 2024->2025, whose land use changed between rolls, or whose unit
  count does not fit the land use are set aside in "Likely special valuation
  or data issue" and never ranked in the main top 1,000 (they stay in the tax
  pool). The shareable portfolio page carries no parcel rows, addresses or tax
  keys; no page shows owner names. Pages are not published without Owen's go.
  Only the portfolio page is ever published.
- **No identifiers in tracked files (Owen, 2026-10-20):** parcel-level outputs with
  addresses or tax keys go to `<data>/reports_private/` (config
  `PRIVATE_REPORTS_DIR`); summaries and logs drop tax key, address and ZIP; the data
  dictionary redacts sample values for sensitive fields. `python -m avm.ci_checks
  privacy` (in CI) must pass. This repo's history still holds identifiers, so it
  stays private; any public repo is a fresh redacted snapshot.

- **Comparable sales** come only from `avm.comps` (Owen, 2026-10-20): same
  property type; 0.5 / 1 / 1.25 mi within 12 months, then 24; stop at the
  first tier with >= 3; exclude first-sale-of-flip, IAAO ratio and price/sq ft
  outliers; SHAP-weighted ranking; tier recorded per subject.
- **Write-up numbers** come only from `python -m avm.writeup` placeholders
  (README.md and docs/methods.md are generated; edit `docs/templates/`). Tax
  dollars use the 2025 NET rate of 20.80 (Owen, 2026-10-20: confirmed from the
  Milwaukee County comparative rate table, manifest key
  tax_rates_county_2016_2025; 22.93 is the 2024 rate). The school levy credit
  scales with value (Wis. Stat. 79.10); lottery and first dollar credits are
  excluded. Duplex yields (HUD SAFMR, issue #1) are optional and do not block publishing.

## Working style
- Plan first, then STOP and wait for "go". If Claude asks a question, it waits
  for Owen's answer; it never answers its own question and proceeds.
- One stage at a time. Run on a small sample first, show real output, then the
  full run.
- After each stage: commit, update docs/decision-log.md (status line and stage log), STOP.
- Every value traces to a source. Conflicts in data are surfaced, not hidden.
  Code does the logic. Failures are loud (raise, don't silently drop).
- Concise reports. No invented numbers. No em dashes in written docs.
- Do not assume field names. Check them against the downloaded data and the
  MPROP reference PDFs.
