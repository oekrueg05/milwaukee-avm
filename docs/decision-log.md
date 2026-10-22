# Milwaukee Property Valuation Model: Handoff / Kickoff Doc

_Created: 2026-10-17. Revised: 2026-10-17 (v2, after plan review). Owner: Owen Krueger. Status: public snapshot of the project. Stages 0-10 done; the prospective 2026 predictions are frozen, hashed and timestamped (Bitcoin block 972397)._

Purpose of this doc: give a fresh Claude Code session everything it needs to start
this project without re-deciding the basics. Read this whole file before writing code.

---

## 1. What this project is
An automated valuation model (AVM) for City of Milwaukee residential properties,
built on public data, that:
1. Predicts a market value for each property, with a prediction interval.
2. Proves its accuracy against real sale prices (held-out, time-based test).
3. Benchmarks itself against the City Assessor's assessed values using
   standard assessment-quality metrics (IAAO ratio study).
4. Explains each valuation two ways: model drivers (SHAP) and comparable sales.
5. Audits fairness: are errors and assessment ratios biased across neighborhoods?
6. Surfaces properties where assessed value and predicted market value diverge
   the most, and translates that into dollars (tax impact, investor yield).

The point is a project with **ground truth**: actual sale prices let us prove it
works with a number, not a demo.

## 2. Why (context for design choices)
- Owner's direction: AI + real estate valuation, real estate private equity.
- Owner is on the Machine Learning track (BUAN 3065/3066), so the ML should be
  real: proper validation, fair baselines, no leakage, honest uncertainty.
- Audience for the final write-up: RE PE / acquisitions teams and AI valuation
  teams. They care about accuracy, uncertainty, comps, and dollars, not just a
  low error metric.
- Prior project (deadline tracker) established a working style worth keeping:
  every value traces to a source, conflicts are surfaced not hidden, code does
  the logic, failures are loud.

## 3. Data sources
All public. Verify each link, license, and field list in Stage 1; do not assume
field names.

| Dataset | Source | What it gives us | Notes |
|---|---|---|---|
| Property Sales Data | data.milwaukee.gov | Sale price, date, property type | Updated yearly. The ground truth. |
| Master Property File (MPROP) | data.milwaukee.gov | ~90 fields per property: assessed values, land use, building characteristics | Updated daily. CURRENT snapshot only. |
| MPROP Historical | data.milwaukee.gov | Past yearly MPROP snapshots | Needed to avoid leakage (section 5). |
| Parcel shapefiles | data.milwaukee.gov | Parcel polygons | Centroids, geospatial features. |
| Land use code definitions | data.milwaukee.gov | Decodes MPROP land use codes | |
| Permit Work Data | data.milwaukee.gov | Permit dates, construction cost | Optional: renovation activity. |
| WI Real Estate Transfer Return (eRETR) | WI Dept. of Revenue | Transfer type / conveyance codes, sale details | May give real arm's-length flags. Check availability and join key. |
| Census tract boundaries + ACS demographics | US Census | Tract demographics | EVALUATION ONLY (equity audit). Never a model feature. |
| HUD Fair Market Rents | HUD | Rent by bedroom count, Milwaukee metro | Duplex income view in Stage 9. |

Join key between sales and MPROP is almost certainly the tax key (confirm in Stage 1).

**Prior art to read before Stage 5:** Cook County Assessor's Office open-source
residential model (GitHub org `ccao-data`). LightGBM-based, public, well documented.
Cite it and state where this project differs.

## 4. Scope
**In (v1):** single-family and duplex residential sales, City of Milwaukee only.
**Later:** condos, small multifamily, commercial.
**Out:** anything requiring paid data (MLS, CoStar), scraping private sites.

## 5. Non-negotiable rules
1. **No leakage.** For a sale in year Y, features come from the MPROP snapshot
   as of before the sale, never the current MPROP. If a historical snapshot isn't
   available for a year, exclude those sales and say so. Neighborhood price-trend
   features use only sales dated before the target sale.
2. **Time-based validation.** Train on older sales, test on the most recent full
   year. Never random-split. Report results on the held-out year only.
3. **Fair assessor comparison.** Compare each sale to the assessment in force at
   the time of sale (the Jan 1 roll for that year). Time-adjust sale prices to
   that valuation date for the ratio study. The model is trained only on sales
   the assessor could also have seen (before the valuation date).
4. **Arm's-length sales only.** Filter out non-market transfers: family
   transfers, $1 or nominal sales, multi-parcel sales, foreclosure/sheriff and
   city in-rem resales, and quick flips (resale within 12 months, since MPROP
   won't reflect the renovation). Every filter is documented with a count of rows
   removed.
5. **Baselines first, and fair ones.** Before any ML, measure (a) the assessor's
   value and (b) a hedonic regression that INCLUDES location (neighborhood or
   tract fixed effects). A fancy model only counts if it beats both.
6. **Assessed value as a feature is a deliberate choice.** Run two model
   variants: "independent AVM" (no prior assessed value) and
   "assessment-informed" (prior-year assessed value as a feature). Report both.
7. **Fair housing.** No protected-class demographics or obvious proxies as
   features. Location features (distance, neighborhood) are fine. Document every
   feature and why it's included; flag any possible proxy. Demographics are used
   ONLY to evaluate fairness (Stage 8).
8. **Reproducible.** One command rebuilds data -> features -> model -> report
   from raw downloads. Random seeds fixed. Raw data never edited in place.
9. **Every number in the final report traces to a script output.** No
   hand-typed metrics.

## 6. Evaluation metrics
Model accuracy on held-out sales:
- Median absolute percentage error (MdAPE)
- Share of predictions within 10% and 20% of sale price
- Prediction interval coverage (e.g. does the 80% interval contain the sale ~80%
  of the time?) and average interval width

Assessment-quality metrics (IAAO ratio study), for BOTH assessor and model,
ratio = value / time-adjusted sale price:
- Median ratio (level). Target 0.90 to 1.10.
- COD, coefficient of dispersion (uniformity). Target <= 15 for SFR.
- PRD, price-related differential. Target 0.98 to 1.03.
- PRB, price-related bias (current IAAO-preferred vertical equity measure).
  Target -0.05 to +0.05.

Note in the write-up: PRD/PRB computed against sale price are biased toward
showing regressivity, because noise in sale prices sits in the denominator.
State this caveat with the finding. Bootstrap confidence intervals for all
metrics (held-out sample may be modest).

## 7. Suggested stack
- Python 3.13 (same interpreter as the deadline tracker:
  <python>\python.exe)
- pandas / polars, geopandas + shapely for parcels
- scikit-learn and statsmodels (baselines, hedonic), LightGBM (main model,
  quantile objective or conformal for intervals), SHAP (explanations)
- DuckDB or Parquet files for intermediate data
- Minimal UI at the end (static HTML report with a map, or Streamlit). Decide in
  Stage 9, not before. The write-up is the real deliverable.
- Git repo from day one, private on GitHub (oekrueg05). Raw and processed data
  gitignored (size), code and docs tracked.

## 8. Stage plan
Work stage by stage. After each stage: show real output, stop, wait for approval.
Test on a small sample before the full dataset.

0. **Setup.** Repo, folder layout (data/raw, data/interim, data/processed, src/,
   reports/, notebooks/), .gitignore, requirements, CLAUDE.md with the rules in
   section 5 and the working style in section 10.
1. **Data acquisition + data dictionary.** Download each dataset (record source
   URL and download date in a manifest). Data dictionary: every field we might
   use, type, null rate, example values. Confirm the join key, how far back
   MPROP Historical goes, whether field names change across years, and whether
   eRETR data is obtainable and joinable. Skim the CCAO model repo; note ideas.
2. **Cleaning + arm's-length filter + leakage-safe join.** Join each sale to the
   MPROP snapshot from before the sale. Apply filters from rule 4 with row
   counts. Pick the held-out test year. Output: one clean sales table plus a
   filter log.
3. **Baseline 1: assessor ratio study.** Time-adjust sale prices, then compute
   median ratio, COD, PRD, PRB (with bootstrap CIs) for the assessor on the
   held-out year. This is the number to beat.
4. **Baseline 2: hedonic regression.** Log price on size, age, lot,
   beds/baths, condition, time trend, AND neighborhood/tract fixed effects.
   Same metrics, same held-out year.
5. **Main model.** LightGBM with time-based CV, two variants (rule 6). Add
   prediction intervals (quantile LightGBM or conformal). Compare against
   Stages 3 and 4 on the same held-out year.
6. **Geospatial features.** Parcel centroids, distance to downtown, lakefront,
   major corridors and transit; neighborhood recent sale-price trends (prior
   sales only). Re-evaluate. Report the lift from each feature group.
7. **Explainability.** SHAP per property and globally. Sanity-check that the
   drivers make real-estate sense. For each property, also pull the 3 to 5
   nearest comparable recent sales.
8. **Equity audit.** Using tract demographics for evaluation only: model error,
   ratio level, and COD by neighborhood / tract income and demographic groups,
   for both the assessor and the model. Report honestly whether the model is
   fairer, worse, or mixed.
9. **Divergence list + dollars + UI.** Properties where model value and assessed
   value differ most, with SHAP drivers, comps, and interval. Translate into
   dollars: estimated tax over/under-payment at Milwaukee's mill rate; for
   duplexes, gross yield using HUD FMR vs model value. Minimal map UI.
10. **Write-up.** README with method, metrics, limitations (selection bias:
    homes that sell aren't the full stock; unobserved interior condition; PRD
    caveat), the PRD/PRB finding, the equity audit, and comparison to CCAO.
    Portfolio-ready.

Optional after Stage 10: grounded LLM valuation memo per property, generated
only from SHAP output, comps, and interval, with every number traced to data.

## 9. Known risks / open questions
- How many years of MPROP Historical exist, and do field names change? (Stage 1)
- Does the city sales data flag non-arm's-length transfers, or does eRETR, or do
  we infer them? (Stage 1)
- Held-out year sale volume may be modest; bootstrap all metrics.
- Market appreciation over the training window: time index in models,
  time-adjusted prices in the ratio study.
- Selection bias: sold homes differ from the full stock. Disclose; don't overclaim.

### Published pages
- The shareable portfolio page is `reports/stage9/page/assessment_check_portfolio.html`. The private review page (addresses and tax keys) is never published or shared.

### Open items (GitHub issues, opened 2026-10-20)
- **DONE 2026-10-22 (commits 52a1835, 9df97ba): comps include the subject's own sale.** Fixed and re-run; results in the stage log row 'self-comp fix'. Original entry: `avm.comps.comps_for` never excludes the subject parcel from its pool. Owen (2026-10-22): exclude the subject parcel's own sales. Impact, counted with a scratch copy of the fixed function and every output written to a temporary folder (real outputs hash-checked unchanged): Stage 7, 68 of 4,197 test subjects get new comps (1 changes search tier), comps (a) level-neutral MdAPE 14.20% to 14.27% (README 14.2% to 14.3%), comps (b) 12.44% to 12.45% (README stays 12.4%), model unchanged, disagreement flags 608 to 619, first-tier share stays 99%. Stage 9, comp value changes for 2,440 parcels (42 change tier, 1 loses comps), listable 12,694 to 12,690 (4 stop being listable, 0 start), top 1,000 and the 479 set-aside parcels unchanged in membership and order (5 listed parcels show a different comp value), relative assessments, tax shifts, groups and tracts unchanged. Serving, all 4,099 self-comp parcels get new comps, 22 change tier, 56 have fewer comps (still at least 3), none lose comps. **The frozen model and the prospective file are unaffected:** `avm.comps` is imported only by Stages 7 and 9, the serving build and the write-up (constants); no training or prediction code uses it.
- [#1](https://github.com/oekrueg05/milwaukee-avm/issues/1) (Optional) Add HUD Small Area FMRs and compute duplex gross yields; not a publishing blocker (the pages hide the duplex section until yields exist).
- [#2](https://github.com/oekrueg05/milwaukee-avm/issues/2) Score the out-of-time 2026 test once the city publishes the 2026 sales file and roll 2026.
- [#3](https://github.com/oekrueg05/milwaukee-avm/issues/3) Income approach for student-rental areas (public data may not be able to test it).
- [#4](https://github.com/oekrueg05/milwaukee-avm/issues/4) Investigate the model's within-tier PRB.
- CI: done (`.github/workflows/ci.yml`).
- [#5](https://github.com/oekrueg05/milwaukee-avm/issues/5) MCP server: read-only `value_parcel` tool (hashed 2026 predictions plus `avm.comps`; no owner names).
- Public repository: snapshots of the private project repository, one squashed commit per update (the first carries the tag prospective-2026).

### Write-up limitations (logged as found; every one goes in the Stage 10 README)
- **Test set excludes resales within 12 months.** Accuracy applies to homes not
  obviously renovated since the last roll. (Stage 2)
- **First-sale flip flags are undercounted in 2025**: 2026 resales are not yet
  published. (Stage 2)
- **No eRETR.** The state transfer-return bulk files need a TAP login; skipped
  for v1, so there is no filer-reported arm's-length flag. Arm's-length status
  rests on the city's undocumented "_valid" filter plus our filters. (Stage 1)
- **Deed type is verifiable only where MPROP conveyance dates survive** (0% for
  2021 and 2025 sales), so it is a flag with a sensitivity check, not a filter.
  (Stage 2)
- **Permit coverage is partial** (about 1,600-2,000 permits a year, from 2017);
  the significant-permit flag under-counts renovation. (Stage 2)
- **Baths before 2019** come from before a city data cleanup (93% same-parcel
  agreement with 2019); carried raw with an era flag, main feature missing.
  (Stage 2)
- **2022 sq ft** is a different measure and is replaced from roll 2021 for 2023
  sales; stories for 2020-2021 sales come from roll 2018. (Stage 1-2)
- **City sales-file characteristics are as of the file extract**, so they are
  never used; MPROP roll Y-1 is the only feature source. (Stage 1)
- **Within-roll sales chasing could only be tested via change dates**, which the
  tax-bill restamp overwrites for about 99.5% of parcels from 2021. (Stage 1)
- **Rooms are stale for some years**: rolls 2015, 2016 and 2018 have no room
  counts, so 2016, 2017 and 2019 sales take rooms from 2-3 rolls earlier.
  (Stage 2)
- **Permit flag cannot cover 2014-2017 sales** (window starts before permit
  data begins in 2017). (Stage 2)
- **Month-only sale dates (2014-2018) cannot detect 3+ parcel bulk sales**, so
  those years may keep some that 2019+ would drop. (Stage 2)
- **Time adjustment moves the level, not the dispersion**: hedonic (seasonal)
  vs IAAO linear trend differ by 0.04 in median ratio; COD/PRD/PRB barely move.
  Vertical-equity results (PRB) do depend on citywide vs tercile indexes.
  (Stage 3)
- **Pre-2020 sales files look less filtered** than the 2020+ "_valid" files
  (more sales far below the prior roll), so training years differ in quality
  from the test year. (Stage 3)
- **PRD/PRB/MKI are not reported within sale-price bands**: selecting on the
  denominator makes them meaningless there. (Stage 3)
- **The assessor's level verdict depends on the valuation-date convention**:
  median ratio 0.917 against the January (seasonal) price, 0.892 against the
  Jan 1 trend level. Dispersion results do not. (Stage 3-4)
- **Hedonic predictions are compressed toward neighborhood means**: within
  neighborhood value tiers, PRB is -0.06 to -0.21. A model can beat the
  assessor on MdAPE while being more regressive within neighborhoods. (Stage 4)
- **Daily-updated city files** (current MPROP, current parcels, permits,
  land-use codes) change every download; now pinned behind
  `--refresh-current`. (Stage 4-5)
- **The GBMs win on uniformity but fail IAAO vertical-equity targets** (PRD
  1.04, PRB -0.09, MKI 0.86) where the assessor meets them; within
  neighborhood value tiers they compress values more than the hedonic.
  (Stage 5)
- **The assessment-informed GBM depends on the assessor's regime**: after
  the 2024 revaluation its level jumped from 1.01 (CV) to 1.056 (2025). (Stage 5)
- **GBM prediction intervals under-cover** (69-72% for a nominal 80%) when
  calibrated on 2024 alone; hedonic normal intervals over-cover high-value
  and under-cover low-value tiers. (Stage 5)
- **No eRETR for v1** (restated): arm's-length status rests on the city
  filter plus ours. (Stage 1, 5)
- **Transit stops are today's (2026 GTFS) for every year**; parks use 2025
  parcel polygons with each roll's land use. (Stage 6)
- **Own-prior-sale features only see clean sales from 2014 on** (21-37% of
  scored sales have one). (Stage 6)
- **City-owned and vacant counts are strong proxies** for neighborhood racial
  composition; kept as allowed location features but flagged for Stage 8.
  (Stage 6)
- **The 2025 test set was viewed across Stages 3-6** (reported at every stage,
  and design choices were made with those reports in view). From the freeze
  onward no selection uses 2025; the clean out-of-sample check is the
  prospective 2026 test (predictions hashed and tagged before any 2026 sale
  was published). (Freeze)
- **Both vertical-equity diagnostics are biased, in opposite directions**:
  PRB/PRD against sale prices lean toward regressivity because sale-price
  noise sits in the denominator; the calibration slope (log price on log
  prediction) is biased below 1 by noise in the prediction (attenuation).
  Stage 8 reports both side by side; prediction-conditional measures lead.
  (Stage 6-8)
- **Group bias gaps mix method error with group-specific appreciation**:
  target prices use a citywide trend, so groups whose prices rose faster look
  under-predicted by every method. Between-method comparisons are unaffected.
  (Stage 8)
- **The audit evaluates a cleaner subset in Black-majority and low-income
  tracts**: more sales are removed there (mostly flips), so accuracy in
  investor-heavy markets is likely overstated. (Stage 8)
- **Models narrow level-bias gaps but not accuracy or dispersion gaps**:
  every method is ~1.7x less accurate in Black-majority and lowest-income
  tracts, and the models' COD gaps are slightly larger than the assessor's.
  (Stage 8)
- **Student-rental areas**: income-producing rentals near Marquette and UWM carry value the model's features cannot see, so divergence there likely reflects the model, not the assessor. 87 of the top 1,000 gaps are within 0.75 mi of either main campus (60 above the model, 27 below), though only 4,213 of 121,471 homes are; 60 of the 106 above-model listings are in those areas. Flagged, not set aside. (Stage 9)
- **Held-flat and mismatched records are set aside, not judged**: 491 listable parcels (433 assessment unchanged from 2024, 22 land use changed, 15 units inconsistent, 37 commercial building type) are listed separately; the model cannot tell a restricted or condition-based value from an error. (Stage 9)
- **Group tax shifts measured against the model mostly reflect the model's own group bias**: no group passes both the bias guardrail and the model-free 2025 sales check, and for White-majority, Q1 and Q5 tracts the sales check points the opposite way. (Stage 9)
- **Detrending trades raw accuracy for level and vertical equity** (baseline
  MdAPE 12.6% -> 13.3%, PRD 1.041 -> 1.016); features recover the accuracy.
  (Stage 6)

## 10. Working style
- Plan first, then stop and wait for "go". A question from Claude waits for
  Owen's answer; Claude never answers its own question and proceeds.
- One stage at a time; test on a sample first; real output before moving on.
- Commit after each stage. Update this docs/decision-log.md (status + stage log below)
  at the end of every stage.
- Concise reports. No invented numbers. No em dashes in written docs.

## 11. Prompts
**First prompt (new session):**
```
Read docs/decision-log.md fully. Then do Stage 0 (setup) and plan Stage 1 (data
acquisition + data dictionary). Show me the folder layout, CLAUDE.md, and the
Stage 1 plan, then STOP and wait for "go".
```

**Every stage after that (reuse):**
```
Stage N. Re-read docs/decision-log.md sections 5, 6, and 8. Give me the plan for Stage N
(steps, files you'll create, what output I'll see), then STOP for "go".
```
After "go": run on a sample, show output, then the full run, show output,
commit, update the stage log, STOP.

## 11b. Story so far (as of Stage 5, 2026-10-18)
- **A public-data GBM roughly matches the City Assessor.** On 2025 sales the
  independent LightGBM (no assessed values) has level-neutral MdAPE 12.6% vs
  13.3% for the assessor (diff -0.7 pts [-1.3, -0.0]) and the same COD (17.8 vs
  17.8). The hedonic baseline is in between (12.9%, COD 17.9).
- **The assessment-informed GBM beats the assessor on uniformity**: MdAPE
  12.1% (-1.2 pts [-1.8, -0.7]) and COD 16.8 (-0.9 [-1.3, -0.5]), in every
  stratum. It inherits the assessor's regime, though: its level jumped from
  1.01 (CV) to 1.056 after the 2024 revaluation.
- **Every model compresses values within neighborhoods**: within neighborhood
  value tiers PRB is -0.06 to -0.25 for the hedonic and both GBMs, vs about
  0.0 to +0.05 for the assessor, and both GBMs fail IAAO PRD/PRB/MKI. Accuracy
  gains so far come partly from pulling homes toward their neighborhood
  average, which overvalues cheaper and undervalues pricier homes.
- The assessor's level (median ratio 0.89 at the Jan 1 trend level) misses the
  0.90 target only under the trend-level convention; COD 17.6-17.8 misses the
  <= 15 target under every convention.

**Update after Stage 6:**
- **A public-data model now beats the assessor.** Independent GBM on a
  detrended target with location, distance, prior-sales and city/vacant
  features: level-neutral MdAPE 12.0% vs 13.3% (-1.3 pts [-1.9, -0.7]), COD
  16.6 vs 17.8 (-1.2 [-1.7, -0.7]), level 1.00, and it meets the IAAO PRD, PRB
  and MKI targets. Only COD <= 15 is missed, by both.
- **The "compression" signal was mostly a measurement artifact.** Calibration
  slopes of log price on log prediction are below 1 for every method, so
  given the prediction, values are not compressed. The negative within-tier
  PRB is the known sale-price-noise bias in PRB; the assessor's ~0 PRB with
  the lowest slopes (0.70) points to noisier assessments, not better
  vertical equity. Stage 8 should lean on prediction-conditional measures.
- The assessment-informed variant needs level-free prior values; once fixed
  it is no better than the public-data model.

## 12. Stage log
| Stage | Status | Date | Key output / decision |
|---|---|---|---|
| 0 | done | 2026-10-17 | Repo, folder layout, .gitignore, requirements.txt, CLAUDE.md. Catalog check: MPROP Historical 1975 to 2025, sales 2002 to 2025, parcel shapefiles 2001 to 2025 (gaps 2003 to 2006). |
| 1 | done | 2026-10-17 | 52 raw files (1.63 GB) in `<data>\raw` with manifest; repo moved to `<home>\Projects\milwaukee-avm`, venv outside a synced folder. Data dictionary for 49 tabular files. **Decisions (Owen):** joins use confirmed rolls only (2013-2025) via `mprop_assessment_year_lookup.csv`; training sales 2014-2024, test 2025; bedrooms from MPROP only. **Findings:** every MPROP file holds its own year's roll; four raw layouts, crosswalk validated by same-parcel continuity (all feature-roll pairs 2013-2024 pass >= 0.985 on SF/duplex) with documented exclusions (2022 sq ft, stories 2019-2020, baths before 2019, building type 2025); 2016 file blank on characteristics for 16% of SF/duplex; MPROP dates corrupt in 2008-2013, 2020 and conveyance dates in 2021, 2025; sales-file characteristics are as of extract, not sale; sale dates month-only 2002-2018; city "_valid" filter undocumented, but deed types of sold parcels are 90-96% warranty, ~0% sheriff; Y-1 tax key match >= 99.75% (2009+). Sales chasing: within roll Y, sold vs unsold individual post-sale changes 0.3-0.8% vs 0.3-0.4% (2014-2018), 0 sold in 2021-2025, so negligible for the benchmark; Y->Y+1 shows sold parcels reassessed more (write-up finding). eRETR, Census key and HUD token need Owen (see `reports/notes/external_sources.md`). CCAO notes in `reports/notes/ccao_notes.md`. |
| 2 | done (1 open item) | 2026-10-17 | `interim/sales_clean.parquet`: 42,073 train (2014-2024; 31,900 SF, 10,173 duplex) and 4,162 test (2025; 2,940 SF, 1,222 duplex). Filter log, multi-parcel rule counts, fallback usage, flags in `reports/stage2/`. 12 unit tests pass; leakage check (feature roll < sale year) passes on every row. **Decisions (Owen):** drop resales < 12 months, flag first sale and ambiguous 12-month resales; deed type and permits are flags, not filters; multi-parcel drop = within 200 m or 3+ parcels (month-only years: 3+ at a non-round price); baths pre-2019 missing + raw value + era flag; training outlier = price/sq ft 3x IQR within neighborhood-year; IAAO ratio trim flagged for Stage 3. **Bug found and fixed:** 2020-2021 sale prices ("$357,000") were silently coerced to missing; all model-input parsing now strict (`io.to_number`), Stage 1 re-run. **Open item:** the month-only 3+ non-round rule removes ~1,700 residential rows in 2014-2018 that look like coincidences (median 13 km apart, 0.2% within 1 km, 81% at $1,000 multiples), awaiting Owen's decision on a stricter rule. |
| 2 (rev) | done | 2026-10-18 | Month-only years (2014-2018) now drop multi-parcel groups only within 200 m (Owen). Drops 0.05-0.24% of residential rows vs 1.2-2.7% in 2019+; the whole gap is the 3+ parcel rule in day years, whose groups also look like coincidences (233 groups, median 11.7 km apart, 1.7% within 1 km, 93% round prices). Open: whether to drop the 3+ rule for day years too. Final counts: 2014-2018 now 2,057 / 2,638 / 3,224 / 3,590 / 4,047. |
| 3 | done | 2026-10-18 | Assessor on 2025 test sales (n = 4,148 IAAO-trimmed; hedonic index, Jan 2025 = 1): median ratio 0.921 [0.914, 0.927] (meets), **COD 17.6 [17.1, 18.1] (misses <= 15)**, PRD 1.009 (meets), PRB 0.034 [0.021, 0.046] (meets), MKI 0.985 (meets). SF COD 16.9, duplex 19.3; lowest price quartile COD 22.2. Tercile-specific indexes move PRB to 0.059 (misses) and MKI to 1.005: vertical-equity results depend on the index. No sign roll 2025 used early-2025 sales (before Apr 1: COD 17.2, median 0.917; after: 17.7, 0.921). Context: COD 13-14 in 2014-2018 and 2020, 17-20 in 2019 and 2021-2025; median ratio 0.80-0.87 in 2021-2023 (roll lagging 15-18% annual price growth); 2014-2016 regressive (PRD 1.03-1.04, PRB -0.07 to -0.08). 2019 outliers (116) are low prices, not roll errors (0 value spikes); pre-2020 non-"_valid" files hold more sales below half the prior roll (1.3% in 2019 vs 0.02-0.23% in 2020-2025). Hedonic vs IAAO-trend adjustment differ by 0.04 in median ratio (seasonality); COD/PRD/PRB barely move. 17 tests pass. |
| 2-3 (rev) | done | 2026-10-18 | Multi-parcel: 200 m rule everywhere, 3+ groups at a non-$1,000-multiple price flagged only (F9 now removes 88 rows in total). Test set 4,232 sales (4,229 with benchmark). Main time adjustment now trend-only (month-of-year effects from 2014-2024: Jan -7.6%, Jun +3.7%; 2025 trend +0.19%/month). Assessor 2025 headline: data change alone moves median ratio 0.921 -> 0.917, COD 17.60 -> 17.54 (barely); the method change moves median ratio to **0.892 [0.885, 0.899], now missing the 0.90 target**; COD 17.6, PRD 1.009, PRB 0.036, MKI 0.984 unchanged. The level verdict depends on whether Jan 1 value is the winter price or the trend level. |
| 4 | done | 2026-10-18 | Hedonic (log price; time spline + sum-to-zero month-of-year; sq ft, lot, age bins, beds, baths, stories, style, duplex; location FE with units < 30 training sales folded to the assessor region). CV 2021-2024 picked neighborhood FE, training start 2020 (mean MdAPE 14.4%; all six configs within 14.4-15.2%). Sensible coefficients (sq ft elasticity 0.52; age monotone; +6% per full bath; beds ~0 given sq ft). 2025 test (n = 4,229): MdAPE 13.0%, within 10% 39.3%, within 20% 66.5%, median ratio 1.001, COD 18.0, 80% PI coverage 84.1% (median width 65% of value). **Head-to-head vs assessor:** COD diff 0.0 [-0.5, 0.5] untrimmed, -0.3 [-0.8, 0.1] union trim: no uniformity win in any stratum. MdAPE -1.6 pts [-2.3, -0.9], mainly from level (assessor 0.89). Hedonic PRD 1.021 vs 1.012, MKI 0.955 vs 0.980; within neighborhood value tiers hedonic PRB -0.06 to -0.21 (value compression toward neighborhood means). 23 tests pass. |
| 5 | done | 2026-10-18 | LightGBM, 40 random configs per variant, CV 2021-2024, selection = smallest mean within-tier \|PRB\| among configs within 0.2 pts of best level-neutral MdAPE (independent: cfg 20, 2014 start, tract, 0.1340 vs best 0.1333; informed: cfg 8, 2014 start, tract, 0.1274 vs best 0.1254). Monotone constraints (sq ft, lot, baths) adopted (CV cost +0.01 / -0.03 pts). **2025 head-to-head (n = 4,197; level-neutral MdAPE / COD):** assessor 13.3% / 17.8; hedonic 12.9% / 17.9; GBM independent 12.6% / 17.8 (MdAPE diff -0.7 [-1.3, -0.0]; COD 0.0 untrimmed, -0.6 [-1.1, -0.1] union trim); **GBM informed 12.1% / 16.8 (MdAPE -1.2 [-1.8, -0.7], COD -0.9 [-1.3, -0.5]; union trim -1.3)**: the only clear uniformity win. **But both GBMs fail vertical equity**: PRD 1.04, PRB -0.09/-0.10, MKI 0.86-0.87 (assessor 1.013 / 0.022 / 0.974 meets all); within-tier PRB -0.14 to -0.25. Levels: GBM independent 0.955 (tree trend lag at the Dec 2024 boundary; 0.96 in CV too), informed 1.056 (1.01 in CV: roll 2024 revaluation shifted the prior-value/price relation). 80% CQR coverage 71.7% (independent) / 69.1% (informed), even across tiers (no Mondrian needed) but uniformly low; hedonic OLS intervals 84% overall but uneven (low tier 68%, high 94%). Sensitivities (flip first sales, non-market deeds, raw baths + era, < 50% prior roll, training start): all within +/-0.1 pt MdAPE and +/-0.1 COD; start 2020 lowers within-tier \|PRB\| 0.19 -> 0.18. Daily city files pinned behind --refresh-current. Bug fixed: value_tiers returned NaN for unrated neighborhoods (Stage 4 unrated stratum had been dropped; Stage 4 re-run). 29 tests pass. |
| 6 | done | 2026-10-18 | Detrended-target GBM (training-only hedonic time effect; Stage 5 hyperparameters fixed). New sources: MCTS GTFS (pinned), TIGER 2023 coastline and roads; parks and city-owned (municipal exemption codes 480-571) and vacant (8880/8885) parcels from roll Y-1. Five feature groups, each added alone. **Independent variant: every group helps; all groups together: level-neutral MdAPE 13.6% -> 12.9% (CV), 13.3% -> 12.0% on 2025 (-1.2 pts [-1.6, -0.7] vs its baseline), COD 17.9 -> 16.6.** Informed variant: groups add nothing (CV +/-0.1 pt). **2025 vs assessor: independent + all groups MdAPE 12.0% (-1.3 [-1.9, -0.7]), COD 16.6 (-1.2 [-1.7, -0.7]), median ratio 1.001, PRD 1.022, PRB -0.033, MKI 0.953: meets every IAAO target except COD <= 15**, where the assessor (17.8) also fails. The level fix also fixed Stage 5's vertical-equity failures (PRD 1.041 -> 1.016, PRB -0.092 -> 0.001 for the baseline) at a cost in raw accuracy (baseline MdAPE 12.6% -> 13.3%). Informed variant level 1.13 in 2025 (raw roll values carry the roll's assessment level into the detrended model); prior value relative to neighborhood median fixes it (1.005, MdAPE 12.2%). Calibration slopes < 1 for every method in CV and 2025 (assessor 0.70-0.82, models 0.80-0.98): no compression given the prediction, so no spread correction; neighborhood-relative target costs +1.7 pts MdAPE (rejected). Intervals after the level fix: independent CQR 84% (tiers 81-86%), informed 76-77% (level bias), hedonic tier-wise conformal 82.5% (tiers 80-84%, vs 66-93% global). 32 tests pass. |
| freeze | done | 2026-10-18 | Re-tune on the Stage 6 feature set, 20 configs per variant (config sampler seed 20261025), CV 2021-2024 ONLY, Stage 5 selection rule. **Frozen primary** (independent, all feature groups, detrended target, monotone sq ft/lot/baths): config 18 = tract FE, train_start 2014, num_leaves 127, learning_rate 0.02, 1,000 rounds, min_data_in_leaf 10, feature_fraction 0.8, bagging_fraction 0.7 (freq 1), lambda_l2 10, cat_smooth 30; LightGBM seed 20261018, deterministic, 4 threads. CV mean level-neutral MdAPE 12.80%, COD 18.58, within-tier \|PRB\| 0.166 (best on both among the 3 configs within 0.2 pts). **Frozen secondary** (informed, prior assessed value relative to neighborhood median): config 16 = neighborhood FE, start 2014, 127 leaves, lr 0.08, 300 rounds, min leaf 50, ff 0.8, bf 0.7, l2 5, cat_smooth 10 (CV 12.85%). Full spec: `reports/frozen/model_spec.json` (sha256 2e91821ef56b9b194d21444380c9c497a1e1e926a0793f798e3a314ba06c6b1f). 2025, reported only: primary MdAPE 11.9%, COD 16.5, median ratio 0.997, PRD 1.021, PRB -0.026, MKI 0.960. |
| prospective | done | 2026-10-18 | Frozen primary refit on clean sales 2014-2025 (47,897 rows; reference Dec 2025 trend level) predicts Jan 1 2026 values for 122,222 of 123,182 SF/duplex parcels in roll 2025 (960 lack features); 80% CQR intervals calibrated on pooled 2023-2025 (Q = 0.0487), 164 widened to contain the point. **`reports/prospective/predictions_2026.parquet` sha256 de73656af10f90732b4f0d9e9f967cb4e1ba30bce0f95179a8244d0689b6fc2a**, git tag `prospective-2026`. Score with `python -m avm.prospective score` once 2026 sales and roll 2026 are published (it verifies the hash first). |
| 7 | done | 2026-10-18 | Frozen primary refit through 2024 reproduces its frozen 2025 predictions exactly (max rel diff 0). TreeSHAP on 2025 test sales. **Group importance (mean \|SHAP\|, log price): nearby prior sales 0.228 (40%), structure 0.141 (25%), tract effect 0.107 (19%), own prior sale 0.029, distances 0.025, coordinates 0.022, city-owned/vacant 0.015 (3%).** Sanity: every feature with an expected sign matches (sq ft rho 0.99, lot 0.97, baths 0.73-0.77, age -0.92, lake distance -0.31, parks -0.26, nearby $/sq ft 0.96-0.98, own prior sale 0.85, city-owned -0.62, vacant -0.43) except distance to a major road (+0.09, below the 0.1 bar: near-zero, not reversed). Nearby $/sq ft effects jump ~10% around $145/sq ft (learned threshold). Comps for all 4,197 test sales (0.1% needed the 2 km radius): comps (a) time-adjusted only MdAPE 14.5%, COD 20.8; comps (b) model-adjusted 12.7%, 17.6; model 11.9%, 16.5; assessor 13.3%, 17.8. Disagreement flag (model vs comps (a) > 25%): 618 sales (14.7%), model MdAPE 17.2% flagged vs 11.4% not flagged, so the flag finds harder cases. |
| 8 | done | 2026-10-19 | Tests whether ERRORS differ by group, not values. ACS 2019-2023 5-year (key from .env, never written; scan of 160 output files found it nowhere). City tracts only (207): Black majority 94, White majority 65, Hispanic majority 27, mixed 21 (no Asian-majority city tract); parcel-weighted income quintiles 31-55 tracts each. CV pooled 2021-2024, 18,944 common sales; 2025, 4,197; no group under 200. **Level bias gap (mean log residual, Black minus White majority): assessor -6.2 pts, hedonic -2.2, primary -4.1, secondary -3.3; every model narrows it (primary -2.1 pts [-2.7, -1.5] vs assessor). Q1 minus Q5 income: assessor -10.1, primary -6.4 (-3.6 [-4.7, -2.6]).** Accuracy gap unchanged: MdAPE against each method's overall level is ~7.4 pts worse in Black-majority tracts and ~12.5 pts worse in Q1 for EVERY method (primary 17.4% vs 10.0%; assessor 19.0% vs 11.6%). Dispersion gaps are LARGER for the models: COD gap Black-White assessor 9.6, primary 10.6 (+0.9 [0.3, 1.5]); Q1-Q5 assessor 15.5, primary 16.9 (+1.4 [0.4, 2.6]). Calibration slopes < 1 in every group for every method (assessor 0.68-0.81, primary 0.85-0.94, lowest in Q1 and Hispanic-majority). PRB more negative for models within low-income/Black-majority groups (biased toward regressivity; shown beside the slopes). Cleaning rate (F6-F11 on in-scope residential sales, 2021-2025) is higher in Black-majority (7.5% vs 4.9% White) and Q1 (8.4% vs 4.4% Q5) tracts, mostly flips (F10 5.4% vs 2.6%): the audit evaluates a cleaner subset there. **City-owned/vacant features: KEEP** (removing them narrows the CV level gap by only 0.002, MdAPE gap +0.0001, COD gap +0.01, CV MdAPE cost 0.02 pts). |
| 9 (review) | built, not published | 2026-10-19 | Frozen primary values every SF/duplex parcel in roll 2024 at Jan 1 2025 (verified identical to the frozen 2025 predictions), 80% CQR ranges (pooled 2022-2024, Q 0.0622), compared with roll 2025. Exclusions from the list and tax pool: sq ft changed between rolls 482, exempt 34, assessment collapsed 17, building value gone 14; 121,471 included. Citywide level (median assessed / model) 0.893. Outside the 80% range 13,092; comps (a) agree 12,682; top 1,000 by |log relative gap|: 935 below, 65 above. **356 of the 935 below-model listings have a 2025 assessment unchanged from 2024, while citywide the median assessment rose 14.6% and only 0.5% stayed flat** (for example 41 adjacent parcels in tract 2200 all at $51,200): likely special valuations (condition-based or legally restricted) the model cannot see; shown as context flags, not excluded. Group tax shifts (median assessed / market share, gross rate $22.42 provisional): Black majority +2.7% [2.5, 2.9] clears the model's CV bias (-1.6 [-2.2, -0.9]) and Q3 +1.6 clears its bias; all others within the model's bias. **Model-free check (assessor median ratio on 2025 sales vs citywide 0.892): Black majority -1.0 [-2.2, +0.3] (inconclusive), Q3 +0.3 (inconclusive); White majority +1.5 and Q5 +1.8, opposite to their model-based shifts.** The parcel-level group shifts are dominated by model bias; the page states no group claim. Page: `reports/stage9/page/assessment_check.html` (built by `python -m avm.stage9_page`). |
| 9 (rev) | built, not published | 2026-10-19 | Owen's decisions: (1) a group claim needs BOTH the bias guardrail and the 2025 sales check; none passes, so the page states no group claim. (2) Set aside, not ranked (they stay in the tax pool): assessment unchanged from 2024 (433), land use changed in roll 2025 (22), units inconsistent with land use (15; for example a parcel coded duplex with 12 units in roll 2025); 468 listable parcels in all (446 below, 22 above). Re-ranked main top 1,000: **893 below the model, 107 above** (was 935/65); relative gap 0.115 to 4.568; median |tax shift| $1,740. (3) Tax rate: Milwaukee NNS (Dec 2025) reports net 22.93 with no tax year, against 20.80; recorded as a conflict, still provisional until Owen confirms the 2025 gross rate. (4) SAFMR file not yet in raw/hud_safmr. (5) Two pages from one template: `assessment_check_private.html` (addresses, tax keys, 1,000 ranked + 468 set aside) and `assessment_check_portfolio.html` (map, aggregate counts, group table, methods; the builder raises if any listed tax key or address appears in it). Tracts under 30 homes ship no values in either. |
| 9 (rev 2) | done, not published | 2026-10-20 | Commercial building type rule added to the set-aside section (roll 2024 or 2025 building type starts with C, not a documented residential style code 01-22): 37. Set aside now 491 (460 below, 31 above). Main top 1,000: **894 below, 106 above**; median abs tax shift $1,725. Student-rental-area context flag (centroid within 0.75 mi / 1,207 m of the largest TIGER 2023 area-landmark polygon for Marquette Univ and Univ of Wisconsin Milwaukee; new source `tiger_arealm_2023`, fetched with new `acquire --only`): **87 of the top 1,000 (60 above, 27 below)**; 4,213 of 121,471 homes citywide. Publishing held until Owen confirms the 2025 gross rate; SAFMR file pending. |
| comps fix | done | 2026-10-20 | Shared `avm.comps`: same property type; tiers 0.5 / 1 / 1.25 mi within 12 months, then 24 (window first), stop at the first tier with >= 3; flips, IAAO ratio outliers and price/sq ft outliers excluded; SHAP-weighted ranking kept; tier recorded per subject. 2025 re-score (same 4,197 sales, all covered before and after): comps (a) level-neutral MdAPE 14.51% -> 14.20%, COD 20.81 -> 20.79, within 10% 36.5% -> 37.6%; comps (b) 12.68% -> 12.44%. 99.0% of test sales use the tightest tier. Stage 9: listable 12,682 -> 12,696, set aside 491 -> 479, top 1,000 split unchanged (894 below, 106 above; 997 of 1,000 the same parcels; 961 found comps at 0.5 mi / 12 months, 39 at 1 mi). Stage 7 now also writes `interval_coverage_2025.csv` (frozen primary: 85.2% for nominal 80%; low tier 81.0%). |
| 10 (sample) | review | 2026-10-20 | `avm.writeup` renders from `docs/templates/` with every number a placeholder traced in `reports/writeup/numbers.json` (80 numbers); lint refuses digits outside placeholders and whitelisted method constants, and em or en dashes. Headline head-to-head computed on identical 2025 sales (`reports/writeup/headline_2025.csv`). Sample = Key results, Summary, Results (~900 words) at `reports/writeup/sample/README_sample.md`. |
| 10 | review | 2026-10-20 | README.md (~2,400 words) and docs/methods.md generated by `python -m avm.writeup` (169 traced numbers). Independent check by a fresh subagent: 25 findings, no number mismatches; all fixed and re-checked. Main corrections: vertical equity (frozen model fails PRD/PRB in CV, within-tier PRB 2025 low -0.230 / mid -0.147 vs assessor -0.054 / +0.019); fairness level gap narrows in CV but is larger than the assessor's on 2025 (+1.4 pts [+0.1, +2.7]); hedonic baseline is neighborhood FE from 2020; feature table filtered to the primary; comps described as using no model prediction (SHAP-weighted selection); student-rental attribution softened to untested; tax credits: school levy credit likely scales with value (net rate would make shifts 7.2% smaller), lottery credit marked to verify. Pipeline now runs Stages 9-10 and `freeze.main(verify=True)`, which re-tunes, requires the committed spec to be reproduced exactly and writes only to reports/frozen/verify/. |
| rebuild | passed | 2026-10-20 | `python -m avm.pipeline` from a copy of raw/ into an empty data folder (`AVM_DATA_DIR=<rebuild-data>`), 12:11 to 13:31. Clean sales table byte-identical (sha256 401f593c...); freeze verify reproduced the committed spec exactly (all fields but frozen_at_commit) and the 2025 predictions (max rel diff 0.0); every CSV from Stage 2 to 9, README, methods.md and numbers.json identical; prospective hash verified. Found and fixed: (1) profile_data could not read the GTFS zip (Stage 1 never re-run after Stage 6); (2) Stage 3 set matplotlib rcParams globally, restyling Stage 7-8 figure axes when run in one process (now rc_context; committed figures unchanged). Stage 1 reports refreshed (stale: written before later sources and the exempt_type field, and before the daily files were re-pinned on 2026-10-18). Log: `reports/rebuild/rebuild_log.txt` (checked: no key). |
| 10 (rev 2) | review | 2026-10-20 | Owen: between-parcel tax shifts use the NET rate (school levy credit reduces each bill in proportion to assessed value, Wis. Stat. 79.10; supersedes the gross-rate entries above); lottery and first dollar credits excluded (capped value); 2025 net rate 20.80 vs 22.93 still to confirm. Median ranked shift $1,725 (gross) -> $1,600 (net). Student-rental note keeps 'may reflect the model; untested' and adds concentration (3.5% of homes, 60 of 106 above-model listings). Second fresh check (no number mismatches; 21 findings, all addressed) found: the group guardrail is size-only (ignores direction); corrected for direction, Hispanic-majority and Q1 look under-assessed and White-majority and Q5 over-assessed, agreeing with the model-free sales check (rule change pending Owen); the Black-White level-gap narrowing holds for mean log ratio but not median ratio (2025 model gap larger by +3.6 pts [+2.0, +5.0]). Code fixes: building-value-gone rule now uses roll 2024 land (14 -> 19); homes with no positive roll 2025 assessment (196) now reported as an exclusion. Stage 9 re-run in the rebuild folder after the fixes: all six tables match (reports/rebuild/stage9_recheck.txt). README trimmed to ~2,370 words. |
| 10 (rev 3) | review | 2026-10-20 | Tax rate CONFIRMED by Owen: Milwaukee County Treasurer comparative rate table (dated 2/18/2026, 'rate to determine taxes due per property') lists City of Milwaukee $22.93 for 2024 and $20.80 for 2025; 20.80 used, 22.93 noted as the 2024 rate; table added to the manifest (tax_rates_county_2016_2025, sha256 38763019...). Guardrail switched to direction-aware (Owen): claim needs (a) bootstrap CI of shift + model CV bias excluding zero, (b) model-free 2025 sales check agreeing with a CI excluding zero, (c) both surviving tercile time adjustment (new `stage3_assessor.tercile_target_prices`, applied to 2021-2025). Result: Hispanic-majority (corrected -3.4% [-4.5, -2.1], sales -4.6% [-7.9, -2.3]) and Q1 (-2.7% [-3.8, -1.6], sales -6.0% [-8.8, -1.7]) under-assessed, all three checks pass; White-majority and Q5 pass (a) and (b) but fail (c) (tercile-adjusted corrected shifts -0.2% and -0.1%, intervals include zero): reported as consistent with assessments lagging faster appreciation, not a valuation pattern; Black-majority: corrected +1.1% [0.6, 1.6] but sales check -1.3% [-2.4, -0.4], opposite direction, no claim. README: four plain summary bullets, new 'What this is good for' section, the 2025 fairness negative stated once in the fairness section. Stage 9 re-run in the rebuild folder: all six tables match. |
| 10 (rev 4) | review | 2026-10-20 | Persistence (Owen): the model-free sales check with tercile time adjustment on rolls 2021-2024 (`reports/stage9/group_sales_check_by_year.csv`; own seed, 2025 results unchanged). Hispanic-majority holds in 4 of 4 earlier rolls (2021 -5.0, 2022 -3.4, 2023 -4.4, 2024 -8.0, 2025 -4.7; all intervals below zero): PERSISTENT. Lowest income quintile holds in 2 of 4 (2021 -6.3, 2023 -3.6; 2022 and 2024 intervals include zero): described as a 2025 result. Dollars at $20.80: Hispanic-majority typical home (median assessment $158,300) about $111 a year less, $1.66M across 14,476 homes, about $15 a year more on each of the other 106,991; Q1 typical home ($114,400) about $65 a year less, $1.66M across 23,609 homes, about $17 more on each other home; groups overlap, so amounts do not add. README (2,485 words), methods (by-year table, dollar method) and page (passing-groups card) updated. Stage 9 rebuild recheck: all seven tables match. |
| CI (#5) | done | 2026-10-20 | `.github/workflows/ci.yml` (Ubuntu, Python 3.13, `requirements.lock`): empty `AVM_DATA_DIR` (no real data), frozen and prospective files made read-only, prospective hash check, ruff 0.17.1 with the minimal rule set (E9, F63, F7, F82, F401; 9 unused imports removed), docs dash check (`avm.ci_checks`), pytest (53 tests; 21 new synthetic tests for comps tiers and exclusions, the guardrail rule, portfolio privacy, the frozen-spec diff, the write-up lint and the dash check), then data folder still empty and frozen/prospective unchanged. Guardrail verdict, portfolio leak check and spec diff extracted into pure functions; after the refactor all seven Stage 9 tables and both page files are byte-identical to before, and both published Artifacts match the local files apart from the publishing wrapper. README has a CI badge (via the template). |
| privacy | done | 2026-10-20 | Owen: no history rewrite; history still holds parcel addresses, tax keys and owner names (scan of all 533 blobs, 2026-10-20), so this private repo stays private and a fresh public repo is planned. Code now keeps identifiers out of tracked outputs: the data dictionary blanks sample values for owner, mailing, address, tax-key, geometry and legal fields; Stage 7 examples drop addresses; Stage 2/3 outlier logs drop the tax key; Stage 9 summaries drop tax key, address and ZIP; the divergence lists and the private page are written to `<data>/reports_private/stage9/` (gitignored at their old paths, untracked). Regenerated: only the three outlier logs changed among the tables (tax-key column gone); the clean table, target prices, Stage 7 and Stage 9 tables and the portfolio page are byte-identical; the private page in the data folder is byte-identical to the published one. CI gained `python -m avm.ci_checks privacy` (tax keys, street addresses, owner-field values in tracked files, outside reports/prospective/; locally also exact-matches the 3,023 private-list values). Three real-looking addresses in a docstring and a test replaced by synthetic ones. OpenTimestamps: `reports/prospective/predictions_2026.parquet.ots` stamps sha256 de73656a... via 4 public calendars (`python -m avm.timestamp`; the `ots` client fails on Windows, so the library is used directly). Proof is pending Bitcoin confirmation: run `python -m avm.timestamp upgrade` after a few hours and commit the upgraded .ots. |
| public snapshot | built and verified, not pushed | 2026-10-20 | The project repository stays private; the public repo is oekrueg05/milwaukee-avm. Owen: include the decision log (this file, as docs/decision-log.md) and CLAUDE.md, scrubbed of Artifact URLs and personal paths; MIT licence for code; README 'Data and license' section; recreate open issues #1-#4 and #6 there (as #1-#5). Snapshot built outside both repos by `<data>/public-build-tools/build_public.py` (git archive of private HEAD, scrub, one squashed commit, tag prospective-2026) at `<home>/public-build/milwaukee-avm`; verified by `verify_public.sh`: history scan of every blob (no addresses, real tax keys or owner values; only synthetic test values and redacted field names), no Artifact URLs or personal paths, prospective hash check, ruff, dashes, privacy, 57 tests, data folder untouched. Next (tomorrow): `python -m avm.timestamp upgrade`, commit the upgraded .ots here, rebuild and re-verify the snapshot, then Owen's go to create and push the public repo. |
| timestamp | done | 2026-10-21 | OpenTimestamps proof upgraded (about 14 hours after stamping): all four calendars returned Bitcoin attestations, blocks 972397 (two calendars), 972423 and 972434. Checked against public block data (blockstream.info) with `python -m avm.timestamp verify`: the proof's digest equals the file's sha256 (de73656a...) and each commitment equals its block's Merkle root. Earliest block time 2026-10-21 00:35:33 UTC, so predictions_2026.parquet provably existed by then. `avm.timestamp upgrade` fixed (the first version called a method the library does not have). external_sources.md cites Johnson (2023) in full. |
| history check | done | 2026-10-21 | `python -m avm.ci_checks history` applies the privacy rules to every text blob reachable from any ref (every committed version of every file, reports/prospective/ excluded). New CI job `history` (full fetch, all tags) runs only in oekrueg05/milwaukee-avm: on the private project repository it fails by design (49,283 findings in past versions, mostly the old Stage 1 data dictionaries), which is why that repository stays private. Tests in `tests/test_ci_checks.py`. CI on c303d38: 57 passed, prospective check verified. |
| snapshot repo | created private | 2026-10-21 | oekrueg05/milwaukee-avm created PRIVATE (the old name redirected to this repo after the rename, so the new repo was created and confirmed empty before any push). Pushed snapshot cf7aea4 (built from 79f8213) and tag prospective-2026. CI on both pushes: test job 59 passed, prospective check verified (122,222 rows, sha256 de73656a...), privacy check passed across 248 tracked files; history job passed on 241 text blobs across all refs. On this repo the history job is skipped as designed. Not public; no issues created. |
| public | done | 2026-10-21 | Clone audit: 13 git repos under the user folder (only drive C:); two hold either history: this project (origin the private project repository) and the public-build snapshot (origin milwaukee-avm). The rebuild and data folders are not repos; no extra worktrees or bundles. Pre-push hook added here (untracked) refusing oekrueg05/milwaukee-avm in https, ssh and mixed-case forms; the private project repository allowed. Pushed 8c9c862. Switched oekrueg05/milwaukee-avm to public; created issues #1-#5 from the drafts; description and topics (avm, real-estate, machine-learning, lightgbm, property-tax, fairness) set. CI on the public repo (manual run): test 59 passed, prospective check verified, privacy passed; history passed on 241 text blobs. |
| MCP (#6) gate 1 | done | 2026-10-21 | `avm.address`: MAI loaders (Stage 9's `addresses()` moved here as `display_addresses`, unchanged) and pure-Python parsing and matching for the planned `value_parcel` tool. Parsing: units dropped, full words to MAI abbreviations, ordinals, aliases checked against the MAI (MARTIN L KING JR, CESAR E CHAVEZ, ST PAUL, ST CLAIR), short ranges (2876-78). Matching: same side of the street only (odd and even numbers face each other; without this 8,952 valued parcels matched a neighbor across the street), unsuffixed numbers win, ZIP as tie-breaker, ranges wider than 20 match their end numbers only. Round trip on the 122,144 valued parcels with an MAI address: 121,947 resolve to their own parcel, 36 share an address with another valued parcel (listed), 161 have only suffixed numbers that the Stage 9 display drops (the server will show suffixes; Stage 9 unchanged); 78 valued parcels have no MAI address. Stage 9 full run plus pages after the move: all 11 outputs rewritten and byte-identical (sha256), private lists and both pages included. Tests: `tests/test_address.py`. |
| MCP (#6) gate 2 | done (sample) | 2026-10-21 | Owen's gate 1 change: a plain number with suffixed variants on other parcels (124 and 124A) returns multiple_matches listing all (`address.resolve`). Round trip on 122,144 valued parcels with an MAI address: 121,520 resolve to their own parcel, 611 suffix-variant and 13 shared-address lists include it, 0 wrong or missing. `avm.serving build --sample` (500 parcels): refit of the frozen primary matches all 122222 of 122222 values exactly (train rows 47897), booster sha256 265ccacade7f5be5... in meta; SHAP 6.72 ms per parcel, additivity checked; rows and frame 129.7s, refit 19.1s. Sample database 33.2 MB; projected full 144 MB and SHAP 13.7 min. `avm.lookup` (stdlib only) builds responses: values and ranges rounded to $1,000, hybrid drivers (location, nearby sale prices, own prior sale and city/vacant grouped; structure individual), comps with suffix-aware addresses. Tests for serving and lookup come in gate 4 as planned. |
| MCP (#6) gate 3 | done | 2026-10-21 | Owen: own prior sale stays a separate driver and shows that sale's date and price when it is in the top 3; city/vacant driver labelled "nearby vacant or city-owned lots". Full `avm.serving build` from clean commit 3d374e3: refit exact match 122222 of 122222 (train rows 47897), booster sha256 265ccacade7f5be5...; timings rows and frame 383.6s, refit 37.9s, SHAP 2141.5s (17.52 ms per parcel, slower than the first sample's 6.7 under machine load), comps and addresses 41.3s. Own prior sale recovered for 36,579 of 36,579 parcels with one; comps for 122,207 of 122,222. Database 143.5 MB in <data>/serving/; `python -m avm.serving check` passed. |
| MCP (#6) gate 4 | done | 2026-10-22 | `avm.mcp_server`: MCP Python SDK 2.3.0 (`MCPServer`, stdio), one read-only tool `value_parcel` with the description Owen specified; `avm.serving_db` holds the schema and writer so tests need no model stack. `tests/test_mcp_server.py` (17 tests, synthetic database): every status, tampered predictions and tampered database refused, database from another prediction file refused, response key whitelist, schema has no owner fields, calls through the SDK in-memory client. `docs/mcp.md` (connect, startup checks, booster recovery, driver grouping, address rules, demo). README: Reproduce sentence on pinned daily files moved to methods (with the `--refresh-current` flag), link to docs/mcp.md added. `mcp` and 26 dependencies pinned in requirements.lock (no existing pin changed; pywin32 Windows-only). Connected in Claude Code (local scope) and the three demo prompts run live from a neutral folder with only this server. The live run found a pre-existing bug in `avm.comps`: the subject's own sale can be its comp (see open items). |
| self-comp fix + subject details | done | 2026-10-22 | Owen: exclude the subject parcel's own sales from its comp pool; add subject details and a separate subject_last_sale to value_parcel. `avm.comps.comps_for` drops pool sales with the subject's tax key (raises if a row has no tax key); 2 new synthetic tests. **Actual results equal the impact counts** (Stage 7 and Stage 9 outputs, both private lists and the served comps are byte-identical to the counterfactual run): comps (a) level-neutral MdAPE 14.20% -> 14.27% (README 14.3%), comps (b) 12.44% -> 12.45%, model 11.94% unchanged, disagreement flags 608 -> 619; listable 12,694 -> 12,690; top 1,000 same parcels in the same order (5 show a different comp value); set aside 479 same parcels, one comp value changed, and one tied pair (identical relative gap 0.2584, all fields identical) swapped places because the Stage 9 ranking sort is not stable (pre-existing; not changed, needs a tie-breaker if exact order matters). Write-up: 183 numbers traced, lint and dash check pass, templates and page state the exclusion. Both pages republished to the same Artifact URLs (version 2 each; live versions matched the local files before regeneration). Serving rebuilt from clean commit 9df97ba: refit exact match 122222 of 122222, booster sha256 265ccacade7f5be5... (unchanged), SHAP 1353.7s; 0 self-comps; all 4,099 former self-comp parcels have new comps (22 change tier, 56 fewer comps, minimum 3, none lose comps); comps for 122,207 of 122,222 as before. `value_parcel` adds subject_details (finished sq ft, bedrooms, baths, year built, lot size from roll 2025; 192 parcels take lot size from roll 2022-2024, noted) and subject_last_sale (most recent clean sale, 39,394 parcels); for 1,441 parcels a later or only sale removed by Stage 2 cleaning is named in a note with date and filter, never price (1,139 resales within 12 months). Key whitelist, tests (18 MCP tests, 92 in all) and docs/mcp.md updated; CI green. Live demo prompt 1 (the duplex range address in docs/mcp.md, neutral folder, only this server): comp 1 was the home's own 2025 sale at 0 mi; it is now another duplex 0.28 mi away, and the home's sale appears as subject_last_sale. Frozen model, spec and prospective file untouched. |
| tie-breaker + excluded last sale | done | 2026-10-22 | Owen: (1) Stage 9 ranks by gap descending, then tax key ascending (stable sort). Re-run of Stage 9 and both pages: every tracked Stage 9 output and both page files byte-identical; top 1,000 and set-aside lists same parcels, same positions, same contents (one tied group, 2 set-aside rows, already in tax-key order from the previous run); both pages republished to the same URLs (version 3). (2) When the latest sale on record was removed by cleaning, `subject_last_sale` is that sale (date, recorded price) with the note "excluded from the model's training data: <reason>; shown for context, not used in the estimate", and the latest clean sale follows as `latest_clean_sale` (1,441 parcels, all with a price; 1,139 resales within 12 months). Serving rebuilt from clean commit 3138e61: refit exact match 122222 of 122222, booster unchanged, 0 self-comps. Tests, whitelist and docs/mcp.md updated. |
| 9 | planned | 2026-10-19 | Plan presented; awaiting "go". |
| (ops) | done | 2026-10-18 | Repo moved from `<home>\Projects\milwaukee-avm` back to the a synced folder Desktop (`<repo>`) at Owen's request. Data (`<data>`) and venv (`<venv>`) unchanged; editable install re-pointed. |
