<!-- Generated file. Edit docs/templates/methods.template.md and run `python -m avm.writeup`. -->
# Methods

Detail behind the [README](../README.md). Every number is generated from a report file or a code constant; sources are in [`reports/writeup/numbers.json`](../reports/writeup/numbers.json). The dated decision log is in [`docs/decision-log.md`](decision-log.md).

## Scope and data

City of Milwaukee single-family (land use 8810) and duplex (8820) homes. All inputs are public downloads, recorded with their source URL and sha256 in `reports/stage1/manifest.csv`:

- **Sales** (City of Milwaukee open data): yearly property sales files, 2002 to 2025. The 2020-and-later files are the city's filtered ("valid") extracts; the city does not document the filter.
- **Assessment rolls** (Master Property File, historical): yearly snapshots 2008 to 2025, read through an explicit field crosswalk across four file layouts. A field is used for a year only if it passes a same-parcel continuity check against neighboring years, or is excluded with the evidence recorded.
- **Parcels, assessment neighborhoods, building permits, Master Address Index** (city open data); **TIGER/Line 2023** tracts, coastline, roads and area landmarks (US Census); **MCTS GTFS** transit stops; **ACS 2019-2023** tract demographics, for the fairness audit only.

## Leakage design

- **Assessment-year lookup.** Rolls are matched to assessment years by an explicit lookup built from consecutive-roll checks, never by file name; only rolls confirmed by that check (2013 to 2025) are used.
- **Features from the roll before the sale.** A sale in year Y takes its characteristics from roll Y-1; if a field is missing there, only earlier rolls may fill it. The assessor's benchmark for the same sale is roll Y (valuation date Jan 1 of Y). Building characteristics in the sales file reflect the file extract, not the sale, and are never used.
- **Prior-sales features** use only sales before the cutoff: for training rows, before the sale's own month; for scored rows, before Jan 1 of the scored year. A home's own prior sale must be more than 12 months before the sale and before the cutoff; the code asserts both.
- **Known look-ahead in location features.** Transit stops come from the current schedule and park locations from one year's parcel outlines, for every year, so they are not as of the sale; how much they changed over the period is not measured.
- **Time-based validation.** Training uses 2014 to 2024; 2025 is the held-out year. Cross-validation rolls forward a year at a time over 2021 to 2024. No random splits.
- **Test-set discipline.** 2025 results were reported at Stages 3 to 6 and influenced design choices then. From the freeze on, every model selection used cross-validation on 2021 to 2024 only; the comparable-sales rule was specified after the freeze and has been evaluated only on 2025. The frozen spec is `reports/frozen/model_spec.json` (sha256 begins 2e91821ef56b).

## Cleaning

64,855 sales-file rows from 2014 to 2025 enter; 48,270 cleaned single-family and duplex sales remain. Each filter is logged with the rows it removes (`reports/stage2/filter_log.csv`):

| filter | removed | remaining |
|---|---|---|
| F0 loaded (blank rows already dropped) | 107 | 64,748 |
| F1 sale date outside file year | 6 | 64,742 |
| F2 exact duplicate row | 210 | 64,532 |
| F3 missing tax key, date or price | 9 | 64,523 |
| F4 price < $10,000 | 99 | 64,424 |
| F5 property type not Residential | 13,676 | 50,748 |
| F6 no roll Y-1 record | 32 | 50,716 |
| F7 roll Y-1 land use not SF/duplex | 891 | 49,825 |
| F8 new construction or no building in roll Y-1 | 61 | 49,764 |
| F9 multi-parcel sale (same date and price, all parcels within 200 m) | 88 | 49,676 |
| F10 resale < 12 months (flip) | 1,346 | 48,330 |
| F11 two sales of one parcel in the same day/month | 60 | 48,270 |

Quick flips are found from repeated tax keys across the sales files (a resale within 12 months); the resale is dropped and the first sale is flagged. Multi-parcel sales are dropped only when every parcel lies within 200 m of the others. Deed type, permit activity and price-per-square-foot outliers are kept as flags for sensitivity checks rather than filters (training drops price-per-square-foot outliers).

## Target prices and the ratio study

Every method is scored against one shared target per sale: its price moved to the Jan 1 trend level of the sale year, with a neutral month-of-year. The time adjustment is a piecewise trend with month-of-year effects estimated on 2014 to 2024; the 2025 part of the index is used only to adjust 2025 prices for evaluation and never feeds training.

IAAO ratio statistics (median ratio, COD, PRD, PRB, MKI) come with 2,000-draw bootstrap intervals; method differences use paired bootstraps on the same draws. The IAAO trim (3 times the interquartile range on the assessor's ratio) is used only for the assessor's standards check; head-to-head comparisons are untrimmed. The headline accuracy measure is **level-neutral** median absolute error: each method's ratios are divided by that method's own median ratio first, so a method is not rewarded for its overall level.

## Models

**Hedonic baseline.** Ordinary least squares on log price with size, lot, age bands, bedrooms, baths, stories, style, duplex, a centered spline time trend, month-of-year effects and assessor-neighborhood fixed effects (units with few training sales are folded into the assessor region), trained on sales from 2020 on; location unit and start year were chosen by its own cross-validation.

**Main model (independent).** LightGBM on a detrended target: log price minus a time effect estimated from the training years only, so the model predicts directly at the December trend level of the year before the valuation date, with no time features. Increases in finished area, lot size and baths are constrained to never lower the value. Its 27 features:

| feature | what | proxy note |
|---|---|---|
| l_area | log finished sq ft (roll Y-1) | size; no proxy concern |
| l_lot | log lot sq ft; lot_missing flags a filled median | size; no proxy concern |
| beds | bedrooms, clipped 1-7 (MPROP only, never the sales file) | no proxy concern |
| full_baths | full baths (2019+ rolls; 0 + baths_missing before) | no proxy concern |
| half_baths | half baths (as above) | no proxy concern |
| style | assessor building type code; codes with < 30 training sales -> other | no proxy concern |
| duplex | roll Y-1 land use 8820 | property type; no proxy concern |
| loc | location fixed effect: assessor neighborhood or census tract, units with < 30 training sales folded into the assessor region | LOCATION: allowed by rule 7 but can proxy for protected-class composition; equity audited in Stage 8 |
| age | numeric age (GBM; the hedonic uses age_bin) | building vintage; correlates with neighborhood history, monitored in Stage 8 |
| stories_num | numeric stories, NaN if missing (GBM) | no proxy concern |
| x_m, y_m | parcel centroid (UTM m) | LOCATION: allowed; fine-grained location can proxy for neighborhood composition; audited in Stage 8 |
| d_downtown_m, d_lake_m, d_major_road_m, d_transit_m, d_park_m | distances to amenities/disamenities | location; low proxy risk; audited in Stage 8 |
| knn_lppsf, knn_dist_m, nbhd_lppsf_24m | prior nearby sale prices per sq ft (before the cutoff, detrended) | market evidence; inherits any historical price discrimination in the area; audited in Stage 8 |
| prior_lp_detr, months_since_prior, has_prior_sale | the parcel's own prior sale (> 12 months before, before the cutoff) | no proxy concern |
| n_city_200m, n_vacant_200m | city-owned (municipal exemption incl. tax-deed foreclosure) and vacant parcels within 200 m, roll Y-1 | STRONG PROXY RISK: disinvestment and foreclosure track racial composition in Milwaukee; flagged, audited in Stage 8 |

**Selection and freeze.** 20 configurations per variant were sampled and scored on cross-validation 2021 to 2024 by mean level-neutral error; among configurations within 0.2 pts of the best, the one with the smallest mean absolute within-tier PRB was chosen. The frozen primary uses 127 leaves, learning rate 0.02, 1000 rounds, minimum 10 rows per leaf, feature fraction 0.8, bagging fraction 0.7 and L2 penalty 10, with LightGBM seed 20261018 and deterministic training.

**Assessment-informed variant.** A separately tuned LightGBM (its own configuration and location unit, in `reports/frozen/model_spec.json`) with the prior roll's total and land assessed values added, each expressed relative to its neighborhood's median so that each roll's overall assessment level does not leak into a detrended model.

**Nearby-sales features.** The 10 nearest prior sales within 24 months (their time-adjusted price per square foot and distance), a neighborhood price-per-square-foot level, and the home's own prior sale.

## Intervals

Conformal quantile regression: LightGBM quantile models for the tenth and ninetieth percentiles, widened by a margin (0.0622 in log price) calibrated on pooled out-of-sample predictions for 2022 to 2024, then widened where needed so every range contains its point value. Coverage on 2025: 85.2% overall; 81.0%, 86.7% and 87.4% by value tier.

## Comparable sales

From the eligible pool (clean sales of the same property type before the valuation date, excluding first sales of a flip, IAAO ratio outliers and price-per-square-foot outliers), the search tries 0.5, 1, 1.25 mi within the most recent 12 then 24 months, and stops at the first step with at least three sales. Up to five are kept, ranked by a structural distance weighted by each feature's SHAP importance. The step used is recorded for each home as a confidence field. Comp value (a) is the median comp price moved to the valuation date; (b) also adjusts each comp by the model's view of the difference. Only (a) uses no model prediction, though it is not fully independent: comps are ranked with the model's SHAP weights, and the pool drops sales flagged as outliers on the assessor's own ratio. Under the previous rule (fixed two-year pool, no exclusions) comp value (a) had a level-neutral error of 14.5%; it is now 14.2%.

## Explanations

TreeSHAP contributions from LightGBM, in log price. Global importance is reported by feature group first, because coordinates, distances and nearby-sales features are correlated and split credit between them.

## Fairness audit

City census tracts are grouped by majority race or ethnicity (more than half of residents; groups with fewer than 5 tracts are folded into "no majority") and by median household income quintile, weighted by parcels. For each group and method: valuation level (mean log ratio), median error against the method's overall level, COD, PRD and PRB, and calibration slopes, with bootstrap intervals and paired differences against the assessor. The share of in-scope sales removed by the cleaning filters (F6 to F11, sales of 2021 to 2025) is reported by group: in Black-majority tracts 7.5% of in-scope sales are removed (flips 5.4%) against 4.9% in White-majority tracts (flips 2.6%); lowest income quintile 8.4% against highest 4.4%.

## Divergence and tax dollars

Each single-family and duplex home in roll 2024 that the model can score (122,214 of 123,168) is valued by the frozen model at Jan 1 2025 with its 80% range and compared with roll 2025. **Relative assessment** = (assessed / model value) / the citywide median of that ratio (0.893). A home is **listable** when its assessment divided by the citywide level falls outside the model's 80% range and the median of its time-adjusted comps points the same way; the main list ranks the largest gaps by the size of the log relative assessment.

Excluded from both the list and the tax pool:

| rule | definition | parcels |
|---|---|---|
| no positive roll 2025 assessment | roll 2025 total missing or zero (dropped before the rules below) | 196 |
| exempt in roll 2025 | any exemption code in roll 2025 (pays no or partial tax) | 34 |
| building value gone in roll 2025 | improvements < 10% of roll 2025 total while roll 2024 had a building (likely demolition, fire or condemnation; model features are stale) | 19 |
| assessment collapsed | roll 2025 total < 50% of roll 2024 total | 17 |
| sq ft changed between rolls | finished sq ft differs between rolls 2024 and 2025 (model features are stale) | 481 |

Set aside in their own section (still in the tax pool). The median home's assessment changed by 14.6% from 2024 to 2025 and 0.5% of homes kept exactly the same value. Counts below are among the listable homes, and a home can meet several rules (479 unique homes):

| rule | definition | parcels |
|---|---|---|
| assessment unchanged from 2024 | roll 2025 total equals roll 2024 total, while almost every assessment changed citywide (likely a condition-based or legally restricted value) | 423 |
| land use changed in roll 2025 | roll 2025 land use differs from roll 2024, which the model used | 22 |
| units inconsistent with land use | residential units in roll 2024 or 2025 above 2 for single-family (8810) or above 4 for duplex (8820) | 14 |
| commercial building type | building type in roll 2024 or 2025 is a commercial code (starts with C), not one of the documented residential style codes 01-22 (MPROP documentation, Appendix E) | 36 |

Shown next to a listed home, without moving it:

| flag | definition |
|---|---|
| N parcels in this tract have this exact assessment | 5 or more included parcels in the same tract share this exact roll 2025 total |
| student-rental area (within 0.75 mi of the Marquette or UWM main campus) | parcel centroid within 0.75 mi of the main campus polygon (largest TIGER 2023 area landmark, MTFCC K2540) of Marquette University or UW-Milwaukee. Income-producing student rentals carry value the model's features cannot see, so divergence here may reflect the model (untested; comps there point the same way as the model) |

**Tax.** Revenue-neutral within the single-family and duplex class: a home's fair assessment is its model value times (total assessed / total model value), and its shift is (assessed minus fair) times the 2025 net rate ($20.80 per thousand dollars, from the Milwaukee County comparative rate table, manifest key `tax_rates_county_2016_2025`; $22.93 in the same table is the 2024 rate; gross $22.42 for context). School levy credit: included through the net rate: it reduces each bill in proportion to assessed value (Wis. Stat. 79.10), so the net rate is the marginal tax per assessed dollar. Not applied: the first dollar credit (every improved parcel) and the lottery and gaming credit (owner-occupied principal dwellings only); both are computed on value only up to a statutory cap (Wis. Stat. 79.10(9)(bm), (9)(bn), (11)), so above the cap they do not grow with the assessment ([statute](https://docs.legis.wisconsin.gov/statutes/statutes/79/ii/10)).

**Groups.** A group's shift is the median of log(assessed / fair). Because the fair share comes from the model, a model that under-values a group by b makes it look over-assessed by about b, so the corrected shift is the shift plus the model's own cross-validated bias for the group (relative to its citywide bias), with a bootstrap interval from independent draws of both. The model-free check is the assessor's median ratio on 2025 sales in the group relative to the citywide ratio. A group is called over- or under-assessed only if (a) the corrected interval excludes zero, (b) the sales check's interval excludes zero in the same direction, and (c) both still hold when every target price is recomputed with separate trend-only time adjustments for each tercile of prior assessed value (the Stage 3 sensitivity, applied to every year used). A result that passes (a) and (b) but not (c) is reported as consistent with assessments lagging faster appreciation in some areas, not a valuation pattern. Passing all three: Hispanic-majority tracts and the lowest income quintile are under-assessed, assessed and taxed below their market-value share. Hispanic-majority tracts: persistent (the tier-adjusted sales check holds in 4 of 4 earlier rolls); a typical home (assessed $158,300) pays about $111 a year less, $1.7 million across 14,476 homes, and the fixed levy puts about $15 a year more on each of the other 106,991. The lowest income quintile: a 2025 result (the tier-adjusted sales check holds in 2 of 4 earlier rolls); a typical home (assessed $114,400) pays about $65 a year less, $1.7 million across 23,609 homes, and the fixed levy puts about $17 a year more on each of the other 97,858. The groups overlap, so the amounts do not add. White-majority tracts and the highest income quintile fail only the tier-specific time adjustment: consistent with assessments lagging faster appreciation in some areas, not a valuation pattern. Black-majority tracts: the two checks conflict (the bias-corrected shift, +1.1% [+0.6%, +1.6%], says over-assessed; the 2025 sales check, -1.3% [-2.4%, -0.4%], says under-assessed; both intervals exclude zero), so no claim is made.

| group | homes | shift [95% CI] | model bias [95% CI] | corrected shift [95% CI] | assessor ratio on 2025 sales vs city | corrected, tercile time adj. | sales check, tercile | (a) / (b) / (c) | result |
|---|---|---|---|---|---|---|---|---|---|
| Black majority | 54,589 | +2.7% [+2.5%, +2.9%] | -1.6% [-2.0%, -1.1%] | +1.1% [+0.6%, +1.6%] | -1.3% [-2.4, -0.4] | +1.8% [+1.3%, +2.3%] | -1.9% [-2.8, -0.6] | pass / fail / fail | no claim: 2025 sales check does not agree with a CI excluding one |
| Hispanic majority | 14,476 | -1.0% [-1.3%, -0.6%] | -2.4% [-3.6%, -1.2%] | -3.4% [-4.5%, -2.1%] | -4.6% [-7.9, -2.3] | -2.8% [-4.0%, -1.5%] | -4.7% [-8.3, -2.8] | pass / pass / pass | under-assessed vs market share |
| White majority | 38,658 | -1.8% [-2.0%, -1.7%] | +2.5% [+2.1%, +3.0%] | +0.7% [+0.2%, +1.1%] | +1.5% [+0.6, +2.2] | -0.2% [-0.6%, +0.3%] | +1.7% [+0.8, +2.5] | pass / pass / fail | consistent with assessments lagging faster appreciation in some areas, not a valuation pattern (does not survive tier-specific time adjustment) |
| no majority (mixed) | 13,744 | -0.7% [-0.9%, -0.5%] | +0.5% [-0.3%, +1.3%] | -0.2% [-1.0%, +0.6%] | +1.5% [-0.2, +3.7] | -0.9% [-1.7%, -0.1%] | +1.3% [-0.4, +3.9] | fail / fail / fail | no claim: corrected shift's CI includes zero |
| Q1 (lowest income) | 23,609 | +1.2% [+0.9%, +1.5%] | -3.9% [-4.9%, -2.8%] | -2.7% [-3.8%, -1.6%] | -6.0% [-8.8, -1.7] | -1.4% [-2.5%, -0.2%] | -6.3% [-8.7, -2.3] | pass / pass / pass | under-assessed vs market share |
| Q2 | 24,047 | +1.6% [+1.4%, +1.8%] | -1.4% [-2.3%, -0.6%] | +0.2% [-0.8%, +1.1%] | -1.8% [-3.1, +0.8] | +1.0% [+0.0%, +1.9%] | -1.9% [-3.6, +0.3] | fail / fail / fail | no claim: corrected shift's CI includes zero |
| Q3 | 24,692 | +1.6% [+1.4%, +1.8%] | +0.3% [-0.3%, +1.0%] | +1.9% [+1.3%, +2.6%] | -0.0% [-1.3, +1.5] | +1.7% [+1.1%, +2.4%] | -0.1% [-1.4, +1.4] | pass / fail / fail | no claim: 2025 sales check does not agree with a CI excluding one |
| Q4 | 24,270 | -0.5% [-0.7%, -0.3%] | +1.4% [+0.9%, +2.0%] | +0.9% [+0.4%, +1.5%] | +1.1% [-0.1, +2.2] | +0.2% [-0.4%, +0.8%] | +1.1% [-0.2, +2.3] | pass / fail / fail | no claim: 2025 sales check does not agree with a CI excluding one |
| Q5 (highest income) | 24,849 | -1.8% [-2.0%, -1.6%] | +2.6% [+2.1%, +3.1%] | +0.7% [+0.2%, +1.3%] | +1.8% [+0.5, +2.9] | -0.1% [-0.7%, +0.4%] | +2.0% [+0.9, +3.3] | pass / pass / fail | consistent with assessments lagging faster appreciation in some areas, not a valuation pattern (does not survive tier-specific time adjustment) |

**Persistence.** The same model-free sales check, with tier-specific time adjustment, on the assessor's rolls 2021 to 2025 (assessor ratio on that year's sales, by group, relative to the city, percent with 95% interval). A star marks a year where the interval excludes zero in the direction of the 2025 corrected shift. A passing group is called persistent only if that holds in most of the four earlier rolls; otherwise the result is described as a 2025 one.

| group | roll 2021 | roll 2022 | roll 2023 | roll 2024 | roll 2025 |
|---|---|---|---|---|---|
| Black majority | -3.6 [-4.6, -2.6] | -0.9 [-1.9, +0.1] | -0.6 [-1.8, +0.5] | +0.4 [-0.8, +1.5] | -1.9 [-2.8, -0.7] |
| Hispanic majority | -5.0 [-7.2, -2.3] * | -3.4 [-6.3, -0.8] * | -4.4 [-9.2, -1.4] * | -8.0 [-10.4, -4.5] * | -4.7 [-8.1, -2.6] * |
| White majority | +3.2 [+2.2, +4.0] * | +1.1 [+0.4, +2.0] * | +0.8 [-0.0, +1.6] | +0.7 [-0.0, +1.6] | +1.7 [+0.9, +2.5] * |
| no majority (mixed) | +1.5 [+0.2, +3.2] | +0.2 [-1.5, +1.7] | +1.1 [-0.2, +2.8] | +0.2 [-0.9, +1.7] | +1.3 [-0.4, +3.8] |
| Q1 (lowest income) | -6.3 [-8.5, -3.7] * | -1.9 [-4.0, +0.2] | -3.6 [-6.1, -0.8] * | -2.4 [-5.0, +0.4] | -6.3 [-9.1, -2.4] * |
| Q2 | -3.5 [-5.9, -1.5] | -2.8 [-4.1, -0.6] | -2.0 [-4.2, -0.4] | -0.3 [-1.8, +1.1] | -1.9 [-3.5, -0.0] |
| Q3 | -0.9 [-2.3, +0.4] | +0.4 [-0.6, +1.8] | +0.2 [-1.0, +1.9] | +0.3 [-0.8, +1.5] | -0.1 [-1.3, +1.5] |
| Q4 | +0.8 [-0.2, +2.0] | -0.1 [-1.2, +1.0] | +0.4 [-1.0, +1.5] | +0.6 [-0.7, +1.5] | +1.1 [-0.2, +2.2] |
| Q5 (highest income) | +3.5 [+2.3, +4.6] * | +1.7 [+0.7, +2.7] * | +1.9 [+0.8, +3.9] * | +0.3 [-0.7, +1.4] | +2.0 [+0.9, +3.3] * |

**Dollars.** For a passing group, a typical home's amount is its median 2025 assessment times the corrected shift times the 2025 net rate, and the group total applies the corrected shift to the group's total assessment. The levy is fixed, so the same amount with the opposite sign is spread over every other home in the pool.

## Limitations

- **No interior condition** in public records: renovations, damage and deferred maintenance are invisible, which may be a large part of the remaining error and of the width of the ranges.
- **Sold homes are not all homes**, and the test excludes quick resales, so accuracy applies to homes not obviously renovated since the last roll. First-sale flip flags are undercounted for 2025 because later resales are not yet published.
- **No state transfer returns** (their bulk files need a login); arm's-length status rests on the city's filtered sales files and this project's filters. Deed type is verifiable only where conveyance dates survive in the roll, so it is a flag, not a filter.
- **Uneven field history.** Baths before 2019 predate a city data cleanup and are treated as missing; one year's finished area uses a different measure and is replaced from an earlier roll; stories and rooms are missing or stale for some rolls. Sale dates before 2019 are month-only.
- **Partial permit coverage**, starting in 2017, so the renovation flag undercounts.
- **Pre-2020 sales files look less filtered** than the later filtered extracts, so training years differ in quality from the test year.
- **The assessor's level verdict depends on the valuation-date convention**; dispersion results do not.
- **Daily-updated city files change with every download**; they are pinned once downloaded.
- **Transit stops are current, not historical**, for every year; parks use one year's parcel outlines with each roll's land use.
- **City-owned and vacant parcel counts can proxy for neighborhood composition**; they are allowed location features, and removing them barely changes the fairness gaps.
- **2025 was viewed across Stages 3 to 6**; the out-of-time 2026 test is the cleaner check.
- **Vertical equity is unresolved.** Both diagnostics are biased, in opposite directions; the frozen model fails PRD and PRB in cross-validation, and within neighborhood value tiers its PRB is far more negative than the assessor's.
- **Group gaps mix method error with group-specific appreciation**, because target prices use a citywide trend; differences between methods in the level gap are roughly unaffected, but accuracy and dispersion gaps are not, and the model's nearby-sales features may track local appreciation better than a roll can.
- **The audit evaluates a cleaner subset** in Black-majority and low-income tracts, so accuracy there is likely overstated for every method.
- **Student-rental areas** near Marquette and UW-Milwaukee hold 3.5% of the homes compared but 60 of the 106 listings above the model. Rentals there earn income the model cannot see, so divergence there may reflect the model; this is untested.
- **Held-flat and mismatched records are set aside, not judged.**
- **Duplex yields are not yet computed**; they need HUD Small Area Fair Market Rents by ZIP code.

## Reproducibility

`python -m avm.pipeline` rebuilds everything from the raw downloads: acquisition (unchanged files skipped, daily files pinned), cleaning, every stage, the pages and this document. Seeds are fixed and LightGBM runs deterministically. The pipeline re-runs the tuning and freeze in a verify mode that must reproduce the committed `reports/frozen/model_spec.json` exactly (every field but the commit stamp) and never rewrites it. The prospective 2026 file is never regenerated; the pipeline verifies its hash (de73656af10f90732b4f0d9e9f967cb4e1ba30bce0f95179a8244d0689b6fc2a). The last full rebuild, from a copy of the raw downloads into an otherwise empty data folder, completed without error, reproduced the frozen spec and the frozen model's 2025 predictions exactly, and verified the prospective file's hash (log: [`reports/rebuild/rebuild_log.txt`](../reports/rebuild/rebuild_log.txt)). The comparison of its outputs with the committed ones (every table from Stage 2 on identical; Stage 1 reports refreshed) is recorded in [`docs/decision-log.md`](decision-log.md). After later Stage 9 code fixes the same day, Stage 9 was re-run in the rebuild folder and every table matched the live run ([`reports/rebuild/stage9_recheck.txt`](../reports/rebuild/stage9_recheck.txt)).
