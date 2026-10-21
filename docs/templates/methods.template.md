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
- **Prior-sales features** use only sales before the cutoff: for training rows, before the sale's own month; for scored rows, before Jan 1 of the scored year. A home's own prior sale must be more than {{own_min}} before the sale and before the cutoff; the code asserts both.
- **Known look-ahead in location features.** Transit stops come from the current schedule and park locations from one year's parcel outlines, for every year, so they are not as of the sale; how much they changed over the period is not measured.
- **Time-based validation.** Training uses 2014 to 2024; 2025 is the held-out year. Cross-validation rolls forward a year at a time over 2021 to 2024. No random splits.
- **Test-set discipline.** 2025 results were reported at Stages 3 to 6 and influenced design choices then. From the freeze on, every model selection used cross-validation on 2021 to 2024 only; the comparable-sales rule was specified after the freeze and has been evaluated only on 2025. The frozen spec is `reports/frozen/model_spec.json` (sha256 begins {{spec_sha}}).

## Cleaning

{{raw_sales}} sales-file rows from 2014 to 2025 enter; {{clean_sales}} cleaned single-family and duplex sales remain. Each filter is logged with the rows it removes (`reports/stage2/filter_log.csv`):

{{filter_table}}

Quick flips are found from repeated tax keys across the sales files (a resale within {{flip_months}}); the resale is dropped and the first sale is flagged. Multi-parcel sales are dropped only when every parcel lies within {{multi_radius}} of the others. Deed type, permit activity and price-per-square-foot outliers are kept as flags for sensitivity checks rather than filters (training drops price-per-square-foot outliers).

## Target prices and the ratio study

Every method is scored against one shared target per sale: its price moved to the Jan 1 trend level of the sale year, with a neutral month-of-year. The time adjustment is a piecewise trend with month-of-year effects estimated on 2014 to 2024; the 2025 part of the index is used only to adjust 2025 prices for evaluation and never feeds training.

IAAO ratio statistics (median ratio, COD, PRD, PRB, MKI) come with {{n_boot}}-draw bootstrap intervals; method differences use paired bootstraps on the same draws. The IAAO trim ({{iqr_k}} times the interquartile range on the assessor's ratio) is used only for the assessor's standards check; head-to-head comparisons are untrimmed. The headline accuracy measure is **level-neutral** median absolute error: each method's ratios are divided by that method's own median ratio first, so a method is not rewarded for its overall level.

## Models

**Hedonic baseline.** Ordinary least squares on log price with size, lot, age bands, bedrooms, baths, stories, style, duplex, a centered spline time trend, month-of-year effects and {{hedonic_loc}} fixed effects (units with few training sales are folded into the assessor region), trained on sales from {{hedonic_start}} on; location unit and start year were chosen by its own cross-validation.

**Main model (independent).** LightGBM on a detrended target: log price minus a time effect estimated from the training years only, so the model predicts directly at the December trend level of the year before the valuation date, with no time features. Increases in finished area, lot size and baths are constrained to never lower the value. Its {{n_features}} features:

{{feature_table}}

**Selection and freeze.** {{n_configs}} configurations per variant were sampled and scored on cross-validation 2021 to 2024 by mean level-neutral error; among configurations within {{mdape_tol}} of the best, the one with the smallest mean absolute within-tier PRB was chosen. The frozen primary uses {{cfg_num_leaves}} leaves, learning rate {{cfg_learning_rate}}, {{cfg_num_boost_round}} rounds, minimum {{cfg_min_data_in_leaf}} rows per leaf, feature fraction {{cfg_feature_fraction}}, bagging fraction {{cfg_bagging_fraction}} and L2 penalty {{cfg_lambda_l2}}, with LightGBM seed {{seed}} and deterministic training.

**Assessment-informed variant.** A separately tuned LightGBM (its own configuration and location unit, in `reports/frozen/model_spec.json`) with the prior roll's total and land assessed values added, each expressed relative to its neighborhood's median so that each roll's overall assessment level does not leak into a detrended model.

**Nearby-sales features.** The {{knn_k}} nearest prior sales within {{knn_window}} (their time-adjusted price per square foot and distance), a neighborhood price-per-square-foot level, and the home's own prior sale.

## Intervals

Conformal quantile regression: LightGBM quantile models for the tenth and ninetieth percentiles, widened by a margin ({{cqr_q}} in log price) calibrated on pooled out-of-sample predictions for 2022 to 2024, then widened where needed so every range contains its point value. Coverage on 2025: {{cover_all}} overall; {{cover_low}}, {{cover_mid}} and {{cover_high}} by value tier.

## Comparable sales

From the eligible pool (clean sales of the same property type before the valuation date, excluding first sales of a flip, IAAO ratio outliers and price-per-square-foot outliers), the search tries {{comp_radii}} within the most recent {{comp_windows}}, and stops at the first step with at least three sales. Up to five are kept, ranked by a structural distance weighted by each feature's SHAP importance. The step used is recorded for each home as a confidence field. Comp value (a) is the median comp price moved to the valuation date; (b) also adjusts each comp by the model's view of the difference. Only (a) uses no model prediction, though it is not fully independent: comps are ranked with the model's SHAP weights, and the pool drops sales flagged as outliers on the assessor's own ratio. Under the previous rule (fixed two-year pool, no exclusions) comp value (a) had a level-neutral error of {{comp_a_mdape_before}}; it is now {{comp_a_mdape}}.

## Explanations

TreeSHAP contributions from LightGBM, in log price. Global importance is reported by feature group first, because coordinates, distances and nearby-sales features are correlated and split credit between them.

## Fairness audit

City census tracts are grouped by majority race or ethnicity (more than half of residents; groups with fewer than {{min_tracts_group}} tracts are folded into "no majority") and by median household income quintile, weighted by parcels. For each group and method: valuation level (mean log ratio), median error against the method's overall level, COD, PRD and PRB, and calibration slopes, with bootstrap intervals and paired differences against the assessor. The share of in-scope sales removed by the cleaning filters (F6 to F11, sales of 2021 to 2025) is reported by group: in Black-majority tracts {{clean_rate_black}} of in-scope sales are removed (flips {{flip_rate_black}}) against {{clean_rate_white}} in White-majority tracts (flips {{flip_rate_white}}); lowest income quintile {{clean_rate_q1}} against highest {{clean_rate_q5}}.

## Divergence and tax dollars

Each single-family and duplex home in roll 2024 that the model can score ({{s9_parcels_model_scored}} of {{s9_parcels_sf_duplex_roll_2024}}) is valued by the frozen model at Jan 1 2025 with its 80% range and compared with roll 2025. **Relative assessment** = (assessed / model value) / the citywide median of that ratio ({{s9_level}}). A home is **listable** when its assessment divided by the citywide level falls outside the model's 80% range and the median of its time-adjusted comps points the same way; the main list ranks the largest gaps by the size of the log relative assessment.

Excluded from both the list and the tax pool:

{{s9_excl_table}}

Set aside in their own section (still in the tax pool). The median home's assessment changed by {{s9_change}} from 2024 to 2025 and {{s9_flat_share}} of homes kept exactly the same value. Counts below are among the listable homes, and a home can meet several rules ({{s9_special_listable}} unique homes):

{{s9_rules_table}}

Shown next to a listed home, without moving it:

{{s9_flags_table}}

**Tax.** Revenue-neutral within the single-family and duplex class: a home's fair assessment is its model value times (total assessed / total model value), and its shift is (assessed minus fair) times the 2025 {{tax_used_kind}} rate ({{tax_used}} per thousand dollars, from the Milwaukee County comparative rate table, manifest key `tax_rates_county_2016_2025`; {{tax_prior}} in the same table is the 2024 rate; gross {{tax_gross}} for context). School levy credit: {{tax_levy_credit}}. Not applied: {{tax_excluded}} ([statute]({{tax_statute}})).

**Groups.** A group's shift is the median of log(assessed / fair). Because the fair share comes from the model, a model that under-values a group by b makes it look over-assessed by about b, so the corrected shift is the shift plus the model's own cross-validated bias for the group (relative to its citywide bias), with a bootstrap interval from independent draws of both. The model-free check is the assessor's median ratio on 2025 sales in the group relative to the citywide ratio. A group is called over- or under-assessed only if (a) the corrected interval excludes zero, (b) the sales check's interval excludes zero in the same direction, and (c) both still hold when every target price is recomputed with separate trend-only time adjustments for each tercile of prior assessed value (the Stage 3 sensitivity, applied to every year used). A result that passes (a) and (b) but not (c) is reported as consistent with assessments lagging faster appreciation in some areas, not a valuation pattern. {{s9_group_result}}

{{s9_group_table}}

**Persistence.** The same model-free sales check, with tier-specific time adjustment, on the assessor's rolls 2021 to 2025 (assessor ratio on that year's sales, by group, relative to the city, percent with 95% interval). A star marks a year where the interval excludes zero in the direction of the 2025 corrected shift. A passing group is called persistent only if that holds in most of the four earlier rolls; otherwise the result is described as a 2025 one.

{{s9_by_year_table}}

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
- **Student-rental areas** near Marquette and UW-Milwaukee hold {{s9_student_share}} of the homes compared but {{s9_listed_student_area_above_model}} of the {{s9_listed_over_assessed}} listings above the model. Rentals there earn income the model cannot see, so divergence there may reflect the model; this is untested.
- **Held-flat and mismatched records are set aside, not judged.**
- **Duplex yields are not yet computed**; they need HUD Small Area Fair Market Rents by ZIP code.

## Reproducibility

`python -m avm.pipeline` rebuilds everything from the raw downloads: acquisition (unchanged files skipped, daily files pinned), cleaning, every stage, the pages and this document. Seeds are fixed and LightGBM runs deterministically. The pipeline re-runs the tuning and freeze in a verify mode that must reproduce the committed `reports/frozen/model_spec.json` exactly (every field but the commit stamp) and never rewrites it. The prospective 2026 file is never regenerated; the pipeline verifies its hash ({{pros_sha_full}}). The last full rebuild, from a copy of the raw downloads into an otherwise empty data folder, {{rebuild_status}} (log: [`reports/rebuild/rebuild_log.txt`](../reports/rebuild/rebuild_log.txt)). The comparison of its outputs with the committed ones (every table from Stage 2 on identical; Stage 1 reports refreshed) is recorded in [`docs/decision-log.md`](decision-log.md). {{stage9_recheck}} ([`reports/rebuild/stage9_recheck.txt`](../reports/rebuild/stage9_recheck.txt)).
