<!-- Generated file. Edit docs/templates/README.template.md and run `python -m avm.writeup`. -->
# Milwaukee AVM

[![CI](https://github.com/{{repo_slug}}/actions/workflows/ci.yml/badge.svg)](https://github.com/{{repo_slug}}/actions/workflows/ci.yml)

**A public-data valuation model for Milwaukee single-family and duplex homes, tested against real sale prices and benchmarked against the City Assessor.**

Every result below is generated from a report file by `python -m avm.writeup` (method constants such as the 0.75 mi radius are written in); each one, with its source, is listed in [`reports/writeup/numbers.json`](reports/writeup/numbers.json). Method detail and the full list of limitations are in [`docs/methods.md`](docs/methods.md).

## Key results

Held-out test: {{n_test}} cleaned 2025 sales (passing the city's sales filter and this project's) that the model was not trained on, scored against the same time-adjusted prices for every method. Model = the frozen independent model (no assessed values used). Brackets are 95% paired-bootstrap intervals. Results on 2025 were viewed during development, so the cleaner test is the out-of-time one in the last row.

| Measure | Model (public data) | City Assessor | Model minus Assessor |
|---|---|---|---|
| Accuracy: level-neutral median absolute error (lower is better) | {{mdape_ln_primary}} | {{mdape_ln_assessor}} | {{mdape_ln_primary_diff}} {{mdape_ln_primary_diff_ci}} |
| Uniformity: COD (IAAO target {{target_cod}}) | {{cod_primary}} | {{cod_assessor}} | {{cod_primary_diff}} {{cod_primary_diff_ci}} |
| Vertical equity: meets IAAO PRD ({{target_prd}}) and PRB ({{target_prb}}) targets, citywide 2025? | PRD {{meets_prd_primary}} ({{prd_primary}}), PRB {{meets_prb_primary}} ({{prb_primary}}); fails both in cross-validation | PRD {{meets_prd_assessor}} ({{prd_assessor}}), PRB {{meets_prb_assessor}} ({{prb_assessor}}) | |
| 80% prediction interval | covers {{cover_all}} of prices; wide, median {{width_all}} of value | no interval published | |
| Out-of-time test, values as of Jan 1 2026 | {{pros_rows}} homes valued, fit only on data through 2025 and committed (`sha256` {{pros_sha}}, git tag `prospective-2026`) on {{pros_date}}, before the city released its 2026 sales file | roll 2026 | **pending** |

## Summary

The City Assessor values every Milwaukee home each year. This project asks how close a model built only from public records gets to real sale prices, and where it disagrees with the assessor. It draws on the {{n_train}} sales of 2014 to 2024 that pass its cleaning filters, takes every property characteristic from the roll in force before the sale, and was frozen before its final scoring on 2025.

- **It is more accurate than the assessor on the same sales.** Level-neutral median error {{mdape_ln_primary}} against {{mdape_ln_assessor}} ({{mdape_ln_primary_diff}} {{mdape_ln_primary_diff_ci}}), and dispersion (COD) {{cod_primary}} against {{cod_assessor}}.
- **Its ranges are wide but honest.** The 80% ranges contain {{cover_all}} of 2025 sale prices; the median range spans {{width_all}} of the value.
- **Fairness is mixed.** Every method is less accurate in Black-majority and lowest-income tracts, and on 2025 the model's Black-White gap in valuation level is larger than the assessor's.
- **An out-of-time test is pending** on values for Jan 1 2026, committed before the city released its 2026 sales file.

## What this is good for

- **Screening leads for assessment review.** The ranked list points to homes whose assessments sit far from both the model and nearby sales, with likely special valuations set aside. Each is a lead to check, not a finding.
- **A starting value for underwriting.** Any single-family or duplex home gets a value, an 80% range and comparable sales from public data alone. The range is wide, so it starts diligence; it does not replace an appraisal or an inspection.
- **A test bed for AVM fairness methods.** The group guardrail (bias correction, model-free sales check, tier-specific time adjustment) and the paired vertical-equity diagnostics can be reused on other models.

Ask Claude about any Milwaukee home: [`docs/mcp.md`](docs/mcp.md).

## Results

### The benchmark: the City Assessor in 2025

On the IAAO ratio study ({{a3_n}} sales after the IAAO outlier trim, prices moved to the Jan 1 2025 trend level), the assessor's median ratio is {{a3_median_ratio}} {{a3_median_ratio_ci}}, just under the IAAO band of {{target_median_ratio}}. Dispersion is the larger problem: COD {{a3_cod}} {{a3_cod_ci}} against a target of {{target_cod}}. Vertical equity meets the targets (PRD {{a3_prd}}, PRB {{a3_prb}}). The level verdict depends on the valuation-date convention; the dispersion verdict does not.

### Head-to-head on identical 2025 sales

All four methods are scored on the same {{n_test}} sales and the same target prices, untrimmed. Level-neutral error divides each method's ratios by its own median first, so it measures accuracy rather than overall level, which a single factor would fix.

| | City Assessor | Hedonic baseline | Model (public data) | Model (assessment-informed) |
|---|---|---|---|---|
| Level-neutral median error | {{mdape_ln_assessor}} | {{mdape_ln_hedonic}} | **{{mdape_ln_primary}}** | {{mdape_ln_secondary}} |
| Difference vs assessor | | {{mdape_ln_hedonic_diff}} {{mdape_ln_hedonic_diff_ci}} | {{mdape_ln_primary_diff}} {{mdape_ln_primary_diff_ci}} | {{mdape_ln_secondary_diff}} {{mdape_ln_secondary_diff_ci}} |
| Within 10% of price (raw ratios, so level counts) | {{within_10_assessor}} | {{within_10_hedonic}} | {{within_10_primary}} | {{within_10_secondary}} |
| Within 20% of price (raw ratios) | {{within_20_assessor}} | {{within_20_hedonic}} | {{within_20_primary}} | {{within_20_secondary}} |
| COD | {{cod_assessor}} | {{cod_hedonic}} | {{cod_primary}} | {{cod_secondary}} |
| Median ratio (level) | {{median_ratio_assessor}} | {{median_ratio_hedonic}} | {{median_ratio_primary}} | {{median_ratio_secondary}} |
| Meets PRD / PRB targets? | {{meets_prd_assessor}} / {{meets_prb_assessor}} | {{meets_prd_hedonic}} / {{meets_prb_hedonic}} | {{meets_prd_primary}} / {{meets_prb_primary}} | {{meets_prd_secondary}} / {{meets_prb_secondary}} |
| Calibration slope | {{calibration_slope_assessor}} | {{calibration_slope_hedonic}} | {{calibration_slope_primary}} | {{calibration_slope_secondary}} |

The hedonic baseline (log price on size, lot, age, bedrooms, baths, stories, style, a time trend and {{hedonic_loc}} fixed effects, trained on sales from {{hedonic_start}} on, the window its own cross-validation chose) already matches the assessor (its difference interval includes zero); only the gradient-boosted models clearly beat it. Giving the model the prior assessed value (the assessment-informed variant) does not help once that value is made level-free.

### Where the gain came from

Each feature group was added alone to a baseline and reported on cross-validation (2021 to 2024), with 2025 also viewed then. Pre-freeze, the cross-validated error falls from {{lift_cv_base}} (structure and tract) to {{lift_cv_all}} with every group; {{lift_best_group}} gives the largest single gain ({{lift_best_gain}}). The re-tuned frozen model reaches {{cv_mdape_frozen}}.

### Uncertainty

The 80% ranges come from conformal quantile regression calibrated on 2022 to 2024. On 2025 they contain {{cover_all}} of prices: {{cover_low}}, {{cover_mid}} and {{cover_high}} in the low, middle and high value tiers, so {{cover_excess_mid}} to {{cover_excess_high}} above nominal outside the low tier. The median range spans {{width_all}} of the value ({{width_low}} in the low tier). Interior condition, invisible in public records, may be a large part of the width, alongside sale-price noise (the split is unmeasured).

### Comparable sales as a check

Each home gets up to five same-type comparable sales when at least three qualify, searched at {{comp_radii}} over {{comp_windows}}, excluding the home's own sales, quick flips and outliers; {{comp_first_tier_share}} of 2025 test sales find them at the tightest step. Time-adjusted comp values use no model prediction and reach a level-neutral error of {{comp_a_mdape}} on {{comp_n}} sales, against {{comp_model_mdape}} for the model. Model-adjusted comps ({{comp_b_mdape}}) borrow the model's judgement, so only time-adjusted comps drive disagreement flags.

### Reading vertical equity carefully

Vertical equity is the model's weak point. Both methods meet the IAAO PRD and PRB targets citywide on 2025, but in cross-validation the model fails both (PRD {{cv_prd_primary}}, PRB {{cv_prb_primary}}). Noise in sale prices makes PRD and PRB lean toward regressivity; noise in values pushes the calibration slope below one. The model's slope is closer to one than the assessor's ({{calibration_slope_primary}} against {{calibration_slope_assessor}}). But both face the same price noise, and within value tiers the model's PRB ({{tier_prb_low_primary}}, {{tier_prb_mid_primary}}, {{tier_prb_high_primary}}, low to high) is far more negative than the assessor's ({{tier_prb_low_assessor}}, {{tier_prb_mid_assessor}}, {{tier_prb_high_assessor}}), consistent with compression across similarly priced neighborhoods. Neither diagnostic settles it alone.

## Fairness audit

The audit asks whether **errors** differ by neighborhood group, not whether values do. Census tracts are grouped by majority race or ethnicity and by median household income quintile (ACS 2019-2023); demographics are never model inputs.

| Gap (group minus group) | Split | City Assessor | Model (public data) |
|---|---|---|---|
| Valuation level (mean log ratio), Black-majority minus White-majority | cross-validation 2021-2024 | {{lvl_bw_assessor_cv}} | {{lvl_bw_primary_cv}} |
| Valuation level (mean log ratio), Black-majority minus White-majority | 2025 | {{lvl_bw_assessor_y25}} | {{lvl_bw_primary_y25}} |
| Valuation level (median ratio), Black-majority minus White-majority | cross-validation 2021-2024 | {{mr_bw_assessor_cv}} | {{mr_bw_primary_cv}} |
| Valuation level (median ratio), Black-majority minus White-majority | 2025 | {{mr_bw_assessor_y25}} | {{mr_bw_primary_y25}} |
| Valuation level (mean log ratio), lowest minus highest income quintile | cross-validation 2021-2024 | {{lvl_q15_assessor_cv}} | {{lvl_q15_primary_cv}} |
| Valuation level (mean log ratio), lowest minus highest income quintile | 2025 | {{lvl_q15_assessor_y25}} | {{lvl_q15_primary_y25}} |
| Median error, Black-majority minus White-majority tracts | cross-validation 2021-2024 | {{gap_bw_mdape_vs_overall_level_assessor}} | {{gap_bw_mdape_vs_overall_level_primary}} |
| Median error, lowest minus highest income quintile | cross-validation 2021-2024 | {{gap_q15_mdape_vs_overall_level_assessor}} | {{gap_q15_mdape_vs_overall_level_primary}} |
| COD, Black-majority minus White-majority tracts | cross-validation 2021-2024 | {{gap_bw_cod_assessor}} | {{gap_bw_cod_primary}} |

**On 2025 the model's Black-White gap in valuation level is larger than the assessor's**, by {{lvl_bw_diff_y25}} by mean log ratio and {{mr_bw_diff_y25}} by median ratio. Relative to White-majority tracts, both methods value Black-majority tracts lower against their sale prices. In cross-validation (mean log ratio of value to price) the model is about level in Black-majority tracts ({{grp_black_primary}}) and high in White-majority tracts ({{grp_white_primary}}), while the assessor is low in both ({{grp_black_assessor}} and {{grp_white_assessor}}). In cross-validation the model's gap is smaller by mean log ratio (change in its size {{lvl_bw_diff_cv}}) but not by median ratio ({{mr_bw_diff_cv}}); on 2025 the income gaps do not clearly differ ({{lvl_q15_diff_y25}}). Level gaps are differences in the ratio statistic times one hundred (for the mean log ratio, log points), not percentage points. Neither method closes the accuracy gap, and in cross-validation the model's dispersion gap is slightly larger.

The audit also evaluates a cleaner sample in some groups. On 2021 to 2025 sales, the cleaning filters (F6 to F11) remove {{clean_rate_black}} of in-scope sales in Black-majority tracts against {{clean_rate_white}} in White-majority tracts; quick resales alone are {{flip_rate_black}} against {{flip_rate_white}}. Accuracy is therefore measured on fewer flipped homes there, and is likely overstated for every method.

## Where the assessor and the model disagree

The frozen model values every home it can score ({{s9_parcels_model_scored}} of {{s9_parcels_sf_duplex_roll_2024}}) at Jan 1 2025; after data-quality exclusions {{s9_parcels_scored}} are compared with their 2025 assessment, net of the assessor's citywide level ({{s9_level}}). A home is listable when its assessment sits outside the model's 80% range **and** its comparable sales agree: {{s9_listable_outside_and_comps_agree}} homes.

- **Set aside, not ranked:** {{s9_special_listable}} of those look like a special valuation or a data issue (an assessment held flat from 2024 while the median home's assessment rose {{s9_change}}, a land-use change, a unit count or commercial building type that does not fit the land use).
- **Ranked:** of the {{s9_listed}} largest remaining gaps, {{s9_listed_under_assessed}} are assessed below the model and {{s9_listed_over_assessed}} above. The median tax difference is {{s9_median_shift}} a year at the 2025 {{tax_used_kind}} rate of {{tax_used}} per thousand dollars of assessed value (Milwaukee County comparative rate table; {{tax_prior}} was the 2024 rate). The net rate is used because the school levy credit scales with assessed value under Wisconsin statute ([`docs/methods.md`](docs/methods.md#divergence-and-tax-dollars)).
- **Student-rental areas:** within 0.75 mi of the Marquette or UW-Milwaukee main campus lie {{s9_student_share}} of the homes compared but {{s9_listed_student_area_above_model}} of the {{s9_listed_over_assessed}} listings above the model ({{s9_listed_student_area}} ranked gaps in all). Rental income the model cannot see may explain them; this is untested, and their comparable sales point the same way as the model.
- **Groups:** each group's tax shift is corrected for the model's own bias for that group. A group is called over- or under-assessed only if (a) the corrected shift's 95% interval excludes zero, (b) the assessor's ratio on 2025 sales, which uses no model, agrees with an interval excluding zero, and (c) both survive tier-specific time adjustment. {{s9_group_result}} Details: [`docs/methods.md`](docs/methods.md#divergence-and-tax-dollars).

A private review page (with addresses) and a shareable page (map, group table, method) are built.

## The out-of-time test

Because 2025 was viewed during development, the cleaner test is out of time: the frozen model, refit on {{pros_train}} sales through 2025 with no 2026 data, valued {{pros_rows}} homes as of Jan 1 2026. The file was committed with its hash and git tag `prospective-2026` on {{pros_date}}, before the city released its 2026 sales file. It is not a pre-registration: it was committed well after the valuation date, the design was settled with 2025 in view, and transit stops are current. Once the 2026 sales file and roll are out, they are added to `sources.SOURCES` by hand and downloaded, and `python -m avm.prospective score` checks the hash and scores both against 2026 prices.

## How this compares to Cook County's model

The Cook County Assessor's Office publishes its residential model ([ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm), README read {{ccao_read}}). Both use LightGBM, value homes at Jan 1, explain values with SHAP and report IAAO ratio statistics. The differences:

- **Demographics.** Cook County's feature set includes ACS census variables. Here demographics are excluded from the model and used only to audit it, though location features that can act as proxies (tract, nearby city-owned and vacant parcels) are used and flagged.
- **Data.** Cook County uses internal assessor data (characteristics, land rates, analyst sales review); this project uses only public downloads.
- **Comparable sales.** Cook County's model does not explicitly produce comparables (its leaf-node comps are experimental); here comps are a model-free check that drives disagreement flags.
- **Uncertainty and an out-of-time test.** This project computes conformal 80% ranges for every home (they over-cover on 2025) and a hashed out-of-time test. The Cook County README does not describe either.

## Limitations

The most important ones (the full list is in [`docs/methods.md`](docs/methods.md#limitations)):

- **No interior condition.** Public records cannot see renovations, damage or deferred maintenance; this may be a large part of the remaining error and of the width of the ranges.
- **Sold homes are not all homes.** Accuracy is measured on sales, and quick flips are excluded, so it applies to homes not obviously renovated since the last roll.
- **No state transfer returns.** Arm's-length status rests on the city's undocumented filter (from 2020) plus this project's; deed type is only a flag.
- **2025 was not untouched.** It was viewed across Stages 3 to 6; model selection used cross-validation only from the freeze on, but the comparable-sales rule was specified after the freeze and evaluated only on 2025. The out-of-time test is the cleaner check.
- **Fairness results mix method error with group-specific appreciation**, because target prices use a citywide trend; only the level-gap differences between methods are roughly unaffected.

## What I'd do next

- **Add interior condition.** MLS listing data (photos, remarks, condition fields) or inspection records would address what may be the largest source of error and narrow the ranges.
- **Add an income approach for rentals.** Near the campuses homes are priced partly on rent, so a rent-based model (by-the-bed leases, local rent comps) belongs beside the sales model, and would test whether the student-rental gaps come from the model.
- **Investigate the within-tier PRB gap** before any use for individual values, and re-check it on the prospective test.
- **Score the out-of-time test** as soon as the 2026 sales file and roll are released, and report it whatever it shows.
- **Fill the duplex income view** with HUD Small Area Fair Market Rents by ZIP code.

## Reproduce

```
python -m venv venv && venv\Scripts\activate
pip install -r requirements.lock && pip install -e .
python -m avm.pipeline
```

Data go to `C:\Users\<you>\data\milwaukee-avm` (override with `AVM_DATA_DIR`), outside the repo. Stage 8 needs a free Census API key in `.env` as `CENSUS_API_KEY`. Seeds are fixed. The pipeline re-runs the tuning and requires it to reproduce the committed frozen spec exactly; the prospective file is checked, never regenerated.

## Data and license

Code is under the MIT License ([`LICENSE`](LICENSE)). **Data are not covered by the MIT License**; each source keeps its own terms, and raw downloads are not stored in this repository (`python -m avm.pipeline` fetches them; `reports/stage1/manifest.csv` lists every URL and sha256):

- **City of Milwaukee open data** (Master Property File, property sales, parcel outlines, Master Address Index, building permits, assessment neighborhoods), under CC-BY: attribution to the City of Milwaukee.
- **US Census Bureau**: ACS 2019-2023 five-year estimates (fairness audit only) and TIGER/Line 2023 outlines.
- **Milwaukee County Transit System** GTFS feed, and the **Milwaukee County Treasurer's** comparative tax rate table.

## Repository

- `src/avm/`: pipeline code, one module per stage; `tests/`: synthetic tests run by CI
- `reports/`: every table and figure by stage; `reports/frozen/` the frozen spec; `reports/prospective/` the hashed 2026 file and its timestamp proof
- `docs/`: `methods.md` and the README templates; `docs/decision-log.md`: the dated decision log
