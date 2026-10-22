<!-- Generated file. Edit docs/templates/README.template.md and run `python -m avm.writeup`. -->
# Milwaukee AVM

[![CI](https://github.com/oekrueg05/milwaukee-avm/actions/workflows/ci.yml/badge.svg)](https://github.com/oekrueg05/milwaukee-avm/actions/workflows/ci.yml)

**A public-data valuation model for Milwaukee single-family and duplex homes, tested against real sale prices and benchmarked against the City Assessor.**

Every result below is generated from a report file by `python -m avm.writeup` (method constants such as the 0.75 mi radius are written in); each one, with its source, is listed in [`reports/writeup/numbers.json`](reports/writeup/numbers.json). Method detail and the full list of limitations are in [`docs/methods.md`](docs/methods.md).

## Key results

Held-out test: 4,197 cleaned 2025 sales (passing the city's sales filter and this project's) that the model was not trained on, scored against the same time-adjusted prices for every method. Model = the frozen independent model (no assessed values used). Brackets are 95% paired-bootstrap intervals. Results on 2025 were viewed during development, so the cleaner test is the out-of-time one in the last row.

| Measure | Model (public data) | City Assessor | Model minus Assessor |
|---|---|---|---|
| Accuracy: level-neutral median absolute error (lower is better) | 11.9% | 13.3% | -1.4 pts [-2.0, -0.9] |
| Uniformity: COD (IAAO target <= 15) | 16.52 | 17.76 | -1.23 [-1.72, -0.72] |
| Vertical equity: meets IAAO PRD (0.98 to 1.03) and PRB (-0.05 to 0.05) targets, citywide 2025? | PRD yes (1.020), PRB yes (-0.026); fails both in cross-validation | PRD yes (1.013), PRB yes (+0.022) | |
| 80% prediction interval | covers 85.2% of prices; wide, median 53% of value | no interval published | |
| Out-of-time test, values as of Jan 1 2026 | 122,222 homes valued, fit only on data through 2025 and committed (`sha256` de73656af10f, git tag `prospective-2026`) on 2026-10-18, before the city released its 2026 sales file | roll 2026 | **pending** |

## Summary

The City Assessor values every Milwaukee home each year. This project asks how close a model built only from public records gets to real sale prices, and where it disagrees with the assessor. It draws on the 44,038 sales of 2014 to 2024 that pass its cleaning filters, takes every property characteristic from the roll in force before the sale, and was frozen before its final scoring on 2025.

- **It is more accurate than the assessor on the same sales.** Level-neutral median error 11.9% against 13.3% (-1.4 pts [-2.0, -0.9]), and dispersion (COD) 16.52 against 17.76.
- **Its ranges are wide but honest.** The 80% ranges contain 85.2% of 2025 sale prices; the median range spans 53% of the value.
- **Fairness is mixed.** Every method is less accurate in Black-majority and lowest-income tracts, and on 2025 the model's Black-White gap in valuation level is larger than the assessor's.
- **An out-of-time test is pending** on values for Jan 1 2026, committed before the city released its 2026 sales file.

## What this is good for

- **Screening leads for assessment review.** The ranked list points to homes whose assessments sit far from both the model and nearby sales, with likely special valuations set aside. Each is a lead to check, not a finding.
- **A starting value for underwriting.** Any single-family or duplex home gets a value, an 80% range and comparable sales from public data alone. The range is wide, so it starts diligence; it does not replace an appraisal or an inspection.
- **A test bed for AVM fairness methods.** The group guardrail (bias correction, model-free sales check, tier-specific time adjustment) and the paired vertical-equity diagnostics can be reused on other models.

Ask Claude about any Milwaukee home: [`docs/mcp.md`](docs/mcp.md).

## Results

### The benchmark: the City Assessor in 2025

On the IAAO ratio study (4,215 sales after the IAAO outlier trim, prices moved to the Jan 1 2025 trend level), the assessor's median ratio is 0.892 [0.885, 0.899], just under the IAAO band of 0.90 to 1.10. Dispersion is the larger problem: COD 17.6 [17.1, 18.1] against a target of <= 15. Vertical equity meets the targets (PRD 1.009, PRB +0.036). The level verdict depends on the valuation-date convention; the dispersion verdict does not.

### Head-to-head on identical 2025 sales

All four methods are scored on the same 4,197 sales and the same target prices, untrimmed. Level-neutral error divides each method's ratios by its own median first, so it measures accuracy rather than overall level, which a single factor would fix.

| | City Assessor | Hedonic baseline | Model (public data) | Model (assessment-informed) |
|---|---|---|---|---|
| Level-neutral median error | 13.3% | 12.9% | **11.9%** | 12.1% |
| Difference vs assessor | | -0.4 pts [-1.0, +0.2] | -1.4 pts [-2.0, -0.9] | -1.2 pts [-1.7, -0.6] |
| Within 10% of price (raw ratios, so level counts) | 35% | 39% | 44% | 42% |
| Within 20% of price (raw ratios) | 64% | 67% | 71% | 70% |
| COD | 17.76 | 17.86 | 16.52 | 16.67 |
| Median ratio (level) | 0.894 | 1.001 | 0.997 | 1.007 |
| Meets PRD / PRB targets? | yes / yes | yes / yes | yes / yes | yes / yes |
| Calibration slope | 0.81 | 0.87 | 0.89 | 0.88 |

The hedonic baseline (log price on size, lot, age, bedrooms, baths, stories, style, a time trend and assessor-neighborhood fixed effects, trained on sales from 2020 on, the window its own cross-validation chose) already matches the assessor (its difference interval includes zero); only the gradient-boosted models clearly beat it. Giving the model the prior assessed value (the assessment-informed variant) does not help once that value is made level-free.

### Where the gain came from

Each feature group was added alone to a baseline and reported on cross-validation (2021 to 2024), with 2025 also viewed then. Pre-freeze, the cross-validated error falls from 13.59% (structure and tract) to 12.87% with every group; prior sales nearby gives the largest single gain (0.35 pts). The re-tuned frozen model reaches 12.80%.

### Uncertainty

The 80% ranges come from conformal quantile regression calibrated on 2022 to 2024. On 2025 they contain 85.2% of prices: 81.0%, 86.7% and 87.4% in the low, middle and high value tiers, so 6.7 pts to 7.4 pts above nominal outside the low tier. The median range spans 53% of the value (82% in the low tier). Interior condition, invisible in public records, may be a large part of the width, alongside sale-price noise (the split is unmeasured).

### Comparable sales as a check

Each home gets up to five same-type comparable sales when at least three qualify, searched at 0.5, 1, 1.25 mi over 12 then 24 months, excluding the home's own sales, quick flips and outliers; 99% of 2025 test sales find them at the tightest step. Time-adjusted comp values use no model prediction and reach a level-neutral error of 14.3% on 4,197 sales, against 11.9% for the model. Model-adjusted comps (12.4%) borrow the model's judgement, so only time-adjusted comps drive disagreement flags.

### Reading vertical equity carefully

Vertical equity is the model's weak point. Both methods meet the IAAO PRD and PRB targets citywide on 2025, but in cross-validation the model fails both (PRD 1.037, PRB -0.059). Noise in sale prices makes PRD and PRB lean toward regressivity; noise in values pushes the calibration slope below one. The model's slope is closer to one than the assessor's (0.89 against 0.81). But both face the same price noise, and within value tiers the model's PRB (-0.230, -0.147, -0.003, low to high) is far more negative than the assessor's (-0.054, +0.019, +0.018), consistent with compression across similarly priced neighborhoods. Neither diagnostic settles it alone.

## Fairness audit

The audit asks whether **errors** differ by neighborhood group, not whether values do. Census tracts are grouped by majority race or ethnicity and by median household income quintile (ACS 2019-2023); demographics are never model inputs.

| Gap (group minus group) | Split | City Assessor | Model (public data) |
|---|---|---|---|
| Valuation level (mean log ratio), Black-majority minus White-majority | cross-validation 2021-2024 | -6.2 | -4.1 |
| Valuation level (mean log ratio), Black-majority minus White-majority | 2025 | -5.9 | -7.3 |
| Valuation level (median ratio), Black-majority minus White-majority | cross-validation 2021-2024 | -3.1 | -3.2 |
| Valuation level (median ratio), Black-majority minus White-majority | 2025 | -2.2 | -5.8 |
| Valuation level (mean log ratio), lowest minus highest income quintile | cross-validation 2021-2024 | -10.1 | -6.4 |
| Valuation level (mean log ratio), lowest minus highest income quintile | 2025 | -11.1 | -10.5 |
| Median error, Black-majority minus White-majority tracts | cross-validation 2021-2024 | +7.3 pts | +7.4 pts |
| Median error, lowest minus highest income quintile | cross-validation 2021-2024 | +12.4 pts | +13.1 pts |
| COD, Black-majority minus White-majority tracts | cross-validation 2021-2024 | +9.6 | +10.6 |

**On 2025 the model's Black-White gap in valuation level is larger than the assessor's**, by +1.4 [+0.1, +2.7] by mean log ratio and +3.6 [+2.0, +5.0] by median ratio. Relative to White-majority tracts, both methods value Black-majority tracts lower against their sale prices. In cross-validation (mean log ratio of value to price) the model is about level in Black-majority tracts (+0.001) and high in White-majority tracts (+0.042), while the assessor is low in both (-0.247 and -0.185). In cross-validation the model's gap is smaller by mean log ratio (change in its size -2.1 [-2.7, -1.5]) but not by median ratio (+0.2 [-0.7, +0.9]); on 2025 the income gaps do not clearly differ (-0.6 [-2.5, +1.5]). Level gaps are differences in the ratio statistic times one hundred (for the mean log ratio, log points), not percentage points. Neither method closes the accuracy gap, and in cross-validation the model's dispersion gap is slightly larger.

The audit also evaluates a cleaner sample in some groups. On 2021 to 2025 sales, the cleaning filters (F6 to F11) remove 7.5% of in-scope sales in Black-majority tracts against 4.9% in White-majority tracts; quick resales alone are 5.4% against 2.6%. Accuracy is therefore measured on fewer flipped homes there, and is likely overstated for every method.

## Where the assessor and the model disagree

The frozen model values every home it can score (122,214 of 123,168) at Jan 1 2025; after data-quality exclusions 121,467 are compared with their 2025 assessment, net of the assessor's citywide level (0.893). A home is listable when its assessment sits outside the model's 80% range **and** its comparable sales agree: 12,690 homes.

- **Set aside, not ranked:** 479 of those look like a special valuation or a data issue (an assessment held flat from 2024 while the median home's assessment rose 14.6%, a land-use change, a unit count or commercial building type that does not fit the land use).
- **Ranked:** of the 1,000 largest remaining gaps, 894 are assessed below the model and 106 above. The median tax difference is $1,600 a year at the 2025 net rate of $20.80 per thousand dollars of assessed value (Milwaukee County comparative rate table; $22.93 was the 2024 rate). The net rate is used because the school levy credit scales with assessed value under Wisconsin statute ([`docs/methods.md`](docs/methods.md#divergence-and-tax-dollars)).
- **Student-rental areas:** within 0.75 mi of the Marquette or UW-Milwaukee main campus lie 3.5% of the homes compared but 60 of the 106 listings above the model (87 ranked gaps in all). Rental income the model cannot see may explain them; this is untested, and their comparable sales point the same way as the model.
- **Groups:** each group's tax shift is corrected for the model's own bias for that group. A group is called over- or under-assessed only if (a) the corrected shift's 95% interval excludes zero, (b) the assessor's ratio on 2025 sales, which uses no model, agrees with an interval excluding zero, and (c) both survive tier-specific time adjustment. Passing all three: Hispanic-majority tracts and the lowest income quintile are under-assessed, assessed and taxed below their market-value share. Hispanic-majority tracts: persistent (the tier-adjusted sales check holds in 4 of 4 earlier rolls); a typical home (assessed $158,300) pays about $111 a year less, $1.7 million across 14,476 homes, and the fixed levy puts about $15 a year more on each of the other 106,991. The lowest income quintile: a 2025 result (the tier-adjusted sales check holds in 2 of 4 earlier rolls); a typical home (assessed $114,400) pays about $65 a year less, $1.7 million across 23,609 homes, and the fixed levy puts about $17 a year more on each of the other 97,858. The groups overlap, so the amounts do not add. White-majority tracts and the highest income quintile fail only the tier-specific time adjustment: consistent with assessments lagging faster appreciation in some areas, not a valuation pattern. Black-majority tracts: the two checks conflict (the bias-corrected shift, +1.1% [+0.6%, +1.6%], says over-assessed; the 2025 sales check, -1.3% [-2.4%, -0.4%], says under-assessed; both intervals exclude zero), so no claim is made. Details: [`docs/methods.md`](docs/methods.md#divergence-and-tax-dollars).

A private review page (with addresses) and a shareable page (map, group table, method) are built.

## The out-of-time test

Because 2025 was viewed during development, the cleaner test is out of time: the frozen model, refit on 47,897 sales through 2025 with no 2026 data, valued 122,222 homes as of Jan 1 2026. The file was committed with its hash and git tag `prospective-2026` on 2026-10-18, before the city released its 2026 sales file. It is not a pre-registration: it was committed well after the valuation date, the design was settled with 2025 in view, and transit stops are current. Once the 2026 sales file and roll are out, they are added to `sources.SOURCES` by hand and downloaded, and `python -m avm.prospective score` checks the hash and scores both against 2026 prices.

## How this compares to Cook County's model

The Cook County Assessor's Office publishes its residential model ([ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm), README read 2026-10-20). Both use LightGBM, value homes at Jan 1, explain values with SHAP and report IAAO ratio statistics. The differences:

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
