# Milwaukee AVM

**A public-data valuation model for Milwaukee single-family and duplex homes, tested against real sale prices and benchmarked against the City Assessor.**

## Key results

Held-out test: {{n_test}} arm's-length 2025 sales the model never saw, scored against the same time-adjusted prices for every method. Model = the frozen independent model (no assessed values used). Brackets are 95% paired-bootstrap intervals.

| Measure | Model (public data) | City Assessor | Model minus Assessor |
|---|---|---|---|
| Accuracy: level-neutral median absolute error (lower is better) | {{mdape_ln_primary}} | {{mdape_ln_assessor}} | {{mdape_ln_primary_diff}} {{mdape_ln_primary_diff_ci}} |
| Uniformity: COD (IAAO target {{target_cod}}) | {{cod_primary}} | {{cod_assessor}} | {{cod_primary_diff}} {{cod_primary_diff_ci}} |
| Vertical equity: PRD / PRB (targets {{target_prd}} / {{target_prb}}) | {{prd_primary}} / {{prb_primary}} | {{prd_assessor}} / {{prb_assessor}} | {{prd_primary_diff}} / {{prb_primary_diff}} |
| 80% prediction interval coverage | {{cover_all}} (median width {{width_all}} of value) | no interval published | |
| Prospective test, values as of Jan 1 2026 | {{pros_rows}} homes valued, hashed (`sha256` {{pros_sha}}) and git-tagged on {{pros_date}}, before any 2026 sale was published | roll 2026 | **pending**: scored once 2026 sales and roll 2026 are released |

## Summary

The City Assessor values every Milwaukee home each year, and the tax bill follows. This project asks how close a model built only from public records gets to real sale prices, and where it disagrees with the assessor. It is built from the {{n_train}} cleaned arm's-length sales of 2014 to 2024, with every feature taken from the assessment roll in force before the sale, and is frozen before it is scored on 2025.

- **It is more accurate than the assessor on the same sales.** Level-neutral median error {{mdape_ln_primary}} against {{mdape_ln_assessor}}, and dispersion (COD) {{cod_primary}} against {{cod_assessor}}. Neither meets the IAAO COD target of {{target_cod}}.
- **It meets the IAAO vertical-equity targets** (PRD {{prd_primary}}, PRB {{prb_primary}}), as does the assessor (PRD {{prd_assessor}}, PRB {{prb_assessor}}). PRD and PRB measured against sale prices lean toward showing regressivity, so they are read next to calibration slopes, which are below one for every method (model {{calibration_slope_primary}}, assessor {{calibration_slope_assessor}}).
- **Its ranges are conservative.** The 80% ranges contain {{cover_all}} of 2025 sale prices, a little more than promised, and {{cover_low}} in the lowest value tier.
- **Fairness is mixed.** Against the assessor, the model narrows the gap in valuation level between Black-majority and White-majority tracts ({{gap_bw_bias_mean_log_primary}} against {{gap_bw_bias_mean_log_assessor}}), but not the gap in accuracy: every method is less accurate in Black-majority and lowest-income tracts (cross-validated on 2021 to 2024 sales).
- **Divergence is a place to look, not a verdict.** Of {{s9_parcels_scored}} homes valued at Jan 1 2025, {{s9_listable_outside_and_comps_agree}} have assessments outside the model's 80% range with nearby comparable sales pointing the same way. Tax dollars use a {{tax_status}} rate. After checking for the model's own bias, {{s9_group_claims}} neighborhood group is called over- or under-assessed.
- **The real test is still ahead.** Values for Jan 1 2026 were published as a hashed, tagged file before any 2026 sale existed.

## Results

### The benchmark: the City Assessor in 2025

On the IAAO ratio study ({{a3_n}} sales after the IAAO outlier trim, prices moved to the Jan 1 2025 trend level), the assessor's median ratio is {{a3_median_ratio}} {{a3_median_ratio_ci}}, just under the IAAO band of {{target_median_ratio}}. Dispersion is the larger problem: COD {{a3_cod}} {{a3_cod_ci}} against a target of {{target_cod}}. Vertical equity meets the targets (PRD {{a3_prd}}, PRB {{a3_prb}}). The level verdict depends on the valuation-date convention; the dispersion verdict does not.

### Head-to-head on identical 2025 sales

All four methods are scored on the same {{n_test}} sales and the same target prices, untrimmed. Level-neutral error divides each method's ratios by its own median first, so it measures accuracy rather than overall level, which a single factor would fix.

| | City Assessor | Hedonic baseline | Model (public data) | Model (assessment-informed) |
|---|---|---|---|---|
| Level-neutral median error | {{mdape_ln_assessor}} | {{mdape_ln_hedonic}} | **{{mdape_ln_primary}}** | {{mdape_ln_secondary}} |
| Difference vs assessor | | {{mdape_ln_hedonic_diff}} {{mdape_ln_hedonic_diff_ci}} | {{mdape_ln_primary_diff}} {{mdape_ln_primary_diff_ci}} | {{mdape_ln_secondary_diff}} {{mdape_ln_secondary_diff_ci}} |
| Within 10% of price | {{within_10_assessor}} | {{within_10_hedonic}} | {{within_10_primary}} | {{within_10_secondary}} |
| Within 20% of price | {{within_20_assessor}} | {{within_20_hedonic}} | {{within_20_primary}} | {{within_20_secondary}} |
| COD | {{cod_assessor}} | {{cod_hedonic}} | {{cod_primary}} | {{cod_secondary}} |
| Median ratio (level) | {{median_ratio_assessor}} | {{median_ratio_hedonic}} | {{median_ratio_primary}} | {{median_ratio_secondary}} |
| PRD | {{prd_assessor}} | {{prd_hedonic}} | {{prd_primary}} | {{prd_secondary}} |
| PRB | {{prb_assessor}} | {{prb_hedonic}} | {{prb_primary}} | {{prb_secondary}} |
| Calibration slope | {{calibration_slope_assessor}} | {{calibration_slope_hedonic}} | {{calibration_slope_primary}} | {{calibration_slope_secondary}} |

The hedonic baseline (log price on size, lot, age, bedrooms, baths, stories, style, a time trend and tract fixed effects) already matches the assessor (its difference interval includes zero); only the gradient-boosted models clearly beat it. Giving the model the prior assessed value (the assessment-informed variant) does not help once that value is made level-free.

### Where the gain came from

Feature groups were added one at a time and judged on cross-validation (2021 to 2024), never on 2025. Starting from structure and tract alone, the cross-validated error falls from {{lift_cv_base}} to {{lift_cv_all}} with every group; the single largest gain ({{lift_best_gain}}) comes from {{lift_best_group}}. The model learns on a detrended target, so it predicts at the valuation date's price level directly instead of extrapolating a time trend.

### Uncertainty

The 80% ranges come from conformal quantile regression calibrated on 2022 to 2024. On 2025 they contain {{cover_all}} of prices overall, and {{cover_low}}, {{cover_mid}} and {{cover_high}} in the low, middle and high neighborhood value tiers. Coverage runs above the nominal 80% overall, so the ranges are slightly conservative; the low tier is closest to nominal and has the widest ranges. The median range is {{width_all}} of the value: wide, because the model has never seen inside the home.

### Comparable sales as an independent check

Each home also gets 3 to 5 comparable sales of the same property type, searched at {{comp_radii}} over {{comp_windows}} before the valuation date, stopping at the first step with enough sales; quick-flip purchases and outlier sales are excluded. {{comp_first_tier_share}} of 2025 test sales find their comps at the tightest step. Time-adjusted comps alone (no model) reach a level-neutral error of {{comp_a_mdape}} on {{comp_n}} sales, against {{comp_model_mdape}} for the model on the same sales. Model-adjusted comps reach {{comp_b_mdape}}, but they borrow the model's own judgement, so only the time-adjusted comps drive disagreement flags.

### Reading vertical equity carefully

PRD and PRB put sale price in the denominator, so noise in prices makes any method look regressive. The calibration slope (log price on log value) is biased the other way, below one when the value is noisy. Every method's slope is below one, which means values are not compressed given the value, and the assessor's low slope ({{calibration_slope_assessor}}) points to noisier assessments rather than better vertical equity. Both diagnostics are shown, and neither is read alone.
