# Milwaukee AVM

**A public-data valuation model for Milwaukee single-family and duplex homes, tested against real sale prices and benchmarked against the City Assessor.**

## Key results

Held-out test: 4,197 arm's-length 2025 sales the model never saw, scored against the same time-adjusted prices for every method. Model = the frozen independent model (no assessed values used). Brackets are 95% paired-bootstrap intervals.

| Measure | Model (public data) | City Assessor | Model minus Assessor |
|---|---|---|---|
| Accuracy: level-neutral median absolute error (lower is better) | 11.9% | 13.3% | -1.4 pts [-2.0, -0.9] |
| Uniformity: COD (IAAO target <= 15) | 16.5 | 17.8 | -1.2 [-1.7, -0.7] |
| Vertical equity: PRD / PRB (targets 0.98 to 1.03 / -0.05 to 0.05) | 1.020 / -0.026 | 1.013 / +0.022 | +0.007 / -0.047 |
| 80% prediction interval coverage | 85.2% (median width 53% of value) | no interval published | |
| Prospective test, values as of Jan 1 2026 | 122,222 homes valued, hashed (`sha256` de73656af10f) and git-tagged on 2026-10-18, before any 2026 sale was published | roll 2026 | **pending**: scored once 2026 sales and roll 2026 are released |

## Summary

The City Assessor values every Milwaukee home each year, and the tax bill follows. This project asks how close a model built only from public records gets to real sale prices, and where it disagrees with the assessor. It is built from the 44,038 cleaned arm's-length sales of 2014 to 2024, with every feature taken from the assessment roll in force before the sale, and is frozen before it is scored on 2025.

- **It is more accurate than the assessor on the same sales.** Level-neutral median error 11.9% against 13.3%, and dispersion (COD) 16.5 against 17.8. Neither meets the IAAO COD target of <= 15.
- **It meets the IAAO vertical-equity targets** (PRD 1.020, PRB -0.026), as does the assessor (PRD 1.013, PRB +0.022). PRD and PRB measured against sale prices lean toward showing regressivity, so they are read next to calibration slopes, which are below one for every method (model 0.89, assessor 0.81).
- **Its ranges are conservative.** The 80% ranges contain 85.2% of 2025 sale prices, a little more than promised, and 81.0% in the lowest value tier.
- **Fairness is mixed.** Against the assessor, the model narrows the gap in valuation level between Black-majority and White-majority tracts (-4.1 pts against -6.2 pts), but not the gap in accuracy: every method is less accurate in Black-majority and lowest-income tracts (cross-validated on 2021 to 2024 sales).
- **Divergence is a place to look, not a verdict.** Of 121,471 homes valued at Jan 1 2025, 12,696 have assessments outside the model's 80% range with nearby comparable sales pointing the same way. Tax dollars use a provisional rate. After checking for the model's own bias, no neighborhood group is called over- or under-assessed.
- **The real test is still ahead.** Values for Jan 1 2026 were published as a hashed, tagged file before any 2026 sale existed.

## Results

### The benchmark: the City Assessor in 2025

On the IAAO ratio study (4,215 sales after the IAAO outlier trim, prices moved to the Jan 1 2025 trend level), the assessor's median ratio is 0.892 [0.885, 0.899], just under the IAAO band of 0.90 to 1.10. Dispersion is the larger problem: COD 17.6 [17.1, 18.1] against a target of <= 15. Vertical equity meets the targets (PRD 1.009, PRB +0.036). The level verdict depends on the valuation-date convention; the dispersion verdict does not.

### Head-to-head on identical 2025 sales

All four methods are scored on the same 4,197 sales and the same target prices, untrimmed. Level-neutral error divides each method's ratios by its own median first, so it measures accuracy rather than overall level, which a single factor would fix.

| | City Assessor | Hedonic baseline | Model (public data) | Model (assessment-informed) |
|---|---|---|---|---|
| Level-neutral median error | 13.3% | 12.9% | **11.9%** | 12.1% |
| Difference vs assessor | | -0.4 pts [-1.0, +0.2] | -1.4 pts [-2.0, -0.9] | -1.2 pts [-1.7, -0.6] |
| Within 10% of price | 35% | 39% | 44% | 42% |
| Within 20% of price | 64% | 67% | 71% | 70% |
| COD | 17.8 | 17.9 | 16.5 | 16.7 |
| Median ratio (level) | 0.894 | 1.001 | 0.997 | 1.007 |
| PRD | 1.013 | 1.021 | 1.020 | 1.021 |
| PRB | +0.022 | -0.015 | -0.026 | -0.021 |
| Calibration slope | 0.81 | 0.87 | 0.89 | 0.88 |

The hedonic baseline (log price on size, lot, age, bedrooms, baths, stories, style, a time trend and tract fixed effects) already matches the assessor (its difference interval includes zero); only the gradient-boosted models clearly beat it. Giving the model the prior assessed value (the assessment-informed variant) does not help once that value is made level-free.

### Where the gain came from

Feature groups were added one at a time and judged on cross-validation (2021 to 2024), never on 2025. Starting from structure and tract alone, the cross-validated error falls from 13.59% to 12.87% with every group; the single largest gain (0.35 pts) comes from prior sales nearby. The model learns on a detrended target, so it predicts at the valuation date's price level directly instead of extrapolating a time trend.

### Uncertainty

The 80% ranges come from conformal quantile regression calibrated on 2022 to 2024. On 2025 they contain 85.2% of prices overall, and 81.0%, 86.7% and 87.4% in the low, middle and high neighborhood value tiers. Coverage runs above the nominal 80% overall, so the ranges are slightly conservative; the low tier is closest to nominal and has the widest ranges. The median range is 53% of the value: wide, because the model has never seen inside the home.

### Comparable sales as an independent check

Each home also gets 3 to 5 comparable sales of the same property type, searched at 0.5, 1, 1.25 mi over 12 then 24 months before the valuation date, stopping at the first step with enough sales; quick-flip purchases and outlier sales are excluded. 99% of 2025 test sales find their comps at the tightest step. Time-adjusted comps alone (no model) reach a level-neutral error of 14.2% on 4,197 sales, against 11.9% for the model on the same sales. Model-adjusted comps reach 12.4%, but they borrow the model's own judgement, so only the time-adjusted comps drive disagreement flags.

### Reading vertical equity carefully

PRD and PRB put sale price in the denominator, so noise in prices makes any method look regressive. The calibration slope (log price on log value) is biased the other way, below one when the value is noisy. Every method's slope is below one, which means values are not compressed given the value, and the assessor's low slope (0.81) points to noisier assessments rather than better vertical equity. Both diagnostics are shown, and neither is read alone.
