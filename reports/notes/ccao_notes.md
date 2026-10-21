# Cook County Assessor (CCAO) residential model: notes

Source: https://github.com/ccao-data/model-res-avm (README, read 2026-10-17).
Sales validation code: https://github.com/ccao-data/model-sales-val

## What they do
- **Model:** LightGBM (via Tidymodels / their Lightsnip package), tuned with
  Bayesian optimization, early stopping on a validation set.
- **Training sales:** 9 years before the assessment date (about 400K rows).
  Drops sales under $10k, multi-PIN and multi-building sales, quit claim,
  executor and beneficial-interest deeds, same-price back-to-back sales of one
  PIN in a year, related-party, bank REO, government and financial-institution
  sales. Statistical price outliers flagged by a separate sales-validation repo.
- **Split:** most recent 10% of sales held out as test. The README describes
  rolling-origin CV, but a 2024 changelog entry says CV switched to V-fold.
  The page is inconsistent on this.
- **Time:** sale date encoded as days since 1997 plus year, month, quarter,
  day of year, post-COVID flag. For assessment, every parcel's sale date is
  set to Jan 1 of the assessment year and the model predicts at that date.
- **Features:** characteristics, lat/lon, neighborhood, township, tract,
  school district, proximity features (transit, parks, roads), walkability,
  flood factor, parcel shape, tax rate, number of prior sales, and **ACS
  census features (e.g. median income, median age)**.
- **Metrics:** COD, PRD, PRB, MKI plus R2, RMSE, MAE, MAPE, by geography.
- **Explainability:** SHAP per card. Comps from LightGBM leaf co-membership
  are experimental and not used in production.
- **Multi-card PINs:** card-level predictions aggregated to the PIN.

## Ideas to borrow
1. Predict at the valuation date (Jan 1) for the assessor comparison: train
   with a sale-date feature, then score every parcel with date = Jan 1. This
   is the clean way to do rule 3 for the model side of the ratio study.
2. Their exclusion list maps well onto rule 4. Same-price back-to-back sales
   and multi-parcel sales are detectable from our files; deed types only for
   years where MPROP conveyance dates survive (see stage1 convey-type table).
3. Leaf co-membership comps: a model-native comps method to compare against
   our nearest-neighbor comps in Stage 7.
4. MKI (Modified Kakwani Index) as a second vertical-equity measure next to
   PRD/PRB.
5. Prior-sale count is a sales-derived feature that is leakage-safe if built
   from sales before the target sale only.

## Where this project differs
- **No census demographics as features.** CCAO uses ACS income/age; we do not
  (rule 7). Tract demographics are used only in the Stage 8 equity audit.
- **Strict time-based CV only.** CCAO moved to V-fold; we keep time-based
  splits (rule 2).
- **Explicit assessor benchmark.** We compare against the city's own roll
  with the same metrics and the same held-out year, and run an
  assessment-informed variant (rule 6). CCAO is the assessor, so it has no
  outside benchmark.
- **Smaller data:** about 3,000 to 4,500 SF/duplex sales per year here vs
  about 45K per year in Cook County, so bootstrap CIs matter more.
