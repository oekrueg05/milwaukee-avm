# Stage 11: within-tier PRB (CV 2021-2024; 2025 reported only)

**Model verdict (rule as amended 2026-10-22): mixed, not resolved**; failing tiers ['low', 'mid', 'high']. Assessor (same rule): mixed, not resolved; failing tiers ['low'].

**Verdict.** Under the pre-registered rule (amended after the 2021 sample, before full results), the verdict is 'mixed, not resolved'. Substantively, no tier shows compression: the model's negative PRB is about what sale-price noise produces for a calibrated valuation of its accuracy, apart from a small mid-tier excess (E = -0.045). The model is calibrated for prediction; reaching the IAAO PRB band requires spreading values past calibration at a cost in accuracy, which is where the assessor sits. Caveat: the prior-sale subset over-represents White-majority and higher-income tracts.

No v2 model was built: the pre-registered rule allowed fixes only after a finding of real compression.

Model, per tier (CV 2021-2024):
- low: PRB -0.198 [-0.253, -0.161], calibrated null -0.209, E +0.011 [-0.046, +0.050] (includes 0 or within +/-0.05); D +0.172 [+0.111, +0.228] (above 0: less regressive against true value than a calibrated valuation)
- mid: PRB -0.188 [-0.224, -0.159], calibrated null -0.143, E -0.045 [-0.084, -0.016] (below 0); D +0.051 [-0.008, +0.101] (includes 0 or within +/-0.05)
- high: PRB -0.053 [-0.116, -0.011], calibrated null -0.071, E +0.018 [-0.047, +0.060] (includes 0 or within +/-0.05); D +0.040 [+0.004, +0.072] (above 0: less regressive against true value than a calibrated valuation)

Model without its own-prior-sale feature:
- low: PRB -0.199 [-0.255, -0.159], calibrated null -0.205, E +0.007 [-0.052, +0.048] (includes 0 or within +/-0.05); D +0.166 [+0.102, +0.223] (above 0: less regressive against true value than a calibrated valuation)
- mid: PRB -0.196 [-0.231, -0.167], calibrated null -0.151, E -0.045 [-0.082, -0.017] (below 0); D +0.052 [-0.014, +0.107] (includes 0 or within +/-0.05)
- high: PRB -0.062 [-0.134, -0.015], calibrated null -0.077, E +0.015 [-0.059, +0.061] (includes 0 or within +/-0.05); D +0.036 [-0.011, +0.074] (includes 0 or within +/-0.05)

Assessor:
- low: PRB -0.074 [-0.126, -0.032], calibrated null -0.172, E +0.098 [+0.046, +0.141] (above 0); D +0.254 [+0.185, +0.314] (above 0: less regressive against true value than a calibrated valuation)
- mid: PRB -0.038 [-0.067, -0.013], calibrated null -0.135, E +0.097 [+0.066, +0.124] (above 0); D +0.198 [+0.146, +0.253] (above 0: less regressive against true value than a calibrated valuation)
- high: PRB +0.012 [-0.020, +0.033], calibrated null -0.072, E +0.084 [+0.049, +0.106] (above 0); D +0.115 [+0.080, +0.150] (above 0: less regressive against true value than a calibrated valuation)

Rule reasons (model): low: noise explains >= half: True; E within/includes 0: True; E below 0: False; D within/includes 0: False; D below 0: False; D_noown within/includes 0: False; D_noown below 0: False | mid: noise explains >= half: True; E within/includes 0: False; E below 0: True; D within/includes 0: True; D below 0: False; D_noown within/includes 0: True; D_noown below 0: False | high: noise explains >= half: True; E within/includes 0: True; E below 0: False; D within/includes 0: False; D below 0: False; D_noown within/includes 0: True; D_noown below 0: False; D and D_noown disagree in direction

Rule reasons (assessor): low: noise explains >= half: True; E within/includes 0: False; E below 0: False; D within/includes 0: False; D below 0: False

**The tradeoff.** Underwriting needs calibrated predictions: within a tier, a home valued at V sells for about V on average (calibration slope of log price on log value = 1). IAAO vertical equity asks for equal ratios across value (within-tier PRB within +/-0.05). Scaling the model's values within each tier by k moves along one curve between the two (CV folds 2022-2024; the k grid for each fold is fit on earlier folds):
- low tier: rescaled to calibration (k = 0.81): slope 1.06, PRB -0.433, level-neutral MdAPE 21.0%. Frozen model (k = 1): slope 0.85, PRB -0.238, level-neutral MdAPE 21.2%. Spread to k = 1.24: slope 0.68, PRB -0.049, level-neutral MdAPE 22.6%. Assessor: slope 0.69, PRB -0.081, level-neutral MdAPE 22.4%.
- mid tier: rescaled to calibration (k = 0.93): slope 0.98, PRB -0.297, level-neutral MdAPE 10.5%. Frozen model (k = 1): slope 0.91, PRB -0.231, level-neutral MdAPE 10.5%. Spread to k = 1.08: slope 0.84, PRB -0.166, level-neutral MdAPE 10.8%. Assessor: slope 0.68, PRB -0.050, level-neutral MdAPE 12.3%.
- high tier: rescaled to calibration (k = 0.93): slope 0.98, PRB -0.110, level-neutral MdAPE 9.4%. Frozen model (k = 1): slope 0.92, PRB -0.055, level-neutral MdAPE 9.6%. Spread to k = 1.07: slope 0.85, PRB +0.002, level-neutral MdAPE 9.9%. Assessor: slope 0.80, PRB +0.022, level-neutral MdAPE 10.5%.
In every tier, PRB rises toward the IAAO band only as values are spread past calibration (the calibration slope falls as k grows), and the calibrated rescale has the most negative PRB. The frozen model sits nearer calibration (slope 0.85 to 0.92); the assessor sits at the spread end (slope 0.68 to 0.80).

## Per tier: observed PRB, calibrated null, E, Q, Q_null, D

| split | tier | method | robustness | n | mdape_ln | prb_obs | prb_lo | prb_hi | prb_null | prb_null_resid | E | e_lo | e_hi | n_proxy | q_raw | leak_bias | reliability | q | q_lo | q_hi | q_null | D | d_lo | d_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV 2021-2024 | low | model |  | 6193 | 0.216 | -0.198 | -0.253 | -0.161 | -0.209 | -0.207 | 0.011 | -0.046 | 0.05 | 884 | -0.006 | 0.011 | 0.78 | -0.022 | -0.062 | 0.012 | -0.193 | 0.172 | 0.111 | 0.228 |
| CV 2021-2024 | low | assessor |  | 6193 | 0.228 | -0.074 | -0.126 | -0.032 | -0.172 | -0.17 | 0.098 | 0.046 | 0.141 | 884 | 0.072 | 0 | 0.78 | 0.092 | 0.04 | 0.134 | -0.161 | 0.254 | 0.185 | 0.314 |
| CV 2021-2024 | low | model_no_own_sale |  | 6193 | 0.215 | -0.199 | -0.255 | -0.159 | -0.205 | -0.203 | 0.007 | -0.052 | 0.048 | 884 | -0.015 | 0 | 0.78 | -0.019 | -0.061 | 0.015 | -0.185 | 0.166 | 0.102 | 0.223 |
| CV 2021-2024 | low | model | older never-seen sale | 6193 | 0.216 | -0.198 | -0.253 | -0.161 | -0.209 | -0.207 | 0.011 | -0.045 | 0.051 | 99 | -0.016 | 0 | 0.784 | -0.02 | -0.109 | 0.116 | -0.163 | 0.142 | 0.002 | 0.308 |
| CV 2021-2024 | low | model | no significant permit | 6190 | 0.216 | -0.198 | -0.253 | -0.161 | -0.209 | -0.207 | 0.011 | -0.047 | 0.051 | 884 | -0.006 | 0.011 | 0.78 | -0.022 | -0.062 | 0.012 | -0.194 | 0.173 | 0.112 | 0.229 |
| CV 2021-2024 | mid | model |  | 6443 | 0.104 | -0.188 | -0.224 | -0.159 | -0.143 | -0.138 | -0.045 | -0.084 | -0.016 | 1575 | 0.002 | 0.031 | 0.777 | -0.037 | -0.074 | -0.006 | -0.088 | 0.051 | -0.008 | 0.101 |
| CV 2021-2024 | mid | assessor |  | 6443 | 0.121 | -0.038 | -0.067 | -0.013 | -0.135 | -0.132 | 0.097 | 0.066 | 0.124 | 1575 | 0.052 | 0 | 0.777 | 0.067 | 0.035 | 0.102 | -0.131 | 0.198 | 0.146 | 0.253 |
| CV 2021-2024 | mid | model_no_own_sale |  | 6443 | 0.107 | -0.196 | -0.231 | -0.167 | -0.151 | -0.146 | -0.045 | -0.082 | -0.017 | 1575 | -0.041 | 0 | 0.777 | -0.052 | -0.086 | -0.024 | -0.104 | 0.052 | -0.014 | 0.107 |
| CV 2021-2024 | mid | model | older never-seen sale | 6443 | 0.104 | -0.188 | -0.224 | -0.159 | -0.143 | -0.138 | -0.045 | -0.082 | -0.014 | 216 | -0.022 | 0 | 0.787 | -0.029 | -0.091 | 0.018 | -0.088 | 0.06 | -0.043 | 0.15 |
| CV 2021-2024 | mid | model | no significant permit | 6434 | 0.104 | -0.188 | -0.224 | -0.159 | -0.143 | -0.138 | -0.045 | -0.082 | -0.015 | 1573 | 0.002 | 0.031 | 0.777 | -0.037 | -0.074 | -0.006 | -0.088 | 0.051 | -0.009 | 0.101 |
| CV 2021-2024 | high | model |  | 5246 | 0.096 | -0.053 | -0.116 | -0.011 | -0.071 | -0.069 | 0.018 | -0.047 | 0.06 | 1649 | 0.021 | 0.013 | 0.899 | 0.009 | -0.017 | 0.026 | -0.031 | 0.04 | 0.004 | 0.072 |
| CV 2021-2024 | high | assessor |  | 5246 | 0.106 | 0.012 | -0.02 | 0.033 | -0.072 | -0.07 | 0.084 | 0.049 | 0.106 | 1649 | 0.057 | 0 | 0.899 | 0.064 | 0.043 | 0.086 | -0.051 | 0.115 | 0.08 | 0.15 |
| CV 2021-2024 | high | model_no_own_sale |  | 5246 | 0.099 | -0.062 | -0.134 | -0.015 | -0.077 | -0.075 | 0.015 | -0.059 | 0.061 | 1649 | -0.006 | 0 | 0.899 | -0.006 | -0.045 | 0.015 | -0.042 | 0.036 | -0.011 | 0.074 |
| CV 2021-2024 | high | model | older never-seen sale | 5246 | 0.096 | -0.053 | -0.116 | -0.011 | -0.071 | -0.069 | 0.018 | -0.046 | 0.061 | 292 | 0.009 | 0 | 0.912 | 0.009 | -0.024 | 0.055 | -0.03 | 0.04 | -0.02 | 0.102 |
| CV 2021-2024 | high | model | no significant permit | 5225 | 0.096 | -0.052 | -0.114 | -0.01 | -0.071 | -0.069 | 0.019 | -0.044 | 0.062 | 1643 | 0.021 | 0.013 | 0.899 | 0.009 | -0.017 | 0.026 | -0.031 | 0.04 | 0.002 | 0.073 |
| 2025 | low | model |  | 1354 | 0.19 | -0.23 | -0.286 | -0.182 | -0.204 | -0.201 | -0.026 | -0.088 | 0.026 | 336 | -0.005 | 0.013 | 0.775 | -0.023 | -0.092 | 0.028 | -0.136 | 0.112 | 0.013 | 0.199 |
| 2025 | low | assessor |  | 1354 | 0.214 | -0.054 | -0.095 | -0.014 | -0.163 | -0.161 | 0.109 | 0.061 | 0.154 | 336 | 0.019 | 0 | 0.775 | 0.025 | -0.036 | 0.082 | -0.145 | 0.171 | 0.082 | 0.253 |
| 2025 | low | model_no_own_sale |  | 1354 | 0.198 | -0.235 | -0.293 | -0.184 | -0.211 | -0.208 | -0.023 | -0.086 | 0.033 | 336 | -0.027 | 0 | 0.775 | -0.034 | -0.112 | 0.024 | -0.15 | 0.115 | 0.011 | 0.207 |
| 2025 | low | model | older never-seen sale | 1354 | 0.19 | -0.23 | -0.286 | -0.182 | -0.205 | -0.201 | -0.026 | -0.088 | 0.028 | 58 | 0.044 | 0 | 0.698 | 0.062 | -0.189 | 0.245 | -0.161 | 0.223 | -0.072 | 0.496 |
| 2025 | low | model | no significant permit | 1354 | 0.19 | -0.23 | -0.286 | -0.182 | -0.204 | -0.201 | -0.026 | -0.088 | 0.026 | 336 | -0.005 | 0.013 | 0.775 | -0.023 | -0.092 | 0.028 | -0.136 | 0.112 | 0.013 | 0.199 |
| 2025 | mid | model |  | 1413 | 0.106 | -0.147 | -0.219 | -0.09 | -0.137 | -0.132 | -0.01 | -0.084 | 0.052 | 467 | -0.003 | 0.036 | 0.765 | -0.05 | -0.137 | 0.028 | -0.102 | 0.051 | -0.051 | 0.148 |
| 2025 | mid | assessor |  | 1413 | 0.116 | 0.019 | -0.05 | 0.084 | -0.109 | -0.106 | 0.128 | 0.057 | 0.196 | 467 | 0.042 | 0 | 0.765 | 0.055 | -0.046 | 0.157 | -0.117 | 0.172 | 0.059 | 0.296 |
| 2025 | mid | model_no_own_sale |  | 1413 | 0.108 | -0.157 | -0.229 | -0.098 | -0.144 | -0.139 | -0.013 | -0.088 | 0.051 | 467 | -0.043 | 0 | 0.765 | -0.056 | -0.128 | 0.015 | -0.114 | 0.059 | -0.041 | 0.157 |
| 2025 | mid | model | older never-seen sale | 1413 | 0.106 | -0.147 | -0.219 | -0.09 | -0.137 | -0.132 | -0.01 | -0.087 | 0.051 | 102 | 0.022 | 0 | 0.773 | 0.028 | -0.056 | 0.113 | -0.108 | 0.136 | -0.006 | 0.29 |
| 2025 | mid | model | no significant permit | 1413 | 0.106 | -0.147 | -0.219 | -0.09 | -0.137 | -0.132 | -0.01 | -0.084 | 0.052 | 467 | -0.003 | 0.036 | 0.765 | -0.05 | -0.137 | 0.028 | -0.102 | 0.051 | -0.051 | 0.148 |
| 2025 | high | model |  | 1267 | 0.086 | -0.003 | -0.078 | 0.031 | -0.075 | -0.072 | 0.071 | -0.007 | 0.109 | 540 | 0.051 | 0.018 | 0.88 | 0.038 | 0.003 | 0.074 | -0.012 | 0.05 | 0.001 | 0.102 |
| 2025 | high | assessor |  | 1267 | 0.105 | 0.018 | -0.024 | 0.054 | -0.094 | -0.091 | 0.111 | 0.069 | 0.154 | 540 | 0.034 | 0 | 0.88 | 0.039 | -0.002 | 0.093 | -0.064 | 0.103 | 0.043 | 0.173 |
| 2025 | high | model_no_own_sale |  | 1267 | 0.088 | -0.017 | -0.102 | 0.02 | -0.077 | -0.074 | 0.06 | -0.028 | 0.101 | 540 | 0.014 | 0 | 0.88 | 0.016 | -0.028 | 0.052 | -0.011 | 0.028 | -0.031 | 0.084 |
| 2025 | high | model | older never-seen sale | 1267 | 0.086 | -0.003 | -0.078 | 0.031 | -0.075 | -0.072 | 0.071 | -0.005 | 0.109 | 127 | 0.023 | 0 | 0.925 | 0.025 | -0.019 | 0.104 | -0.009 | 0.034 | -0.045 | 0.136 |
| 2025 | high | model | no significant permit | 1264 | 0.086 | -0.002 | -0.077 | 0.032 | -0.074 | -0.072 | 0.072 | -0.005 | 0.11 | 539 | 0.051 | 0.018 | 0.88 | 0.038 | 0.004 | 0.075 | -0.011 | 0.049 | 0.001 | 0.101 |

## Calibration slopes

| split | tier | method | n | slope | hc1_lo | hc1_hi | cluster_lo | cluster_hi |
|---|---|---|---|---|---|---|---|---|
| CV 2021-2024 | low | model | 6193 | 0.829 | 0.804 | 0.854 | 0.798 | 0.859 |
| CV 2021-2024 | low | assessor | 6193 | 0.692 | 0.668 | 0.716 | 0.65 | 0.734 |
| CV 2021-2024 | low | model_no_own_sale | 6193 | 0.828 | 0.802 | 0.853 | 0.798 | 0.858 |
| CV 2021-2024 | mid | model | 6443 | 0.914 | 0.889 | 0.938 | 0.887 | 0.941 |
| CV 2021-2024 | mid | assessor | 6443 | 0.701 | 0.677 | 0.724 | 0.664 | 0.737 |
| CV 2021-2024 | mid | model_no_own_sale | 6443 | 0.915 | 0.89 | 0.941 | 0.888 | 0.942 |
| CV 2021-2024 | high | model | 5246 | 0.919 | 0.898 | 0.94 | 0.896 | 0.942 |
| CV 2021-2024 | high | assessor | 5246 | 0.814 | 0.794 | 0.833 | 0.744 | 0.883 |
| CV 2021-2024 | high | model_no_own_sale | 5246 | 0.921 | 0.899 | 0.942 | 0.897 | 0.944 |
| 2025 | low | model | 1354 | 0.869 | 0.814 | 0.923 | 0.814 | 0.923 |
| 2025 | low | assessor | 1354 | 0.676 | 0.626 | 0.726 | 0.615 | 0.737 |
| 2025 | low | model_no_own_sale | 1354 | 0.864 | 0.81 | 0.919 | 0.809 | 0.92 |
| 2025 | mid | model | 1413 | 0.866 | 0.815 | 0.917 | 0.796 | 0.936 |
| 2025 | mid | assessor | 1413 | 0.67 | 0.621 | 0.719 | 0.587 | 0.753 |
| 2025 | mid | model_no_own_sale | 1413 | 0.865 | 0.812 | 0.919 | 0.792 | 0.938 |
| 2025 | high | model | 1267 | 0.869 | 0.837 | 0.901 | 0.823 | 0.916 |
| 2025 | high | assessor | 1267 | 0.793 | 0.754 | 0.832 | 0.687 | 0.899 |
| 2025 | high | model_no_own_sale | 1267 | 0.877 | 0.843 | 0.911 | 0.828 | 0.925 |

## Tradeoff (CV folds 2022-2024): values scaled within tier by k

| tier | point | k_mean | k_min | k_max | slope_earlier_folds_mean | calibration_slope | mdape_ln | prb_obs | prb_null | E | q | q_null | D | d_lo | d_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | step 0 | 1 | 1 | 1 | 0.806 | 0.851 | 0.212 | -0.238 | -0.229 | -0.008 | -0.023 | -0.186 | 0.164 | 0.098 | 0.229 |
| low | step 1 | 1.035 | 1.03 | 1.041 | 0.806 | 0.823 | 0.213 | -0.207 | -0.217 | 0.009 | -0.004 | -0.177 | 0.173 | 0.108 | 0.237 |
| low | step 2 | 1.069 | 1.06 | 1.082 | 0.806 | 0.796 | 0.215 | -0.178 | -0.207 | 0.029 | 0.014 | -0.172 | 0.187 | 0.125 | 0.25 |
| low | step 3 | 1.104 | 1.09 | 1.123 | 0.806 | 0.771 | 0.215 | -0.15 | -0.195 | 0.045 | 0.032 | -0.163 | 0.195 | 0.135 | 0.258 |
| low | step 4 | 1.138 | 1.12 | 1.164 | 0.806 | 0.748 | 0.217 | -0.124 | -0.191 | 0.067 | 0.05 | -0.164 | 0.214 | 0.154 | 0.277 |
| low | step 5 | 1.173 | 1.151 | 1.205 | 0.806 | 0.725 | 0.219 | -0.098 | -0.186 | 0.088 | 0.068 | -0.164 | 0.232 | 0.171 | 0.295 |
| low | step 6 | 1.207 | 1.181 | 1.246 | 0.806 | 0.705 | 0.222 | -0.073 | -0.181 | 0.108 | 0.085 | -0.163 | 0.248 | 0.185 | 0.31 |
| low | step 7 | 1.242 | 1.211 | 1.287 | 0.806 | 0.685 | 0.226 | -0.049 | -0.177 | 0.128 | 0.102 | -0.164 | 0.266 | 0.202 | 0.328 |
| low | calibrated (k = b) | 0.806 | 0.777 | 0.826 | 0.806 | 1.056 | 0.21 | -0.433 | -0.32 | -0.113 | -0.132 | -0.246 | 0.114 | 0.042 | 0.19 |
| low | assessor |  |  |  |  | 0.692 | 0.224 | -0.081 | -0.176 | 0.095 | 0.086 | -0.156 | 0.242 | 0.171 | 0.312 |
| mid | step 0 | 1 | 1 | 1 | 0.926 | 0.911 | 0.105 | -0.231 | -0.17 | -0.061 | -0.021 | -0.112 | 0.092 | 0.023 | 0.151 |
| mid | step 1 | 1.011 | 1.01 | 1.012 | 0.926 | 0.901 | 0.106 | -0.221 | -0.166 | -0.055 | -0.015 | -0.109 | 0.095 | 0.027 | 0.153 |
| mid | step 2 | 1.023 | 1.019 | 1.024 | 0.926 | 0.891 | 0.105 | -0.212 | -0.164 | -0.048 | -0.009 | -0.109 | 0.1 | 0.033 | 0.158 |
| mid | step 3 | 1.034 | 1.029 | 1.037 | 0.926 | 0.881 | 0.106 | -0.202 | -0.163 | -0.039 | -0.003 | -0.112 | 0.109 | 0.044 | 0.166 |
| mid | step 4 | 1.045 | 1.038 | 1.049 | 0.926 | 0.872 | 0.107 | -0.193 | -0.161 | -0.032 | 0.003 | -0.111 | 0.114 | 0.05 | 0.172 |
| mid | step 5 | 1.057 | 1.048 | 1.061 | 0.926 | 0.863 | 0.107 | -0.184 | -0.158 | -0.026 | 0.009 | -0.111 | 0.119 | 0.056 | 0.176 |
| mid | step 6 | 1.068 | 1.058 | 1.073 | 0.926 | 0.853 | 0.108 | -0.175 | -0.155 | -0.02 | 0.015 | -0.108 | 0.123 | 0.059 | 0.179 |
| mid | step 7 | 1.079 | 1.067 | 1.086 | 0.926 | 0.845 | 0.108 | -0.166 | -0.153 | -0.013 | 0.021 | -0.107 | 0.128 | 0.066 | 0.184 |
| mid | calibrated (k = b) | 0.926 | 0.921 | 0.937 | 0.926 | 0.983 | 0.105 | -0.297 | -0.189 | -0.108 | -0.059 | -0.117 | 0.058 | -0.02 | 0.123 |
| mid | assessor |  |  |  |  | 0.684 | 0.123 | -0.05 | -0.149 | 0.099 | 0.085 | -0.155 | 0.24 | 0.176 | 0.311 |
| high | step 0 | 1 | 1 | 1 | 0.931 | 0.917 | 0.096 | -0.055 | -0.077 | 0.022 | 0.008 | -0.032 | 0.04 | -0.002 | 0.077 |
| high | step 1 | 1.011 | 1.009 | 1.012 | 0.931 | 0.907 | 0.096 | -0.047 | -0.076 | 0.029 | 0.014 | -0.032 | 0.046 | 0.004 | 0.082 |
| high | step 2 | 1.021 | 1.017 | 1.023 | 0.931 | 0.898 | 0.096 | -0.039 | -0.075 | 0.036 | 0.02 | -0.033 | 0.054 | 0.011 | 0.089 |
| high | step 3 | 1.032 | 1.026 | 1.035 | 0.931 | 0.888 | 0.096 | -0.03 | -0.075 | 0.045 | 0.026 | -0.036 | 0.062 | 0.019 | 0.098 |
| high | step 4 | 1.042 | 1.035 | 1.046 | 0.931 | 0.879 | 0.097 | -0.022 | -0.074 | 0.051 | 0.033 | -0.035 | 0.068 | 0.025 | 0.104 |
| high | step 5 | 1.053 | 1.044 | 1.058 | 0.931 | 0.871 | 0.098 | -0.014 | -0.072 | 0.058 | 0.039 | -0.034 | 0.073 | 0.03 | 0.109 |
| high | step 6 | 1.063 | 1.052 | 1.07 | 0.931 | 0.862 | 0.099 | -0.006 | -0.072 | 0.065 | 0.045 | -0.035 | 0.08 | 0.038 | 0.116 |
| high | step 7 | 1.074 | 1.061 | 1.081 | 0.931 | 0.853 | 0.099 | 0.002 | -0.072 | 0.074 | 0.051 | -0.038 | 0.09 | 0.048 | 0.125 |
| high | calibrated (k = b) | 0.931 | 0.925 | 0.942 | 0.931 | 0.984 | 0.094 | -0.11 | -0.084 | -0.026 | -0.032 | -0.028 | -0.004 | -0.046 | 0.037 |
| high | assessor |  |  |  |  | 0.801 | 0.105 | 0.022 | -0.075 | 0.097 | 0.069 | -0.052 | 0.121 | 0.084 | 0.164 |

## Sale-price noise (repeat sales)

| tier | pairs | parcels | sigma_p2 | sigma_p2_lo | sigma_p2_hi | sigma_p | drift_per_year | drift_lo | drift_hi | sigma_p2_robust_mad | drift_robust_mad | residual_kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 1214 | 1145 | 0.053 | 0.044 | 0.062 | 0.229 | 0.001 | -0.004 | 0.006 | 0.038 | -0.002 | -0.233 |
| mid | 2617 | 2410 | 0.017 | 0.012 | 0.021 | 0.129 | 0.001 | -0.001 | 0.003 | 0.007 | 0.001 | 2.431 |
| high | 2480 | 2236 | 0.016 | 0.012 | 0.02 | 0.128 | -0.001 | -0.002 | 0.001 | 0.005 | 0.002 | 2.697 |

## Tier-specific time adjustment

| split | tier | method | n | prb_citywide_time_adj | prb_tercile_time_adj |
|---|---|---|---|---|---|
| CV | low | model | 6193 | -0.198 | -0.205 |
| CV | low | assessor | 6193 | -0.074 | -0.081 |
| CV | low | model_no_own_sale | 6193 | -0.199 | -0.204 |
| CV | mid | model | 6443 | -0.188 | -0.198 |
| CV | mid | assessor | 6443 | -0.038 | -0.045 |
| CV | mid | model_no_own_sale | 6443 | -0.196 | -0.205 |
| CV | high | model | 5246 | -0.053 | -0.053 |
| CV | high | assessor | 5246 | 0.012 | 0.012 |
| CV | high | model_no_own_sale | 5246 | -0.062 | -0.062 |
| 2025 | low | model | 1354 | -0.23 | -0.229 |
| 2025 | low | assessor | 1354 | -0.054 | -0.053 |
| 2025 | low | model_no_own_sale | 1354 | -0.235 | -0.234 |
| 2025 | mid | model | 1413 | -0.147 | -0.139 |
| 2025 | mid | assessor | 1413 | 0.019 | 0.027 |
| 2025 | mid | model_no_own_sale | 1413 | -0.157 | -0.149 |
| 2025 | high | model | 1267 | -0.003 | 0 |
| 2025 | high | assessor | 1267 | 0.018 | 0.021 |
| 2025 | high | model_no_own_sale | 1267 | -0.017 | -0.013 |

Flags: none
