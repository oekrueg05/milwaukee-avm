# Stage 11: within-tier PRB (CV 2021 (sample))

**Model verdict (rule as amended 2026-10-22): mixed, not resolved**; failing tiers ['low', 'mid', 'high']. Assessor (same rule): mixed, not resolved; failing tiers ['low'].

Reasons (model): low: noise explains >= half: True; E within/includes 0: True; E below 0: False; D within/includes 0: False; D below 0: False; D_noown within/includes 0: False; D_noown below 0: False | mid: noise explains >= half: True; E within/includes 0: True; E below 0: False; D within/includes 0: True; D below 0: False; D_noown within/includes 0: True; D_noown below 0: False | high: noise explains >= half: True; E within/includes 0: True; E below 0: False; D within/includes 0: True; D below 0: False; D_noown within/includes 0: True; D_noown below 0: False

Reasons (assessor): low: noise explains >= half: True; E within/includes 0: False; E below 0: False; D within/includes 0: False; D below 0: False

## Per tier: observed PRB, calibrated null, E, Q, Q_null, D

| split | tier | method | robustness | n | mdape_ln | prb_obs | prb_lo | prb_hi | prb_null | prb_null_resid | E | e_lo | e_hi | n_proxy | q_raw | leak_bias | reliability | q | q_lo | q_hi | q_null | D | d_lo | d_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV 2021 (sample) | low | model |  | 1558 | 0.226 | -0.156 | -0.257 | -0.082 | -0.211 | -0.209 | 0.055 | -0.048 | 0.132 | 132 | 0.015 | 0.01 | 0.726 | 0.006 | -0.115 | 0.118 | -0.228 | 0.234 | 0.078 | 0.385 |
| CV 2021 (sample) | low | assessor |  | 1558 | 0.242 | -0.062 | -0.142 | 0.004 | -0.192 | -0.191 | 0.13 | 0.046 | 0.202 | 132 | 0.107 | 0 | 0.726 | 0.147 | 0.032 | 0.266 | -0.194 | 0.341 | 0.204 | 0.493 |
| CV 2021 (sample) | low | model_no_own_sale |  | 1558 | 0.225 | -0.158 | -0.261 | -0.081 | -0.208 | -0.206 | 0.05 | -0.055 | 0.13 | 132 | 0.022 | 0 | 0.726 | 0.031 | -0.078 | 0.142 | -0.213 | 0.244 | 0.099 | 0.394 |
| CV 2021 (sample) | low | model | older never-seen sale | 1558 | 0.226 | -0.156 | -0.257 | -0.082 | -0.21 | -0.208 | 0.054 | -0.047 | 0.134 | 11 | 0.038 | 0 | 0.664 | 0.057 |  |  | -0.123 | 0.179 |  |  |
| CV 2021 (sample) | low | model | no significant permit | 1556 | 0.226 | -0.155 | -0.256 | -0.082 | -0.21 | -0.209 | 0.055 | -0.048 | 0.132 | 132 | 0.015 | 0.01 | 0.726 | 0.006 | -0.115 | 0.118 | -0.227 | 0.233 | 0.075 | 0.387 |
| CV 2021 (sample) | mid | model |  | 1793 | 0.1 | -0.19 | -0.251 | -0.14 | -0.151 | -0.145 | -0.039 | -0.103 | 0.015 | 350 | -0.056 | 0.022 | 0.789 | -0.099 | -0.159 | -0.061 | -0.074 | -0.025 | -0.114 | 0.053 |
| CV 2021 (sample) | mid | assessor |  | 1793 | 0.116 | -0.02 | -0.076 | 0.025 | -0.139 | -0.135 | 0.119 | 0.06 | 0.169 | 350 | 0.022 | 0 | 0.789 | 0.028 | -0.033 | 0.082 | -0.117 | 0.145 | 0.062 | 0.227 |
| CV 2021 (sample) | mid | model_no_own_sale |  | 1793 | 0.1 | -0.191 | -0.257 | -0.138 | -0.152 | -0.146 | -0.039 | -0.107 | 0.019 | 350 | -0.07 | 0 | 0.789 | -0.089 | -0.153 | -0.047 | -0.076 | -0.013 | -0.109 | 0.068 |
| CV 2021 (sample) | mid | model | older never-seen sale | 1793 | 0.1 | -0.19 | -0.251 | -0.14 | -0.151 | -0.145 | -0.04 | -0.105 | 0.014 | 30 | -0.027 | 0 | 0.862 | -0.032 | -0.112 | 0.257 | -0.07 | 0.038 | -0.193 | 0.426 |
| CV 2021 (sample) | mid | model | no significant permit | 1791 | 0.1 | -0.19 | -0.251 | -0.14 | -0.151 | -0.145 | -0.039 | -0.103 | 0.015 | 350 | -0.056 | 0.022 | 0.789 | -0.099 | -0.159 | -0.061 | -0.074 | -0.025 | -0.115 | 0.053 |
| CV 2021 (sample) | high | model |  | 1421 | 0.098 | -0.065 | -0.154 | -0.017 | -0.075 | -0.073 | 0.01 | -0.08 | 0.061 | 402 | 0.028 | 0.012 | 0.893 | 0.018 | -0.018 | 0.041 | -0.038 | 0.057 | -0.001 | 0.105 |
| CV 2021 (sample) | high | assessor |  | 1421 | 0.108 | -0.002 | -0.043 | 0.023 | -0.078 | -0.076 | 0.076 | 0.033 | 0.105 | 402 | 0.053 | 0 | 0.893 | 0.06 | 0.03 | 0.101 | -0.062 | 0.122 | 0.068 | 0.181 |
| CV 2021 (sample) | high | model_no_own_sale |  | 1421 | 0.104 | -0.076 | -0.179 | -0.021 | -0.084 | -0.082 | 0.008 | -0.096 | 0.066 | 402 | 0.001 | 0 | 0.893 | 0.001 | -0.06 | 0.036 | -0.054 | 0.055 | -0.023 | 0.113 |
| CV 2021 (sample) | high | model | older never-seen sale | 1421 | 0.098 | -0.065 | -0.154 | -0.017 | -0.075 | -0.073 | 0.01 | -0.081 | 0.062 | 44 | 0.024 | 0 | 0.908 | 0.027 | -0.071 | 0.08 | -0.035 | 0.062 | -0.074 | 0.18 |
| CV 2021 (sample) | high | model | no significant permit | 1413 | 0.098 | -0.061 | -0.145 | -0.015 | -0.075 | -0.073 | 0.014 | -0.074 | 0.062 | 401 | 0.029 | 0.012 | 0.893 | 0.019 | -0.017 | 0.043 | -0.038 | 0.057 | 0.001 | 0.108 |

## Calibration slopes

| split | tier | method | n | slope | hc1_lo | hc1_hi | cluster_lo | cluster_hi |
|---|---|---|---|---|---|---|---|---|
| CV 2021 (sample) | low | model | 1558 | 0.777 | 0.731 | 0.822 | 0.711 | 0.842 |
| CV 2021 (sample) | low | assessor | 1558 | 0.693 | 0.651 | 0.736 | 0.632 | 0.755 |
| CV 2021 (sample) | low | model_no_own_sale | 1558 | 0.782 | 0.736 | 0.828 | 0.714 | 0.85 |
| CV 2021 (sample) | mid | model | 1793 | 0.921 | 0.876 | 0.966 | 0.868 | 0.974 |
| CV 2021 (sample) | mid | assessor | 1793 | 0.744 | 0.707 | 0.782 | 0.683 | 0.806 |
| CV 2021 (sample) | mid | model_no_own_sale | 1793 | 0.913 | 0.868 | 0.959 | 0.86 | 0.966 |
| CV 2021 (sample) | high | model | 1421 | 0.925 | 0.881 | 0.969 | 0.89 | 0.96 |
| CV 2021 (sample) | high | assessor | 1421 | 0.848 | 0.816 | 0.879 | 0.779 | 0.916 |
| CV 2021 (sample) | high | model_no_own_sale | 1421 | 0.928 | 0.885 | 0.972 | 0.894 | 0.963 |

## Tradeoff (CV folds 2022-2024): values scaled within tier by k

skipped in the sample

## Sale-price noise (repeat sales)

| tier | pairs | parcels | sigma_p2 | sigma_p2_lo | sigma_p2_hi | sigma_p | drift_per_year | drift_lo | drift_hi | sigma_p2_robust_mad | drift_robust_mad | residual_kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 1214 | 1145 | 0.053 | 0.044 | 0.062 | 0.229 | 0.001 | -0.004 | 0.006 | 0.038 | -0.002 | -0.233 |
| mid | 2617 | 2410 | 0.017 | 0.012 | 0.021 | 0.129 | 0.001 | -0.001 | 0.003 | 0.007 | 0.001 | 2.431 |
| high | 2480 | 2236 | 0.016 | 0.012 | 0.02 | 0.128 | -0.001 | -0.002 | 0.001 | 0.005 | 0.002 | 2.697 |

## Tier-specific time adjustment

| split | tier | method | n | prb_citywide_time_adj | prb_tercile_time_adj |
|---|---|---|---|---|---|
| CV | low | model | 1558 | -0.156 | -0.166 |
| CV | low | assessor | 1558 | -0.062 | -0.074 |
| CV | low | model_no_own_sale | 1558 | -0.158 | -0.168 |
| CV | mid | model | 1793 | -0.19 | -0.208 |
| CV | mid | assessor | 1793 | -0.02 | -0.035 |
| CV | mid | model_no_own_sale | 1793 | -0.191 | -0.209 |
| CV | high | model | 1421 | -0.065 | -0.062 |
| CV | high | assessor | 1421 | -0.002 | 0.001 |
| CV | high | model_no_own_sale | 1421 | -0.076 | -0.073 |

Flags: none
