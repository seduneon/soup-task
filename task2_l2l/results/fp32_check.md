# fp32 merge check (R3 follow-up)

| arm | pass | 95 % CI | >1 answer tag | truncated | mean tokens |
|---|---|---|---|---|---|
| adapter_bf16 | 0.430 | [0.386, 0.472] | 18.0% | 10.8% | 34 |
| merged_bf16 | 0.502 | [0.456, 0.544] | 7.0% | 4.2% | 23 |
| base_bf16 | 0.318 | [0.278, 0.362] | 1.8% | 2.0% | 19 |
| adapter_fp32 | 0.374 | [0.330, 0.418] | 27.0% | 20.2% | 46 |
| merged_fp32 | 0.370 | [0.324, 0.412] | 27.0% | 19.8% | 45 |

| pair (a − b) | diff | 95 % CI | identical text |
|---|---|---|---|
| merged_fp32 − adapter_fp32 | -0.004 | [-0.010, +0.000] | 97.6% |
| merged_bf16 − adapter_bf16 | +0.072 | [+0.040, +0.104] | 64.8% |
| adapter_fp32 − adapter_bf16 | -0.056 | [-0.084, -0.030] | 64.0% |
| merged_fp32 − adapter_bf16 | -0.060 | [-0.088, -0.034] | 64.2% |
| merged_bf16 − merged_fp32 | +0.132 | [+0.094, +0.170] | 48.0% |
| adapter_fp32 − base_bf16 | +0.056 | [+0.004, +0.108] | 11.0% |

R3 at fp32 (merged − adapter CI ∋ 0): **pass**, identical text 97.6%.
