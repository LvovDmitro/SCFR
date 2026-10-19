<!-- Public aggregate copy; source revision 3eef14cce5a3ff3b70babc611dc95378474623d6. Not a newly computed analysis. -->

# Source-Causal Arm Analysis

Common paired samples: 300
Baseline: `matched_shuffled_prior`
Bootstrap unit: `cluster`
Cluster field: `subject_id`
Clusters: `231`
Reference mismatch rows: `0`

## Arm Summary

| Arm | Rows | Non-empty | Unique | Prior Images | Prior Reports | Token-F1 | Finding-F1 | Finding Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matched_shuffled_prior` | 300 | 300 | 295 | 300 | 300 | 0.2781 | 0.3861 | 0.6688 | 1.3433 | 1.0833 |
| `crossseed_meta` | 300 | 300 | 279 | 300 | 300 | 0.2894 | 0.4224 | 0.7045 | 0.7933 | 1.2167 |
| `current_only` | 300 | 300 | 277 | 0 | 0 | 0.2908 | 0.4139 | 0.7039 | 0.7500 | 1.2400 |

## Deltas vs Baseline

### `crossseed_meta`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | 0.0113 | [0.0033, 0.0194] | 0.0056 |
| `finding_f1` | 0.0364 | [-0.0083, 0.0823] | 0.1112 |
| `finding_status_accuracy` | 0.0358 | [0.0203, 0.0517] | 0.0004 |
| `finding_fp_count` | -0.5500 | [-0.6656, -0.4381] | 0.0004 |
| `finding_fn_count` | 0.1333 | [0.0405, 0.2290] | 0.0064 |
| `prediction_tokens` | -0.7700 | [-2.4593, 0.9133] | 0.3491 |

### `current_only`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | 0.0127 | [0.0045, 0.0213] | 0.0028 |
| `finding_f1` | 0.0278 | [-0.0168, 0.0744] | 0.2320 |
| `finding_status_accuracy` | 0.0352 | [0.0204, 0.0502] | 0.0004 |
| `finding_fp_count` | -0.5933 | [-0.7103, -0.4801] | 0.0004 |
| `finding_fn_count` | 0.1567 | [0.0627, 0.2548] | 0.0016 |
| `prediction_tokens` | -0.8767 | [-2.4192, 0.6075] | 0.2615 |

## Per-Finding Summary

### `matched_shuffled_prior`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 100 | 110 | 52 | 58 | 48 | 0.4952 | 0.6433 |
| `cardiomegaly` | 52 | 166 | 39 | 127 | 13 | 0.3578 | 0.5300 |
| `consolidation` | 22 | 21 | 2 | 19 | 20 | 0.0930 | 0.5400 |
| `edema` | 54 | 76 | 32 | 44 | 22 | 0.4923 | 0.5767 |
| `fracture` | 9 | 2 | 0 | 2 | 9 | 0.0000 | 0.9600 |
| `mass_or_nodule` | 13 | 6 | 2 | 4 | 11 | 0.2105 | 0.9467 |
| `opacity` | 87 | 85 | 34 | 51 | 53 | 0.3953 | 0.6367 |
| `pleural_effusion` | 111 | 105 | 77 | 28 | 34 | 0.7130 | 0.6233 |
| `pneumonia` | 90 | 26 | 9 | 17 | 81 | 0.1552 | 0.5700 |
| `pneumothorax` | 24 | 8 | 4 | 4 | 20 | 0.2500 | 0.5400 |
| `support_device` | 60 | 95 | 46 | 49 | 14 | 0.5935 | 0.7900 |

### `crossseed_meta`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 100 | 85 | 40 | 45 | 60 | 0.4324 | 0.6500 |
| `cardiomegaly` | 52 | 79 | 23 | 56 | 29 | 0.3511 | 0.7100 |
| `consolidation` | 22 | 8 | 0 | 8 | 22 | 0.0000 | 0.5567 |
| `edema` | 54 | 49 | 25 | 24 | 29 | 0.4854 | 0.5867 |
| `fracture` | 9 | 0 | 0 | 0 | 9 | 0.0000 | 0.9633 |
| `mass_or_nodule` | 13 | 9 | 3 | 6 | 10 | 0.2727 | 0.9467 |
| `opacity` | 87 | 62 | 30 | 32 | 57 | 0.4027 | 0.6833 |
| `pleural_effusion` | 111 | 95 | 72 | 23 | 39 | 0.6990 | 0.6433 |
| `pneumonia` | 90 | 27 | 16 | 11 | 74 | 0.2735 | 0.5900 |
| `pneumothorax` | 24 | 7 | 4 | 3 | 20 | 0.2581 | 0.5733 |
| `support_device` | 60 | 74 | 44 | 30 | 16 | 0.6567 | 0.8467 |

### `current_only`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 100 | 75 | 34 | 41 | 66 | 0.3886 | 0.6433 |
| `cardiomegaly` | 52 | 69 | 25 | 44 | 27 | 0.4132 | 0.7567 |
| `consolidation` | 22 | 17 | 4 | 13 | 18 | 0.2051 | 0.5467 |
| `edema` | 54 | 59 | 29 | 30 | 25 | 0.5133 | 0.5867 |
| `fracture` | 9 | 0 | 0 | 0 | 9 | 0.0000 | 0.9633 |
| `mass_or_nodule` | 13 | 5 | 3 | 2 | 10 | 0.3333 | 0.9600 |
| `opacity` | 87 | 50 | 24 | 26 | 63 | 0.3504 | 0.6833 |
| `pleural_effusion` | 111 | 97 | 72 | 25 | 39 | 0.6923 | 0.6367 |
| `pneumonia` | 90 | 22 | 11 | 11 | 79 | 0.1964 | 0.5567 |
| `pneumothorax` | 24 | 5 | 4 | 1 | 20 | 0.2759 | 0.5700 |
| `support_device` | 60 | 76 | 44 | 32 | 16 | 0.6471 | 0.8400 |

## Gate

Hard invalidity: PASS
