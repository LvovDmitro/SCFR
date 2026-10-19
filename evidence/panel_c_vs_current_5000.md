<!-- Public aggregate copy; source revision 3eef14cce5a3ff3b70babc611dc95378474623d6. Not a newly computed analysis. -->

# Source-Causal Arm Analysis

Common paired samples: 300
Baseline: `current_only`
Bootstrap unit: `cluster`
Cluster field: `subject_id`
Clusters: `226`
Reference mismatch rows: `0`

## Arm Summary

| Arm | Rows | Non-empty | Unique | Prior Images | Prior Reports | Token-F1 | Finding-F1 | Finding Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_only` | 300 | 300 | 270 | 0 | 0 | 0.2923 | 0.4201 | 0.7061 | 0.6800 | 1.2433 |
| `crossseed_meta` | 300 | 300 | 272 | 300 | 300 | 0.2931 | 0.4103 | 0.7036 | 0.7033 | 1.2533 |
| `matched_shuffled_prior` | 300 | 300 | 292 | 300 | 300 | 0.2793 | 0.4028 | 0.6715 | 1.1600 | 1.1033 |

## Deltas vs Baseline

### `crossseed_meta`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | 0.0008 | [-0.0026, 0.0040] | 0.6335 |
| `finding_f1` | -0.0097 | [-0.0288, 0.0091] | 0.3271 |
| `finding_status_accuracy` | -0.0024 | [-0.0118, 0.0066] | 0.5859 |
| `finding_fp_count` | 0.0233 | [-0.0572, 0.1096] | 0.6075 |
| `finding_fn_count` | 0.0100 | [-0.0511, 0.0695] | 0.7810 |
| `prediction_tokens` | -1.0133 | [-2.3789, 0.0375] | 0.0624 |

### `matched_shuffled_prior`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | -0.0131 | [-0.0213, -0.0048] | 0.0016 |
| `finding_f1` | -0.0172 | [-0.0525, 0.0176] | 0.3491 |
| `finding_status_accuracy` | -0.0345 | [-0.0500, -0.0195] | 0.0004 |
| `finding_fp_count` | 0.4800 | [0.3587, 0.6090] | 0.0004 |
| `finding_fn_count` | -0.1400 | [-0.2244, -0.0528] | 0.0016 |
| `prediction_tokens` | -0.8400 | [-2.8553, 1.1014] | 0.4075 |

## Per-Finding Summary

### `current_only`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 104 | 73 | 46 | 27 | 58 | 0.5198 | 0.7067 |
| `cardiomegaly` | 53 | 92 | 33 | 59 | 20 | 0.4552 | 0.7333 |
| `consolidation` | 30 | 9 | 3 | 6 | 27 | 0.1538 | 0.5267 |
| `edema` | 73 | 66 | 40 | 26 | 33 | 0.5755 | 0.5967 |
| `fracture` | 5 | 0 | 0 | 0 | 5 | 0.0000 | 0.9733 |
| `mass_or_nodule` | 12 | 3 | 0 | 3 | 12 | 0.0000 | 0.9500 |
| `opacity` | 93 | 54 | 24 | 30 | 69 | 0.3265 | 0.6467 |
| `pleural_effusion` | 119 | 98 | 84 | 14 | 35 | 0.7742 | 0.7000 |
| `pneumonia` | 92 | 19 | 9 | 10 | 83 | 0.1622 | 0.5433 |
| `pneumothorax` | 17 | 9 | 6 | 3 | 11 | 0.4615 | 0.5467 |
| `support_device` | 60 | 66 | 40 | 26 | 20 | 0.6349 | 0.8433 |

### `crossseed_meta`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 104 | 78 | 46 | 32 | 58 | 0.5055 | 0.6900 |
| `cardiomegaly` | 53 | 91 | 35 | 56 | 18 | 0.4861 | 0.7500 |
| `consolidation` | 30 | 9 | 1 | 8 | 29 | 0.0513 | 0.5133 |
| `edema` | 73 | 66 | 37 | 29 | 36 | 0.5324 | 0.5733 |
| `fracture` | 5 | 2 | 1 | 1 | 4 | 0.2857 | 0.9733 |
| `mass_or_nodule` | 12 | 4 | 2 | 2 | 10 | 0.2500 | 0.9600 |
| `opacity` | 93 | 49 | 25 | 24 | 68 | 0.3521 | 0.6700 |
| `pleural_effusion` | 119 | 100 | 83 | 17 | 36 | 0.7580 | 0.6900 |
| `pneumonia` | 92 | 18 | 7 | 11 | 85 | 0.1273 | 0.5400 |
| `pneumothorax` | 17 | 9 | 5 | 4 | 12 | 0.3846 | 0.5400 |
| `support_device` | 60 | 67 | 40 | 27 | 20 | 0.6299 | 0.8400 |

### `matched_shuffled_prior`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 104 | 98 | 54 | 44 | 50 | 0.5347 | 0.6767 |
| `cardiomegaly` | 53 | 168 | 44 | 124 | 9 | 0.3982 | 0.5533 |
| `consolidation` | 30 | 14 | 2 | 12 | 28 | 0.0909 | 0.4867 |
| `edema` | 73 | 86 | 48 | 38 | 25 | 0.6038 | 0.5633 |
| `fracture` | 5 | 0 | 0 | 0 | 5 | 0.0000 | 0.9733 |
| `mass_or_nodule` | 12 | 9 | 2 | 7 | 10 | 0.1905 | 0.9433 |
| `opacity` | 93 | 74 | 29 | 45 | 64 | 0.3473 | 0.6200 |
| `pleural_effusion` | 119 | 112 | 89 | 23 | 30 | 0.7706 | 0.6633 |
| `pneumonia` | 92 | 22 | 10 | 12 | 82 | 0.1754 | 0.5600 |
| `pneumothorax` | 17 | 7 | 3 | 4 | 14 | 0.2500 | 0.5233 |
| `support_device` | 60 | 85 | 46 | 39 | 14 | 0.6345 | 0.8233 |

## Gate

Hard invalidity: PASS
