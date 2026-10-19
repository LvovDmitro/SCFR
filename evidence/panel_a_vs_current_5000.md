<!-- Public aggregate copy; source revision 3eef14cce5a3ff3b70babc611dc95378474623d6. Not a newly computed analysis. -->

# Source-Causal Arm Analysis

Common paired samples: 300
Baseline: `current_only`
Bootstrap unit: `cluster`
Cluster field: `subject_id`
Clusters: `230`
Reference mismatch rows: `0`

## Arm Summary

| Arm | Rows | Non-empty | Unique | Prior Images | Prior Reports | Token-F1 | Finding-F1 | Finding Acc | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_only` | 300 | 300 | 252 | 0 | 0 | 0.2858 | 0.4193 | 0.7018 | 0.7433 | 1.2400 |
| `crossseed_meta` | 300 | 300 | 252 | 300 | 300 | 0.2863 | 0.4220 | 0.7052 | 0.7167 | 1.2400 |
| `matched_shuffled_prior` | 300 | 300 | 291 | 300 | 300 | 0.2715 | 0.4215 | 0.6930 | 1.0600 | 1.0700 |

## Deltas vs Baseline

### `crossseed_meta`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | 0.0005 | [-0.0018, 0.0030] | 0.6995 |
| `finding_f1` | 0.0027 | [-0.0109, 0.0177] | 0.7111 |
| `finding_status_accuracy` | 0.0033 | [-0.0039, 0.0111] | 0.3847 |
| `finding_fp_count` | -0.0267 | [-0.0875, 0.0337] | 0.4007 |
| `finding_fn_count` | 0.0000 | [-0.0525, 0.0489] | 1.0000 |
| `prediction_tokens` | -0.0633 | [-0.5331, 0.3887] | 0.7942 |

### `matched_shuffled_prior`

| Metric | Delta | 95% CI | p(two-sided) |
|---|---:|---:|---:|
| `token_f1` | -0.0142 | [-0.0228, -0.0059] | 0.0008 |
| `finding_f1` | 0.0022 | [-0.0368, 0.0393] | 0.9098 |
| `finding_status_accuracy` | -0.0088 | [-0.0239, 0.0064] | 0.2456 |
| `finding_fp_count` | 0.3167 | [0.2047, 0.4343] | 0.0004 |
| `finding_fn_count` | -0.1700 | [-0.2517, -0.0860] | 0.0004 |
| `prediction_tokens` | -0.5200 | [-2.1461, 1.0469] | 0.5239 |

## Per-Finding Summary

### `current_only`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 97 | 83 | 45 | 38 | 52 | 0.5000 | 0.7000 |
| `cardiomegaly` | 51 | 88 | 35 | 53 | 16 | 0.5036 | 0.7633 |
| `consolidation` | 23 | 12 | 3 | 9 | 20 | 0.1714 | 0.5633 |
| `edema` | 65 | 63 | 31 | 32 | 34 | 0.4844 | 0.5667 |
| `fracture` | 5 | 1 | 0 | 1 | 5 | 0.0000 | 0.9800 |
| `mass_or_nodule` | 13 | 3 | 2 | 1 | 11 | 0.2500 | 0.9567 |
| `opacity` | 101 | 53 | 22 | 31 | 79 | 0.2857 | 0.6033 |
| `pleural_effusion` | 95 | 87 | 73 | 14 | 22 | 0.8022 | 0.7100 |
| `pneumonia` | 122 | 23 | 14 | 9 | 108 | 0.1931 | 0.4700 |
| `pneumothorax` | 17 | 6 | 3 | 3 | 14 | 0.2609 | 0.5500 |
| `support_device` | 45 | 66 | 34 | 32 | 11 | 0.6126 | 0.8567 |

### `crossseed_meta`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 97 | 82 | 47 | 35 | 50 | 0.5251 | 0.7167 |
| `cardiomegaly` | 51 | 83 | 36 | 47 | 15 | 0.5373 | 0.7833 |
| `consolidation` | 23 | 12 | 3 | 9 | 20 | 0.1714 | 0.5733 |
| `edema` | 65 | 60 | 32 | 28 | 33 | 0.5120 | 0.5867 |
| `fracture` | 5 | 1 | 0 | 1 | 5 | 0.0000 | 0.9800 |
| `mass_or_nodule` | 13 | 2 | 2 | 0 | 11 | 0.2667 | 0.9600 |
| `opacity` | 101 | 57 | 22 | 35 | 79 | 0.2785 | 0.5900 |
| `pleural_effusion` | 95 | 87 | 70 | 17 | 25 | 0.7692 | 0.6967 |
| `pneumonia` | 122 | 22 | 13 | 9 | 109 | 0.1806 | 0.4600 |
| `pneumothorax` | 17 | 5 | 3 | 2 | 14 | 0.2727 | 0.5533 |
| `support_device` | 45 | 66 | 34 | 32 | 11 | 0.6126 | 0.8567 |

### `matched_shuffled_prior`

| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `atelectasis` | 97 | 99 | 58 | 41 | 39 | 0.5918 | 0.7300 |
| `cardiomegaly` | 51 | 151 | 46 | 105 | 5 | 0.4554 | 0.6267 |
| `consolidation` | 23 | 13 | 3 | 10 | 20 | 0.1667 | 0.5400 |
| `edema` | 65 | 78 | 41 | 37 | 24 | 0.5734 | 0.5967 |
| `fracture` | 5 | 1 | 0 | 1 | 5 | 0.0000 | 0.9800 |
| `mass_or_nodule` | 13 | 2 | 0 | 2 | 13 | 0.0000 | 0.9500 |
| `opacity` | 101 | 80 | 37 | 43 | 64 | 0.4088 | 0.6267 |
| `pleural_effusion` | 95 | 89 | 72 | 17 | 23 | 0.7826 | 0.7000 |
| `pneumonia` | 122 | 31 | 17 | 14 | 105 | 0.2222 | 0.5067 |
| `pneumothorax` | 17 | 4 | 2 | 2 | 15 | 0.1905 | 0.5467 |
| `support_device` | 45 | 83 | 37 | 46 | 8 | 0.5781 | 0.8200 |

## Gate

Hard invalidity: PASS
