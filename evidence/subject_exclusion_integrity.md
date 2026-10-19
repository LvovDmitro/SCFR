<!-- Public aggregate copy; source revision 3eef14cce5a3ff3b70babc611dc95378474623d6. Not a newly computed analysis. -->

# P12 Cross-Seed Leakage Audit

Purpose: verify split integrity for subject-excluded cross-seed SCFR evidence.

## Gate Summary

| Target | Train rows | Validate rows | Prediction rows | Train subjects | Validate subjects | Subject overlap | Sample mismatch | Missing subject IDs | Hard invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `seed1221` | 1466 | 300 | 300 | 953 | 230 | 0 | 0 | 0 | `False` |
| `seed911` | 1343 | 300 | 300 | 943 | 231 | 0 | 0 | 0 | `False` |
| `locked1000` | 1330 | 300 | 300 | 930 | 226 | 0 | 0 | 0 | `False` |

The original detailed source-input list was omitted from this public copy. The aggregate integrity table is unchanged.
