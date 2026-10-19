<!-- Public aggregate copy; source revision 3eef14cce5a3ff3b70babc611dc95378474623d6. Not a newly computed analysis. -->

# SCFR Confirmatory Gate

Stage: `RETROSPECTIVE_CONFIRMATORY_GOVERNANCE` on existing cross-seed clustered artifacts.
Report set: `bootstrap5000_2026_05_04`.

This is the gate specification to reuse for future locked holdouts. On
already-produced May 3 artifacts it is retrospective governance, not
post-hoc preregistration.

Verdict: `PASS_CONSERVATIVE_SCFR_CLAIM`

## Primary Family

Primary endpoint family: matched wrong-history FP repair across three p12 targets.
Holm alpha: `0.05`.

| Target | Delta FP vs matched | p | Holm adjusted p | Holm reject |
|---|---:|---:|---:|---|
| `locked1000` | -0.4567 | 0.0004 | 0.0012 | `True` |
| `seed1221` | -0.3433 | 0.0004 | 0.0012 | `True` |
| `seed911` | -0.5500 | 0.0004 | 0.0012 | `True` |

## Current-Only Parity Margins

| Target | Metric | Delta vs current-only | Criterion | Pass |
|---|---|---:|---|---|
| `seed1221` | `finding_f1` | 0.0027 | delta >= -0.0200 | `True` |
| `seed1221` | `finding_status_accuracy` | 0.0033 | delta >= -0.0200 | `True` |
| `seed1221` | `finding_fp_count` | -0.0267 | delta <= +0.1000 | `True` |
| `seed1221` | `finding_fn_count` | 0.0000 | delta <= +0.1000 | `True` |
| `seed911` | `finding_f1` | 0.0085 | delta >= -0.0200 | `True` |
| `seed911` | `finding_status_accuracy` | 0.0006 | delta >= -0.0200 | `True` |
| `seed911` | `finding_fp_count` | 0.0433 | delta <= +0.1000 | `True` |
| `seed911` | `finding_fn_count` | -0.0233 | delta <= +0.1000 | `True` |
| `locked1000` | `finding_f1` | -0.0097 | delta >= -0.0200 | `True` |
| `locked1000` | `finding_status_accuracy` | -0.0024 | delta >= -0.0200 | `True` |
| `locked1000` | `finding_fp_count` | 0.0233 | delta <= +0.1000 | `True` |
| `locked1000` | `finding_fn_count` | 0.0100 | delta <= +0.1000 | `True` |

## Secondary Families

### `secondary_matched_status`

| Target | Delta | p | Holm adjusted p | Holm reject |
|---|---:|---:|---:|---|
| `locked1000` | 0.0321 | 0.0008 | 0.0016 | `True` |
| `seed1221` | 0.0121 | 0.1068 | 0.1068 | `False` |
| `seed911` | 0.0358 | 0.0004 | 0.0012 | `True` |

### `secondary_matched_token_f1`

| Target | Delta | p | Holm adjusted p | Holm reject |
|---|---:|---:|---:|---|
| `locked1000` | 0.0138 | 0.0012 | 0.0036 | `True` |
| `seed1221` | 0.0147 | 0.0012 | 0.0036 | `True` |
| `seed911` | 0.0113 | 0.0056 | 0.0056 | `True` |

### `exploratory_matched_finding_f1`

| Target | Delta | p | Holm adjusted p | Holm reject |
|---|---:|---:|---:|---|
| `locked1000` | 0.0075 | 0.6631 | 1.0000 | `False` |
| `seed1221` | 0.0006 | 0.9598 | 1.0000 | `False` |
| `seed911` | 0.0364 | 0.1112 | 0.3335 | `False` |

## Claim Boundary

- Safe: conservative SCFR claim if primary FP repair and current-only
  parity margins pass.
- Required caveat: FN increases versus matched wrong-history are a tradeoff.
- Unsafe: generic RRG SOTA, clinical safety, current-only dominance, or
  calling this retrospective gate a preregistered analysis.
