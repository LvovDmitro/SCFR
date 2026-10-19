#!/usr/bin/env python3
"""Audit SCFR against fixed selectors with paired patient-level bootstrap CIs.

This is a retrospective, report-reference-proxy analysis.  It does not provide
image-grounded or radiologist-grounded clinical error estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.analyze_source_causal_arms import finding_vector, row_scores
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from analyze_source_causal_arms import finding_vector, row_scores


DEFAULT_PARITY_MARGINS: Dict[str, float] = {
    "finding_f1": -0.02,
    "finding_status_accuracy": -0.02,
    "finding_fp_count": 0.10,
    "finding_fn_count": 0.10,
}

HIGHER_IS_BETTER = {"token_f1", "finding_f1", "finding_status_accuracy", "objective"}
LOWER_IS_BETTER = {"finding_fp_count", "finding_fn_count", "prediction_tokens"}


def as_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list or {{'samples': [...]}}: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        rows.append(dict(row))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_input(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected PANEL=PATH")
    panel, raw_path = raw.split("=", 1)
    panel = panel.strip()
    raw_path = raw_path.strip()
    if not panel or not raw_path:
        raise argparse.ArgumentTypeError("Expected non-empty PANEL=PATH")
    return panel, Path(raw_path)


def sample_key(row: Mapping[str, Any], *, panel: str, idx: int) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = as_text(row.get(key))
        if value:
            return value
    raise RuntimeError(
        f"{panel}: row {idx} has no sample_id/id/study_id; "
        "synthetic row-order IDs are not allowed"
    )


def positive_findings(text: str) -> set[str]:
    return {
        finding
        for finding, status in finding_vector(text).items()
        if status == "positive"
    }


def metric_objective(
    score: Mapping[str, float],
    *,
    fp_weight: float,
    fn_weight: float,
) -> float:
    return (
        float(score["finding_f1"])
        - fp_weight * float(score["finding_fp_count"])
        - fn_weight * float(score["finding_fn_count"])
    )


def score_text(
    prediction: str,
    reference: str,
    *,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, float]:
    score = dict(row_scores({"prediction": prediction, "reference": reference}))
    score["objective"] = metric_objective(
        score,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    return score


def choose_nonlearned_gate(
    row: Mapping[str, Any],
    *,
    gate: str,
    current_field: str,
    real_field: str,
    prior_field: str,
) -> str:
    """Return ``current`` or ``real`` using reference-free fixed rules."""
    current = as_text(row.get(current_field))
    real = as_text(row.get(real_field))
    prior = as_text(row.get(prior_field))
    current_vector = finding_vector(current)
    real_vector = finding_vector(real)
    current_positive = {key for key, value in current_vector.items() if value == "positive"}
    real_positive = {key for key, value in real_vector.items() if value == "positive"}

    if gate == "status_agreement":
        return "real" if real_vector == current_vector else "current"
    if gate == "no_positive_additions":
        return "real" if real_positive.issubset(current_positive) else "current"
    if gate == "block_prior_only_additions":
        prior_positive = positive_findings(prior)
        suspect = (real_positive - current_positive) & prior_positive
        return "current" if suspect else "real"
    raise ValueError(f"Unknown non-learned gate: {gate}")


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated sample percentile (R/NumPy default type-7 form)."""
    if len(values) == 0:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class PatientBootstrapPlan:
    subjects: Tuple[str, ...]
    subject_by_row: Tuple[int, ...]
    draw_counts: Optional[np.ndarray]
    n_bootstrap: int
    seed: int


def make_patient_bootstrap_plan(
    subject_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> PatientBootstrapPlan:
    if not subject_ids:
        raise ValueError("patient bootstrap requires at least one row")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    normalized = [as_text(subject_id) for subject_id in subject_ids]
    if any(not subject_id for subject_id in normalized):
        raise ValueError("subject_ids must be non-empty")
    subjects = tuple(sorted(set(normalized)))
    subject_index = {subject: idx for idx, subject in enumerate(subjects)}
    subject_by_row = tuple(subject_index[subject] for subject in normalized)
    draw_counts: Optional[np.ndarray] = None
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        probabilities = np.full(len(subjects), 1.0 / len(subjects), dtype=np.float64)
        # A multinomial count vector is exactly equivalent to sampling the same
        # number of patients with replacement, but permits fast matrix-vector
        # evaluation and can be reused for every metric/method in a subset.
        draw_counts = rng.multinomial(
            len(subjects),
            probabilities,
            size=n_bootstrap,
        ).astype(np.int32, copy=False)
    return PatientBootstrapPlan(
        subjects=subjects,
        subject_by_row=subject_by_row,
        draw_counts=draw_counts,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def paired_patient_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    subject_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    plan: Optional[PatientBootstrapPlan] = None,
) -> Dict[str, Any]:
    """Return row-weighted cluster and patient-balanced paired bootstrap CIs.

    The row-weighted estimate matches the ordinary mean across reports while
    resampling whole patients.  The patient-balanced sensitivity first averages
    each patient's paired deltas and gives every patient equal weight.
    """
    if not (len(left) == len(right) == len(subject_ids)):
        raise ValueError("left, right, and subject_ids must have equal lengths")
    if not left:
        raise ValueError("paired bootstrap requires at least one row")
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    active_plan = plan or make_patient_bootstrap_plan(
        subject_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    normalized_subjects = tuple(as_text(subject_id) for subject_id in subject_ids)
    expected_subject_by_row = tuple(
        {subject: idx for idx, subject in enumerate(active_plan.subjects)}[subject]
        for subject in normalized_subjects
    )
    if (
        active_plan.n_bootstrap != n_bootstrap
        or active_plan.seed != seed
        or active_plan.subject_by_row != expected_subject_by_row
    ):
        raise ValueError("bootstrap plan does not match subject_ids/n_bootstrap/seed")

    subject_sums = np.zeros(len(active_plan.subjects), dtype=np.float64)
    subject_sizes = np.zeros(len(active_plan.subjects), dtype=np.float64)
    for delta, subject_idx in zip(deltas, active_plan.subject_by_row):
        subject_sums[subject_idx] += delta
        subject_sizes[subject_idx] += 1.0
    per_patient_deltas = subject_sums / subject_sizes
    row_observed = statistics.mean(deltas)
    patient_observed = float(per_patient_deltas.mean())
    if n_bootstrap == 0:
        row_ci: List[Optional[float]] = [None, None]
        patient_ci: List[Optional[float]] = [None, None]
    else:
        assert active_plan.draw_counts is not None
        row_draws = (
            active_plan.draw_counts @ subject_sums
        ) / (active_plan.draw_counts @ subject_sizes)
        patient_draws = (
            active_plan.draw_counts @ per_patient_deltas
        ) / len(active_plan.subjects)
        row_ci = [percentile(row_draws, 0.025), percentile(row_draws, 0.975)]
        patient_ci = [
            percentile(patient_draws, 0.025),
            percentile(patient_draws, 0.975),
        ]
    return {
        "paired_rows": len(deltas),
        "patients": len(active_plan.subjects),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "row_weighted_patient_cluster": {
            "observed_delta": row_observed,
            "ci95_percentile": row_ci,
            "estimand": "mean report-level paired delta; patients resampled as intact clusters",
        },
        "patient_balanced_sensitivity": {
            "observed_delta": patient_observed,
            "ci95_percentile": patient_ci,
            "estimand": "mean within-patient paired delta; patients receive equal weight",
        },
    }


def mean_scores(scores: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    return {
        metric: statistics.mean(float(score[metric]) for score in scores)
        for metric in sorted(scores[0])
    }


def route_for_scfr(
    row: Mapping[str, Any],
    *,
    selected_arm_field: str,
    current_field: str,
    real_field: str,
    scfr_field: str,
) -> str:
    explicit = as_text(row.get(selected_arm_field))
    if explicit:
        return explicit
    scfr = as_text(row.get(scfr_field))
    current = as_text(row.get(current_field))
    real = as_text(row.get(real_field))
    if scfr == current == real:
        return "current_real_identical"
    if scfr == current:
        return current_field
    if scfr == real:
        return real_field
    return "unknown_scfr_source"


def prepare_rows(
    input_specs: Sequence[Tuple[str, Path]],
    *,
    current_field: str,
    real_field: str,
    scfr_field: str,
    prior_field: str,
    subject_field: str,
    selected_arm_field: str,
    split_field: str,
    splits: Sequence[str],
    fp_weight: float,
    fn_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requested_splits = {as_text(value) for value in splits if as_text(value)}
    prepared: List[Dict[str, Any]] = []
    seen_samples: set[str] = set()
    provenance: Dict[str, Any] = {
        "inputs": [],
        "requested_splits": sorted(requested_splits),
        "selected_arm_checks": 0,
        "selected_arm_unverifiable": 0,
        "selected_arm_mismatches": 0,
    }
    for panel, path in input_specs:
        rows = load_json_rows(path)
        kept = 0
        for idx, row in enumerate(rows):
            if requested_splits and as_text(row.get(split_field)) not in requested_splits:
                continue
            sample_id = sample_key(row, panel=panel, idx=idx)
            if sample_id in seen_samples:
                raise RuntimeError(
                    f"Duplicate sample key across pooled inputs: {sample_id}. "
                    "Do not treat repeated evaluations of one study as independent rows."
                )
            seen_samples.add(sample_id)
            subject_id = as_text(row.get(subject_field))
            if not subject_id:
                raise RuntimeError(
                    f"{panel}/{sample_id}: missing {subject_field}; "
                    "patient-level bootstrap cannot fall back to row IDs"
                )
            required_text = {
                "reference": as_text(row.get("reference")),
                current_field: as_text(row.get(current_field)),
                real_field: as_text(row.get(real_field)),
                scfr_field: as_text(row.get(scfr_field)),
                prior_field: as_text(row.get(prior_field)),
            }
            missing = [name for name, value in required_text.items() if not value]
            if missing:
                raise RuntimeError(
                    f"{panel}/{sample_id}: missing required non-empty fields: "
                    f"{','.join(missing)}"
                )

            selected_arm = as_text(row.get(selected_arm_field))
            if selected_arm:
                provenance["selected_arm_checks"] += 1
                selected_candidate = as_text(row.get(selected_arm))
                if not selected_candidate:
                    provenance["selected_arm_unverifiable"] += 1
                elif selected_candidate != required_text[scfr_field]:
                    provenance["selected_arm_mismatches"] += 1

            current_score = score_text(
                required_text[current_field],
                required_text["reference"],
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            real_score = score_text(
                required_text[real_field],
                required_text["reference"],
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            scfr_score = score_text(
                required_text[scfr_field],
                required_text["reference"],
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            prepared.append(
                {
                    "panel": panel,
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "row": row,
                    "texts": {
                        "current": required_text[current_field],
                        "real": required_text[real_field],
                        "scfr": required_text[scfr_field],
                    },
                    "scores": {
                        "current": current_score,
                        "real": real_score,
                        "scfr": scfr_score,
                    },
                    "scfr_route": route_for_scfr(
                        row,
                        selected_arm_field=selected_arm_field,
                        current_field=current_field,
                        real_field=real_field,
                        scfr_field=scfr_field,
                    ),
                }
            )
            kept += 1
        provenance["inputs"].append(
            {
                "panel": panel,
                "path": str(path),
                "sha256": sha256_file(path),
                "input_rows": len(rows),
                "kept_rows": kept,
            }
        )
    if provenance["selected_arm_mismatches"]:
        raise RuntimeError(
            "SCFR selected_arm/prediction provenance mismatch on "
            f"{provenance['selected_arm_mismatches']} rows"
        )
    if not prepared:
        raise RuntimeError("No rows remained after input loading and split filtering")
    return prepared, provenance


def materialize_methods(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_field: str,
    real_field: str,
    prior_field: str,
) -> Dict[str, Dict[str, Any]]:
    methods: Dict[str, Dict[str, Any]] = {
        "always_current_only": {
            "classification": "deployable_reference_free_fixed_baseline",
            "choices": ["current"] * len(rows),
            "scores": [row["scores"]["current"] for row in rows],
        },
        "always_real_history": {
            "classification": "deployable_reference_free_fixed_baseline",
            "choices": ["real"] * len(rows),
            "scores": [row["scores"]["real"] for row in rows],
        },
        "scfr": {
            "classification": "evaluated_system_requires_external_training_provenance",
            "choices": [str(row["scfr_route"]) for row in rows],
            "scores": [row["scores"]["scfr"] for row in rows],
        },
    }
    gate_definitions = {
        "gate_status_agreement": "status_agreement",
        "gate_no_positive_additions": "no_positive_additions",
        "gate_block_prior_only_additions": "block_prior_only_additions",
    }
    for method_name, gate in gate_definitions.items():
        choices = [
            choose_nonlearned_gate(
                row["row"],
                gate=gate,
                current_field=current_field,
                real_field=real_field,
                prior_field=prior_field,
            )
            for row in rows
        ]
        methods[method_name] = {
            "classification": "post_hoc_reference_free_nonlearned_heuristic",
            "gate": gate,
            "choices": choices,
            "scores": [
                row["scores"][choice]
                for row, choice in zip(rows, choices)
            ],
        }

    oracle_choices: List[str] = []
    oracle_scores: List[Mapping[str, float]] = []
    for row in rows:
        current_score = row["scores"]["current"]
        real_score = row["scores"]["real"]
        # Conservative deterministic tie-break to current-only.
        choice = (
            "real"
            if real_score["objective"] - current_score["objective"] > 1e-12
            else "current"
        )
        oracle_choices.append(choice)
        oracle_scores.append(row["scores"][choice])
    methods["oracle_current_vs_real_reference_proxy"] = {
        "classification": "analysis_only_outcome_conditioned_reference_leakage",
        "restriction": "upper bound only for selectors choosing between current and real",
        "choices": oracle_choices,
        "scores": oracle_scores,
        "uses_reference_at_selection": True,
    }
    return methods


def reference_proxy_strata(
    rows: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    strata: List[str] = []
    for row in rows:
        delta = float(row["scores"]["real"]["objective"]) - float(
            row["scores"]["current"]["objective"]
        )
        if delta > 1e-12:
            strata.append("real_better")
        elif delta < -1e-12:
            strata.append("current_better")
        else:
            strata.append("tie")
    counts = Counter(strata)
    routes: Dict[str, Any] = {}
    for method_name, payload in methods.items():
        method_routes: Dict[str, Dict[str, Any]] = {}
        choices = payload["choices"]
        for stratum in ("real_better", "current_better", "tie"):
            indices = [idx for idx, value in enumerate(strata) if value == stratum]
            choice_counts = Counter(str(choices[idx]) for idx in indices)
            method_routes[stratum] = {
                "rows": len(indices),
                "choice_counts": dict(sorted(choice_counts.items())),
                "real_choice_rate": (
                    sum(choices[idx] == "real" for idx in indices) / len(indices)
                    if indices
                    else None
                ),
            }
        routes[method_name] = method_routes
    return {
        "classification": "retrospective_outcome_conditioned_diagnostic",
        "definition": (
            "Strata compare real-history and current-only using the report-reference "
            "proxy objective; they are not labels of clinically valid/invalid history."
        ),
        "counts": dict(sorted(counts.items())),
        "method_routes": routes,
    }


def ci_margin_audit(
    deltas_vs_current: Mapping[str, Mapping[str, Any]],
    *,
    margins: Mapping[str, float],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    for metric, margin in margins.items():
        payload = deltas_vs_current[metric]["row_weighted_patient_cluster"]
        observed = float(payload["observed_delta"])
        lower, upper = payload["ci95_percentile"]
        if metric in HIGHER_IS_BETTER:
            point_pass = observed >= margin
            ci_pass = lower is not None and float(lower) >= margin
            criterion = f"lower 95% CI >= {margin:+.4f}"
            critical_bound = lower
        elif metric in LOWER_IS_BETTER:
            point_pass = observed <= margin
            ci_pass = upper is not None and float(upper) <= margin
            criterion = f"upper 95% CI <= {margin:+.4f}"
            critical_bound = upper
        else:  # pragma: no cover - protected by the fixed margin dictionary
            raise ValueError(f"Unknown metric direction: {metric}")
        checks.append(
            {
                "metric": metric,
                "margin": margin,
                "observed_delta": observed,
                "critical_ci_bound": critical_bound,
                "criterion": criterion,
                "point_estimate_pass": bool(point_pass),
                "ci_supports_margin": bool(ci_pass),
            }
        )
    return {
        "classification": "retrospective_ci_margin_audit_not_preregistered_noninferiority",
        "checks": checks,
        "all_ci_support_margins": all(check["ci_supports_margin"] for check in checks),
    }


def analyze_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_field: str,
    real_field: str,
    prior_field: str,
    fp_weight: float,
    fn_weight: float,
    n_bootstrap: int,
    seed: int,
    parity_margins: Mapping[str, float],
) -> Dict[str, Any]:
    methods = materialize_methods(
        rows,
        current_field=current_field,
        real_field=real_field,
        prior_field=prior_field,
    )
    subjects = [str(row["subject_id"]) for row in rows]
    bootstrap_plan = make_patient_bootstrap_plan(
        subjects,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    baseline_scores = methods["always_current_only"]["scores"]
    method_reports: Dict[str, Any] = {}
    for method_name, payload in methods.items():
        scores = payload["scores"]
        summary: Dict[str, Any] = {
            key: value
            for key, value in payload.items()
            if key not in {"scores", "choices"}
        }
        summary["rows"] = len(rows)
        summary["choice_counts"] = dict(sorted(Counter(payload["choices"]).items()))
        summary["metrics"] = mean_scores(scores)
        if method_name != "always_current_only":
            summary["deltas_vs_current"] = {
                metric: paired_patient_bootstrap(
                    [float(score[metric]) for score in scores],
                    [float(score[metric]) for score in baseline_scores],
                    subjects,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    plan=bootstrap_plan,
                )
                for metric in sorted(baseline_scores[0])
            }
        method_reports[method_name] = summary
    scfr_deltas = method_reports["scfr"]["deltas_vs_current"]
    return {
        "rows": len(rows),
        "patients": len(set(subjects)),
        "panels": dict(sorted(Counter(str(row["panel"]) for row in rows).items())),
        "methods": method_reports,
        "scfr_vs_current_retrospective_margin_audit": ci_margin_audit(
            scfr_deltas,
            margins=parity_margins,
        ),
        "reference_proxy_utility_strata": reference_proxy_strata(rows, methods),
    }


def build_report(
    input_specs: Sequence[Tuple[str, Path]],
    *,
    current_field: str = "current_only",
    real_field: str = "real_prior",
    scfr_field: str = "prediction",
    prior_field: str = "prior_report",
    subject_field: str = "subject_id",
    selected_arm_field: str = "selected_arm",
    split_field: str = "split",
    splits: Sequence[str] = (),
    fp_weight: float = 0.2,
    fn_weight: float = 0.4,
    n_bootstrap: int = 5000,
    seed: int = 20260724,
    min_rows: int = 20,
    min_patients: int = 20,
    parity_margins: Mapping[str, float] = DEFAULT_PARITY_MARGINS,
) -> Dict[str, Any]:
    rows, provenance = prepare_rows(
        input_specs,
        current_field=current_field,
        real_field=real_field,
        scfr_field=scfr_field,
        prior_field=prior_field,
        subject_field=subject_field,
        selected_arm_field=selected_arm_field,
        split_field=split_field,
        splits=splits,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    if len(rows) < min_rows:
        raise RuntimeError(f"rows below minimum: {len(rows)} < {min_rows}")
    if len({row['subject_id'] for row in rows}) < min_patients:
        raise RuntimeError(
            "patients below minimum: "
            f"{len({row['subject_id'] for row in rows})} < {min_patients}"
        )
    panels = sorted({str(row["panel"]) for row in rows})
    by_panel: Dict[str, Any] = {}
    for panel_index, panel in enumerate(panels):
        panel_rows = [row for row in rows if row["panel"] == panel]
        by_panel[panel] = analyze_subset(
            panel_rows,
            current_field=current_field,
            real_field=real_field,
            prior_field=prior_field,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            n_bootstrap=n_bootstrap,
            seed=seed + panel_index + 1,
            parity_margins=parity_margins,
        )
    warnings: List[str] = []
    if not splits:
        warnings.append(
            "no_split_filter_applied: verify all supplied rows are held out from selector training"
        )
    if provenance["selected_arm_checks"] == 0:
        warnings.append(
            "scfr_selected_arm_field_absent: prediction provenance cannot be checked from this file"
        )
    if provenance["selected_arm_unverifiable"]:
        warnings.append(
            "some_scfr_selected_arm_values_lack_corresponding_candidate_text"
        )
    return {
        "analysis_stage": "RETROSPECTIVE_REBUTTAL_AUDIT",
        "claim_boundary": {
            "metrics": "11-finding report-reference regex proxy, not image truth",
            "bootstrap": "paired by study and resampled by patient",
            "simple_gates": "post-hoc, fixed, non-learned, and reference-free",
            "oracle": (
                "uses the reference outcome and is analysis-only; it upper-bounds only "
                "current-vs-real selection under the stated scalar proxy objective"
            ),
            "valid_history": (
                "reference-proxy utility strata must not be described as clinically "
                "valid-history labels"
            ),
        },
        "config": {
            "current_field": current_field,
            "real_field": real_field,
            "scfr_field": scfr_field,
            "prior_field": prior_field,
            "subject_field": subject_field,
            "selected_arm_field": selected_arm_field,
            "split_field": split_field,
            "splits": list(splits),
            "objective": {
                "formula": "finding_f1 - fp_weight*fp_count - fn_weight*fn_count",
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "parity_margins": dict(parity_margins),
            "parity_margin_status": "retrospective_existing_governance_not_preregistered",
        },
        "provenance": provenance,
        "warnings": warnings,
        "pooled": analyze_subset(
            rows,
            current_field=current_field,
            real_field=real_field,
            prior_field=prior_field,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            n_bootstrap=n_bootstrap,
            seed=seed,
            parity_margins=parity_margins,
        ),
        "by_panel": by_panel,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_analysis_table(name: str, analysis: Mapping[str, Any]) -> List[str]:
    lines = [
        f"## {name}",
        "",
        f"Rows: `{analysis['rows']}`; patients: `{analysis['patients']}`.",
        "",
        "| Method | Class | Current picks | Real picks | F1 | Status | FP | FN | Objective | dF1 vs current (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method_name, payload in analysis["methods"].items():
        metrics = payload["metrics"]
        choices = payload["choice_counts"]
        delta_text = "baseline"
        if "deltas_vs_current" in payload:
            delta = payload["deltas_vs_current"]["finding_f1"][
                "row_weighted_patient_cluster"
            ]
            ci = delta["ci95_percentile"]
            delta_text = (
                f"{fmt(delta['observed_delta'])} "
                f"[{fmt(ci[0])}, {fmt(ci[1])}]"
            )
        lines.append(
            f"| `{method_name}` | `{payload['classification']}` | "
            f"{choices.get('current', choices.get('current_only', 0))} | "
            f"{choices.get('real', choices.get('real_prior', 0))} | "
            f"{fmt(metrics.get('finding_f1'))} | "
            f"{fmt(metrics.get('finding_status_accuracy'))} | "
            f"{fmt(metrics.get('finding_fp_count'))} | "
            f"{fmt(metrics.get('finding_fn_count'))} | "
            f"{fmt(metrics.get('objective'))} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            "### SCFR vs current-only: retrospective CI margin audit",
            "",
            "| Metric | Delta | Critical CI bound | Criterion | Point pass | CI supports margin |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    margin = analysis["scfr_vs_current_retrospective_margin_audit"]
    for check in margin["checks"]:
        lines.append(
            f"| `{check['metric']}` | {fmt(check['observed_delta'])} | "
            f"{fmt(check['critical_ci_bound'])} | {check['criterion']} | "
            f"`{check['point_estimate_pass']}` | `{check['ci_supports_margin']}` |"
        )
    lines.extend(
        [
            "",
            "The margin check is retrospective and is not a preregistered",
            "non-inferiority test. `False` means the CI does not establish the",
            "chosen margin; it is not proof of inferiority.",
            "",
        ]
    )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SCFR Selector Baselines and Patient-Level Bootstrap",
        "",
        "Stage: `RETROSPECTIVE_REBUTTAL_AUDIT`.",
        "",
        "All reported correctness/error quantities below use the 11-finding",
        "report-reference regex proxy. They are not image-grounded clinical error",
        "estimates. The oracle uses the reference at selection time, is analysis-only,",
        "and is never a deployable baseline.",
        "",
    ]
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    lines.extend(render_analysis_table("Pooled", report["pooled"]))
    for panel, analysis in report["by_panel"].items():
        lines.extend(render_analysis_table(f"Panel `{panel}`", analysis))
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- `always_current_only` and `always_real_history` are absolute fixed baselines.",
            "- The three gates are post-hoc, non-learned, reference-free heuristics.",
            "- `reference_proxy_utility_strata` are outcome-conditioned diagnostics;",
            "  they do not identify clinically valid history.",
            "- The oracle is a restricted current-vs-real reference-proxy ceiling,",
            "  not a deployable method and not an image-grounded ceiling.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="PANEL=PATH",
        help="Wide JSON rows; repeat for independent evaluation panels.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--current-field", default="current_only")
    parser.add_argument("--real-field", default="real_prior")
    parser.add_argument("--scfr-field", default="prediction")
    parser.add_argument("--prior-field", default="prior_report")
    parser.add_argument("--subject-field", default="subject_id")
    parser.add_argument("--selected-arm-field", default="selected_arm")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--fp-weight", type=float, default=0.2)
    parser.add_argument("--fn-weight", type=float, default=0.4)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-patients", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(
        args.input,
        current_field=str(args.current_field),
        real_field=str(args.real_field),
        scfr_field=str(args.scfr_field),
        prior_field=str(args.prior_field),
        subject_field=str(args.subject_field),
        selected_arm_field=str(args.selected_arm_field),
        split_field=str(args.split_field),
        splits=list(args.split),
        fp_weight=float(args.fp_weight),
        fn_weight=float(args.fn_weight),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        min_rows=int(args.min_rows),
        min_patients=int(args.min_patients),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

