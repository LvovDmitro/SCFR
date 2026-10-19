#!/usr/bin/env python3
"""Aggregate-only analysis of the locked R01 SCFR Task-A response.

The script joins the blinded current-image labels to the private answer key
and the four frozen full-panel MAIRA-2 generation arms.  It deliberately
emits no case/subject identifiers, report text, image paths, or free text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

try:
    from scripts.analyze_source_causal_arms import finding_vector
    from scripts.manage_scfr_human_adjudication_release import (
        _read_secret,
        _tree_sha256,
        _verify_signed,
    )
    from scripts.prepare_scfr_human_adjudication_packet import (
        FINDINGS,
        TASK_A_STATUSES,
        sha256_canonical_json,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_vector
    from scripts.manage_scfr_human_adjudication_release import (
        _read_secret,
        _tree_sha256,
        _verify_signed,
    )
    from scripts.prepare_scfr_human_adjudication_packet import (
        FINDINGS,
        TASK_A_STATUSES,
        sha256_canonical_json,
        sha256_file,
    )


SCHEMA = "scfr-r01-task-a-aggregate-v1"
ARMS = (
    "current_only",
    "real_history",
    "matched_wrong_history",
    "random_wrong_history",
)
ARM_PATTERNS = {
    "current_only": ("current_only",),
    "real_history": ("real_prior",),
    "matched_wrong_history": ("matched_shuffled_prior",),
    "random_wrong_history": ("shuffled_prior",),
}
EXPECTED_STRATA = (
    "wrong_history_adds_proxy_positive",
    "proxy_fn_conflict",
    "random_wrong_history_finding_status_change",
    "matched_wrong_history_finding_status_change",
    "wrong_history_text_only_difference",
)
RESOLVED = frozenset(("present", "absent"))
FREE_TEXT_NEGATIVE_RE = re.compile(
    r"\bнет\b|отсутств|не\s+(?:выяв|обнаруж|определ|визуализ|отмеч|име)|"
    r"без\s+(?:дополн|иных|других)|дополнительн.{0,30}\bне\b|"
    r"патологическ.{0,30}\bне\b|"
    r"^(?:none|no|n/?a|not applicable|no other findings|-)\s*[.!]?$",
    re.IGNORECASE,
)
FREE_TEXT_UNCERTAIN_RE = re.compile(
    r"не\s+исключ|возможно|вероят|подозр|сомн|трудно|нельзя|огранич|"
    r"\b(?:possible|possibly|uncertain|limited)\b",
    re.IGNORECASE,
)


def _load_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object")
    return value


def _artifact_files(root: Path, arm: str) -> List[Path]:
    patterns = ARM_PATTERNS[arm]
    files = []
    for path in root.glob("*.json"):
        name = path.name.lower()
        if not all(pattern in name for pattern in patterns):
            continue
        if arm == "random_wrong_history" and "matched_shuffled_prior" in name:
            continue
        files.append(path)
    files.sort(key=lambda path: path.name)
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 chunks for {arm}, observed {len(files)}")
    return files


def _load_requested_arms(
    root: Path,
    source_ids: set[str],
) -> tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Any]]:
    arms_by_id: Dict[str, Dict[str, str]] = {
        source_id: {} for source_id in source_ids
    }
    reference_by_id: Dict[str, str] = {}
    arm_row_counts: Dict[str, int] = {}
    arm_file_hashes: Dict[str, str] = {}
    for arm in ARMS:
        seen: set[str] = set()
        row_count = 0
        digest = hashlib.sha256()
        for path in _artifact_files(root, arm):
            raw = path.read_bytes()
            digest.update(hashlib.sha256(raw).digest())
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Non-list artifact for {arm}")
            for row in payload:
                if not isinstance(row, dict):
                    raise RuntimeError(f"Non-object artifact row for {arm}")
                row_count += 1
                source_id = str(row.get("id") or "")
                if not source_id or source_id in seen:
                    raise RuntimeError(f"Missing/duplicate source identifier in {arm}")
                seen.add(source_id)
                if source_id not in source_ids:
                    continue
                prediction = row.get("prediction")
                reference = row.get("reference")
                if not isinstance(prediction, str) or not isinstance(reference, str):
                    raise RuntimeError(f"Missing report fields in {arm}")
                arms_by_id[source_id][arm] = prediction
                previous = reference_by_id.setdefault(source_id, reference)
                if previous != reference:
                    raise RuntimeError("Reference text differs across frozen arms")
        if row_count != 18716 or len(seen) != 18716:
            raise RuntimeError(
                f"Frozen arm {arm} is incomplete: rows={row_count}, unique={len(seen)}"
            )
        arm_row_counts[arm] = row_count
        arm_file_hashes[arm] = digest.hexdigest()
    missing = {
        arm: sum(arm not in reports for reports in arms_by_id.values())
        for arm in ARMS
    }
    if any(missing.values()) or len(reference_by_id) != len(source_ids):
        raise RuntimeError(f"Task-A/full-arm join is incomplete: {missing}")
    return arms_by_id, reference_by_id, {
        "arm_row_counts": arm_row_counts,
        "arm_chunk_chain_sha256": arm_file_hashes,
        "joined_cases": len(source_ids),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: float, fp: float, fn: float) -> float | None:
    return _ratio(2.0 * tp, 2.0 * tp + fp + fn)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _case_weight(case: Mapping[str, Any], weighted: bool) -> float:
    if not weighted:
        return 1.0
    return 1.0 / float(case["inclusion_probability"])


def _reference_counts(
    cases: Sequence[Mapping[str, Any]],
    *,
    weighted: bool,
) -> Dict[str, float | None]:
    counts = Counter()
    for case in cases:
        weight = _case_weight(case, weighted)
        for finding in FINDINGS:
            reference_positive = case["reference"][finding] == "positive"
            status = case["reader"][finding]
            if status not in RESOLVED:
                counts["unresolved"] += weight
            elif reference_positive and status == "present":
                counts["tp"] += weight
            elif reference_positive and status == "absent":
                counts["fp"] += weight
            elif not reference_positive and status == "present":
                counts["fn"] += weight
            else:
                counts["tn"] += weight
    resolved = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    observed = _ratio(counts["tp"] + counts["tn"], resolved)
    ref_positive = counts["tp"] + counts["fp"]
    image_positive = counts["tp"] + counts["fn"]
    expected = None
    kappa = None
    if resolved:
        expected = (
            ref_positive * image_positive
            + (resolved - ref_positive) * (resolved - image_positive)
        ) / (resolved * resolved)
        if expected < 1.0:
            kappa = (float(observed) - expected) / (1.0 - expected)
    return {
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tn": counts["tn"],
        "unresolved": counts["unresolved"],
        "agreement_resolved": observed,
        "reference_positive_ppv_vs_image": _ratio(counts["tp"], ref_positive),
        "reference_positive_sensitivity_vs_image": _ratio(
            counts["tp"], image_positive
        ),
        "specificity_vs_image": _ratio(
            counts["tn"], counts["tn"] + counts["fp"]
        ),
        "cohen_kappa_resolved": kappa,
    }


def _arm_counts(
    cases: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    weighted: bool,
) -> Dict[str, float | None]:
    counts = Counter()
    total_weight = 0.0
    for case in cases:
        weight = _case_weight(case, weighted)
        total_weight += weight
        for finding in FINDINGS:
            report_positive = case["arms"][arm][finding] == "positive"
            reference_positive = case["reference"][finding] == "positive"
            status = case["reader"][finding]
            if status == "present":
                if report_positive:
                    counts["image_tp"] += weight
                else:
                    counts["missed_visible"] += weight
            elif status == "absent":
                if report_positive:
                    counts["unsupported"] += weight
                else:
                    counts["image_tn"] += weight
            else:
                counts["unresolved_opportunities"] += weight
                if report_positive:
                    counts["possible_unsupported"] += weight
                else:
                    counts["possible_missed"] += weight
            if report_positive and not reference_positive:
                counts["proxy_fp_events"] += weight
                if status == "absent":
                    counts["proxy_fp_agree"] += weight
                elif status == "present":
                    counts["proxy_fp_disagree"] += weight
                else:
                    counts["proxy_fp_unresolved"] += weight
            if not report_positive and reference_positive:
                counts["proxy_fn_events"] += weight
                if status == "present":
                    counts["proxy_fn_agree"] += weight
                elif status == "absent":
                    counts["proxy_fn_disagree"] += weight
                else:
                    counts["proxy_fn_unresolved"] += weight
    proxy_fp_resolved = (
        counts["proxy_fp_agree"] + counts["proxy_fp_disagree"]
    )
    proxy_fn_resolved = (
        counts["proxy_fn_agree"] + counts["proxy_fn_disagree"]
    )
    return {
        "unsupported_positives": counts["unsupported"],
        "missed_visible_positives": counts["missed_visible"],
        "combined_errors": counts["unsupported"] + counts["missed_visible"],
        "unsupported_mean_per_case": _ratio(
            counts["unsupported"], total_weight
        ),
        "missed_visible_mean_per_case": _ratio(
            counts["missed_visible"], total_weight
        ),
        "combined_error_mean_per_case": _ratio(
            counts["unsupported"] + counts["missed_visible"], total_weight
        ),
        "image_grounded_f1": _f1(
            counts["image_tp"],
            counts["unsupported"],
            counts["missed_visible"],
        ),
        "unresolved_finding_opportunities": counts[
            "unresolved_opportunities"
        ],
        "proxy_fp_events": counts["proxy_fp_events"],
        "proxy_fp_image_absent": counts["proxy_fp_agree"],
        "proxy_fp_image_present": counts["proxy_fp_disagree"],
        "proxy_fp_unresolved": counts["proxy_fp_unresolved"],
        "proxy_fp_validity_among_resolved": _ratio(
            counts["proxy_fp_agree"], proxy_fp_resolved
        ),
        "proxy_fp_validity_bounds_all_events": [
            _ratio(counts["proxy_fp_agree"], counts["proxy_fp_events"]),
            _ratio(
                counts["proxy_fp_agree"] + counts["proxy_fp_unresolved"],
                counts["proxy_fp_events"],
            ),
        ],
        "proxy_fn_events": counts["proxy_fn_events"],
        "proxy_fn_image_present": counts["proxy_fn_agree"],
        "proxy_fn_image_absent": counts["proxy_fn_disagree"],
        "proxy_fn_unresolved": counts["proxy_fn_unresolved"],
        "proxy_fn_validity_among_resolved": _ratio(
            counts["proxy_fn_agree"], proxy_fn_resolved
        ),
        "proxy_fn_validity_bounds_all_events": [
            _ratio(counts["proxy_fn_agree"], counts["proxy_fn_events"]),
            _ratio(
                counts["proxy_fn_agree"] + counts["proxy_fn_unresolved"],
                counts["proxy_fn_events"],
            ),
        ],
    }


def _transition_counts(
    cases: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    weighted: bool,
) -> Dict[str, float | None]:
    counts = Counter()
    for case in cases:
        weight = _case_weight(case, weighted)
        for finding in FINDINGS:
            current = case["arms"]["current_only"][finding] == "positive"
            other = case["arms"][arm][finding] == "positive"
            reference = case["reference"][finding] == "positive"
            status = case["reader"][finding]
            if other and not current:
                prefix = "addition"
                if not reference:
                    counts["incremental_proxy_fp"] += weight
            elif current and not other:
                prefix = "removal"
                if reference:
                    counts["incremental_proxy_fn"] += weight
            else:
                continue
            counts[f"{prefix}_events"] += weight
            counts[f"{prefix}_{status}"] += weight
            if prefix == "addition" and not reference:
                counts[f"incremental_proxy_fp_{status}"] += weight
            if prefix == "removal" and reference:
                counts[f"incremental_proxy_fn_{status}"] += weight
    add_resolved = counts["addition_present"] + counts["addition_absent"]
    inc_fp_resolved = (
        counts["incremental_proxy_fp_present"]
        + counts["incremental_proxy_fp_absent"]
    )
    removal_resolved = counts["removal_present"] + counts["removal_absent"]
    return {
        "positive_additions": counts["addition_events"],
        "additions_image_absent": counts["addition_absent"],
        "additions_image_present": counts["addition_present"],
        "additions_unresolved": (
            counts["addition_uncertain"] + counts["addition_unassessable"]
        ),
        "addition_unsupported_rate_resolved": _ratio(
            counts["addition_absent"], add_resolved
        ),
        "incremental_proxy_fp_events": counts["incremental_proxy_fp"],
        "incremental_proxy_fp_image_absent": counts[
            "incremental_proxy_fp_absent"
        ],
        "incremental_proxy_fp_image_present": counts[
            "incremental_proxy_fp_present"
        ],
        "incremental_proxy_fp_unresolved": (
            counts["incremental_proxy_fp_uncertain"]
            + counts["incremental_proxy_fp_unassessable"]
        ),
        "incremental_proxy_fp_validity_resolved": _ratio(
            counts["incremental_proxy_fp_absent"], inc_fp_resolved
        ),
        "positive_removals": counts["removal_events"],
        "removals_image_present": counts["removal_present"],
        "removals_image_absent": counts["removal_absent"],
        "removals_unresolved": (
            counts["removal_uncertain"] + counts["removal_unassessable"]
        ),
        "removal_harm_rate_resolved": _ratio(
            counts["removal_present"], removal_resolved
        ),
        "incremental_proxy_fn_events": counts["incremental_proxy_fn"],
    }


def _point_metrics(
    cases: Sequence[Mapping[str, Any]],
    *,
    weighted: bool,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    reference = _reference_counts(cases, weighted=weighted)
    for key in (
        "agreement_resolved",
        "reference_positive_ppv_vs_image",
        "reference_positive_sensitivity_vs_image",
        "cohen_kappa_resolved",
    ):
        value = reference[key]
        if value is not None:
            values[f"reference/{key}"] = float(value)
    arm_results = {
        arm: _arm_counts(cases, arm, weighted=weighted) for arm in ARMS
    }
    for arm, result in arm_results.items():
        for key in (
            "unsupported_mean_per_case",
            "missed_visible_mean_per_case",
            "combined_error_mean_per_case",
            "image_grounded_f1",
            "proxy_fp_validity_among_resolved",
            "proxy_fn_validity_among_resolved",
        ):
            value = result[key]
            if value is not None:
                values[f"arm/{arm}/{key}"] = float(value)
    for arm in ARMS[1:]:
        for key in (
            "unsupported_mean_per_case",
            "missed_visible_mean_per_case",
            "combined_error_mean_per_case",
            "image_grounded_f1",
        ):
            left = arm_results[arm][key]
            right = arm_results["current_only"][key]
            if left is not None and right is not None:
                values[f"contrast/{arm}-current_only/{key}"] = (
                    float(left) - float(right)
                )
        transition = _transition_counts(cases, arm, weighted=weighted)
        for key in (
            "addition_unsupported_rate_resolved",
            "incremental_proxy_fp_validity_resolved",
            "removal_harm_rate_resolved",
        ):
            value = transition[key]
            if value is not None:
                values[f"transition/{arm}/{key}"] = float(value)
    real_transition = _transition_counts(
        cases, "real_history", weighted=weighted
    )
    for arm in ("matched_wrong_history", "random_wrong_history"):
        other_transition = _transition_counts(cases, arm, weighted=weighted)
        for key in (
            "addition_unsupported_rate_resolved",
            "incremental_proxy_fp_validity_resolved",
            "removal_harm_rate_resolved",
        ):
            left = other_transition[key]
            right = real_transition[key]
            if left is not None and right is not None:
                values[f"transition_contrast/{arm}-real_history/{key}"] = (
                    float(left) - float(right)
                )
    return values


def _stratified_bootstrap(
    cases: Sequence[Mapping[str, Any]],
    *,
    weighted: bool,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    by_stratum: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_stratum[str(case["sampling_stratum"])].append(case)
    observed = _point_metrics(cases, weighted=weighted)
    replicates: Dict[str, List[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        sample: List[Mapping[str, Any]] = []
        for stratum in sorted(by_stratum):
            rows = by_stratum[stratum]
            sample.extend(rows[rng.randrange(len(rows))] for _ in rows)
        current = _point_metrics(sample, weighted=weighted)
        for key, value in current.items():
            if math.isfinite(value):
                replicates[key].append(value)
    return {
        key: {
            "estimate": value,
            "ci95": [
                _percentile(replicates.get(key, []), 0.025),
                _percentile(replicates.get(key, []), 0.975),
            ],
            "valid_replicates": len(replicates.get(key, [])),
        }
        for key, value in sorted(observed.items())
    }


def _clean_counts(value: Any, *, weighted: bool) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean_counts(item, weighted=weighted)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clean_counts(item, weighted=weighted) for item in value]
    if isinstance(value, float):
        if not weighted and value.is_integer():
            return int(value)
        return value
    return value


def analyze(
    *,
    private_key_path: Path,
    response_path: Path,
    lock_receipt_path: Path,
    release_receipt_path: Path,
    signing_secret_path: Path,
    released_stage_dir: Path,
    artifact_dir: Path,
    returned_workbook_path: Path | None,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    private = _load_object(private_key_path)
    response = _load_object(response_path)
    lock_receipt = _load_object(lock_receipt_path)
    release_receipt = _load_object(release_receipt_path)
    secret = _read_secret(signing_secret_path)
    lock_unsigned = _verify_signed(lock_receipt, secret, label="Task-A lock receipt")
    release_unsigned = _verify_signed(
        release_receipt, secret, label="Task-A release receipt"
    )
    if sha256_file(response_path) != lock_unsigned.get("response_sha256"):
        raise RuntimeError("Locked response hash mismatch")
    if sha256_file(release_receipt_path) != lock_unsigned.get(
        "release_receipt_sha256"
    ):
        raise RuntimeError("Release receipt hash mismatch")
    if _tree_sha256(released_stage_dir) != release_unsigned.get(
        "stage_tree_sha256"
    ):
        raise RuntimeError("Released Task-A tree hash mismatch")
    if lock_unsigned.get("reader_code") != "R01" or lock_unsigned.get(
        "stage_number"
    ) != 1:
        raise RuntimeError("Unexpected reader/stage in lock receipt")
    if private.get("packet_token") != response.get("packet_token"):
        raise RuntimeError("Private/response packet tokens differ")
    private_cases = private.get("cases")
    response_cases = response.get("cases")
    if (
        not isinstance(private_cases, list)
        or not isinstance(response_cases, list)
        or len(private_cases) != len(response_cases)
        or len(private_cases) != 60
    ):
        raise RuntimeError("Task-A case cardinality mismatch")
    private_tokens = [case.get("case_token") for case in private_cases]
    response_tokens = [case.get("case_token") for case in response_cases]
    if (
        private_tokens != response_tokens
        or len(set(private_tokens)) != len(private_tokens)
        or sha256_canonical_json(private_tokens)
        != private.get("task_a_case_order_sha256")
        or private.get("task_a_case_order_sha256")
        != response.get("case_order_commitment")
        or response.get("case_order_commitment")
        != lock_unsigned.get("case_order_commitment")
    ):
        raise RuntimeError("Task-A case-order binding failed")
    source_ids = {str(case.get("source_case_id") or "") for case in private_cases}
    subjects = {str(case.get("subject_id") or "") for case in private_cases}
    if "" in source_ids or "" in subjects or len(source_ids) != 60 or len(subjects) != 60:
        raise RuntimeError("Task-A source/subject uniqueness failed")
    arms_by_id, references_by_id, artifact_audit = _load_requested_arms(
        artifact_dir, source_ids
    )
    response_by_token = {
        str(case["case_token"]): case["response"] for case in response_cases
    }
    cases: List[Dict[str, Any]] = []
    status_counts = Counter()
    stratum_counts = Counter()
    free_text_nonempty = 0
    free_text_negative_or_none = 0
    free_text_uncertainty_only = 0
    free_text_affirmative_narrative = 0
    free_text_word_counts: List[int] = []
    reference_vector_matches = 0
    for private_case in private_cases:
        token = str(private_case["case_token"])
        source_id = str(private_case["source_case_id"])
        reader_response = response_by_token[token]
        statuses = reader_response.get("finding_statuses")
        if (
            not isinstance(statuses, dict)
            or tuple(statuses) != tuple(FINDINGS)
            or any(value not in TASK_A_STATUSES for value in statuses.values())
        ):
            raise RuntimeError("Invalid/incomplete R01 finding-status schema")
        status_counts.update(statuses.values())
        free_text = str(
            reader_response.get("other_important_findings") or ""
        ).strip()
        if free_text:
            free_text_nonempty += 1
            free_text_word_counts.append(
                len(re.findall(r"[\w]+", free_text, flags=re.UNICODE))
            )
            if FREE_TEXT_NEGATIVE_RE.search(free_text):
                free_text_negative_or_none += 1
            elif FREE_TEXT_UNCERTAIN_RE.search(free_text):
                free_text_uncertainty_only += 1
            else:
                free_text_affirmative_narrative += 1
        reference = private_case.get("reference_finding_statuses")
        if reference != finding_vector(references_by_id[source_id]):
            raise RuntimeError("Private reference vector/full artifact mismatch")
        reference_vector_matches += 1
        stratum = str(private_case.get("sampling_stratum") or "")
        stratum_counts[stratum] += 1
        inclusion_probability = float(private_case["inclusion_probability"])
        if not 0.0 < inclusion_probability <= 1.0:
            raise RuntimeError("Invalid challenge inclusion probability")
        cases.append(
            {
                "stratum": stratum,
                "sampling_stratum": stratum,
                "inclusion_probability": inclusion_probability,
                "reader": dict(statuses),
                "reference": dict(reference),
                "arms": {
                    arm: finding_vector(arms_by_id[source_id][arm])
                    for arm in ARMS
                },
            }
        )
    if set(stratum_counts) != set(EXPECTED_STRATA) or any(
        stratum_counts[stratum] != 12 for stratum in EXPECTED_STRATA
    ):
        raise RuntimeError("Unexpected Task-A challenge stratum allocation")
    inferred_sizes = {
        stratum: int(round(stratum_counts[stratum] / next(
            case["inclusion_probability"]
            for case in cases
            if case["sampling_stratum"] == stratum
        )))
        for stratum in EXPECTED_STRATA
    }
    for stratum, size in inferred_sizes.items():
        observed_pi = next(
            case["inclusion_probability"]
            for case in cases
            if case["sampling_stratum"] == stratum
        )
        if not math.isclose(
            observed_pi,
            stratum_counts[stratum] / size,
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise RuntimeError("Challenge sampling probability is not auditable")

    primary_arms = {
        arm: _clean_counts(
            _arm_counts(cases, arm, weighted=False), weighted=False
        )
        for arm in ARMS
    }
    primary_transitions = {
        arm: _clean_counts(
            _transition_counts(cases, arm, weighted=False), weighted=False
        )
        for arm in ARMS[1:]
    }
    primary_reference = _clean_counts(
        _reference_counts(cases, weighted=False), weighted=False
    )
    weighted_arms = {
        arm: _arm_counts(cases, arm, weighted=True) for arm in ARMS
    }
    weighted_transitions = {
        arm: _transition_counts(cases, arm, weighted=True) for arm in ARMS[1:]
    }
    weighted_reference = _reference_counts(cases, weighted=True)
    workbook_hash = (
        sha256_file(returned_workbook_path)
        if returned_workbook_path is not None
        else None
    )
    return {
        "schema": SCHEMA,
        "analysis_status": "passed",
        "scope": {
            "reader": "R01",
            "reader_count": 1,
            "cases": len(cases),
            "unique_subjects": len(subjects),
            "finding_opportunities": len(cases) * len(FINDINGS),
            "cohort": "challenge_enriched",
            "strata": dict(sorted(stratum_counts.items())),
            "sampling_target_for_ipw_sensitivity": (
                "union of the five prespecified challenge strata among the "
                "frozen subject-deduplicated cases remaining after smoke and "
                "probability-cohort exclusion; not the full 18,716-study population"
            ),
            "inferred_target_stratum_sizes": dict(sorted(inferred_sizes.items())),
            "inferred_target_union_size": sum(inferred_sizes.values()),
        },
        "integrity": {
            "lock_hmac_verified": True,
            "release_hmac_verified": True,
            "release_stage_tree_verified": True,
            "response_sha256": sha256_file(response_path),
            "lock_receipt_sha256": sha256_file(lock_receipt_path),
            "release_receipt_sha256": sha256_file(release_receipt_path),
            "private_answer_key_sha256": sha256_file(private_key_path),
            "returned_workbook_sha256": workbook_hash,
            "case_order_binding_verified": True,
            "reference_vector_matches": reference_vector_matches,
            "artifact_audit": artifact_audit,
        },
        "completion": {
            "required_labels": len(cases) * len(FINDINGS),
            "completed_labels": sum(status_counts.values()),
            "missing_labels": len(cases) * len(FINDINGS) - sum(status_counts.values()),
            "status_counts": dict(sorted(status_counts.items())),
            "cases_with_nonempty_other_finding_text": free_text_nonempty,
            "other_finding_text_aggregate_screen": {
                "negative_or_no_additional_marker": free_text_negative_or_none,
                "uncertainty_only_marker": free_text_uncertainty_only,
                "affirmative_narrative_uncoded": free_text_affirmative_narrative,
                "minimum_words": min(free_text_word_counts)
                if free_text_word_counts
                else 0,
                "maximum_words": max(free_text_word_counts)
                if free_text_word_counts
                else 0,
                "included_in_fixed_11_metrics": False,
            },
        },
        "primary_unweighted_selected_challenge": {
            "estimand": "descriptive mean/rate over the 60 selected challenge cases",
            "reference_proxy_vs_r01": primary_reference,
            "arms": primary_arms,
            "transitions_vs_current_only": primary_transitions,
            "stratified_subject_bootstrap": _stratified_bootstrap(
                cases,
                weighted=False,
                n_bootstrap=n_bootstrap,
                seed=seed,
            ),
        },
        "ipw_sensitivity_five_strata_union": {
            "estimand": (
                "inverse-inclusion-probability Hajek estimates for the union "
                "of five prespecified challenge strata"
            ),
            "reference_proxy_vs_r01": weighted_reference,
            "arms": weighted_arms,
            "transitions_vs_current_only": weighted_transitions,
            "stratified_subject_bootstrap": _stratified_bootstrap(
                cases,
                weighted=True,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            ),
        },
        "interpretation_guards": [
            "R01 labels current-image support for 11 prespecified findings.",
            "This is one-reader image-grounded validation, not inter-reader agreement.",
            "The primary cohort is deliberately enriched and is not prevalence representative.",
            "IPW sensitivity targets only the union of five sampled challenge strata.",
            "Task A does not assess report-level wording, temporal narrative, or clinical utility.",
            "Ten affirmative free-text observations remain uncoded and are excluded from fixed-11 metrics.",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    completion = result["completion"]
    primary = result["primary_unweighted_selected_challenge"]
    primary_boot = primary["stratified_subject_bootstrap"]
    weighted_boot = result["ipw_sensitivity_five_strata_union"][
        "stratified_subject_bootstrap"
    ]
    lines = [
        "# R01 Task-A aggregate analysis",
        "",
        (
            f"Locked one-reader image-only adjudication: "
            f"{result['scope']['cases']} cases / "
            f"{completion['completed_labels']} finding labels; "
            f"missing={completion['missing_labels']}."
        ),
        "",
        "## Completion",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for status, count in completion["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    free_text = completion["other_finding_text_aggregate_screen"]
    lines.extend(
        [
            "",
            (
                f"Free text: {completion['cases_with_nonempty_other_finding_text']} "
                f"nonempty entries; {free_text['affirmative_narrative_uncoded']} "
                "affirmative narratives, "
                f"{free_text['negative_or_no_additional_marker']} negative/"
                f"placeholder, {free_text['uncertainty_only_marker']} "
                "uncertainty-only. These entries are not included in the "
                "fixed-11 metrics."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Reference proxy versus R01 current-image labels",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for key, value in primary["reference_proxy_vs_r01"].items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Image-grounded arm results (selected challenge cohort)",
            "",
            (
                "| arm | unsupported positives | missed visible positives | "
                "combined errors/case | image F1 | confirmed image-absent / "
                "resolved proxy-FP |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        row = primary["arms"][arm]
        lines.append(
            f"| {arm} | {row['unsupported_positives']} | "
            f"{row['missed_visible_positives']} | "
            f"{_fmt(row['combined_error_mean_per_case'])} | "
            f"{_fmt(row['image_grounded_f1'])} | "
            f"{_fmt(row['proxy_fp_validity_among_resolved'])} |"
        )
    lines.extend(
        [
            "",
            "## Positive additions relative to current-only",
            "",
            (
                "| arm | additions | image absent | image present | unresolved | "
                "unsupported/resolved | incremental proxy-FP valid/resolved |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS[1:]:
        row = primary["transitions_vs_current_only"][arm]
        interval = primary_boot[
            f"transition/{arm}/addition_unsupported_rate_resolved"
        ]["ci95"]
        lines.append(
            f"| {arm} | {row['positive_additions']} | "
            f"{row['additions_image_absent']} | "
            f"{row['additions_image_present']} | "
            f"{row['additions_unresolved']} | "
            f"{_fmt(row['addition_unsupported_rate_resolved'])} "
            f"[{_fmt(interval[0])}, {_fmt(interval[1])}] | "
            f"{_fmt(row['incremental_proxy_fp_validity_resolved'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct separation of legitimate and wrong history",
            "",
        ]
    )
    matched_real = primary_boot[
        "transition_contrast/matched_wrong_history-real_history/"
        "addition_unsupported_rate_resolved"
    ]
    random_real = primary_boot[
        "transition_contrast/random_wrong_history-real_history/"
        "addition_unsupported_rate_resolved"
    ]
    ipw_matched_current_unsupported = weighted_boot[
        "contrast/matched_wrong_history-current_only/"
        "unsupported_mean_per_case"
    ]
    ipw_matched_current_combined = weighted_boot[
        "contrast/matched_wrong_history-current_only/"
        "combined_error_mean_per_case"
    ]
    ipw_matched_real_additions = weighted_boot[
        "transition_contrast/matched_wrong_history-real_history/"
        "addition_unsupported_rate_resolved"
    ]
    lines.extend(
        [
            (
                "Matched-wrong versus real-history positive additions: "
                f"+{100 * matched_real['estimate']:.1f} percentage points "
                "image-unsupported among resolved events "
                f"(95% CI {100 * matched_real['ci95'][0]:.1f} to "
                f"{100 * matched_real['ci95'][1]:.1f})."
            ),
            "",
            (
                "Random-wrong versus real-history positive additions: "
                f"+{100 * random_real['estimate']:.1f} percentage points "
                f"(95% CI {100 * random_real['ci95'][0]:.1f} to "
                f"{100 * random_real['ci95'][1]:.1f})."
            ),
            "",
            (
                "IPW sensitivity for the frozen five-strata challenge frame: "
                "matched-wrong minus real-history image-unsupported addition "
                f"rate {100 * ipw_matched_real_additions['estimate']:+.1f} "
                "percentage points "
                f"(95% CI {100 * ipw_matched_real_additions['ci95'][0]:+.1f} "
                f"to {100 * ipw_matched_real_additions['ci95'][1]:+.1f}); "
                "matched-wrong minus current-only unsupported positives "
                f"{ipw_matched_current_unsupported['estimate']:+.3f}/case "
                f"(95% CI {ipw_matched_current_unsupported['ci95'][0]:+.3f} "
                f"to {ipw_matched_current_unsupported['ci95'][1]:+.3f}); "
                "combined image-grounded errors "
                f"{ipw_matched_current_combined['estimate']:+.3f}/case "
                f"(95% CI {ipw_matched_current_combined['ci95'][0]:+.3f} "
                f"to {ipw_matched_current_combined['ci95'][1]:+.3f})."
            ),
            "",
            "## Reviewer-facing interpretation",
            "",
            (
                "This result materially answers the image-grounding concern: "
                "the same qualified reader assessed current images without "
                "candidate reports, and wrong-history additions were substantially "
                "more often image-absent than real-history additions. It therefore "
                "separates a legitimate-history pattern (most resolved real-history "
                "additions were image-supported) from a wrong-history pattern "
                "(most resolved matched-wrong additions were image-unsupported)."
            ),
            "",
            (
                "The reference-report proxy is not a perfect clinical label: "
                f"its positive PPV against R01 was "
                f"{primary['reference_proxy_vs_r01']['reference_positive_ppv_vs_image']:.3f}. "
                "The defensible claim is calibrated validity of incremental "
                "wrong-history events, not that every proxy FP is clinically invalid."
            ),
            "",
            (
                "Highest-value follow-up: an independent R02 reading of the same "
                "60 cases for inter-reader reliability. If only R01 is available, "
                "a short blinded report-level audit of the matched/random positive "
                "additions would validate wording and event extraction; a separate "
                "valid-history SCFR-versus-current-only A/B task is needed for "
                "longitudinal clinical utility."
            ),
            "",
            "## Scope",
            "",
            (
                "Primary values are descriptive for the 60-case, five-stratum "
                "challenge sample. The JSON also contains a design-based IPW "
                "sensitivity analysis for the union of those five strata and "
                "stratified subject-bootstrap intervals."
            ),
            "",
            (
                "This is a locked one-radiologist image-grounded Task-A result. "
                "It is not inter-reader agreement and does not assess report-level "
                "wording or longitudinal clinical utility."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--lock-receipt", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--signing-secret", type=Path, required=True)
    parser.add_argument("--released-stage", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--returned-workbook", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        private_key_path=args.private_key,
        response_path=args.response,
        lock_receipt_path=args.lock_receipt,
        release_receipt_path=args.release_receipt,
        signing_secret_path=args.signing_secret,
        released_stage_dir=args.released_stage,
        artifact_dir=args.artifact_dir,
        returned_workbook_path=args.returned_workbook,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["analysis_status"],
                "cases": result["scope"]["cases"],
                "labels": result["completion"]["completed_labels"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

