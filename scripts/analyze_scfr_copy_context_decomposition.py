#!/usr/bin/env python3
"""Decompose history-arm positive additions into copy/context signatures.

The analysis is deliberately text-grounded.  It identifies finding categories
that become positive relative to the aligned current-only output, then checks
whether the supplied prior contains the same positive finding and whether the
positive output sentence resembles any sentence in that prior.  It does not
infer a generator's internal mechanism or establish image-grounded validity.

Only aggregate counts, proportions, and subject-cluster bootstrap intervals
are written.  Row identifiers, subject identifiers, paths, and report text
remain in the authenticated local cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.analyze_source_causal_arms import (
        FINDING_PATTERNS,
        finding_status,
        finding_vector,
        split_sentences,
        token_f1,
        tokenize,
    )
    from src.eval.scfr_proxy_cache import (
        EXPECTED_CHUNKS,
        EXPECTED_ROWS,
        INVENTORY_SCHEMA_VERSION,
        canonical_mimic_subject_id,
        canonical_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from analyze_source_causal_arms import (
        FINDING_PATTERNS,
        finding_status,
        finding_vector,
        split_sentences,
        token_f1,
        tokenize,
    )
    from src.eval.scfr_proxy_cache import (
        EXPECTED_CHUNKS,
        EXPECTED_ROWS,
        INVENTORY_SCHEMA_VERSION,
        canonical_mimic_subject_id,
        canonical_sha256,
    )


ARMS: Tuple[str, ...] = (
    "real_prior",
    "matched_shuffled_prior",
    "shuffled_prior",
)
ALL_INPUT_ARMS: Tuple[str, ...] = ("current_only",) + ARMS
STRATA: Tuple[str, ...] = (
    "all",
    "reference_supported",
    "reference_discordant",
)
CRITERIA: Tuple[str, ...] = (
    "same_finding_positive_in_supplied_prior",
    "exact_normalized_source_sentence_match",
    "near_copy_token_f1_ge_0_8",
    "shared_5gram",
    "same_finding_non_near_copy_reformulation",
)
VECTOR_WIDTH = 1 + len(CRITERIA)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized_sentence(text: str) -> str:
    return " ".join(tokenize(text))


def ngrams(tokens: Sequence[str], n: int) -> set[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return {
        tuple(tokens[index : index + n])
        for index in range(0, len(tokens) - n + 1)
    }


def positive_sentences(text: str, finding: str) -> List[str]:
    patterns = FINDING_PATTERNS[finding]
    return [
        sentence
        for sentence in split_sentences(text)
        if finding_status(sentence, patterns) == "positive"
    ]


def sentence_copy_flags(
    output_positive_sentences: Sequence[str],
    prior_sentences: Sequence[str],
    *,
    near_copy_threshold: float = 0.8,
    shared_ngram_n: int = 5,
) -> Dict[str, bool]:
    """Compare positive output sentences with every supplied-prior sentence."""

    if not 0.0 <= near_copy_threshold <= 1.0:
        raise ValueError("near_copy_threshold must lie in [0, 1]")
    if shared_ngram_n <= 0:
        raise ValueError("shared_ngram_n must be positive")
    output = [
        (sentence, normalized_sentence(sentence), tokenize(sentence))
        for sentence in output_positive_sentences
        if normalized_sentence(sentence)
    ]
    prior = [
        (sentence, normalized_sentence(sentence), tokenize(sentence))
        for sentence in prior_sentences
        if normalized_sentence(sentence)
    ]
    exact = False
    near = False
    shared = False
    prior_ngrams = [ngrams(tokens, shared_ngram_n) for _, _, tokens in prior]
    for output_sentence, output_norm, output_tokens in output:
        output_ngrams = ngrams(output_tokens, shared_ngram_n)
        for index, (prior_sentence, prior_norm, _) in enumerate(prior):
            exact = exact or output_norm == prior_norm
            near = near or (
                token_f1(output_sentence, prior_sentence) >= near_copy_threshold
            )
            shared = shared or bool(output_ngrams & prior_ngrams[index])
            if exact and near and shared:
                break
        if exact and near and shared:
            break
    return {
        "exact_normalized_source_sentence_match": exact,
        "near_copy_token_f1_ge_0_8": near,
        "shared_5gram": shared,
    }


def decompose_case(
    current_output: str,
    history_output: str,
    supplied_prior: str,
    reference: str,
    *,
    current_vector: Optional[Mapping[str, str]] = None,
    reference_vector: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Return one private in-memory record per newly positive finding."""

    current_vector = (
        dict(current_vector)
        if current_vector is not None
        else finding_vector(current_output)
    )
    history_vector = finding_vector(history_output)
    prior_vector = finding_vector(supplied_prior)
    reference_vector = (
        dict(reference_vector)
        if reference_vector is not None
        else finding_vector(reference)
    )
    prior_sentences = split_sentences(supplied_prior)
    events: List[Dict[str, Any]] = []
    for finding in FINDING_PATTERNS:
        if (
            history_vector[finding] != "positive"
            or current_vector[finding] == "positive"
        ):
            continue
        same_prior_finding = prior_vector[finding] == "positive"
        flags = sentence_copy_flags(
            positive_sentences(history_output, finding),
            prior_sentences,
        )
        events.append(
            {
                "stratum": (
                    "reference_supported"
                    if reference_vector[finding] == "positive"
                    else "reference_discordant"
                ),
                "flags": {
                    "same_finding_positive_in_supplied_prior": same_prior_finding,
                    **flags,
                    "same_finding_non_near_copy_reformulation": (
                        same_prior_finding
                        and not flags["near_copy_token_f1_ge_0_8"]
                    ),
                },
            }
        )
    return events


def _safe_artifact_path(cache_root: Path, artifact: Mapping[str, Any]) -> Path:
    cache_key = str(artifact.get("cache_object_key") or "")
    logical_name = str(artifact.get("logical_name") or "")
    if (
        len(cache_key) != 32
        or any(char not in "0123456789abcdef" for char in cache_key)
        or not logical_name
        or Path(logical_name).name != logical_name
    ):
        raise RuntimeError("Unsafe cache artifact identity")
    expected = (cache_root / f"{cache_key}.{logical_name}").resolve()
    source = Path(str(artifact.get("source_path") or "")).resolve()
    try:
        expected.relative_to(cache_root)
    except ValueError as exc:
        raise RuntimeError("Cache artifact escapes certified cache root") from exc
    if source != expected:
        raise RuntimeError("Inventory source path does not match cache identity")
    return expected


def validate_and_index_inventory(
    inventory_path: Path,
    *,
    recovery_root: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Any]]:
    inventory = load_json(inventory_path)
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported or invalid cache inventory")
    cache_root = Path(str(inventory.get("cache_root") or "")).resolve()
    if not cache_root.is_dir():
        raise RuntimeError("Certified local cache root is unavailable")
    full = inventory.get("full18716")
    if (
        not isinstance(full, Mapping)
        or full.get("expected_rows_per_arm") != EXPECTED_ROWS
        or full.get("expected_chunks") != list(EXPECTED_CHUNKS)
        or full.get("all_five_arms_complete_and_aligned") is not True
    ):
        raise RuntimeError("Inventory does not certify the full aligned cohort")
    arm_summaries = full.get("arms")
    alignment = full.get("cross_arm_alignment")
    if not isinstance(arm_summaries, Mapping) or not isinstance(alignment, Mapping):
        raise RuntimeError("Inventory arm summaries are missing")
    for arm in ALL_INPUT_ARMS:
        if (
            not isinstance(arm_summaries.get(arm), Mapping)
            or arm_summaries[arm].get("rows") != EXPECTED_ROWS
            or arm_summaries[arm].get("unique_row_ids") != EXPECTED_ROWS
            or arm_summaries[arm].get("expected_complete") is not True
            or not isinstance(alignment.get(arm), Mapping)
            or not all(
                alignment[arm].get(key) is True
                for key in (
                    "same_row_ids_as_current_only",
                    "same_references_as_current_only",
                    "same_current_images_as_current_only",
                )
            )
        ):
            raise RuntimeError(f"Inventory does not certify arm {arm}")

    indexed: Dict[str, Dict[int, Dict[str, Any]]] = {
        arm: {} for arm in ALL_INPUT_ARMS
    }
    for value in inventory.get("artifacts") or []:
        if not isinstance(value, Mapping):
            continue
        marker = value.get("full18716")
        if not isinstance(marker, Mapping):
            continue
        arm = str(marker.get("arm") or "")
        chunk = marker.get("chunk")
        if arm not in indexed:
            continue
        if chunk not in EXPECTED_CHUNKS or chunk in indexed[arm]:
            raise RuntimeError("Duplicate or invalid selected arm/chunk")
        artifact = dict(value)
        certified_path = _safe_artifact_path(cache_root, artifact)
        if certified_path.is_file():
            artifact["_path"] = certified_path
        elif recovery_root is not None:
            recovery_path = (
                recovery_root.resolve()
                / "artifacts"
                / str(artifact["logical_name"])
            ).resolve()
            try:
                recovery_path.relative_to(recovery_root.resolve())
            except ValueError as exc:
                raise RuntimeError("Recovery artifact escapes recovery root") from exc
            artifact["_path"] = recovery_path
        else:
            artifact["_path"] = certified_path
        indexed[arm][int(chunk)] = artifact
    for arm in ALL_INPUT_ARMS:
        if tuple(sorted(indexed[arm])) != EXPECTED_CHUNKS:
            raise RuntimeError(f"Arm {arm} does not contain chunks 1..10")
    return indexed, {
        "inventory_sha256": sha256_file(inventory_path),
        "certified_rows_per_arm": EXPECTED_ROWS,
        "selected_arms": len(ALL_INPUT_ARMS),
        "selected_artifacts": len(ALL_INPUT_ARMS) * len(EXPECTED_CHUNKS),
        "inventory_certifies_all_five_arms_aligned": True,
    }


def load_verified_rows(artifact: Mapping[str, Any], *, label: str) -> List[Dict[str, Any]]:
    path = artifact["_path"]
    if not isinstance(path, Path) or not path.is_file():
        raise RuntimeError(f"{label}: cached artifact is unavailable")
    if path.stat().st_size != int(artifact.get("bytes") or -1):
        raise RuntimeError(f"{label}: byte-size mismatch")
    if sha256_file(path) != str(artifact.get("sha256") or ""):
        raise RuntimeError(f"{label}: SHA-256 mismatch")
    payload = load_json(path)
    if (
        not isinstance(payload, list)
        or len(payload) != int(artifact.get("json_list_rows") or -1)
    ):
        raise RuntimeError(f"{label}: invalid JSON row census")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"{label}: row {index} is not an object")
        rows.append(dict(row))
    return rows


def index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    indexed: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(rows):
        row = dict(value)
        identifier = str(row.get("id") or "").strip()
        images = row.get("image_paths")
        if (
            not identifier
            or identifier in indexed
            or not isinstance(images, list)
            or not images
        ):
            raise RuntimeError(f"{label}: invalid/duplicate row {index}")
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        if not reference.strip() or not prediction.strip():
            raise RuntimeError(f"{label}: empty reference/prediction")
        subject = canonical_mimic_subject_id(
            row,
            current_images=[str(path) for path in images],
            label=f"{label}/row{index}",
        )
        row["_subject"] = subject
        row["_images"] = [str(path) for path in images]
        row["_reference"] = reference
        row["_prediction"] = prediction
        order.append(identifier)
        indexed[identifier] = row
    return order, indexed


def _empty_cell() -> np.ndarray:
    return np.zeros(VECTOR_WIDTH, dtype=np.int64)


def _event_vector(flags: Mapping[str, bool]) -> np.ndarray:
    return np.asarray(
        [1] + [int(bool(flags[criterion])) for criterion in CRITERIA],
        dtype=np.int64,
    )


def _percentile_ci(values: np.ndarray) -> List[float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [None, None]
    return [
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    ]


def bootstrap_proportions(
    subject_matrix: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    batch_size: int = 50,
) -> Dict[str, List[float | None]]:
    """Subject-cluster percentile CIs for event-weighted criterion fractions."""

    if subject_matrix.ndim != 3 or subject_matrix.shape[2] != VECTOR_WIDTH:
        raise ValueError("subject_matrix has an invalid shape")
    n_subjects, n_cells, _ = subject_matrix.shape
    draws = np.full(
        (n_bootstrap, n_cells, len(CRITERIA)),
        np.nan,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_subjects, 1.0 / n_subjects)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        weights = rng.multinomial(
            n_subjects,
            probabilities,
            size=stop - start,
        )
        totals = np.einsum("bs,scv->bcv", weights, subject_matrix)
        denominators = totals[:, :, 0:1]
        with np.errstate(divide="ignore", invalid="ignore"):
            draws[start:stop] = np.where(
                denominators > 0,
                totals[:, :, 1:] / denominators,
                np.nan,
            )
    output = {
        f"{cell}:{criterion}": _percentile_ci(draws[:, cell, criterion_index])
        for cell in range(n_cells)
        for criterion_index, criterion in enumerate(CRITERIA)
    }
    same_prior_index = CRITERIA.index(
        "same_finding_positive_in_supplied_prior"
    )
    non_near_index = CRITERIA.index(
        "same_finding_non_near_copy_reformulation"
    )
    same_prior = draws[:, :, same_prior_index]
    # ``draws`` contains fractions over all additions.  Their ratio therefore
    # equals the desired conditional fraction among same-prior events.
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional_non_near = np.where(
            same_prior > 0,
            draws[:, :, non_near_index] / same_prior,
            np.nan,
        )
        conditional_near = np.where(
            same_prior > 0,
            1.0 - conditional_non_near,
            np.nan,
        )
    for cell in range(n_cells):
        output[f"{cell}:conditional_near_among_same_prior"] = _percentile_ci(
            conditional_near[:, cell]
        )
        output[
            f"{cell}:conditional_non_near_among_same_prior"
        ] = _percentile_ci(conditional_non_near[:, cell])
    return output


def run_analysis(
    inventory_path: Path,
    *,
    recovery_root: Optional[Path] = None,
    n_bootstrap: int = 5000,
    seed: int = 20260728,
) -> Dict[str, Any]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    artifacts, validation = validate_and_index_inventory(
        inventory_path,
        recovery_root=recovery_root,
    )
    cells = [(arm, stratum) for arm in ARMS for stratum in STRATA]
    cell_index = {cell: index for index, cell in enumerate(cells)}
    subject_counts: MutableMapping[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(cells), VECTOR_WIDTH), dtype=np.int64)
    )
    aggregate_counts = {
        cell: _empty_cell() for cell in cells
    }
    report_counts = {
        cell: {
            "reports_with_additions": 0,
            **{f"reports_with_{criterion}": 0 for criterion in CRITERIA},
        }
        for cell in cells
    }
    seen_ids: Dict[str, set[str]] = {arm: set() for arm in ALL_INPUT_ARMS}
    identity_rows: Dict[str, List[Tuple[str, str, str]]] = {
        arm: [] for arm in ALL_INPUT_ARMS
    }

    for chunk in EXPECTED_CHUNKS:
        current_order, current_by_id = index_rows(
            load_verified_rows(
                artifacts["current_only"][chunk],
                label=f"current_only/chunk{chunk:03d}",
            ),
            label=f"current_only/chunk{chunk:03d}",
        )
        current_id_set = set(current_order)
        if seen_ids["current_only"] & current_id_set:
            raise RuntimeError("Duplicate current-only IDs across chunks")
        seen_ids["current_only"].update(current_id_set)
        for identifier in current_order:
            row = current_by_id[identifier]
            # Include subjects with zero addition events in the resampling
            # frame; otherwise the cluster bootstrap conditions on having at
            # least one addition and understates the certified cohort census.
            subject_counts[row["_subject"]]
            row["_current_vector"] = finding_vector(row["_prediction"])
            row["_reference_vector"] = finding_vector(row["_reference"])
            identity_rows["current_only"].append(
                (
                    identifier,
                    canonical_sha256(row["_reference"]),
                    canonical_sha256(row["_images"]),
                )
            )

        for arm in ARMS:
            arm_order, arm_by_id = index_rows(
                load_verified_rows(
                    artifacts[arm][chunk],
                    label=f"{arm}/chunk{chunk:03d}",
                ),
                label=f"{arm}/chunk{chunk:03d}",
            )
            arm_id_set = set(arm_order)
            if arm_id_set != current_id_set:
                raise RuntimeError(f"{arm}/chunk{chunk:03d}: ID alignment failed")
            if seen_ids[arm] & arm_id_set:
                raise RuntimeError(f"{arm}: duplicate IDs across chunks")
            seen_ids[arm].update(arm_id_set)
            for identifier in current_order:
                current = current_by_id[identifier]
                history = arm_by_id[identifier]
                if (
                    history["_reference"] != current["_reference"]
                    or history["_images"] != current["_images"]
                    or history["_subject"] != current["_subject"]
                ):
                    raise RuntimeError(
                        f"{arm}/chunk{chunk:03d}: aligned row identity failed"
                    )
                prior_report = str(
                    history.get("prior_report") or history.get("context") or ""
                )
                if not prior_report.strip():
                    raise RuntimeError(f"{arm}/chunk{chunk:03d}: empty supplied prior")
                if (
                    history.get("prior_report") not in (None, "")
                    and history.get("context") not in (None, "")
                    and str(history["prior_report"]) != str(history["context"])
                ):
                    raise RuntimeError(
                        f"{arm}/chunk{chunk:03d}: context/prior_report mismatch"
                    )
                events = decompose_case(
                    current["_prediction"],
                    history["_prediction"],
                    prior_report,
                    current["_reference"],
                    current_vector=current["_current_vector"],
                    reference_vector=current["_reference_vector"],
                )
                events_by_stratum: Dict[str, List[Mapping[str, bool]]] = {
                    "all": [event["flags"] for event in events],
                    "reference_supported": [
                        event["flags"]
                        for event in events
                        if event["stratum"] == "reference_supported"
                    ],
                    "reference_discordant": [
                        event["flags"]
                        for event in events
                        if event["stratum"] == "reference_discordant"
                    ],
                }
                for stratum, flags_list in events_by_stratum.items():
                    if not flags_list:
                        continue
                    cell = (arm, stratum)
                    report_counts[cell]["reports_with_additions"] += 1
                    summed = sum(
                        (_event_vector(flags) for flags in flags_list),
                        start=_empty_cell(),
                    )
                    aggregate_counts[cell] += summed
                    subject_counts[current["_subject"]][cell_index[cell]] += summed
                    for criterion in CRITERIA:
                        if any(flags[criterion] for flags in flags_list):
                            report_counts[cell][f"reports_with_{criterion}"] += 1
                identity_rows[arm].append(
                    (
                        identifier,
                        canonical_sha256(history["_reference"]),
                        canonical_sha256(history["_images"]),
                    )
                )
            del arm_by_id
        del current_by_id

    if any(len(values) != EXPECTED_ROWS for values in seen_ids.values()):
        raise RuntimeError("Actual selected-arm row census is not 18,716")
    if any(seen_ids[arm] != seen_ids["current_only"] for arm in ARMS):
        raise RuntimeError("Actual selected-arm ID sets are not aligned")

    inventory = load_json(inventory_path)
    summaries = inventory["full18716"]["arms"]
    for arm in ALL_INPUT_ARMS:
        identities = identity_rows[arm]
        observed = {
            "row_id_set_sha256": canonical_sha256(
                sorted(identifier for identifier, _, _ in identities)
            ),
            "id_reference_map_sha256": canonical_sha256(
                sorted((identifier, reference_hash) for identifier, reference_hash, _ in identities)
            ),
            "id_current_image_map_sha256": canonical_sha256(
                sorted((identifier, image_hash) for identifier, _, image_hash in identities)
            ),
        }
        for key, value in observed.items():
            if value != summaries[arm].get(key):
                raise RuntimeError(f"{arm}: aggregate {key} mismatch")

    subjects = sorted(subject_counts)
    subject_matrix = np.stack([subject_counts[subject] for subject in subjects])
    intervals = bootstrap_proportions(
        subject_matrix,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    results: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        results[arm] = {}
        for stratum in STRATA:
            cell = (arm, stratum)
            index = cell_index[cell]
            counts = aggregate_counts[cell]
            denominator = int(counts[0])
            criteria: Dict[str, Any] = {}
            for criterion_index, criterion in enumerate(CRITERIA, start=1):
                numerator = int(counts[criterion_index])
                criteria[criterion] = {
                    "addition_events_meeting_criterion": numerator,
                    "fraction_of_addition_events": (
                        numerator / denominator if denominator else None
                    ),
                    "subject_cluster_bootstrap_ci95": intervals[
                        f"{index}:{criterion}"
                    ],
                    "reports_with_at_least_one_matching_addition": report_counts[
                        cell
                    ][f"reports_with_{criterion}"],
                }
            results[arm][stratum] = {
                "addition_events": denominator,
                "reports_with_at_least_one_addition": report_counts[cell][
                    "reports_with_additions"
                ],
                "same_prior_finding_copy_form": {
                    "same_prior_finding_events": int(counts[1]),
                    "near_copy_events": int(counts[1] - counts[-1]),
                    "near_copy_fraction_among_same_prior_finding_events": (
                        (counts[1] - counts[-1]) / counts[1]
                        if counts[1]
                        else None
                    ),
                    "near_copy_subject_cluster_bootstrap_ci95": intervals[
                        f"{index}:conditional_near_among_same_prior"
                    ],
                    "non_near_copy_reformulation_events": int(counts[-1]),
                    "non_near_copy_reformulation_fraction_among_same_prior_finding_events": (
                        counts[-1] / counts[1] if counts[1] else None
                    ),
                    "non_near_copy_subject_cluster_bootstrap_ci95": intervals[
                        f"{index}:conditional_non_near_among_same_prior"
                    ],
                },
                "criteria": criteria,
            }

    return {
        "schema_version": "scfr-copy-context-decomposition-v1",
        "status": "complete",
        "scope": {
            "rows": EXPECTED_ROWS,
            "subjects": len(subjects),
            "history_arms": list(ARMS),
            "unit": (
                "11-category finding positive in a history-arm output but not "
                "positive in the aligned current-only output"
            ),
        },
        "method": {
            "finding_extractor": (
                "unchanged rule-based 11-category extractor used by the "
                "full-stress proxy analysis"
            ),
            "reference_strata": {
                "reference_supported": (
                    "the same finding is positive in the fixed current-study "
                    "reference report"
                ),
                "reference_discordant": (
                    "the same finding is not positive in the fixed "
                    "current-study reference report"
                ),
            },
            "copy_tests": {
                "same_finding_positive_in_supplied_prior": (
                    "the same extractor labels the supplied prior positive"
                ),
                "exact_normalized_source_sentence_match": (
                    "a positive output sentence equals a prior sentence after "
                    "lowercase alphabetic token normalization"
                ),
                "near_copy_token_f1_ge_0_8": (
                    "maximum multiset token-F1 against any prior sentence is "
                    "at least 0.8"
                ),
                "shared_5gram": (
                    "a positive output sentence and a prior sentence share at "
                    "least one contiguous five-token sequence"
                ),
                "same_finding_non_near_copy_reformulation": (
                    "the supplied prior contains the same positive finding, "
                    "but no positive output sentence reaches token-F1 0.8 "
                    "against any prior sentence"
                ),
            },
            "bootstrap": {
                "unit": "target subject cluster",
                "estimand": "event-weighted criterion fraction",
                "method": "nonparametric cluster bootstrap with replacement",
                "interval": "two-sided percentile 95%",
                "replicates": n_bootstrap,
                "seed": seed,
            },
        },
        "input_validation": {
            **validation,
            "selected_artifact_sha256_verified": True,
            "actual_cross_arm_row_reference_image_subject_alignment_verified": True,
            "aggregate_inventory_identity_hashes_verified": True,
            "row_ids_subject_ids_paths_or_text_emitted": False,
        },
        "results": results,
        "interpretation_boundary": (
            "These text-grounded signatures distinguish exact/near lexical "
            "reuse from same-finding non-near-copy reformulation.  They do not "
            "identify the model's internal causal mechanism and do not replace "
            "image-grounded radiologist adjudication of clinical validity."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Full-18,716 copy-versus-context decomposition",
        "",
        (
            f"Rows: **{payload['scope']['rows']:,}**; target subjects: "
            f"**{payload['scope']['subjects']:,}**. Fractions are over newly "
            "positive 11-category finding events relative to current-only; "
            "brackets show subject-cluster bootstrap 95% CIs."
        ),
        "",
        "| History arm | Reference stratum | Additions | Reports | Same finding in prior | Exact sentence | Near copy (F1>=0.8) | Shared 5-gram | Non-near-copy reformulation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(metric: Mapping[str, Any]) -> str:
        value = metric["fraction_of_addition_events"]
        ci = metric["subject_cluster_bootstrap_ci95"]
        if value is None:
            return "NA"
        return (
            f"{metric['addition_events_meeting_criterion']:,} "
            f"({100 * value:.1f}%; "
            f"{100 * ci[0]:.1f}-{100 * ci[1]:.1f})"
        )

    arm_labels = {
        "real_prior": "Real history",
        "matched_shuffled_prior": "Matched wrong history",
        "shuffled_prior": "Shuffled wrong history",
    }
    stratum_labels = {
        "all": "All",
        "reference_supported": "Reference-supported",
        "reference_discordant": "Reference-discordant",
    }
    for arm in ARMS:
        for stratum in STRATA:
            row = payload["results"][arm][stratum]
            criteria = row["criteria"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        arm_labels[arm],
                        stratum_labels[stratum],
                        f"{row['addition_events']:,}",
                        f"{row['reports_with_at_least_one_addition']:,}",
                        cell(criteria["same_finding_positive_in_supplied_prior"]),
                        cell(criteria["exact_normalized_source_sentence_match"]),
                        cell(criteria["near_copy_token_f1_ge_0_8"]),
                        cell(criteria["shared_5gram"]),
                        cell(
                            criteria[
                                "same_finding_non_near_copy_reformulation"
                            ]
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Copy form conditional on the same finding being positive in the supplied prior",
            "",
            "| History arm | Reference stratum | Same-prior events | Near copy | Non-near-copy reformulation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for stratum in STRATA:
            copy_form = payload["results"][arm][stratum][
                "same_prior_finding_copy_form"
            ]
            near_ci = copy_form["near_copy_subject_cluster_bootstrap_ci95"]
            non_near_ci = copy_form[
                "non_near_copy_subject_cluster_bootstrap_ci95"
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        arm_labels[arm],
                        stratum_labels[stratum],
                        f"{copy_form['same_prior_finding_events']:,}",
                        (
                            f"{copy_form['near_copy_events']:,} "
                            f"({100 * copy_form['near_copy_fraction_among_same_prior_finding_events']:.1f}%; "
                            f"{100 * near_ci[0]:.1f}-{100 * near_ci[1]:.1f})"
                        ),
                        (
                            f"{copy_form['non_near_copy_reformulation_events']:,} "
                            f"({100 * copy_form['non_near_copy_reformulation_fraction_among_same_prior_finding_events']:.1f}%; "
                            f"{100 * non_near_ci[0]:.1f}-{100 * non_near_ci[1]:.1f})"
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "The decomposition is text-grounded: it separates exact/near "
            "lexical reuse from same-finding non-near-copy reformulation, but "
            "does not by itself establish image-grounded clinical validity or "
            "the generator's internal causal mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.resolve() == args.output_md.resolve():
        raise RuntimeError("JSON and Markdown output paths must differ")
    for output in (args.output_json, args.output_md):
        output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_analysis(
        args.inventory,
        recovery_root=args.recovery_root,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["scope"]["rows"],
                "subjects": payload["scope"]["subjects"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "aggregate_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

