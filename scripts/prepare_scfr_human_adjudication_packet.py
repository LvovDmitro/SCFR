#!/usr/bin/env python3
"""Prepare, but never release, a blinded SCFR human-adjudication packet.

This is an offline packet builder.  It consumes a frozen reader manifest, its
private mapping, and the hash inventory of already staged pixels.  It neither
uses ClearML nor performs network operations.

The full reader handoff is split into three independently releasable
directories:

1. Task A: current images only;
2. Task B stage 1: current and provenance-verified true-prior images, with
   history informativeness recorded before any candidate reports are shown;
3. Task B stage 2: anonymous, re-randomized A/B candidate reports.

Full A+B is the default.  The explicit ``--task-a-only`` scope emits only the
first directory for current-image support adjudication and records that Task B
is not authorized.

The decoding key is written to an explicitly separate private root outside
the repository.  No private directory or key is ever created below the public
packet root.  Every reader-facing artifact is marked
``prepared_not_released``.  This program has no release mode: completing a
computational audit and authorizing qualified human readers are external
protocol gates.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from src.eval.scfr_blinded_reader import (
        FINDINGS,
        normalized_report_hash,
        opaque_token,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.eval.scfr_blinded_reader import (
        FINDINGS,
        normalized_report_hash,
        opaque_token,
    )


PROTOCOL_VERSION = "scfr-human-adjudication-packet-v1"
RELEASE_STATUS = "prepared_not_released"
EXPECTED_PUBLIC_PROTOCOL = "scfr-model-reader-v1"
EXPECTED_PRIVATE_PROTOCOL = "scfr-model-reader-v1-private"
EXPECTED_SAMPLING_PROTOCOL = "scfr-model-reader-sampling-v2"
REQUIRED_TASK_B_ARMS = ("current_only", "scfr")

TOKEN_RE = re.compile(r"^[0-9a-f]{16,64}$")
SLOT_RE = re.compile(r"^[A-Z]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")

PUBLIC_ROOT_KEYS = {
    "protocol_version",
    "sampling_protocol_version",
    "protocol_scope",
    "seed",
    "cases",
    "manifest_sha256",
}
PUBLIC_CASE_REQUIRED_KEYS = {
    "case_token",
    "reader_cohort",
    "current_image_tokens",
    "prior_image_tokens",
    "report_presentations",
    "valid_history_pair",
    "pair_is_identical",
}
PUBLIC_CASE_OPTIONAL_KEYS = {"true_prior_context"}
PUBLIC_PRESENTATION_KEYS = {"slot", "report_token", "report"}

PRIVATE_ROOT_KEYS = {
    "protocol_version",
    "sampling_protocol_version",
    "protocol_scope",
    "arm_schema",
    "arm_fields",
    "seed",
    "blinding_secret",
    "deduplication",
    "sampling",
    "cases",
    "public_manifest_sha256",
}
PRIVATE_CASE_KEYS = {
    "case_token",
    "source_case_id",
    "subject_id",
    "panel",
    "cohort",
    "sampling_stratum",
    "inclusion_probability",
    "report_mapping",
    "valid_history_pair_mapping",
    "reference_finding_statuses",
    "selected_arm",
    "arm_fields",
    "protocol_scope",
    "evaluation_regime",
}
PRIVATE_REPORT_MAPPING_KEYS = {
    "case_token",
    "slot_to_arms",
    "arm_to_slot",
    "slot_to_report_hash",
}
ASSET_INVENTORY_ROOT_KEYS = {
    "public_manifest_sha256",
    "asset_count",
    "assets",
}
ASSET_INVENTORY_ENTRY_KEYS = {
    "relative_path",
    "bytes",
    "sha256",
    "copy_status",
}

TASK_A_STATUSES = ("present", "absent", "uncertain", "unassessable")
HISTORY_INFORMATIVENESS_VALUES = (
    "clinically_informative",
    "not_clinically_informative",
    "uncertain",
    "unassessable",
)
PREFERENCE_VALUES = ("A", "B", "tie", "unassessable")
TEMPORAL_CORRECTNESS_VALUES = (
    "correct",
    "partially_correct",
    "incorrect",
    "not_applicable",
    "unassessable",
)

# These names must never occur as reader-facing JSON keys.  Report text is
# intentionally exempt from value scanning: the reports themselves are the
# anonymous clinical objects being judged.
FORBIDDEN_READER_KEY_WORDS = {
    "arm",
    "source",
    "reference",
    "model",
    "generator",
    "subject",
    "study",
    "patient",
    "cohort",
    "panel",
    "sampling",
    "seed",
    "private",
}
REPORT_TEXT_KEYS = {"report_a", "report_b"}


@dataclass(frozen=True)
class ValidatedInputs:
    """Fully cross-checked frozen inputs and staged pixel locations."""

    reader_manifest: Dict[str, Any]
    private_mapping: Dict[str, Any]
    asset_inventory: Dict[str, Any]
    public_by_token: Dict[str, Dict[str, Any]]
    private_by_token: Dict[str, Dict[str, Any]]
    asset_paths: Dict[str, Path]
    input_hashes: Dict[str, str]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 representation used for commitments."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_canonical_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    location: str,
) -> None:
    observed = set(value)
    expected_set = set(expected)
    if observed != expected_set:
        raise RuntimeError(
            f"{location} has unexpected schema: "
            f"missing={sorted(expected_set-observed)} "
            f"extra={sorted(observed-expected_set)}"
        )


def _require_token(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not TOKEN_RE.fullmatch(value):
        raise RuntimeError(f"{location} is not an opaque token")
    return value


def _validate_presentations(
    value: Any,
    *,
    location: str,
    allow_empty: bool,
    globally_seen_tokens: set,
) -> Dict[str, Dict[str, str]]:
    if not isinstance(value, list):
        raise RuntimeError(f"{location} must be a list")
    if not value and not allow_empty:
        raise RuntimeError(f"{location} must not be empty")
    by_slot: Dict[str, Dict[str, str]] = {}
    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"
        if not isinstance(item, dict):
            raise RuntimeError(f"{item_location} must be an object")
        _require_exact_keys(
            item,
            PUBLIC_PRESENTATION_KEYS,
            location=item_location,
        )
        slot = item["slot"]
        if not isinstance(slot, str) or not SLOT_RE.fullmatch(slot):
            raise RuntimeError(f"{item_location}.slot is invalid")
        if slot in by_slot:
            raise RuntimeError(f"{location} contains a duplicate slot")
        report_token = _require_token(
            item["report_token"],
            location=f"{item_location}.report_token",
        )
        if report_token in globally_seen_tokens:
            raise RuntimeError("Duplicate report token across frozen presentations")
        globally_seen_tokens.add(report_token)
        report = item["report"]
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError(f"{item_location}.report must be non-empty")
        by_slot[slot] = {
            "slot": slot,
            "report_token": report_token,
            "report": report,
        }
    return by_slot


def _validate_report_mapping(
    mapping: Any,
    *,
    public_presentations: Mapping[str, Mapping[str, str]],
    expected_arms: Sequence[str],
    expected_case_token: str,
    location: str,
) -> None:
    if not isinstance(mapping, dict):
        raise RuntimeError(f"{location} must be an object")
    _require_exact_keys(
        mapping,
        PRIVATE_REPORT_MAPPING_KEYS,
        location=location,
    )
    if mapping["case_token"] != expected_case_token:
        raise RuntimeError(f"{location}.case_token mismatch")
    arm_to_slot = mapping["arm_to_slot"]
    slot_to_arms = mapping["slot_to_arms"]
    slot_to_report_hash = mapping["slot_to_report_hash"]
    if not isinstance(arm_to_slot, dict) or set(arm_to_slot) != set(expected_arms):
        raise RuntimeError(f"{location}.arm_to_slot does not match frozen arms")
    if not isinstance(slot_to_arms, dict):
        raise RuntimeError(f"{location}.slot_to_arms must be an object")
    if not isinstance(slot_to_report_hash, dict):
        raise RuntimeError(f"{location}.slot_to_report_hash must be an object")
    public_slots = set(public_presentations)
    if set(slot_to_arms) != public_slots or set(slot_to_report_hash) != public_slots:
        raise RuntimeError(f"{location} slot keys do not match public reports")

    reconstructed: Dict[str, str] = {}
    for slot, arms in slot_to_arms.items():
        if not isinstance(arms, list) or not arms:
            raise RuntimeError(f"{location}.slot_to_arms entries must be lists")
        if len(arms) != len(set(arms)):
            raise RuntimeError(f"{location} contains duplicate arm assignments")
        for arm in arms:
            if arm not in expected_arms or arm in reconstructed:
                raise RuntimeError(f"{location} has invalid arm assignments")
            reconstructed[arm] = slot
        expected_hash = normalized_report_hash(
            public_presentations[slot]["report"]
        )
        if slot_to_report_hash[slot] != expected_hash:
            raise RuntimeError(f"{location} report hash mismatch")
    if reconstructed != arm_to_slot:
        raise RuntimeError(f"{location} inverse arm/slot mappings disagree")


def _validate_asset_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_manifest_hash: str,
    referenced_tokens: set,
    asset_root: Path,
) -> Dict[str, Path]:
    _require_exact_keys(
        inventory,
        ASSET_INVENTORY_ROOT_KEYS,
        location="asset_inventory",
    )
    if inventory["public_manifest_sha256"] != expected_manifest_hash:
        raise RuntimeError("Asset inventory does not match the frozen manifest")
    assets = inventory["assets"]
    if not isinstance(assets, dict):
        raise RuntimeError("asset_inventory.assets must be an object")
    if isinstance(inventory["asset_count"], bool) or not isinstance(
        inventory["asset_count"], int
    ):
        raise RuntimeError("asset_inventory.asset_count must be an integer")
    if inventory["asset_count"] != len(assets):
        raise RuntimeError("Asset inventory count mismatch")
    if set(assets) != referenced_tokens:
        raise RuntimeError(
            "Frozen image references and asset inventory tokens differ"
        )

    root = asset_root.resolve()
    resolved: Dict[str, Path] = {}
    observed_relative_paths = set()
    for token, item in assets.items():
        _require_token(token, location="asset_inventory asset key")
        if not isinstance(item, dict):
            raise RuntimeError("Asset inventory entry must be an object")
        _require_exact_keys(
            item,
            ASSET_INVENTORY_ENTRY_KEYS,
            location="asset_inventory asset entry",
        )
        relative = item["relative_path"]
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Asset inventory relative_path must be non-empty")
        if "\\" in relative:
            raise RuntimeError("Asset inventory paths must use forward slashes")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise RuntimeError("Unsafe asset inventory relative_path")
        if not pure.parts or pure.parts[0] != "images":
            raise RuntimeError("Staged pixels must reside below images/")
        if relative in observed_relative_paths:
            raise RuntimeError("Duplicate asset inventory relative_path")
        observed_relative_paths.add(relative)
        candidate = root.joinpath(*pure.parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Asset inventory path escapes asset root") from exc
        if not candidate.is_file():
            raise RuntimeError("A staged pixel listed in the inventory is missing")
        observed_bytes = candidate.stat().st_size
        expected_bytes = item["bytes"]
        if isinstance(expected_bytes, bool) or not isinstance(
            expected_bytes, int
        ):
            raise RuntimeError("Asset byte count must be an integer")
        if expected_bytes != observed_bytes:
            raise RuntimeError("Staged pixel byte count mismatch")
        expected_sha = item["sha256"]
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(
            expected_sha
        ):
            raise RuntimeError("Asset SHA-256 is invalid")
        if sha256_file(candidate) != expected_sha:
            raise RuntimeError("Staged pixel SHA-256 mismatch")
        if not isinstance(item["copy_status"], str) or not item[
            "copy_status"
        ].strip():
            raise RuntimeError("Asset copy_status must be non-empty")
        resolved[token] = candidate
    return resolved


def validate_inputs(
    *,
    reader_manifest_path: Path,
    private_mapping_path: Path,
    asset_inventory_path: Path,
    asset_root: Optional[Path] = None,
) -> ValidatedInputs:
    """Load and cross-audit every frozen input before creating output."""

    reader = load_object(reader_manifest_path)
    private = load_object(private_mapping_path)
    inventory = load_object(asset_inventory_path)

    _require_exact_keys(reader, PUBLIC_ROOT_KEYS, location="reader_manifest")
    _require_exact_keys(private, PRIVATE_ROOT_KEYS, location="private_mapping")
    if reader["protocol_version"] != EXPECTED_PUBLIC_PROTOCOL:
        raise RuntimeError("Unsupported frozen public protocol")
    if private["protocol_version"] != EXPECTED_PRIVATE_PROTOCOL:
        raise RuntimeError("Unsupported frozen private protocol")
    if (
        reader["sampling_protocol_version"] != EXPECTED_SAMPLING_PROTOCOL
        or private["sampling_protocol_version"] != EXPECTED_SAMPLING_PROTOCOL
    ):
        raise RuntimeError("Unsupported frozen sampling protocol")
    if private["protocol_scope"] != reader["protocol_scope"]:
        raise RuntimeError("Public/private protocol scopes differ")
    if private["seed"] != reader["seed"]:
        raise RuntimeError("Public/private frozen seeds differ")
    if not isinstance(private["blinding_secret"], str) or not private[
        "blinding_secret"
    ]:
        raise RuntimeError("Private frozen blinding secret is missing")

    claimed_manifest_hash = reader["manifest_sha256"]
    if not isinstance(claimed_manifest_hash, str) or not SHA256_RE.fullmatch(
        claimed_manifest_hash
    ):
        raise RuntimeError("Frozen manifest SHA-256 is invalid")
    unhashed_reader = dict(reader)
    unhashed_reader.pop("manifest_sha256")
    if sha256_canonical_json(unhashed_reader) != claimed_manifest_hash:
        raise RuntimeError("Frozen reader manifest semantic SHA-256 mismatch")
    if private["public_manifest_sha256"] != claimed_manifest_hash:
        raise RuntimeError("Private mapping does not match public manifest")

    arm_schema = private["arm_schema"]
    arm_fields = private["arm_fields"]
    if (
        not isinstance(arm_schema, list)
        or not arm_schema
        or not all(isinstance(arm, str) and arm for arm in arm_schema)
        or len(arm_schema) != len(set(arm_schema))
    ):
        raise RuntimeError("Private arm schema is invalid")
    if not isinstance(arm_fields, dict) or set(arm_fields) != set(arm_schema):
        raise RuntimeError("Private arm fields do not match arm schema")

    public_cases = reader["cases"]
    private_cases = private["cases"]
    if not isinstance(public_cases, list) or not public_cases:
        raise RuntimeError("Frozen public cases must be a non-empty list")
    if not isinstance(private_cases, list) or len(private_cases) != len(
        public_cases
    ):
        raise RuntimeError("Public/private frozen case counts differ")

    public_by_token: Dict[str, Dict[str, Any]] = {}
    private_by_token: Dict[str, Dict[str, Any]] = {}
    seen_subjects = set()
    seen_images = set()
    seen_reports = set()
    public_order: List[str] = []
    private_order: List[str] = []
    report_slots_by_case: Dict[str, Dict[str, Dict[str, str]]] = {}
    pair_slots_by_case: Dict[str, Dict[str, Dict[str, str]]] = {}

    for index, public_case in enumerate(public_cases):
        location = f"reader_manifest.cases[{index}]"
        if not isinstance(public_case, dict):
            raise RuntimeError(f"{location} must be an object")
        observed = set(public_case)
        missing = PUBLIC_CASE_REQUIRED_KEYS - observed
        extra = observed - (
            PUBLIC_CASE_REQUIRED_KEYS | PUBLIC_CASE_OPTIONAL_KEYS
        )
        if missing or extra:
            raise RuntimeError(
                f"{location} has unexpected schema: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        case_token = _require_token(
            public_case["case_token"],
            location=f"{location}.case_token",
        )
        if case_token in public_by_token:
            raise RuntimeError("Duplicate frozen public case token")
        public_order.append(case_token)
        cohort = public_case["reader_cohort"]
        if not isinstance(cohort, str) or not cohort.strip():
            raise RuntimeError(f"{location}.reader_cohort must be non-empty")
        for field in ("current_image_tokens", "prior_image_tokens"):
            tokens = public_case[field]
            if not isinstance(tokens, list):
                raise RuntimeError(f"{location}.{field} must be a list")
            if field == "current_image_tokens" and not tokens:
                raise RuntimeError(f"{location}.{field} must not be empty")
            for image_token in tokens:
                _require_token(
                    image_token,
                    location=f"{location}.{field}",
                )
                if image_token in seen_images:
                    raise RuntimeError("Duplicate image token across frozen cases")
                seen_images.add(image_token)

        report_presentations = _validate_presentations(
            public_case["report_presentations"],
            location=f"{location}.report_presentations",
            allow_empty=False,
            globally_seen_tokens=seen_reports,
        )
        pair_presentations = _validate_presentations(
            public_case["valid_history_pair"],
            location=f"{location}.valid_history_pair",
            allow_empty=True,
            globally_seen_tokens=seen_reports,
        )
        pair_is_identical = public_case["pair_is_identical"]
        if pair_presentations:
            if not isinstance(pair_is_identical, bool):
                raise RuntimeError(
                    f"{location}.pair_is_identical must be boolean"
                )
            expected_size = 1 if pair_is_identical else 2
            if len(pair_presentations) != expected_size:
                raise RuntimeError(
                    f"{location} pair size/identity marker mismatch"
                )
            if not isinstance(public_case.get("true_prior_context"), str):
                raise RuntimeError(
                    f"{location}.true_prior_context is required with pair"
                )
        else:
            if pair_is_identical is not None:
                raise RuntimeError(
                    f"{location}.pair_is_identical must be null without pair"
                )
            if "true_prior_context" in public_case:
                raise RuntimeError(
                    f"{location}.true_prior_context is forbidden without pair"
                )
        report_slots_by_case[case_token] = report_presentations
        pair_slots_by_case[case_token] = pair_presentations
        public_by_token[case_token] = public_case

    for index, private_case in enumerate(private_cases):
        location = f"private_mapping.cases[{index}]"
        if not isinstance(private_case, dict):
            raise RuntimeError(f"{location} must be an object")
        _require_exact_keys(
            private_case,
            PRIVATE_CASE_KEYS,
            location=location,
        )
        case_token = _require_token(
            private_case["case_token"],
            location=f"{location}.case_token",
        )
        if case_token in private_by_token:
            raise RuntimeError("Duplicate frozen private case token")
        private_order.append(case_token)
        subject_id = private_case["subject_id"]
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise RuntimeError(f"{location}.subject_id must be non-empty")
        if subject_id != subject_id.strip():
            raise RuntimeError(f"{location}.subject_id has boundary whitespace")
        canonical_subject_id = subject_id.casefold()
        if canonical_subject_id in seen_subjects:
            raise RuntimeError("Duplicate subjects in frozen adjudication cohort")
        seen_subjects.add(canonical_subject_id)
        source_case_id = private_case["source_case_id"]
        if not isinstance(source_case_id, str) or not source_case_id.strip():
            raise RuntimeError(f"{location}.source_case_id must be non-empty")
        if source_case_id != source_case_id.strip():
            raise RuntimeError(
                f"{location}.source_case_id has boundary whitespace"
            )
        public_case = public_by_token.get(case_token)
        if public_case is None:
            raise RuntimeError("Private case is absent from public manifest")
        if private_case["cohort"] != public_case["reader_cohort"]:
            raise RuntimeError("Public/private case cohort mismatch")
        if (
            private_case["protocol_scope"] != reader["protocol_scope"]
            or private_case["arm_fields"] != arm_fields
        ):
            raise RuntimeError("Private per-case frozen protocol mismatch")
        expected_main_case_token = opaque_token(
            private_case["source_case_id"],
            secret_salt=private["blinding_secret"],
            seed=private["seed"],
            namespace="case",
        )
        if case_token != expected_main_case_token:
            raise RuntimeError("Private source case does not decode public token")
        reference_statuses = private_case["reference_finding_statuses"]
        if not isinstance(reference_statuses, dict) or set(
            reference_statuses
        ) != set(FINDINGS):
            raise RuntimeError("Private reference finding schema mismatch")
        if not all(
            isinstance(value, str) and value
            for value in reference_statuses.values()
        ):
            raise RuntimeError("Private reference finding status is invalid")

        _validate_report_mapping(
            private_case["report_mapping"],
            public_presentations=report_slots_by_case[case_token],
            expected_arms=arm_schema,
            expected_case_token=case_token,
            location=f"{location}.report_mapping",
        )
        pair_mapping = private_case["valid_history_pair_mapping"]
        pair_slots = pair_slots_by_case[case_token]
        if pair_mapping is None:
            if pair_slots:
                raise RuntimeError("Public valid-history pair lacks private key")
        else:
            if not pair_slots:
                raise RuntimeError("Private valid-history key lacks public pair")
            expected_pair_case_token = opaque_token(
                private_case["source_case_id"],
                secret_salt=private["blinding_secret"],
                seed=private["seed"] + 1,
                namespace="case",
            )
            _validate_report_mapping(
                pair_mapping,
                public_presentations=pair_slots,
                expected_arms=REQUIRED_TASK_B_ARMS,
                expected_case_token=expected_pair_case_token,
                location=f"{location}.valid_history_pair_mapping",
            )
        private_by_token[case_token] = private_case

    if public_order != private_order:
        raise RuntimeError(
            "Public/private frozen case order mismatch; refusing reordering"
        )

    root = asset_root if asset_root is not None else asset_inventory_path.parent
    asset_paths = _validate_asset_inventory(
        inventory,
        expected_manifest_hash=claimed_manifest_hash,
        referenced_tokens=seen_images,
        asset_root=root,
    )
    return ValidatedInputs(
        reader_manifest=reader,
        private_mapping=private,
        asset_inventory=inventory,
        public_by_token=public_by_token,
        private_by_token=private_by_token,
        asset_paths=asset_paths,
        input_hashes={
            "reader_manifest_file_sha256": sha256_file(reader_manifest_path),
            "private_mapping_file_sha256": sha256_file(private_mapping_path),
            "asset_inventory_file_sha256": sha256_file(asset_inventory_path),
            "public_manifest_semantic_sha256": claimed_manifest_hash,
        },
    )


def _keyed_hex(
    secret: str,
    *,
    namespace: str,
    seed: int,
    value: Any,
) -> str:
    payload = canonical_json_bytes(
        {
            "namespace": namespace,
            "seed": seed,
            "value": value,
        }
    )
    return hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()


def _reader_token(
    secret: str,
    *,
    namespace: str,
    seed: int,
    value: Any,
) -> str:
    return _keyed_hex(
        secret,
        namespace=namespace,
        seed=seed,
        value=value,
    )[:32]


def _report_for_arm(
    public_case: Mapping[str, Any],
    private_case: Mapping[str, Any],
    arm: str,
) -> Dict[str, str]:
    mapping = private_case["valid_history_pair_mapping"]
    slot = mapping["arm_to_slot"][arm]
    presentations = {
        presentation["slot"]: presentation
        for presentation in public_case["valid_history_pair"]
    }
    presentation = presentations[slot]
    return {
        "input_slot": slot,
        "input_report_token": presentation["report_token"],
        "report": presentation["report"],
        "report_hash": normalized_report_hash(presentation["report"]),
    }


def _reader_form_task_a() -> Dict[str, Any]:
    return {
        "finding_statuses": {finding: None for finding in FINDINGS},
        "other_important_findings": None,
    }


def _reader_form_task_b_history() -> Dict[str, Any]:
    return {
        "history_informativeness": None,
        "history_informativeness_reason": None,
    }


def _blank_report_assessment() -> Dict[str, Any]:
    return {
        "supported_findings_recovered": None,
        "supported_findings_lost": None,
        "unsupported_additions": None,
        "important_omissions": None,
        "temporal_correctness": None,
    }


def _reader_form_task_b_reports() -> Dict[str, Any]:
    return {
        "overall_preference": None,
        "preference_reason": None,
        "report_a_assessment": _blank_report_assessment(),
        "report_b_assessment": _blank_report_assessment(),
    }


def audit_reader_facing_payload(
    payload: Any,
    *,
    forbidden_original_tokens: Iterable[str],
    private_arm_names: Iterable[str],
) -> None:
    """Reject provenance keys and direct frozen-token leakage."""

    forbidden_tokens = set(forbidden_original_tokens)
    forbidden_values = set(private_arm_names)

    def walk(value: Any, path: str, *, report_text: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise RuntimeError("Reader-facing JSON keys must be strings")
                lowered = key.lower()
                key_words = set(lowered.split("_"))
                if (
                    key_words & FORBIDDEN_READER_KEY_WORDS
                    or lowered.startswith("input_")
                    or "answer_key" in lowered
                ):
                    raise RuntimeError(
                        f"Forbidden provenance key in reader payload at {path}"
                    )
                walk(
                    child,
                    f"{path}.{key}",
                    report_text=key in REPORT_TEXT_KEYS,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", report_text=report_text)
        elif isinstance(value, str) and not report_text:
            if value in forbidden_tokens or value in forbidden_values:
                raise RuntimeError(
                    f"Frozen identity leaked into reader payload at {path}"
                )
            if (
                WINDOWS_ABSOLUTE_RE.match(value)
                or value.startswith("\\\\")
                or value.startswith("/")
                or "://" in value
            ):
                raise RuntimeError(
                    f"Filesystem provenance leaked into reader payload at {path}"
                )

    walk(payload, "$")


def _safe_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise RuntimeError("Unsupported staged image suffix for human packet")
    return suffix


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_stage_assets(
    *,
    stage_root: Path,
    input_tokens: Sequence[str],
    human_asset_tokens: Mapping[str, str],
    asset_paths: Mapping[str, Path],
    asset_inventory: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    public_assets: Dict[str, str] = {}
    private_assets: Dict[str, Dict[str, Any]] = {}
    for input_token in sorted(set(input_tokens)):
        source = asset_paths[input_token]
        human_token = human_asset_tokens[input_token]
        suffix = _safe_suffix(source)
        relative = PurePosixPath(
            "images",
            human_token[:2],
            f"{human_token}{suffix}",
        )
        destination = stage_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError("Duplicate human image destination")
        shutil.copyfile(source, destination)
        expected = asset_inventory["assets"][input_token]
        if destination.stat().st_size != expected["bytes"]:
            raise RuntimeError("Copied human image byte count mismatch")
        observed_sha = sha256_file(destination)
        if observed_sha != expected["sha256"]:
            raise RuntimeError("Copied human image SHA-256 mismatch")
        public_assets[human_token] = relative.as_posix()
        private_assets[human_token] = {
            "input_image_token": input_token,
            "input_relative_path": expected["relative_path"],
            "bytes": expected["bytes"],
            "sha256": observed_sha,
        }
    return public_assets, private_assets


def _tree_sha256(root: Path) -> str:
    files: Dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    return sha256_canonical_json(files)


def _selected_cases(
    inputs: ValidatedInputs,
    *,
    include_cohorts: Sequence[str],
    packet_secret: str,
    randomization_seed: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not include_cohorts:
        raise RuntimeError("At least one reader cohort must be selected")
    if len(include_cohorts) != len(set(include_cohorts)):
        raise RuntimeError("Duplicate --include-cohort values")
    observed = {
        case["reader_cohort"] for case in inputs.reader_manifest["cases"]
    }
    missing = set(include_cohorts) - observed
    if missing:
        raise RuntimeError("A requested reader cohort is absent from the freeze")
    selected = [
        (case, inputs.private_by_token[case["case_token"]])
        for case in inputs.reader_manifest["cases"]
        if case["reader_cohort"] in set(include_cohorts)
    ]
    token_to_order_key: Dict[str, str] = {}
    seen_order_keys = set()
    for public_case, _private_case in selected:
        token = public_case["case_token"]
        key = _keyed_hex(
            packet_secret,
            namespace="human-case-order",
            seed=randomization_seed,
            value=token,
        )
        if key in seen_order_keys:
            raise RuntimeError("Human case randomization-key collision")
        seen_order_keys.add(key)
        token_to_order_key[token] = key
    return sorted(
        selected,
        key=lambda pair: token_to_order_key[pair[0]["case_token"]],
    )


def _task_b_eligibility(
    public_case: Mapping[str, Any],
    private_case: Mapping[str, Any],
) -> Tuple[bool, str]:
    if not public_case["prior_image_tokens"]:
        return False, "no_true_prior_images"
    if not public_case["valid_history_pair"]:
        return False, "no_valid_history_report_pair"
    if private_case["valid_history_pair_mapping"] is None:
        return False, "no_private_valid_history_key"
    if private_case["protocol_scope"] != "controller":
        return False, "not_controller_scope"
    if private_case["evaluation_regime"] != "valid_history":
        return False, "history_provenance_not_verified"
    mapping = private_case["valid_history_pair_mapping"]
    if set(mapping["arm_to_slot"]) != set(REQUIRED_TASK_B_ARMS):
        return False, "wrong_report_pair_contract"
    return True, ""


def prepare_packet(
    *,
    reader_manifest_path: Path,
    private_mapping_path: Path,
    asset_inventory_path: Path,
    output_dir: Path,
    private_output_dir: Path,
    packet_secret: str,
    randomization_seed: int,
    include_cohorts: Sequence[str],
    asset_root: Optional[Path] = None,
    task_a_only: bool = False,
) -> Dict[str, Any]:
    """Create an immutable local packet and its separate private answer key.

    Full A+B is the default and requires at least one valid-history Task B
    case.  ``task_a_only=True`` is an explicit narrower study scope that emits
    only current-image Task A artifacts and cannot authorize Task B.
    """

    output_dir = output_dir.resolve()
    private_output_dir = private_output_dir.resolve()
    if output_dir.exists() or private_output_dir.exists():
        raise RuntimeError(
            "Public or private output directory already exists; refusing overwrite"
        )
    repository_root = Path(__file__).resolve().parents[1]
    for label, candidate in (
        ("public", output_dir),
        ("private", private_output_dir),
    ):
        try:
            candidate.relative_to(repository_root)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"{label.capitalize()} packet root must be outside the repository"
            )
    for left, right in (
        (output_dir, private_output_dir),
        (private_output_dir, output_dir),
    ):
        try:
            right.relative_to(left)
        except ValueError:
            continue
        raise RuntimeError(
            "Public and private packet roots must be physically disjoint"
        )
    if not isinstance(task_a_only, bool):
        raise RuntimeError("task_a_only must be boolean")
    if not isinstance(packet_secret, str) or len(
        packet_secret.encode("utf-8")
    ) < 32:
        raise RuntimeError("Packet secret must contain at least 32 UTF-8 bytes")
    if isinstance(randomization_seed, bool) or not isinstance(
        randomization_seed, int
    ):
        raise RuntimeError("randomization_seed must be an integer")

    inputs = validate_inputs(
        reader_manifest_path=reader_manifest_path,
        private_mapping_path=private_mapping_path,
        asset_inventory_path=asset_inventory_path,
        asset_root=asset_root,
    )
    if packet_secret == inputs.private_mapping["blinding_secret"]:
        raise RuntimeError(
            "Human packet secret must differ from frozen model-reader secret"
        )
    selected = _selected_cases(
        inputs,
        include_cohorts=include_cohorts,
        packet_secret=packet_secret,
        randomization_seed=randomization_seed,
    )
    eligible: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    task_b_exclusions: Dict[str, str] = {}
    if not task_a_only:
        for public_case, private_case in selected:
            is_eligible, reason = _task_b_eligibility(
                public_case,
                private_case,
            )
            if is_eligible:
                eligible.append((public_case, private_case))
            else:
                task_b_exclusions[public_case["case_token"]] = reason
        if not eligible:
            raise RuntimeError(
                "No Task B cases have current images, true-prior images, a "
                "frozen current_only/SCFR pair, and valid-history provenance"
            )

    study_scope = (
        "task_a_only_current_image_support"
        if task_a_only
        else "full_task_a_and_task_b_valid_history"
    )
    task_b_authorized = not task_a_only

    packet_token = _reader_token(
        packet_secret,
        namespace="human-packet",
        seed=randomization_seed,
        value={
            "manifest": inputs.input_hashes["public_manifest_semantic_sha256"],
            "cohorts": sorted(include_cohorts),
            "study_scope": study_scope,
        },
    )
    human_case_tokens = {
        public_case["case_token"]: _reader_token(
            packet_secret,
            namespace="human-case",
            seed=randomization_seed,
            value={
                "packet": packet_token,
                "case": public_case["case_token"],
            },
        )
        for public_case, _private_case in selected
    }
    if len(set(human_case_tokens.values())) != len(human_case_tokens):
        raise RuntimeError("Duplicate human case tokens")

    selected_input_image_tokens = set()
    for public_case, _private_case in selected:
        selected_input_image_tokens.update(public_case["current_image_tokens"])
        if not task_a_only:
            selected_input_image_tokens.update(
                public_case["prior_image_tokens"]
            )
    human_asset_tokens = {
        input_token: _reader_token(
            packet_secret,
            namespace="human-image",
            seed=randomization_seed,
            value={"packet": packet_token, "image": input_token},
        )
        for input_token in selected_input_image_tokens
    }
    if len(set(human_asset_tokens.values())) != len(human_asset_tokens):
        raise RuntimeError("Duplicate human image tokens")

    task_a_cases: List[Dict[str, Any]] = []
    task_b_history_cases: List[Dict[str, Any]] = []
    task_b_report_cases: List[Dict[str, Any]] = []
    private_cases: List[Dict[str, Any]] = []
    stage_a_input_assets: List[str] = []
    stage_b_input_assets: List[str] = []

    orientation_start: Optional[int] = None
    if not task_a_only:
        orientation_start = int(
            _keyed_hex(
                packet_secret,
                namespace="human-ab-orientation-start",
                seed=randomization_seed,
                value=packet_token,
            )[0],
            16,
        ) % 2
    eligible_index = {
        public_case["case_token"]: index
        for index, (public_case, _private_case) in enumerate(eligible)
    }

    for reader_index, (public_case, private_case) in enumerate(selected):
        input_case_token = public_case["case_token"]
        human_case_token = human_case_tokens[input_case_token]
        current_tokens = [
            human_asset_tokens[token]
            for token in public_case["current_image_tokens"]
        ]
        stage_a_input_assets.extend(public_case["current_image_tokens"])
        task_a_cases.append(
            {
                "case_token": human_case_token,
                "current_image_tokens": current_tokens,
                "response": _reader_form_task_a(),
            }
        )
        private_case_out: Dict[str, Any] = {
            "case_token": human_case_token,
            "reader_order_index": reader_index,
            "input_case_token": input_case_token,
            "input_case_order_index": inputs.reader_manifest["cases"].index(
                public_case
            ),
            "source_case_id": private_case["source_case_id"],
            "subject_id": private_case["subject_id"],
            "panel": private_case["panel"],
            "cohort": private_case["cohort"],
            "sampling_stratum": private_case["sampling_stratum"],
            "inclusion_probability": private_case["inclusion_probability"],
            "reference_finding_statuses": private_case[
                "reference_finding_statuses"
            ],
        }

        if not task_a_only:
            if input_case_token in eligible_index:
                task_b_index = eligible_index[input_case_token]
                prior_tokens = [
                    human_asset_tokens[token]
                    for token in public_case["prior_image_tokens"]
                ]
                stage_b_input_assets.extend(
                    public_case["current_image_tokens"]
                )
                stage_b_input_assets.extend(public_case["prior_image_tokens"])
                task_b_history_cases.append(
                    {
                        "case_token": human_case_token,
                        "current_image_tokens": current_tokens,
                        "prior_image_tokens": prior_tokens,
                        "response": _reader_form_task_b_history(),
                    }
                )
                current_report = _report_for_arm(
                    public_case,
                    private_case,
                    "current_only",
                )
                scfr_report = _report_for_arm(
                    public_case,
                    private_case,
                    "scfr",
                )
                if (task_b_index + int(orientation_start)) % 2 == 0:
                    arm_a, report_a = "current_only", current_report
                    arm_b, report_b = "scfr", scfr_report
                else:
                    arm_a, report_a = "scfr", scfr_report
                    arm_b, report_b = "current_only", current_report
                task_b_report_cases.append(
                    {
                        "case_token": human_case_token,
                        "report_a": report_a["report"],
                        "report_b": report_b["report"],
                        "response": _reader_form_task_b_reports(),
                    }
                )
                private_case_out["task_b"] = {
                    "task_b_order_index": task_b_index,
                    "history_provenance": "valid_history",
                    "prior_context_sha256": hashlib.sha256(
                        public_case["true_prior_context"].encode("utf-8")
                    ).hexdigest(),
                    "report_a": {
                        "arm": arm_a,
                        "input_slot": report_a["input_slot"],
                        "input_report_token": report_a["input_report_token"],
                        "report_sha256": report_a["report_hash"],
                    },
                    "report_b": {
                        "arm": arm_b,
                        "input_slot": report_b["input_slot"],
                        "input_report_token": report_b["input_report_token"],
                        "report_sha256": report_b["report_hash"],
                    },
                    "pair_is_identical": public_case["pair_is_identical"],
                }
            else:
                private_case_out["task_b"] = {
                    "excluded": True,
                    "reason": task_b_exclusions[input_case_token],
                }
        private_cases.append(private_case_out)

    task_a_order = [case["case_token"] for case in task_a_cases]
    task_b_history_order = [
        case["case_token"] for case in task_b_history_cases
    ]
    task_b_report_order = [case["case_token"] for case in task_b_report_cases]
    if not task_a_only:
        if task_b_history_order != task_b_report_order:
            raise RuntimeError("Task B history/report case order mismatch")
        if task_b_history_order != [
            token
            for token in task_a_order
            if token in set(task_b_history_order)
        ]:
            raise RuntimeError("Task B order is not a stable Task A subsequence")

    task_a_payload = {
        "protocol_version": f"{PROTOCOL_VERSION}-task-a",
        "release_status": RELEASE_STATUS,
        "packet_token": packet_token,
        "finding_vocabulary": list(FINDINGS),
        "finding_status_values": list(TASK_A_STATUSES),
        "instructions": [
            "Review the current chest images only.",
            "Record one status for every finding before continuing.",
            "Record any other clinically important finding in free text.",
        ],
        "case_order_commitment": sha256_canonical_json(task_a_order),
        "cases": task_a_cases,
    }
    task_b_history_payload: Optional[Dict[str, Any]] = None
    task_b_reports_payload: Optional[Dict[str, Any]] = None
    if not task_a_only:
        task_b_history_payload = {
            "protocol_version": f"{PROTOCOL_VERSION}-task-b-history",
            "release_status": RELEASE_STATUS,
            "packet_token": packet_token,
            "history_informativeness_values": list(
                HISTORY_INFORMATIVENESS_VALUES
            ),
            "instructions": [
                "Review the current and prior images.",
                "Record and lock history informativeness before reports are "
                "shown.",
            ],
            "case_order_commitment": sha256_canonical_json(
                task_b_history_order
            ),
            "cases": task_b_history_cases,
        }
        task_b_reports_payload = {
            "protocol_version": f"{PROTOCOL_VERSION}-task-b-reports",
            "release_status": RELEASE_STATUS,
            "packet_token": packet_token,
            "preference_values": list(PREFERENCE_VALUES),
            "temporal_correctness_values": list(TEMPORAL_CORRECTNESS_VALUES),
            "instructions": [
                "Judge anonymous reports A and B using the reviewed images.",
                "Choose A, B, tie, or unassessable for overall preference.",
                "List supported findings recovered or lost, unsupported "
                "additions, important omissions, and temporal errors for each "
                "report.",
            ],
            "case_order_commitment": sha256_canonical_json(
                task_b_report_order
            ),
            "cases": task_b_report_cases,
        }

    original_tokens = set(inputs.public_by_token)
    original_tokens.update(inputs.asset_paths)
    for private_case in inputs.private_mapping["cases"]:
        original_tokens.add(private_case["source_case_id"])
        original_tokens.add(private_case["subject_id"])
    for public_case in inputs.reader_manifest["cases"]:
        for field in ("report_presentations", "valid_history_pair"):
            original_tokens.update(
                item["report_token"] for item in public_case[field]
            )
    reader_payloads = [task_a_payload]
    if task_b_history_payload is not None:
        reader_payloads.append(task_b_history_payload)
    if task_b_reports_payload is not None:
        reader_payloads.append(task_b_reports_payload)
    for payload in reader_payloads:
        audit_reader_facing_payload(
            payload,
            forbidden_original_tokens=original_tokens,
            private_arm_names=inputs.private_mapping["arm_schema"],
        )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    private_parent = private_output_dir.parent
    private_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=str(output_parent),
        )
    )
    temporary_private_root = Path(
        tempfile.mkdtemp(
            prefix=f".{private_output_dir.name}.building-",
            dir=str(private_parent),
        )
    )
    private_committed = False
    try:
        stage_a_root = temporary_root / "reader_stage_01_task_a"
        stage_a_root.mkdir(parents=True)
        stage_b_history_root: Optional[Path] = None
        stage_b_reports_root: Optional[Path] = None
        if not task_a_only:
            stage_b_history_root = (
                temporary_root / "reader_stage_02_task_b_history"
            )
            stage_b_reports_root = (
                temporary_root / "reader_stage_03_task_b_reports"
            )
            stage_b_history_root.mkdir(parents=True)
            stage_b_reports_root.mkdir(parents=True)

        stage_a_assets, stage_a_private_assets = _copy_stage_assets(
            stage_root=stage_a_root,
            input_tokens=stage_a_input_assets,
            human_asset_tokens=human_asset_tokens,
            asset_paths=inputs.asset_paths,
            asset_inventory=inputs.asset_inventory,
        )
        stage_b_assets: Dict[str, str] = {}
        stage_b_private_assets: Dict[str, Dict[str, Any]] = {}
        if stage_b_history_root is not None:
            stage_b_assets, stage_b_private_assets = _copy_stage_assets(
                stage_root=stage_b_history_root,
                input_tokens=stage_b_input_assets,
                human_asset_tokens=human_asset_tokens,
                asset_paths=inputs.asset_paths,
                asset_inventory=inputs.asset_inventory,
            )
        task_a_asset_payload = {
            "protocol_version": f"{PROTOCOL_VERSION}-assets-task-a",
            "release_status": RELEASE_STATUS,
            "packet_token": packet_token,
            "assets": stage_a_assets,
        }
        task_b_asset_payload: Optional[Dict[str, Any]] = None
        if not task_a_only:
            task_b_asset_payload = {
                "protocol_version": f"{PROTOCOL_VERSION}-assets-task-b",
                "release_status": RELEASE_STATUS,
                "packet_token": packet_token,
                "assets": stage_b_assets,
            }
        asset_payloads = [task_a_asset_payload]
        if task_b_asset_payload is not None:
            asset_payloads.append(task_b_asset_payload)
        for payload in asset_payloads:
            audit_reader_facing_payload(
                payload,
                forbidden_original_tokens=original_tokens,
                private_arm_names=inputs.private_mapping["arm_schema"],
            )

        _write_json(stage_a_root / "task_a.json", task_a_payload)
        _write_json(
            stage_a_root / "asset_manifest.json",
            task_a_asset_payload,
        )
        if (
            stage_b_history_root is not None
            and stage_b_reports_root is not None
            and task_b_history_payload is not None
            and task_b_reports_payload is not None
            and task_b_asset_payload is not None
        ):
            _write_json(
                stage_b_history_root / "task_b_history.json",
                task_b_history_payload,
            )
            _write_json(
                stage_b_history_root / "asset_manifest.json",
                task_b_asset_payload,
            )
            _write_json(
                stage_b_reports_root / "task_b_reports.json",
                task_b_reports_payload,
            )

        private_answer_key = {
            "protocol_version": f"{PROTOCOL_VERSION}-private-key",
            "release_status": RELEASE_STATUS,
            "study_scope": study_scope,
            "task_b_authorized": task_b_authorized,
            "packet_token": packet_token,
            "randomization_seed": randomization_seed,
            "included_cohorts": list(include_cohorts),
            "input_hashes": inputs.input_hashes,
            "task_a_case_order_sha256": sha256_canonical_json(task_a_order),
            "stage_01_assets": stage_a_private_assets,
            "cases": private_cases,
        }
        if not task_a_only:
            private_answer_key.update(
                {
                    "task_b_case_order_sha256": sha256_canonical_json(
                        task_b_history_order
                    ),
                    "task_b_orientation_start": orientation_start,
                    "task_b_exclusion_counts": {
                        reason: list(task_b_exclusions.values()).count(reason)
                        for reason in sorted(set(task_b_exclusions.values()))
                    },
                    "stage_02_assets": stage_b_private_assets,
                }
            )
        private_root = temporary_private_root
        _write_json(
            private_root / "private_answer_key.json",
            private_answer_key,
        )

        reader_stage_hashes = {
            "reader_stage_01_task_a": _tree_sha256(stage_a_root),
        }
        if (
            stage_b_history_root is not None
            and stage_b_reports_root is not None
        ):
            reader_stage_hashes.update(
                {
                    "reader_stage_02_task_b_history": _tree_sha256(
                        stage_b_history_root
                    ),
                    "reader_stage_03_task_b_reports": _tree_sha256(
                        stage_b_reports_root
                    ),
                }
            )
        receipt = {
            "protocol_version": f"{PROTOCOL_VERSION}-receipt",
            "release_status": RELEASE_STATUS,
            "authorized_for_release": False,
            "study_scope": study_scope,
            "task_b_authorized": task_b_authorized,
            "release_gate": (
                "Release only to qualified readers after the computational "
                "audit is complete and frozen and all institutional access, "
                "privacy, and ethics gates are documented."
            ),
            "validation_claim": (
                "Packet preparation is not human validation and does not "
                "establish any study result."
            ),
            "packet_token": packet_token,
            "input_hashes": inputs.input_hashes,
            "reader_stage_tree_sha256": reader_stage_hashes,
            "private_answer_key_file_sha256": sha256_file(
                private_root / "private_answer_key.json"
            ),
            "counts": {
                "task_a_cases": len(task_a_cases),
                "task_a_images": len(stage_a_assets),
            },
        }
        if not task_a_only:
            receipt["counts"].update(
                {
                    "task_b_cases": len(task_b_history_cases),
                    "task_b_excluded_cases": len(task_b_exclusions),
                    "task_b_images": len(stage_b_assets),
                    "task_b_identical_report_pairs": sum(
                        bool(
                            case["task_b"]
                            and case["task_b"].get("pair_is_identical")
                        )
                        for case in private_cases
                    ),
                }
            )
        _write_json(temporary_root / "packet_receipt.json", receipt)
        if output_dir.exists():
            raise RuntimeError(
                "Output directory appeared during build; refusing overwrite"
            )
        if private_output_dir.exists():
            raise RuntimeError(
                "Private output directory appeared during build; refusing overwrite"
            )
        os.replace(str(temporary_private_root), str(private_output_dir))
        private_committed = True
        os.replace(str(temporary_root), str(output_dir))
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if temporary_private_root.exists():
            shutil.rmtree(temporary_private_root)
        if private_committed and private_output_dir.exists():
            shutil.rmtree(private_output_dir)
        raise

    summary = {
        "release_status": RELEASE_STATUS,
        "authorized_for_release": False,
        "study_scope": study_scope,
        "task_b_authorized": task_b_authorized,
        "packet_token": packet_token,
        "task_a_cases": len(task_a_cases),
        "output_dir": str(output_dir),
        "private_output_dir": str(private_output_dir),
    }
    if not task_a_only:
        summary.update(
            {
                "task_b_cases": len(task_b_history_cases),
                "task_b_excluded_cases": len(task_b_exclusions),
            }
        )
    return summary


def resolve_packet_secret(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError("Packet secret file does not exist")
    secret = path.read_text(encoding="utf-8").strip()
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("Packet secret must contain at least 32 UTF-8 bytes")
    return secret


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-manifest", type=Path, required=True)
    parser.add_argument("--private-mapping", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Dataset root for inventory relative paths (default: inventory parent).",
    )
    parser.add_argument("--packet-secret-file", type=Path, required=True)
    parser.add_argument("--randomization-seed", type=int, required=True)
    parser.add_argument(
        "--include-cohort",
        action="append",
        required=True,
        help="Frozen reader cohort to include; repeat for multiple cohorts.",
    )
    parser.add_argument(
        "--task-a-only",
        action="store_true",
        help=(
            "Prepare current-image Task A only. No Task B directory, form, "
            "report pair, or authorization is granted. Full A+B remains the "
            "default."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        required=True,
        help=(
            "Separate non-repository root for the private answer key; it must "
            "not be nested with --output-dir."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_packet(
            reader_manifest_path=args.reader_manifest,
            private_mapping_path=args.private_mapping,
            asset_inventory_path=args.asset_inventory,
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            private_output_dir=args.private_output_dir,
            packet_secret=resolve_packet_secret(args.packet_secret_file),
            randomization_seed=args.randomization_seed,
            include_cohorts=args.include_cohort,
            task_a_only=args.task_a_only,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

