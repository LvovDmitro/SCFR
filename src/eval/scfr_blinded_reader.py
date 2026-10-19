"""Contracts and blinding utilities for an SCFR model-reader pilot.

This module deliberately contains no model-loading or inference code.  It
provides the small, auditable core shared by launchers and downstream
analyses:

* strict response contracts for image-only, image-grounded report audit, and
  pairwise tasks;
* recovery of the first complete JSON object from otherwise noisy model text;
* cross-platform canonical hashes for JSON payloads and prompts;
* keyed, deterministic blinding of case and report identities; and
* exact-report deduplication with a private mapping back to source arms.

Arm names and source case identifiers belong only in the private assignment
artifact.  ``BlindedCase.to_public_dict()`` is the intended serialization for
reader-facing inputs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union


# Keep this tuple aligned with ``scripts/analyze_source_causal_arms.py``.
FINDINGS: Tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "fracture",
    "mass_or_nodule",
    "opacity",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
    "support_device",
)

IMAGE_STATUSES: Tuple[str, ...] = (
    "present",
    "absent",
    "uncertain",
    "unassessable",
)
IMAGE_QUALITY_VALUES: Tuple[str, ...] = ("adequate", "limited", "unassessable")
REPORT_STATUSES: Tuple[str, ...] = (
    "present",
    "absent",
    "uncertain",
    "not_mentioned",
)
REPORT_AUDIT_JUDGMENTS: Tuple[str, ...] = (
    "supported",
    "unsupported_positive",
    "omitted_positive",
    "contradiction",
    "not_applicable",
    "uncertain",
    "unassessable",
)
REPORT_VERDICTS: Tuple[str, ...] = (
    "acceptable",
    "minor_error",
    "major_error",
    "uncertain",
    "unassessable",
)
PAIRWISE_CHOICES: Tuple[str, ...] = ("A", "B", "tie", "abstain")
PAIRWISE_CRITERIA: Tuple[str, ...] = (
    "image_faithfulness",
    "clinically_important_omissions",
    "unsupported_content",
    "temporal_correctness",
    "history_usefulness",
    "overall_clinical_utility",
)


class SchemaValidationError(ValueError):
    """Raised when a reader response violates its fixed JSON contract."""


def _enum_schema(values: Sequence[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _nonempty_string_schema() -> Dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _confidence_schema() -> Dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def _strict_object_schema(
    properties: Mapping[str, Any],
    required: Sequence[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_IMAGE_FINDING_SCHEMA = _strict_object_schema(
    {
        "status": _enum_schema(IMAGE_STATUSES),
        "confidence": _confidence_schema(),
        "evidence": _nonempty_string_schema(),
    },
    ("status", "confidence", "evidence"),
)

IMAGE_ONLY_RESPONSE_SCHEMA: Dict[str, Any] = _strict_object_schema(
    {
        "findings": _strict_object_schema(
            {finding: _IMAGE_FINDING_SCHEMA for finding in FINDINGS},
            FINDINGS,
        ),
        "image_quality": _enum_schema(IMAGE_QUALITY_VALUES),
        "should_abstain": {"type": "boolean"},
        "limitations": {
            "type": "array",
            "items": _nonempty_string_schema(),
        },
    },
    ("findings", "image_quality", "should_abstain", "limitations"),
)

_REPORT_AUDIT_FINDING_SCHEMA = _strict_object_schema(
    {
        "image_status": _enum_schema(IMAGE_STATUSES),
        "report_status": _enum_schema(REPORT_STATUSES),
        "judgment": _enum_schema(REPORT_AUDIT_JUDGMENTS),
        "confidence": _confidence_schema(),
        "rationale": _nonempty_string_schema(),
    },
    (
        "image_status",
        "report_status",
        "judgment",
        "confidence",
        "rationale",
    ),
)

REPORT_AUDIT_RESPONSE_SCHEMA: Dict[str, Any] = _strict_object_schema(
    {
        "findings": _strict_object_schema(
            {finding: _REPORT_AUDIT_FINDING_SCHEMA for finding in FINDINGS},
            FINDINGS,
        ),
        "overall": _strict_object_schema(
            {
                "verdict": _enum_schema(REPORT_VERDICTS),
                "confidence": _confidence_schema(),
                "rationale": _nonempty_string_schema(),
            },
            ("verdict", "confidence", "rationale"),
        ),
        "should_abstain": {"type": "boolean"},
        "limitations": {
            "type": "array",
            "items": _nonempty_string_schema(),
        },
    },
    ("findings", "overall", "should_abstain", "limitations"),
)

PAIRWISE_RESPONSE_SCHEMA: Dict[str, Any] = _strict_object_schema(
    {
        "preference": _enum_schema(PAIRWISE_CHOICES),
        "confidence": _confidence_schema(),
        "criteria": _strict_object_schema(
            {criterion: _enum_schema(PAIRWISE_CHOICES) for criterion in PAIRWISE_CRITERIA},
            PAIRWISE_CRITERIA,
        ),
        "rationale": _nonempty_string_schema(),
    },
    ("preference", "confidence", "criteria", "rationale"),
)


def _path(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}.{child}"


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{path or 'response'} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise SchemaValidationError(f"{path or 'response'} has a non-string key")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    keys: Sequence[str],
    path: str,
) -> None:
    expected = set(keys)
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise SchemaValidationError(
            f"{path or 'response'} has invalid keys ({', '.join(details)})"
        )


def _require_choice(value: Any, choices: Sequence[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise SchemaValidationError(f"{path} must be one of {list(choices)}")
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _require_confidence(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{path} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SchemaValidationError(f"{path} must be a finite number in [0, 1]")
    return result


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{path} must be a JSON boolean")
    return value


def _require_string_list(value: Any, path: str) -> List[str]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{path} must be a JSON array")
    return [_require_text(item, f"{path}[{index}]") for index, item in enumerate(value)]


def validate_image_only_response(value: Any) -> Dict[str, Any]:
    """Validate and return the canonical image-only response object.

    The output contains exactly the fixed 11 findings, in ``FINDINGS`` order,
    and strips only leading/trailing whitespace from free-text fields.
    """

    response = _require_mapping(value, "")
    top_keys = ("findings", "image_quality", "should_abstain", "limitations")
    _require_exact_keys(response, top_keys, "")
    findings = _require_mapping(response["findings"], "findings")
    _require_exact_keys(findings, FINDINGS, "findings")

    canonical_findings: Dict[str, Any] = {}
    finding_keys = ("status", "confidence", "evidence")
    for finding in FINDINGS:
        item_path = f"findings.{finding}"
        item = _require_mapping(findings[finding], item_path)
        _require_exact_keys(item, finding_keys, item_path)
        canonical_findings[finding] = {
            "status": _require_choice(
                item["status"], IMAGE_STATUSES, _path(item_path, "status")
            ),
            "confidence": _require_confidence(
                item["confidence"], _path(item_path, "confidence")
            ),
            "evidence": _require_text(
                item["evidence"], _path(item_path, "evidence")
            ),
        }

    return {
        "findings": canonical_findings,
        "image_quality": _require_choice(
            response["image_quality"], IMAGE_QUALITY_VALUES, "image_quality"
        ),
        "should_abstain": _require_bool(
            response["should_abstain"], "should_abstain"
        ),
        "limitations": _require_string_list(response["limitations"], "limitations"),
    }


def validate_report_audit_response(value: Any) -> Dict[str, Any]:
    """Validate and return an image-grounded report-audit response."""

    response = _require_mapping(value, "")
    top_keys = ("findings", "overall", "should_abstain", "limitations")
    _require_exact_keys(response, top_keys, "")
    findings = _require_mapping(response["findings"], "findings")
    _require_exact_keys(findings, FINDINGS, "findings")

    canonical_findings: Dict[str, Any] = {}
    finding_keys = (
        "image_status",
        "report_status",
        "judgment",
        "confidence",
        "rationale",
    )
    for finding in FINDINGS:
        item_path = f"findings.{finding}"
        item = _require_mapping(findings[finding], item_path)
        _require_exact_keys(item, finding_keys, item_path)
        canonical_findings[finding] = {
            "image_status": _require_choice(
                item["image_status"],
                IMAGE_STATUSES,
                _path(item_path, "image_status"),
            ),
            "report_status": _require_choice(
                item["report_status"],
                REPORT_STATUSES,
                _path(item_path, "report_status"),
            ),
            "judgment": _require_choice(
                item["judgment"],
                REPORT_AUDIT_JUDGMENTS,
                _path(item_path, "judgment"),
            ),
            "confidence": _require_confidence(
                item["confidence"], _path(item_path, "confidence")
            ),
            "rationale": _require_text(
                item["rationale"], _path(item_path, "rationale")
            ),
        }

    overall = _require_mapping(response["overall"], "overall")
    overall_keys = ("verdict", "confidence", "rationale")
    _require_exact_keys(overall, overall_keys, "overall")
    canonical_overall = {
        "verdict": _require_choice(
            overall["verdict"], REPORT_VERDICTS, "overall.verdict"
        ),
        "confidence": _require_confidence(
            overall["confidence"], "overall.confidence"
        ),
        "rationale": _require_text(overall["rationale"], "overall.rationale"),
    }

    return {
        "findings": canonical_findings,
        "overall": canonical_overall,
        "should_abstain": _require_bool(
            response["should_abstain"], "should_abstain"
        ),
        "limitations": _require_string_list(response["limitations"], "limitations"),
    }


def validate_pairwise_response(value: Any) -> Dict[str, Any]:
    """Validate and return a canonical A/B pairwise reader response."""

    response = _require_mapping(value, "")
    top_keys = ("preference", "confidence", "criteria", "rationale")
    _require_exact_keys(response, top_keys, "")
    criteria = _require_mapping(response["criteria"], "criteria")
    _require_exact_keys(criteria, PAIRWISE_CRITERIA, "criteria")
    return {
        "preference": _require_choice(
            response["preference"], PAIRWISE_CHOICES, "preference"
        ),
        "confidence": _require_confidence(response["confidence"], "confidence"),
        "criteria": {
            criterion: _require_choice(
                criteria[criterion],
                PAIRWISE_CHOICES,
                f"criteria.{criterion}",
            )
            for criterion in PAIRWISE_CRITERIA
        },
        "rationale": _require_text(response["rationale"], "rationale"),
    }


def extract_first_json_object(raw_text: str) -> Dict[str, Any]:
    """Return the first complete JSON object embedded in model output.

    ``json.JSONDecoder.raw_decode`` handles braces and escaped quotes inside
    strings, unlike regular-expression extraction.  We try every object
    opener in source order, so prose and Markdown code fences are harmless.
    Arrays and scalar JSON values are intentionally ignored.
    """

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        try:
            value, _ = decoder.raw_decode(raw_text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No complete JSON object found in model response")


def _object_without_duplicate_keys(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _require_finite_json_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite_json_numbers(item)


def _decode_exact_json_object(raw_text: str) -> Dict[str, Any]:
    """Decode exactly one JSON object, allowing surrounding whitespace only."""

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("Model response is blank")
    decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )
    try:
        value, end = decoder.raw_decode(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Model response is not exact valid JSON: {exc}") from exc
    if end != len(stripped):
        raise ValueError("Model response contains text or another value after JSON")
    if not isinstance(value, dict):
        raise ValueError("Model response must be one JSON object")
    _require_finite_json_numbers(value)
    return value


def extract_exact_model_response(raw_text: str, backend: str) -> Dict[str, Any]:
    """Return the unambiguous final JSON object for one frozen reader backend.

    MedGemma and CheXOne must return a bare JSON object with no Markdown or
    surrounding prose.  NV-Reason-CXR has a documented native reasoning
    envelope, so only the body of exactly one final ``<answer>`` segment is
    decoded.  Any JSON-like text in its reasoning prefix is deliberately
    ignored and can never become the scored response.
    """

    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string")
    canonical = backend.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "chex_one": "chexone",
        "med_gemma": "medgemma",
        "nvreasoncxr": "nv_reason_cxr",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical in {"medgemma", "chexone"}:
        return _decode_exact_json_object(raw_text)
    if canonical != "nv_reason_cxr":
        raise ValueError(f"Unsupported model-reader backend: {backend!r}")
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    stripped = raw_text.strip()
    openings = list(re.finditer(r"<answer>", stripped))
    closings = list(re.finditer(r"</answer>", stripped))
    if len(openings) != 1 or len(closings) != 1:
        raise ValueError(
            "NV-Reason-CXR response must contain exactly one complete "
            "<answer> segment"
        )
    thinking_openings = list(re.finditer(r"<thinking>", stripped))
    thinking_closings = list(re.finditer(r"</thinking>", stripped))
    if len(thinking_openings) != len(thinking_closings) or len(
        thinking_openings
    ) > 1:
        raise ValueError(
            "NV-Reason-CXR response has an ambiguous <thinking> segment"
        )
    envelope = re.fullmatch(
        r"(?:<thinking>.*?</thinking>\s*)?"
        r"<answer>(?P<answer>.*?)</answer>",
        stripped,
        flags=re.DOTALL,
    )
    if envelope is None:
        raise ValueError(
            "NV-Reason-CXR response must be an optional exact <thinking> "
            "segment followed by one final <answer> segment"
        )
    return _decode_exact_json_object(envelope.group("answer"))


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashing and manifests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value is not canonical JSON data: {exc}") from exc


def sha256_canonical_json(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of canonical JSON."""

    payload = canonical_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonicalize_prompt(prompt: str) -> str:
    """Normalize prompt Unicode and line endings without changing spacing."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = unicodedata.normalize("NFC", prompt)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def sha256_prompt(prompt: str) -> str:
    """Return a cross-platform SHA-256 hash of the exact logical prompt."""

    return hashlib.sha256(canonicalize_prompt(prompt).encode("utf-8")).hexdigest()


def normalize_report_text(report: str) -> str:
    """Normalize report text for *exact-content* duplicate detection.

    Unicode is normalized to NFC and every run of whitespace is represented by
    one ASCII space.  Case and punctuation are deliberately preserved: this is
    deduplication, not semantic equivalence.
    """

    if not isinstance(report, str):
        raise TypeError("report must be a string")
    return " ".join(unicodedata.normalize("NFC", report).split())


def normalized_report_hash(report: str) -> str:
    """Hash a normalized report using SHA-256."""

    return hashlib.sha256(normalize_report_text(report).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DeduplicatedReport:
    """One normalized report and all private source arms that produced it."""

    report_hash: str
    report: str
    arms: Tuple[str, ...] = field(repr=False)


def deduplicate_arm_reports(
    arm_reports: Mapping[str, str],
) -> Tuple[DeduplicatedReport, ...]:
    """Collapse normalized-identical reports and retain private arm provenance.

    The returned records and each record's arms are lexically sorted, making
    this step independent of input-dictionary insertion order.
    """

    if not isinstance(arm_reports, Mapping) or not arm_reports:
        raise ValueError("arm_reports must be a non-empty mapping")

    grouped: Dict[str, Dict[str, Any]] = {}
    for arm, raw_report in arm_reports.items():
        if not isinstance(arm, str) or not arm:
            raise ValueError("Each arm name must be a non-empty string")
        report = normalize_report_text(raw_report)
        report_hash = hashlib.sha256(report.encode("utf-8")).hexdigest()
        group = grouped.setdefault(report_hash, {"report": report, "arms": []})
        # A SHA-256 collision should never silently merge distinct normalized
        # reports, even though it is only a theoretical concern here.
        if group["report"] != report:
            raise RuntimeError("SHA-256 collision between distinct reports")
        group["arms"].append(arm)

    return tuple(
        DeduplicatedReport(
            report_hash=report_hash,
            report=grouped[report_hash]["report"],
            arms=tuple(sorted(grouped[report_hash]["arms"])),
        )
        for report_hash in sorted(grouped)
    )


Secret = Union[str, bytes]


def _secret_bytes(secret_salt: Secret) -> bytes:
    if isinstance(secret_salt, str):
        secret = secret_salt.encode("utf-8")
    elif isinstance(secret_salt, bytes):
        secret = secret_salt
    else:
        raise TypeError("secret_salt must be str or bytes")
    if not secret:
        raise ValueError("secret_salt must not be empty")
    return secret


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def _keyed_digest(
    secret_salt: Secret,
    namespace: str,
    seed: int,
    value: Any,
) -> str:
    secret = _secret_bytes(secret_salt)
    _validate_seed(seed)
    payload = canonical_json_dumps(
        {"namespace": namespace, "seed": seed, "value": value}
    ).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def opaque_token(
    identifier: Any,
    *,
    secret_salt: Secret,
    seed: int = 0,
    namespace: str = "item",
    length: int = 24,
) -> str:
    """Create a deterministic, keyed token that does not reveal ``identifier``.

    ``secret_salt`` should be a randomly generated secret stored only in the
    private assignment artifact.  HMAC prevents a reader from confirming
    guesses about case IDs or arm names from the public token.
    """

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if isinstance(length, bool) or not isinstance(length, int) or not 16 <= length <= 64:
        raise ValueError("length must be an integer in [16, 64]")
    digest = _keyed_digest(secret_salt, namespace, seed, identifier)
    return digest[:length]


def deterministic_item_order(
    case_id: Any,
    item_ids: Iterable[str],
    *,
    secret_salt: Secret,
    seed: int = 0,
    namespace: str = "arm-order",
) -> Tuple[str, ...]:
    """Return a keyed, deterministic per-case permutation.

    The returned identifiers are private.  Reader-facing data should expose
    only positional labels (``A``, ``B``, ...) and opaque tokens, as done by
    :func:`build_blinded_case`.
    """

    items = tuple(item_ids)
    if not items:
        raise ValueError("item_ids must not be empty")
    if len(set(items)) != len(items):
        raise ValueError("item_ids must be unique")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("item_ids must contain non-empty strings")
    _secret_bytes(secret_salt)
    _validate_seed(seed)
    return tuple(
        sorted(
            items,
            key=lambda item: (
                _keyed_digest(
                    secret_salt,
                    namespace,
                    seed,
                    {"case_id": case_id, "item_id": item},
                ),
                item,
            ),
        )
    )


def deterministic_arm_order(
    case_id: Any,
    arm_names: Iterable[str],
    *,
    secret_salt: Secret,
    seed: int = 0,
) -> Tuple[str, ...]:
    """Private convenience wrapper for deterministic per-case arm order."""

    return deterministic_item_order(
        case_id,
        arm_names,
        secret_salt=secret_salt,
        seed=seed,
        namespace="arm-order",
    )


@dataclass(frozen=True)
class BlindedPresentation:
    """One public report presentation within a blinded case."""

    slot: str
    report_token: str
    report: str
    report_hash: str = field(repr=False)

    def to_public_dict(self) -> Dict[str, str]:
        """Serialize without the internal normalized-text hash."""

        return {
            "slot": self.slot,
            "report_token": self.report_token,
            "report": self.report,
        }


@dataclass(frozen=True)
class BlindedCase:
    """Reader-facing inputs plus explicitly private decoding maps."""

    case_token: str
    presentations: Tuple[BlindedPresentation, ...]
    slot_to_arms: Mapping[str, Tuple[str, ...]] = field(repr=False)
    arm_to_slot: Mapping[str, str] = field(repr=False)

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a payload containing neither source case ID nor arm names."""

        return {
            "case_token": self.case_token,
            "presentations": [
                presentation.to_public_dict()
                for presentation in self.presentations
            ],
        }

    def to_private_mapping(self) -> Dict[str, Any]:
        """Return the decoding artifact; never include it in reader inputs."""

        return {
            "case_token": self.case_token,
            "slot_to_arms": {
                slot: list(arms) for slot, arms in self.slot_to_arms.items()
            },
            "arm_to_slot": dict(self.arm_to_slot),
            "slot_to_report_hash": {
                presentation.slot: presentation.report_hash
                for presentation in self.presentations
            },
        }


def _slot_label(index: int) -> str:
    # Spreadsheet-like labels keep the public contract usable beyond 26 arms.
    if index < 0:
        raise ValueError("index must be non-negative")
    result = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        result = chr(ord("A") + remainder) + result
        if value == 0:
            return result
        value -= 1


def build_blinded_case(
    case_id: Any,
    arm_reports: Mapping[str, str],
    *,
    secret_salt: Secret,
    seed: int = 0,
) -> BlindedCase:
    """Deduplicate reports and construct public/private blinded artifacts.

    Presentation order is keyed by case, seed, and normalized report hash—not
    by source arm name.  Thus exact-identical arms occupy one reader slot and
    are expanded back to every originating arm only after evaluation.
    """

    deduplicated = deduplicate_arm_reports(arm_reports)
    reports_by_hash = {item.report_hash: item for item in deduplicated}
    ordered_hashes = deterministic_item_order(
        case_id,
        reports_by_hash,
        secret_salt=secret_salt,
        seed=seed,
        namespace="report-order",
    )
    case_token = opaque_token(
        case_id,
        secret_salt=secret_salt,
        seed=seed,
        namespace="case",
    )

    presentations: List[BlindedPresentation] = []
    slot_to_arms: Dict[str, Tuple[str, ...]] = {}
    arm_to_slot: Dict[str, str] = {}
    for index, report_hash in enumerate(ordered_hashes):
        item = reports_by_hash[report_hash]
        slot = _slot_label(index)
        report_token = opaque_token(
            {"case_id": case_id, "report_hash": report_hash},
            secret_salt=secret_salt,
            seed=seed,
            namespace="report",
        )
        presentations.append(
            BlindedPresentation(
                slot=slot,
                report_token=report_token,
                report=item.report,
                report_hash=item.report_hash,
            )
        )
        slot_to_arms[slot] = item.arms
        for arm in item.arms:
            arm_to_slot[arm] = slot

    return BlindedCase(
        case_token=case_token,
        presentations=tuple(presentations),
        slot_to_arms=slot_to_arms,
        arm_to_slot=arm_to_slot,
    )


__all__ = [
    "FINDINGS",
    "IMAGE_STATUSES",
    "IMAGE_QUALITY_VALUES",
    "REPORT_STATUSES",
    "REPORT_AUDIT_JUDGMENTS",
    "REPORT_VERDICTS",
    "PAIRWISE_CHOICES",
    "PAIRWISE_CRITERIA",
    "IMAGE_ONLY_RESPONSE_SCHEMA",
    "REPORT_AUDIT_RESPONSE_SCHEMA",
    "PAIRWISE_RESPONSE_SCHEMA",
    "SchemaValidationError",
    "validate_image_only_response",
    "validate_report_audit_response",
    "validate_pairwise_response",
    "extract_first_json_object",
    "extract_exact_model_response",
    "canonical_json_dumps",
    "sha256_canonical_json",
    "canonicalize_prompt",
    "sha256_prompt",
    "normalize_report_text",
    "normalized_report_hash",
    "DeduplicatedReport",
    "deduplicate_arm_reports",
    "opaque_token",
    "deterministic_item_order",
    "deterministic_arm_order",
    "BlindedPresentation",
    "BlindedCase",
    "build_blinded_case",
]

