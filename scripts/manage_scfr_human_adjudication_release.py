#!/usr/bin/env python3
"""Tamper-evident sequential release and response locking for SCFR readers.

The prepared packet is immutable.  One signed controller state is created per
anonymous reader code.  Stage 2 cannot be released until the stage-1 response
is schema-validated and locked; stage 3 has the analogous stage-2 gate.
Private decoding keys are verified at initialization but are never copied into
reader-facing release directories.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    from scripts.prepare_scfr_human_adjudication_packet import (
        FINDINGS,
        HISTORY_INFORMATIVENESS_VALUES,
        PREFERENCE_VALUES,
        TASK_A_STATUSES,
        TEMPORAL_CORRECTNESS_VALUES,
        sha256_canonical_json,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.prepare_scfr_human_adjudication_packet import (
        FINDINGS,
        HISTORY_INFORMATIVENESS_VALUES,
        PREFERENCE_VALUES,
        TASK_A_STATUSES,
        TEMPORAL_CORRECTNESS_VALUES,
        sha256_canonical_json,
        sha256_file,
    )


PROTOCOL_VERSION = "scfr-human-adjudication-release-v1"
READER_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STAGES = {
    1: ("reader_stage_01_task_a", "task_a.json"),
    2: ("reader_stage_02_task_b_history", "task_b_history.json"),
    3: ("reader_stage_03_task_b_reports", "task_b_reports.json"),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def _validate_utc_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RuntimeError(f"{label} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise RuntimeError(f"{label} is not UTC")
    return value


def _read_secret(path: Path) -> bytes:
    if not path.is_file():
        raise RuntimeError("Release signing secret file does not exist")
    secret = path.read_bytes().strip()
    if len(secret) < 32:
        raise RuntimeError("Release signing secret must contain at least 32 bytes")
    return secret


def _signature(payload: Mapping[str, Any], secret: bytes) -> str:
    return hmac.new(secret, _canonical_bytes(payload), hashlib.sha256).hexdigest()


def _signed(payload: Mapping[str, Any], secret: bytes) -> Dict[str, Any]:
    if "signature_hmac_sha256" in payload:
        raise RuntimeError("Unsigned payload unexpectedly contains a signature")
    output = dict(payload)
    output["signature_hmac_sha256"] = _signature(payload, secret)
    return output


def _verify_signed(
    payload: Mapping[str, Any],
    secret: bytes,
    *,
    label: str,
) -> Dict[str, Any]:
    observed = payload.get("signature_hmac_sha256")
    if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
        raise RuntimeError(f"{label} has no valid HMAC signature")
    unsigned = dict(payload)
    unsigned.pop("signature_hmac_sha256")
    expected = _signature(unsigned, secret)
    if not hmac.compare_digest(observed, expected):
        raise RuntimeError(f"{label} HMAC signature mismatch")
    return unsigned


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_replace: bool,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_replace:
        raise RuntimeError("Refusing to overwrite an existing control artifact")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(_canonical_bytes(payload) + b"\n")
        if path.exists() and not allow_replace:
            raise RuntimeError("Control artifact appeared during write")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise RuntimeError("Expected immutable stage directory is missing")
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    if not files:
        raise RuntimeError("Immutable stage directory is empty")
    return sha256_canonical_json(files)


def _outside_repository(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(repository_root)
    except ValueError:
        return resolved
    raise RuntimeError(f"{label} must be outside the repository")


def _assert_disjoint(left: Path, right: Path, *, label: str) -> None:
    for parent, child in ((left, right), (right, left)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise RuntimeError(f"{label} roots must be physically disjoint")


def _validate_packet(
    public_packet_dir: Path,
) -> tuple[Dict[str, Any], str]:
    root = _outside_repository(public_packet_dir, label="Public packet")
    if (root / "private").exists():
        raise RuntimeError("Public packet illegally contains a private directory")
    receipt_path = root / "packet_receipt.json"
    receipt = _load_object(receipt_path)
    if (
        receipt.get("protocol_version")
        != "scfr-human-adjudication-packet-v1-receipt"
        or receipt.get("release_status") != "prepared_not_released"
        or receipt.get("authorized_for_release") is not False
    ):
        raise RuntimeError("Prepared packet receipt status/protocol mismatch")
    stage_hashes = receipt.get("reader_stage_tree_sha256")
    if not isinstance(stage_hashes, dict) or not stage_hashes:
        raise RuntimeError("Prepared packet has no stage commitments")
    expected_names = {
        STAGES[1][0]
    } | ({STAGES[2][0], STAGES[3][0]} if receipt.get("task_b_authorized") else set())
    if set(stage_hashes) != expected_names:
        raise RuntimeError("Prepared packet stage set does not match study scope")
    for name, digest in stage_hashes.items():
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise RuntimeError("Prepared packet stage hash is invalid")
        if _tree_sha256(root / name) != digest:
            raise RuntimeError("Prepared packet stage tree changed after preparation")
    return receipt, sha256_file(receipt_path)


def initialize_state(
    *,
    public_packet_dir: Path,
    private_answer_key_path: Path,
    state_path: Path,
    signing_secret_path: Path,
    reader_code: str,
    start_stage: int = 1,
) -> Dict[str, Any]:
    """Create one signed, immutable-origin controller state."""

    if not READER_CODE_RE.fullmatch(reader_code):
        raise RuntimeError("reader_code must be a short anonymous code")
    public_root = _outside_repository(public_packet_dir, label="Public packet")
    state_path = _outside_repository(state_path, label="Release state")
    private_key = _outside_repository(
        private_answer_key_path,
        label="Private answer key",
    )
    _assert_disjoint(
        public_root,
        state_path.parent,
        label="Public packet and control",
    )
    receipt, receipt_hash = _validate_packet(public_root)
    if start_stage not in (1, 2):
        raise RuntimeError("start_stage must be 1 or 2")
    if start_stage == 2 and receipt.get("task_b_authorized") is not True:
        raise RuntimeError("Task-B-only workflow requires a Task-B-authorized packet")
    if not private_key.is_file():
        raise RuntimeError("Private answer key is missing")
    private_hash = sha256_file(private_key)
    if private_hash != receipt.get("private_answer_key_file_sha256"):
        raise RuntimeError("Private answer key does not match packet receipt")
    secret = _read_secret(signing_secret_path)
    stages: Dict[str, Any] = {}
    for number, (name, _filename) in STAGES.items():
        if number < start_stage:
            continue
        if name not in receipt["reader_stage_tree_sha256"]:
            continue
        stages[str(number)] = {
            "stage_name": name,
            "stage_tree_sha256": receipt["reader_stage_tree_sha256"][name],
            "status": "pending",
        }
    unsigned = {
        "protocol_version": f"{PROTOCOL_VERSION}-state",
        "reader_code": reader_code,
        "packet_token": receipt["packet_token"],
        "public_packet_receipt_sha256": receipt_hash,
        "private_answer_key_sha256": private_hash,
        "task_b_authorized": bool(receipt["task_b_authorized"]),
        "workflow_scope": (
            "full_task_a_then_task_b"
            if start_stage == 1
            else "task_b_only_history_then_reports"
        ),
        "start_stage": start_stage,
        "state_sequence": 0,
        "updated_at_utc": _utc_now(),
        "stages": stages,
    }
    state = _signed(unsigned, secret)
    _write_json_atomic(state_path, state, allow_replace=False)
    return {
        "status": "initialized",
        "reader_code": reader_code,
        "packet_token": receipt["packet_token"],
        "state_sha256": sha256_file(state_path),
        "next_stage": _next_stage(unsigned),
    }


def _load_state(
    state_path: Path,
    secret: bytes,
    *,
    public_packet_dir: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    state = _load_object(state_path)
    unsigned = _verify_signed(state, secret, label="Release state")
    if unsigned.get("protocol_version") != f"{PROTOCOL_VERSION}-state":
        raise RuntimeError("Release state protocol mismatch")
    _validate_utc_timestamp(
        unsigned.get("updated_at_utc"),
        label="Release state updated_at_utc",
    )
    receipt, receipt_hash = _validate_packet(public_packet_dir)
    if (
        unsigned.get("public_packet_receipt_sha256") != receipt_hash
        or unsigned.get("packet_token") != receipt.get("packet_token")
        or unsigned.get("task_b_authorized")
        is not bool(receipt.get("task_b_authorized"))
    ):
        raise RuntimeError("Release state is bound to a different packet")
    return state, unsigned, receipt_hash


def _next_stage(unsigned_state: Mapping[str, Any]) -> Optional[int]:
    stages = unsigned_state["stages"]
    for number in sorted(int(value) for value in stages):
        if stages[str(number)]["status"] == "pending":
            return number
        if stages[str(number)]["status"] != "locked":
            return None
    return None


def release_stage(
    *,
    public_packet_dir: Path,
    state_path: Path,
    signing_secret_path: Path,
    stage_number: int,
    release_output_dir: Path,
    confirm_qualified_reader: bool,
    confirm_institutional_gates: bool,
    confirm_computational_audit_frozen: bool,
) -> Dict[str, Any]:
    """Copy exactly the next immutable stage and sign its release receipt."""

    if not all(
        (
            confirm_qualified_reader,
            confirm_institutional_gates,
            confirm_computational_audit_frozen,
        )
    ):
        raise RuntimeError(
            "Release requires qualified-reader, institutional, and frozen-audit "
            "confirmations"
        )
    if stage_number not in STAGES:
        raise RuntimeError("Unknown adjudication stage")
    secret = _read_secret(signing_secret_path)
    _state, unsigned, packet_receipt_hash = _load_state(
        state_path,
        secret,
        public_packet_dir=public_packet_dir,
    )
    if _next_stage(unsigned) != stage_number:
        raise RuntimeError(
            "Requested stage is not next; preceding response must be locked"
        )
    stage_state = unsigned["stages"].get(str(stage_number))
    if not isinstance(stage_state, dict) or stage_state["status"] != "pending":
        raise RuntimeError("Requested stage is unavailable or already released")

    public_root = _outside_repository(public_packet_dir, label="Public packet")
    release_root = _outside_repository(
        release_output_dir,
        label="Reader release",
    )
    control_root = _outside_repository(state_path, label="Release state").parent
    _assert_disjoint(public_root, release_root, label="Packet and release")
    _assert_disjoint(control_root, release_root, label="Control and release")
    if release_root.exists():
        raise RuntimeError("Reader release output already exists")
    release_root.parent.mkdir(parents=True, exist_ok=True)

    stage_name, _filename = STAGES[stage_number]
    source_stage = public_root / stage_name
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{release_root.name}.building-",
            dir=str(release_root.parent),
        )
    )
    try:
        released_stage = temporary / stage_name
        shutil.copytree(source_stage, released_stage)
        observed_tree = _tree_sha256(released_stage)
        if observed_tree != stage_state["stage_tree_sha256"]:
            raise RuntimeError("Copied release stage differs from frozen packet")
        receipt_unsigned = {
            "protocol_version": f"{PROTOCOL_VERSION}-stage-receipt",
            "reader_code": unsigned["reader_code"],
            "packet_token": unsigned["packet_token"],
            "public_packet_receipt_sha256": packet_receipt_hash,
            "stage_number": stage_number,
            "stage_name": stage_name,
            "stage_tree_sha256": observed_tree,
            "released_at_utc": _utc_now(),
            "release_authorizations": {
                "qualified_reader_confirmed": True,
                "institutional_gates_confirmed": True,
                "computational_audit_frozen_confirmed": True,
            },
        }
        release_receipt = _signed(receipt_unsigned, secret)
        _write_json_atomic(
            temporary / "release_receipt.json",
            release_receipt,
            allow_replace=False,
        )
        release_receipt_hash = sha256_file(
            temporary / "release_receipt.json"
        )
        if release_root.exists():
            raise RuntimeError("Reader release output appeared during build")
        os.replace(str(temporary), str(release_root))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    stage_state.update(
        {
            "status": "released",
            "release_receipt_sha256": release_receipt_hash,
            "released_stage_tree_sha256": observed_tree,
        }
    )
    unsigned["state_sequence"] += 1
    unsigned["updated_at_utc"] = _utc_now()
    new_state = _signed(unsigned, secret)
    try:
        _write_json_atomic(state_path, new_state, allow_replace=True)
    except BaseException:
        # The release directory was created by this invocation and has not
        # been handed off successfully without a matching signed state.
        if release_root.exists():
            shutil.rmtree(release_root)
        raise
    return {
        "status": "released",
        "reader_code": unsigned["reader_code"],
        "packet_token": unsigned["packet_token"],
        "stage_number": stage_number,
        "stage_tree_sha256": observed_tree,
        "release_receipt_sha256": release_receipt_hash,
    }


def _require_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise RuntimeError(f"{label} must be completed text")
    return value


def _require_string_list(value: Any, *, label: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RuntimeError(f"{label} must be a list of non-empty strings")


def _validate_case_response(stage_number: int, response: Any) -> None:
    if not isinstance(response, dict):
        raise RuntimeError("Reader case response must be an object")
    if stage_number == 1:
        if set(response) != {"finding_statuses", "other_important_findings"}:
            raise RuntimeError("Task A response schema changed")
        findings = response["finding_statuses"]
        if not isinstance(findings, dict) or set(findings) != set(FINDINGS):
            raise RuntimeError("Task A finding vocabulary changed")
        if any(value not in TASK_A_STATUSES for value in findings.values()):
            raise RuntimeError("Task A contains an incomplete/invalid finding status")
        _require_string(
            response["other_important_findings"],
            label="Task A other_important_findings",
            allow_empty=True,
        )
        return
    if stage_number == 2:
        if set(response) != {
            "history_informativeness",
            "history_informativeness_reason",
        }:
            raise RuntimeError("Task B history response schema changed")
        if response["history_informativeness"] not in (
            HISTORY_INFORMATIVENESS_VALUES
        ):
            raise RuntimeError("Task B history informativeness is invalid")
        _require_string(
            response["history_informativeness_reason"],
            label="Task B history reason",
        )
        return
    if set(response) != {
        "overall_preference",
        "preference_reason",
        "report_a_assessment",
        "report_b_assessment",
    }:
        raise RuntimeError("Task B report response schema changed")
    if response["overall_preference"] not in PREFERENCE_VALUES:
        raise RuntimeError("Task B overall preference is invalid")
    _require_string(response["preference_reason"], label="Task B preference reason")
    assessment_keys = {
        "supported_findings_recovered",
        "supported_findings_lost",
        "unsupported_additions",
        "important_omissions",
        "temporal_correctness",
    }
    for label in ("report_a_assessment", "report_b_assessment"):
        assessment = response[label]
        if not isinstance(assessment, dict) or set(assessment) != assessment_keys:
            raise RuntimeError("Task B report assessment schema changed")
        for key in assessment_keys - {"temporal_correctness"}:
            _require_string_list(assessment[key], label=f"{label}.{key}")
        if assessment["temporal_correctness"] not in (
            TEMPORAL_CORRECTNESS_VALUES
        ):
            raise RuntimeError("Task B temporal correctness is invalid")


def _validate_completed_response(
    source_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    *,
    stage_number: int,
) -> str:
    if set(response_payload) != set(source_payload):
        raise RuntimeError("Completed response top-level schema changed")
    for key in source_payload:
        if key == "cases":
            continue
        if response_payload[key] != source_payload[key]:
            raise RuntimeError("Completed response changed immutable task metadata")
    source_cases = source_payload["cases"]
    response_cases = response_payload["cases"]
    if not isinstance(response_cases, list) or len(response_cases) != len(
        source_cases
    ):
        raise RuntimeError("Completed response case cardinality changed")
    tokens = []
    for source_case, response_case in zip(source_cases, response_cases):
        if not isinstance(response_case, dict) or set(response_case) != set(
            source_case
        ):
            raise RuntimeError("Completed response case schema changed")
        for key in source_case:
            if key == "response":
                continue
            if response_case[key] != source_case[key]:
                raise RuntimeError("Completed response changed immutable case data")
        _validate_case_response(stage_number, response_case["response"])
        tokens.append(response_case["case_token"])
    commitment = sha256_canonical_json(tokens)
    if commitment != source_payload.get("case_order_commitment"):
        raise RuntimeError("Completed response case order commitment mismatch")
    return commitment


def lock_response(
    *,
    public_packet_dir: Path,
    state_path: Path,
    signing_secret_path: Path,
    stage_number: int,
    release_output_dir: Path,
    response_json_path: Path,
) -> Dict[str, Any]:
    """Validate, hash, copy, and sign one completed reader response."""

    if stage_number not in STAGES:
        raise RuntimeError("Unknown adjudication stage")
    secret = _read_secret(signing_secret_path)
    _state, unsigned, packet_receipt_hash = _load_state(
        state_path,
        secret,
        public_packet_dir=public_packet_dir,
    )
    stage_state = unsigned["stages"].get(str(stage_number))
    if not isinstance(stage_state, dict) or stage_state.get("status") != "released":
        raise RuntimeError("A response can be locked only after its stage release")
    release_root = _outside_repository(
        release_output_dir,
        label="Reader release",
    )
    release_receipt_path = release_root / "release_receipt.json"
    release_receipt = _load_object(release_receipt_path)
    release_unsigned = _verify_signed(
        release_receipt,
        secret,
        label="Stage release receipt",
    )
    _validate_utc_timestamp(
        release_unsigned.get("released_at_utc"),
        label="Stage release timestamp",
    )
    if (
        sha256_file(release_receipt_path)
        != stage_state["release_receipt_sha256"]
        or release_unsigned.get("reader_code") != unsigned["reader_code"]
        or release_unsigned.get("packet_token") != unsigned["packet_token"]
        or release_unsigned.get("public_packet_receipt_sha256")
        != packet_receipt_hash
        or release_unsigned.get("stage_number") != stage_number
    ):
        raise RuntimeError("Stage release receipt/state binding mismatch")
    stage_name, task_filename = STAGES[stage_number]
    released_stage = release_root / stage_name
    observed_tree = _tree_sha256(released_stage)
    if (
        observed_tree != stage_state["stage_tree_sha256"]
        or observed_tree != release_unsigned.get("stage_tree_sha256")
    ):
        raise RuntimeError("Released stage changed before response lock")

    source_payload = _load_object(released_stage / task_filename)
    response_payload = _load_object(response_json_path)
    case_order_sha = _validate_completed_response(
        source_payload,
        response_payload,
        stage_number=stage_number,
    )
    response_sha = sha256_file(response_json_path)
    locked_root = state_path.resolve().parent / "locked_responses"
    locked_response = (
        locked_root
        / unsigned["reader_code"]
        / f"stage_{stage_number:02d}_response.json"
    )
    if locked_response.exists():
        raise RuntimeError("A locked response already exists for this stage")
    locked_response.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(response_json_path, locked_response)
    if sha256_file(locked_response) != response_sha:
        locked_response.unlink(missing_ok=True)
        raise RuntimeError("Locked response copy hash mismatch")

    lock_receipt_unsigned = {
        "protocol_version": f"{PROTOCOL_VERSION}-response-lock-receipt",
        "reader_code": unsigned["reader_code"],
        "packet_token": unsigned["packet_token"],
        "public_packet_receipt_sha256": packet_receipt_hash,
        "stage_number": stage_number,
        "stage_tree_sha256": observed_tree,
        "release_receipt_sha256": stage_state["release_receipt_sha256"],
        "case_order_commitment": case_order_sha,
        "response_sha256": response_sha,
        "locked_at_utc": _utc_now(),
    }
    lock_receipt = _signed(lock_receipt_unsigned, secret)
    lock_receipt_path = locked_response.with_name(
        f"stage_{stage_number:02d}_lock_receipt.json"
    )
    try:
        _write_json_atomic(
            lock_receipt_path,
            lock_receipt,
            allow_replace=False,
        )
    except BaseException:
        locked_response.unlink(missing_ok=True)
        raise
    lock_receipt_sha = sha256_file(lock_receipt_path)

    stage_state.update(
        {
            "status": "locked",
            "response_sha256": response_sha,
            "lock_receipt_sha256": lock_receipt_sha,
            "case_order_commitment": case_order_sha,
        }
    )
    unsigned["state_sequence"] += 1
    unsigned["updated_at_utc"] = _utc_now()
    try:
        _write_json_atomic(
            state_path,
            _signed(unsigned, secret),
            allow_replace=True,
        )
    except BaseException:
        lock_receipt_path.unlink(missing_ok=True)
        locked_response.unlink(missing_ok=True)
        raise
    return {
        "status": "locked",
        "reader_code": unsigned["reader_code"],
        "packet_token": unsigned["packet_token"],
        "stage_number": stage_number,
        "response_sha256": response_sha,
        "lock_receipt_sha256": lock_receipt_sha,
        "next_stage": _next_stage(unsigned),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--public-packet-dir", type=Path, required=True)
    initialize.add_argument("--private-answer-key", type=Path, required=True)
    initialize.add_argument("--state-path", type=Path, required=True)
    initialize.add_argument("--signing-secret-file", type=Path, required=True)
    initialize.add_argument("--reader-code", required=True)
    initialize.add_argument(
        "--start-stage",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Use 2 for a predeclared Task-B-only workflow: blinded history "
            "assessment followed by A/B reports, without a new Task-A repeat."
        ),
    )

    release = subparsers.add_parser("release")
    release.add_argument("--public-packet-dir", type=Path, required=True)
    release.add_argument("--state-path", type=Path, required=True)
    release.add_argument("--signing-secret-file", type=Path, required=True)
    release.add_argument("--stage", type=int, choices=tuple(STAGES), required=True)
    release.add_argument("--release-output-dir", type=Path, required=True)
    release.add_argument("--confirm-qualified-reader", action="store_true")
    release.add_argument("--confirm-institutional-gates", action="store_true")
    release.add_argument(
        "--confirm-computational-audit-frozen",
        action="store_true",
    )

    lock = subparsers.add_parser("lock")
    lock.add_argument("--public-packet-dir", type=Path, required=True)
    lock.add_argument("--state-path", type=Path, required=True)
    lock.add_argument("--signing-secret-file", type=Path, required=True)
    lock.add_argument("--stage", type=int, choices=tuple(STAGES), required=True)
    lock.add_argument("--release-output-dir", type=Path, required=True)
    lock.add_argument("--response-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "init":
            result = initialize_state(
                public_packet_dir=args.public_packet_dir,
                private_answer_key_path=args.private_answer_key,
                state_path=args.state_path,
                signing_secret_path=args.signing_secret_file,
                reader_code=args.reader_code,
                start_stage=args.start_stage,
            )
        elif args.command == "release":
            result = release_stage(
                public_packet_dir=args.public_packet_dir,
                state_path=args.state_path,
                signing_secret_path=args.signing_secret_file,
                stage_number=args.stage,
                release_output_dir=args.release_output_dir,
                confirm_qualified_reader=args.confirm_qualified_reader,
                confirm_institutional_gates=args.confirm_institutional_gates,
                confirm_computational_audit_frozen=(
                    args.confirm_computational_audit_frozen
                ),
            )
        else:
            result = lock_response(
                public_packet_dir=args.public_packet_dir,
                state_path=args.state_path,
                signing_secret_path=args.signing_secret_file,
                stage_number=args.stage,
                release_output_dir=args.release_output_dir,
                response_json_path=args.response_json,
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

