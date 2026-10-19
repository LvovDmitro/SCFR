"""Read-only loader for the frozen full-18,716 source-stress cache.

The loader verifies a previously generated privacy-safe inventory, resolves
exactly ten cached chunks for each of the five historical source arms, checks
file hashes and row alignment, and returns only the four arms used by the
``proxy_stress`` model-reader scope.  It never writes a merged report table.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


INVENTORY_SCHEMA_VERSION = "scfr-local-clearml-cache-inventory-v1"
FULL_CACHE_ARMS: Tuple[str, ...] = (
    "current_only",
    "real_prior",
    "prior_report_only",
    "matched_shuffled_prior",
    "shuffled_prior",
)
EXPECTED_CHUNKS: Tuple[int, ...] = tuple(range(1, 11))
EXPECTED_ROWS = 18_716
PROXY_OUTPUT_FIELDS: Tuple[str, ...] = (
    "current_only",
    "real_prior",
    "matched_shuffled_prior",
    "shuffled_prior",
)
FULL_LOGICAL_NAME_RE = re.compile(
    r"^results_maira2_scfr_full18716_chunk(?P<chunk>\d{3})"
    r"(?:_rows\d+_\d+)?_"
    r"(?P<arm>matched_shuffled_prior|prior_report_only|"
    r"shuffled_prior|current_only|real_prior)"
    r"(?:_rerun\d+)?\.json$"
)
SUBJECT_VALUE_RE = re.compile(r"^(?:p)?(?P<digits>\d{8})$", re.IGNORECASE)
SUBJECT_PATH_RE = re.compile(
    r"(?:^|[\\/])p(?P<digits>\d{8})(?=[\\/]|$)",
    re.IGNORECASE,
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected an inventory JSON object: {path}")
    return payload


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _resolve_under(root: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise RuntimeError(f"Unsafe cache filename: {filename!r}")
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Cache artifact escapes cache root: {filename}") from exc
    return path


def verify_inventory_artifacts(
    cache_root: Path,
    inventory: Mapping[str, Any],
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Any]]:
    """Verify inventory structure, paths, sizes, and all 50 artifact hashes."""

    expected_rows = _strict_positive_int(expected_rows, "expected_rows")
    root = cache_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ClearML cache root not found: {root}")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise RuntimeError("Unsupported ClearML cache inventory schema")
    inventory_root = Path(str(inventory.get("cache_root") or "")).resolve()
    if inventory_root != root:
        raise RuntimeError(
            "The supplied cache root does not match the frozen inventory"
        )
    errors = inventory.get("errors")
    if errors not in (None, []):
        raise RuntimeError("The frozen inventory contains artifact errors")

    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Inventory has no artifacts list")
    selected = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and isinstance(artifact.get("full18716"), Mapping)
    ]
    expected_artifact_count = len(FULL_CACHE_ARMS) * len(EXPECTED_CHUNKS)
    if len(selected) != expected_artifact_count:
        raise RuntimeError(
            "Expected exactly 10 full18716 chunks for each of 5 arms "
            f"({expected_artifact_count} artifacts), observed={len(selected)}"
        )

    resolved: Dict[str, Dict[int, Dict[str, Any]]] = {
        arm: {} for arm in FULL_CACHE_ARMS
    }
    for artifact in selected:
        classification = artifact["full18716"]
        arm = str(classification.get("arm") or "")
        chunk = classification.get("chunk")
        if arm not in FULL_CACHE_ARMS or chunk not in EXPECTED_CHUNKS:
            raise RuntimeError(
                f"Unexpected full18716 arm/chunk: arm={arm!r}, chunk={chunk!r}"
            )
        if chunk in resolved[arm]:
            raise RuntimeError(
                f"Duplicate full18716 inventory entry: arm={arm}, chunk={chunk}"
            )
        cache_key = str(artifact.get("cache_object_key") or "")
        logical_name = str(artifact.get("logical_name") or "")
        if (
            len(cache_key) != 32
            or any(char not in "0123456789abcdef" for char in cache_key)
        ):
            raise RuntimeError(f"Invalid cache object key for arm={arm}, chunk={chunk}")
        match = FULL_LOGICAL_NAME_RE.fullmatch(logical_name)
        if (
            match is None
            or match.group("arm") != arm
            or int(match.group("chunk")) != chunk
        ):
            raise RuntimeError(
                f"Logical filename does not match arm/chunk: {logical_name!r}"
            )
        filename = f"{cache_key}.{logical_name}"
        path = _resolve_under(root, filename)
        inventory_source = Path(str(artifact.get("source_path") or "")).resolve()
        if inventory_source != path:
            raise RuntimeError(
                f"Inventory source_path mismatch for arm={arm}, chunk={chunk}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Cached artifact missing for arm={arm}, chunk={chunk}"
            )
        expected_bytes = _strict_positive_int(
            artifact.get("bytes"),
            f"bytes for arm={arm}, chunk={chunk}",
        )
        raw_bytes = path.read_bytes()
        if len(raw_bytes) != expected_bytes:
            raise RuntimeError(
                f"Cached artifact size mismatch for arm={arm}, chunk={chunk}"
            )
        expected_sha = str(artifact.get("sha256") or "")
        observed_sha = hashlib.sha256(raw_bytes).hexdigest()
        if len(expected_sha) != 64 or observed_sha != expected_sha:
            raise RuntimeError(
                f"Cached artifact SHA-256 mismatch for arm={arm}, chunk={chunk}"
            )
        item = dict(artifact)
        item["_verified_path"] = path
        # Preserve the exact authenticated bytes for downstream parsing.  A
        # second open here would permit a cache file to change after hashing
        # but before JSON decoding (TOCTOU), while retaining the same
        # IDs/reference/image metadata checked later.
        item["_verified_bytes"] = raw_bytes
        resolved[arm][int(chunk)] = item

    for arm in FULL_CACHE_ARMS:
        if tuple(sorted(resolved[arm])) != EXPECTED_CHUNKS:
            raise RuntimeError(f"Arm {arm} does not contain exactly chunks 1..10")

    full = inventory.get("full18716")
    if not isinstance(full, Mapping):
        raise RuntimeError("Inventory has no full18716 aggregate")
    if full.get("expected_rows_per_arm") != expected_rows:
        raise RuntimeError(
            "Inventory expected_rows_per_arm does not match the loader contract"
        )
    if full.get("expected_chunks") != list(EXPECTED_CHUNKS):
        raise RuntimeError("Inventory expected_chunks must be exactly 1..10")
    if full.get("all_five_arms_complete_and_aligned") is not True:
        raise RuntimeError("Inventory does not mark all five arms complete/aligned")
    arm_summaries = full.get("arms")
    cross_alignment = full.get("cross_arm_alignment")
    if (
        not isinstance(arm_summaries, Mapping)
        or set(arm_summaries) != set(FULL_CACHE_ARMS)
        or not isinstance(cross_alignment, Mapping)
        or set(cross_alignment) != set(FULL_CACHE_ARMS)
    ):
        raise RuntimeError("Inventory full18716 arm summaries are incomplete")
    for arm in FULL_CACHE_ARMS:
        summary = arm_summaries[arm]
        if not isinstance(summary, Mapping):
            raise RuntimeError(f"Invalid inventory summary for arm={arm}")
        expected_shas = [
            str(resolved[arm][chunk]["sha256"]) for chunk in EXPECTED_CHUNKS
        ]
        if (
            summary.get("files") != len(EXPECTED_CHUNKS)
            or summary.get("chunks") != list(EXPECTED_CHUNKS)
            or summary.get("rows") != expected_rows
            or summary.get("unique_row_ids") != expected_rows
            or summary.get("duplicate_row_count") != 0
            or summary.get("expected_complete") is not True
            or summary.get("artifact_sha256") != expected_shas
        ):
            raise RuntimeError(f"Inventory aggregate is incomplete for arm={arm}")
        alignment = cross_alignment[arm]
        if not isinstance(alignment, Mapping) or not all(
            alignment.get(key) is True
            for key in (
                "same_row_ids_as_current_only",
                "same_references_as_current_only",
                "same_current_images_as_current_only",
            )
        ):
            raise RuntimeError(f"Inventory alignment check failed for arm={arm}")

    safe_summary = {
        "source_mode": "read_only_local_clearml_cache",
        "verified_arms": len(FULL_CACHE_ARMS),
        "verified_chunks_per_arm": len(EXPECTED_CHUNKS),
        "verified_artifacts": expected_artifact_count,
        "expected_rows_per_arm": expected_rows,
        "all_five_arms_complete_and_aligned": True,
    }
    return resolved, safe_summary


def load_verified_artifact_json(
    artifact: MutableMapping[str, Any],
    *,
    consume_bytes: bool = False,
) -> List[Any]:
    """Decode one cache artifact only from its already authenticated bytes.

    ``verify_inventory_artifacts`` reads and hashes every artifact exactly
    once.  Consumers must use this helper rather than reopening
    ``_verified_path``.  ``consume_bytes`` releases the potentially large raw
    buffer after parsing when the caller will not need the artifact again.
    """

    raw_bytes = artifact.get("_verified_bytes")
    if not isinstance(raw_bytes, bytes):
        raise RuntimeError("Verified cache artifact bytes are unavailable")
    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Verified cache artifact is not valid JSON") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Verified cache artifact is not a JSON list")
    if consume_bytes:
        artifact.pop("_verified_bytes", None)
    return payload


def _nonempty_identifier(row: Mapping[str, Any], label: str) -> str:
    value = str(row.get("id") or "").strip()
    if not value:
        raise RuntimeError(f"{label}: missing non-empty id")
    return value


def _current_images(row: Mapping[str, Any], label: str) -> List[str]:
    value = row.get("image_paths")
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{label}: image_paths must be a non-empty list")
    images = [str(item).strip() for item in value]
    if any(not item for item in images):
        raise RuntimeError(f"{label}: image_paths contains an empty value")
    return images


def canonical_mimic_subject_id(
    row: Mapping[str, Any],
    *,
    current_images: Optional[Sequence[str]] = None,
    label: str = "row",
) -> str:
    """Return canonical ``p########`` target subject ID or fail closed."""

    explicit: Optional[str] = None
    raw_subject = row.get("subject_id")
    if raw_subject not in (None, ""):
        if isinstance(raw_subject, bool):
            raise RuntimeError(f"{label}: invalid subject_id")
        match = SUBJECT_VALUE_RE.fullmatch(str(raw_subject).strip())
        if match is None:
            raise RuntimeError(
                f"{label}: subject_id must be 8 digits or p########"
            )
        explicit = f"p{match.group('digits')}"

    images = list(current_images) if current_images is not None else _current_images(
        row,
        label,
    )
    path_subjects = {
        f"p{match.group('digits')}"
        for path in images
        for match in SUBJECT_PATH_RE.finditer(str(path))
    }
    if len(path_subjects) > 1:
        raise RuntimeError(f"{label}: current images contain multiple subject IDs")
    path_subject = next(iter(path_subjects), None)
    if explicit and path_subject and explicit != path_subject:
        raise RuntimeError(
            f"{label}: subject_id disagrees with current-image path"
        )
    result = explicit or path_subject
    if result is None:
        raise RuntimeError(
            f"{label}: cannot derive strict MIMIC subject_id from subject_id "
            "or a p######## current-image path"
        )
    return result


def _load_verified_artifact(
    artifact: MutableMapping[str, Any],
    *,
    arm: str,
    chunk: int,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[Tuple[str, str, str]]]:
    payload = load_verified_artifact_json(
        artifact,
        consume_bytes=True,
    )
    if len(payload) != artifact.get("json_list_rows"):
        raise RuntimeError(
            f"Cached row count disagrees with inventory: arm={arm}, chunk={chunk}"
        )
    order: List[str] = []
    indexed: Dict[str, Dict[str, Any]] = {}
    identities: List[Tuple[str, str, str]] = []
    for index, value in enumerate(payload):
        label = f"arm={arm}, chunk={chunk}, row={index}"
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{label}: expected an object")
        row = dict(value)
        identifier = _nonempty_identifier(row, label)
        if identifier in indexed:
            raise RuntimeError(f"{label}: duplicate id within cached chunk")
        images = _current_images(row, label)
        subject = canonical_mimic_subject_id(
            row,
            current_images=images,
            label=label,
        )
        reference = str(row.get("reference") or "")
        if not reference.strip():
            raise RuntimeError(f"{label}: empty reference")
        prediction = str(row.get("prediction") or "")
        if not prediction.strip():
            raise RuntimeError(f"{label}: empty prediction")
        row["_verified_current_images"] = images
        row["_verified_subject_id"] = subject
        row["_verified_reference"] = reference
        row["_verified_prediction"] = prediction
        indexed[identifier] = row
        order.append(identifier)
        identities.append(
            (
                identifier,
                canonical_sha256(reference),
                canonical_sha256(images),
            )
        )
    counts = Counter(order)
    if (
        len(counts) != artifact.get("unique_row_ids")
        or sum(count - 1 for count in counts.values() if count > 1)
        != artifact.get("duplicate_row_count")
        or canonical_sha256(sorted(counts))
        != artifact.get("row_id_set_sha256")
    ):
        raise RuntimeError(
            f"Cached ID integrity disagrees with inventory: arm={arm}, chunk={chunk}"
        )
    return order, indexed, identities


def load_full18716_proxy_stress_rows(
    cache_root: Path,
    inventory_path: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load verified cache chunks and construct proxy-stress rows in memory."""

    inventory = load_inventory(inventory_path)
    artifacts, safe_summary = verify_inventory_artifacts(
        cache_root,
        inventory,
        expected_rows=expected_rows,
    )
    output: List[Dict[str, Any]] = []
    global_ids: Dict[str, set] = {arm: set() for arm in FULL_CACHE_ARMS}
    aggregate_identities: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for chunk in EXPECTED_CHUNKS:
        orders: Dict[str, List[str]] = {}
        rows_by_arm: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for arm in FULL_CACHE_ARMS:
            order, indexed, identities = _load_verified_artifact(
                artifacts[arm][chunk],
                arm=arm,
                chunk=chunk,
            )
            overlap = global_ids[arm] & set(order)
            if overlap:
                raise RuntimeError(
                    f"Duplicate IDs across chunks for arm={arm}, chunk={chunk}"
                )
            global_ids[arm].update(order)
            aggregate_identities[arm].extend(identities)
            orders[arm] = order
            rows_by_arm[arm] = indexed
        current_ids = set(orders["current_only"])
        for arm in FULL_CACHE_ARMS:
            if set(orders[arm]) != current_ids:
                raise RuntimeError(
                    f"Cross-arm ID mismatch within chunk={chunk}, arm={arm}"
                )

        for identifier in orders["current_only"]:
            current = rows_by_arm["current_only"][identifier]
            reference = current["_verified_reference"]
            images = current["_verified_current_images"]
            subject = current["_verified_subject_id"]
            for arm in FULL_CACHE_ARMS:
                candidate = rows_by_arm[arm][identifier]
                if (
                    candidate["_verified_reference"] != reference
                    or candidate["_verified_current_images"] != images
                    or candidate["_verified_subject_id"] != subject
                ):
                    raise RuntimeError(
                        f"Cross-arm reference/image/subject mismatch in chunk={chunk}"
                    )
            real = rows_by_arm["real_prior"][identifier]
            output.append(
                {
                    "id": identifier,
                    "subject_id": subject,
                    "image_paths": list(images),
                    "reference": reference,
                    "current_only": current["_verified_prediction"],
                    "real_prior": real["_verified_prediction"],
                    "matched_shuffled_prior": rows_by_arm[
                        "matched_shuffled_prior"
                    ][identifier]["_verified_prediction"],
                    "shuffled_prior": rows_by_arm["shuffled_prior"][identifier][
                        "_verified_prediction"
                    ],
                }
            )

    if any(len(global_ids[arm]) != expected_rows for arm in FULL_CACHE_ARMS):
        raise RuntimeError("Actual unique row count is not the expected full cohort")
    if any(global_ids[arm] != global_ids["current_only"] for arm in FULL_CACHE_ARMS):
        raise RuntimeError("Actual row IDs are not aligned across all five arms")

    full_summary = inventory["full18716"]["arms"]
    for arm in FULL_CACHE_ARMS:
        identity_rows = aggregate_identities[arm]
        id_counts = Counter(item[0] for item in identity_rows)
        observed = {
            "row_id_set_sha256": canonical_sha256(sorted(id_counts)),
            "id_reference_map_sha256": canonical_sha256(
                sorted((item[0], item[1]) for item in identity_rows)
            ),
            "id_current_image_map_sha256": canonical_sha256(
                sorted((item[0], item[2]) for item in identity_rows)
            ),
        }
        for key, value in observed.items():
            if value != full_summary[arm].get(key):
                raise RuntimeError(
                    f"Actual aggregate {key} disagrees with inventory for arm={arm}"
                )

    safe_summary = {
        **safe_summary,
        "loaded_proxy_stress_rows_in_memory": len(output),
        "intermediate_wide_merge_written": False,
        "proxy_output_arms": 4,
        "excluded_from_reader_scope": "prior_report_only",
    }
    return output, safe_summary


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "FULL_CACHE_ARMS",
    "EXPECTED_CHUNKS",
    "EXPECTED_ROWS",
    "PROXY_OUTPUT_FIELDS",
    "canonical_mimic_subject_id",
    "load_verified_artifact_json",
    "verify_inventory_artifacts",
    "load_full18716_proxy_stress_rows",
]

