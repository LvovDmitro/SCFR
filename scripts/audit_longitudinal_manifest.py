#!/usr/bin/env python3
"""Audit source-valid longitudinal manifest invariants before SMOKE launch."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


SUBJECT_PATH_RE = re.compile(r"(?:^|[\\/])p(\d{8})(?:[\\/]|$)")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(as_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list manifest: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx} in {path}")
        rows.append(row)
    return rows


def parse_key_value_counts(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=COUNT, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"Expected non-empty NAME in: {raw}")
        out[key] = int(value)
    return out


def candidate_paths(raw_path: str, roots: Sequence[Path], strict: bool = False) -> Iterable[Path]:
    path = Path(raw_path)
    yield path
    if not path.is_absolute():
        yield Path.cwd() / path
    for root in roots:
        yield root / raw_path
        if not strict:
            yield root / path.name


def any_existing_path(paths: Sequence[str], roots: Sequence[Path], strict: bool = False) -> bool:
    seen: Set[str] = set()
    for raw in paths:
        for candidate in candidate_paths(raw, roots, strict=strict):
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.is_file():
                return True
    return False


def subject_from_path(path: str) -> str:
    match = SUBJECT_PATH_RE.search(path.replace("\\", "/"))
    return match.group(1) if match else ""


def split_subject_overlaps(split_subjects: Mapping[str, Set[str]]) -> List[Dict[str, Any]]:
    splits = sorted(split_subjects)
    overlaps: List[Dict[str, Any]] = []
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared = split_subjects[left] & split_subjects[right]
            if shared:
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "shared_subjects": len(shared),
                        "sample_subjects": sorted(shared)[:10],
                    }
                )
    return overlaps


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    image_roots: Sequence[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows_n = len(rows)
    source_subject_study = 0
    current_nonempty = 0
    prior_nonempty = 0
    current_existing = 0
    prior_existing = 0
    reference_text = 0
    prior_report_text = 0
    split_rows = 0
    split_provenance_rows = 0
    temporal_gap_rows = 0
    positive_temporal_gap_rows = 0
    current_subject_path_mismatch = 0
    prior_subject_path_mismatch = 0
    duplicate_study_ids = 0
    split_counts: Dict[str, int] = defaultdict(int)
    split_subjects: Dict[str, Set[str]] = defaultdict(set)
    seen_studies: Set[str] = set()
    unique_subjects: Set[str] = set()
    examples: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        subject_id = str(row.get("subject_id") or "").strip().lstrip("p")
        study_id = str(row.get("study_id") or "").strip().lstrip("s")
        split = str(row.get("split") or "").strip().lower()
        current_paths = as_list(row.get("image_paths") or row.get("image_path"))
        prior_paths = as_list(row.get("prior_image_paths") or row.get("prior_image_path"))

        if subject_id:
            unique_subjects.add(subject_id)
        source_subject_study += int(bool(subject_id and study_id))
        current_nonempty += int(bool(current_paths))
        prior_nonempty += int(bool(prior_paths))
        current_exists = any_existing_path(current_paths, image_roots, strict=args.strict_path_resolution)
        prior_exists = any_existing_path(prior_paths, image_roots, strict=args.strict_path_resolution)
        current_existing += int(current_exists)
        prior_existing += int(prior_exists)
        reference_text += int(has_text(row.get("reference")))
        prior_report_text += int(has_text(row.get("prior_report") or row.get("context")))
        split_rows += int(bool(split))
        split_provenance_rows += int(isinstance(row.get("split_provenance"), dict))
        if split and subject_id:
            split_counts[split] += 1
            split_subjects[split].add(subject_id)
        if study_id in seen_studies:
            duplicate_study_ids += 1
        if study_id:
            seen_studies.add(study_id)

        gap = row.get("temporal_gap_days")
        if isinstance(gap, (int, float)):
            temporal_gap_rows += 1
            positive_temporal_gap_rows += int(gap > 0)

        current_subjects = {subject_from_path(path) for path in current_paths}
        current_subjects.discard("")
        prior_subjects = {subject_from_path(path) for path in prior_paths}
        prior_subjects.discard("")
        if subject_id and current_subjects and subject_id not in current_subjects:
            current_subject_path_mismatch += 1
        if subject_id and prior_subjects and subject_id not in prior_subjects:
            prior_subject_path_mismatch += 1

        if len(examples) < 5:
            examples.append(
                {
                    "row_index": idx,
                    "id": row.get("id") or row.get("sample_id"),
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "split": split,
                    "current_exists": current_exists,
                    "prior_exists": prior_exists,
                    "temporal_gap_days": gap,
                }
            )

    overlaps = split_subject_overlaps(split_subjects)
    blockers: List[str] = []
    if rows_n < args.min_rows:
        blockers.append("rows_below_min")
    if source_subject_study < args.min_rows:
        blockers.append("source_subject_study_coverage_below_min")
    if current_nonempty < args.min_rows:
        blockers.append("current_image_paths_below_min")
    if prior_nonempty < args.min_rows and not args.allow_missing_prior_images:
        blockers.append("prior_image_paths_below_min")
    if current_existing < args.min_existing_current_images:
        blockers.append("existing_current_images_below_min")
    if prior_existing < args.min_existing_prior_images:
        blockers.append("existing_prior_images_below_min")
    if reference_text < args.min_rows:
        blockers.append("reference_text_below_min")
    if prior_report_text < args.min_rows and not args.allow_missing_prior_report:
        blockers.append("prior_report_text_below_min")
    if args.require_official_split and split_rows < args.min_rows:
        blockers.append("official_split_rows_below_min")
    if args.require_official_split and split_provenance_rows < args.min_rows:
        blockers.append("split_provenance_rows_below_min")
    if args.require_temporal_gap and positive_temporal_gap_rows < args.min_rows:
        blockers.append("positive_temporal_gap_rows_below_min")
    if duplicate_study_ids:
        blockers.append("duplicate_study_ids")
    if current_subject_path_mismatch:
        blockers.append("current_subject_path_mismatch")
    if prior_subject_path_mismatch and not args.allow_prior_subject_mismatch:
        blockers.append("prior_subject_path_mismatch")
    if overlaps:
        blockers.append("patient_overlap_across_splits")
    split_min_counts = parse_key_value_counts(args.min_split_count)
    for split, required in sorted(split_min_counts.items()):
        if int(split_counts.get(split, 0) or 0) < required:
            blockers.append(f"split_{split}_below_min")

    return {
        "rows": rows_n,
        "source_identity": {
            "subject_and_study_rows": source_subject_study,
            "unique_subjects": len(unique_subjects),
            "unique_studies": len(seen_studies),
            "duplicate_study_ids": duplicate_study_ids,
        },
        "images": {
            "current_nonempty_rows": current_nonempty,
            "prior_nonempty_rows": prior_nonempty,
            "current_existing_rows": current_existing,
            "prior_existing_rows": prior_existing,
            "current_subject_path_mismatch_rows": current_subject_path_mismatch,
            "prior_subject_path_mismatch_rows": prior_subject_path_mismatch,
        },
        "reports": {
            "reference_text_rows": reference_text,
            "prior_report_text_rows": prior_report_text,
        },
        "splits": {
            "split_rows": split_rows,
            "split_provenance_rows": split_provenance_rows,
            "split_counts": dict(sorted(split_counts.items())),
            "subject_overlaps": overlaps,
        },
        "temporal": {
            "temporal_gap_rows": temporal_gap_rows,
            "positive_temporal_gap_rows": positive_temporal_gap_rows,
        },
        "examples": examples,
        "gate": {
            "blocked": bool(blockers),
            "blockers": sorted(set(blockers)),
            "recommendation": "BLOCK_SMOKE" if blockers else "ALLOW_SMOKE",
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--min-existing-current-images", type=int, default=0)
    parser.add_argument("--min-existing-prior-images", type=int, default=0)
    parser.add_argument("--require-official-split", action="store_true")
    parser.add_argument("--require-temporal-gap", action="store_true")
    parser.add_argument(
        "--allow-prior-subject-mismatch",
        action="store_true",
        help="Allow wrong-patient prior image paths for shuffled-prior negative controls.",
    )
    parser.add_argument(
        "--allow-missing-prior-images",
        action="store_true",
        help="Allow control arms that intentionally remove prior images.",
    )
    parser.add_argument(
        "--allow-missing-prior-report",
        action="store_true",
        help="Allow control arms that intentionally remove prior report text.",
    )
    parser.add_argument("--strict-path-resolution", action="store_true")
    parser.add_argument(
        "--min-split-count",
        action="append",
        default=[],
        help="Require at least SPLIT=COUNT rows. May be repeated.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"ERROR: manifest not found: {args.manifest}")
    rows = load_rows(args.manifest)
    report = {
        "generated_at": utc_now_iso(),
        "stage": "SCOPING_DATA_GATE",
        "manifest": str(args.manifest),
        "image_roots": [str(path) for path in args.image_root],
        "config": {
            "min_rows": int(args.min_rows),
            "min_existing_current_images": int(args.min_existing_current_images),
            "min_existing_prior_images": int(args.min_existing_prior_images),
            "require_official_split": bool(args.require_official_split),
            "require_temporal_gap": bool(args.require_temporal_gap),
            "allow_prior_subject_mismatch": bool(args.allow_prior_subject_mismatch),
            "allow_missing_prior_images": bool(args.allow_missing_prior_images),
            "allow_missing_prior_report": bool(args.allow_missing_prior_report),
            "strict_path_resolution": bool(args.strict_path_resolution),
            "min_split_count": parse_key_value_counts(args.min_split_count),
        },
        **audit_rows(rows, args.image_root, args),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_blocker and report["gate"]["blocked"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

