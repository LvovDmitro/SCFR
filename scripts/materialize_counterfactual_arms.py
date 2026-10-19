#!/usr/bin/env python3
"""Materialize longitudinal counterfactual arms from a source manifest."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import finding_vector, tokenize
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_vector, tokenize

ARM_NAMES: Tuple[str, ...] = (
    "current_only",
    "real_prior",
    "shuffled_prior",
    "matched_shuffled_prior",
    "prior_report_only",
)


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


def row_id(row: Mapping[str, Any], idx: int) -> str:
    for key in ("id", "sample_id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row_{idx}"


def subject_id(row: Mapping[str, Any]) -> str:
    return str(row.get("subject_id") or "").strip()


def require_subject_ids(rows: Sequence[Mapping[str, Any]], arm: str) -> None:
    missing = [idx for idx, row in enumerate(rows) if not subject_id(row)]
    if missing:
        preview = ",".join(str(idx) for idx in missing[:10])
        raise RuntimeError(f"{arm}: missing subject_id for shuffled-prior control rows: {preview}")


def prior_report_text(row: Mapping[str, Any]) -> str:
    return str(row.get("prior_report") or row.get("context") or "").strip()


def has_prior_payload(row: Mapping[str, Any]) -> bool:
    return bool(as_list(row.get("prior_image_paths") or row.get("prior_image_path")) or prior_report_text(row))


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected list JSON manifest: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx} in {path}")
        rows.append(dict(row))
    return rows


def row_split(row: Mapping[str, Any]) -> str:
    return str(row.get("split") or "").strip().lower()


def choose_shuffled_donors(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    match_split: bool = True,
) -> Dict[int, int]:
    require_subject_ids(rows, "shuffled_prior")
    donor_indices = [idx for idx, row in enumerate(rows) if has_prior_payload(row)]
    if len(donor_indices) < 2:
        raise RuntimeError("Need at least two rows with prior payload to materialize shuffled_prior")

    rng = random.Random(seed)
    donor_cycle = list(donor_indices)
    rng.shuffle(donor_cycle)

    assignments: Dict[int, int] = {}
    cursor = 0
    for target_idx, target_row in enumerate(rows):
        target_subject = subject_id(target_row)
        target_split = row_split(target_row)
        chosen_idx: Optional[int] = None
        chosen_pos = cursor
        for offset in range(len(donor_cycle)):
            pos = (cursor + offset) % len(donor_cycle)
            donor_idx = donor_cycle[pos]
            donor_row = rows[donor_idx]
            donor_subject = subject_id(donor_row)
            same_subject = bool(target_subject and donor_subject and target_subject == donor_subject)
            split_ok = not match_split or not target_split or row_split(donor_row) == target_split
            if donor_idx != target_idx and not same_subject and split_ok:
                chosen_idx = donor_idx
                chosen_pos = pos
                break
        if chosen_idx is None:
            raise RuntimeError(
                "Unable to assign shuffled prior with different subject"
                f"{' and same split' if match_split else ''} for all rows. "
                "Provide more subjects with prior payload or pass --allow-cross-split-shuffle."
            )
        assignments[target_idx] = chosen_idx
        cursor = (chosen_pos + 1) % len(donor_cycle)
    return assignments


def reference_positive_findings(row: Mapping[str, Any]) -> List[str]:
    vec = finding_vector(str(row.get("reference") or ""))
    return sorted(name for name, status in vec.items() if status == "positive")


def reference_finding_mask(row: Mapping[str, Any], finding_index: Mapping[str, int]) -> int:
    mask = 0
    for finding in reference_positive_findings(row):
        bit = finding_index.get(finding)
        if bit is not None:
            mask |= 1 << bit
    return mask


def popcount(value: int) -> int:
    return bin(int(value)).count("1")


def candidate_split_key(row: Mapping[str, Any], match_split: bool) -> str:
    if not match_split:
        return "*"
    return row_split(row) or "*"


def best_length_matched_donor(
    group: Sequence[Tuple[int, float, int, str]],
    lengths: Sequence[int],
    *,
    target_len: int,
    target_idx: int,
    target_subject: str,
) -> Optional[Tuple[int, float, int]]:
    """Return the best (length_delta, tie_breaker, donor_idx) in one mask group."""
    pos = bisect.bisect_left(lengths, target_len)
    left = pos - 1
    right = pos
    best: Optional[Tuple[int, float, int]] = None

    while left >= 0 or right < len(group):
        left_delta = abs(group[left][0] - target_len) if left >= 0 else None
        right_delta = abs(group[right][0] - target_len) if right < len(group) else None
        next_delta = min(value for value in (left_delta, right_delta) if value is not None)
        if best is not None and next_delta > best[0]:
            break

        if left_delta is not None and left_delta == next_delta:
            length, tie, donor_idx, donor_subject = group[left]
            if donor_idx != target_idx and not (target_subject and donor_subject and target_subject == donor_subject):
                candidate = (abs(length - target_len), tie, donor_idx)
                if best is None or candidate < best:
                    best = candidate
            left -= 1

        if right_delta is not None and right_delta == next_delta:
            length, tie, donor_idx, donor_subject = group[right]
            if donor_idx != target_idx and not (target_subject and donor_subject and target_subject == donor_subject):
                candidate = (abs(length - target_len), tie, donor_idx)
                if best is None or candidate < best:
                    best = candidate
            right += 1

    return best


def choose_matched_shuffled_donors(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    match_split: bool = True,
) -> Dict[int, int]:
    require_subject_ids(rows, "matched_shuffled_prior")
    donor_indices = [idx for idx, row in enumerate(rows) if has_prior_payload(row)]
    if len(donor_indices) < 2:
        raise RuntimeError("Need at least two rows with prior payload to materialize matched_shuffled_prior")

    rng = random.Random(seed)
    tie_breakers = {idx: rng.random() for idx in donor_indices}
    all_findings = sorted({finding for row in rows for finding in reference_positive_findings(row)})
    finding_index = {finding: idx for idx, finding in enumerate(all_findings)}
    finding_masks = {idx: reference_finding_mask(row, finding_index) for idx, row in enumerate(rows)}
    prior_lengths = {idx: len(tokenize(prior_report_text(row))) for idx, row in enumerate(rows)}

    groups: Dict[str, Dict[int, List[Tuple[int, float, int, str]]]] = {}
    for donor_idx in donor_indices:
        donor_row = rows[donor_idx]
        split_key = candidate_split_key(donor_row, match_split)
        groups.setdefault(split_key, {}).setdefault(finding_masks[donor_idx], []).append(
            (prior_lengths[donor_idx], tie_breakers[donor_idx], donor_idx, subject_id(donor_row))
        )
    group_lengths: Dict[str, Dict[int, List[int]]] = {}
    for split_key, by_mask in groups.items():
        group_lengths[split_key] = {}
        for mask, items in by_mask.items():
            items.sort()
            group_lengths[split_key][mask] = [item[0] for item in items]

    assignments: Dict[int, int] = {}
    for target_idx, target_row in enumerate(rows):
        target_subject = subject_id(target_row)
        target_split = row_split(target_row)
        split_key = candidate_split_key(target_row, match_split)
        by_mask = groups.get(split_key, {})
        by_mask_lengths = group_lengths.get(split_key, {})
        target_mask = finding_masks[target_idx]
        target_prior_len = prior_lengths[target_idx]
        best: Optional[Tuple[int, int, float, int]] = None
        for donor_mask in sorted(by_mask, key=lambda mask: (popcount(target_mask ^ mask), mask)):
            finding_distance = popcount(target_mask ^ donor_mask)
            if best is not None and finding_distance > best[0]:
                break
            candidate = best_length_matched_donor(
                by_mask[donor_mask],
                by_mask_lengths[donor_mask],
                target_len=target_prior_len,
                target_idx=target_idx,
                target_subject=target_subject,
            )
            if candidate is None:
                continue
            length_delta, tie, donor_idx = candidate
            scored = (finding_distance, length_delta, tie, donor_idx)
            if best is None or scored < best:
                best = scored
        if best is None:
            raise RuntimeError(
                "Unable to assign matched shuffled prior with different subject"
                f"{' and same split' if match_split else ''} for all rows. "
                "Provide more subjects with prior payload or pass --allow-cross-split-shuffle."
            )
        assignments[target_idx] = best[3]
    return assignments


def annotate_common(row: Dict[str, Any], arm: str, src_idx: int) -> None:
    row["counterfactual_arm"] = arm
    row.setdefault("counterfactual_source", {})
    row["counterfactual_source"].update(
        {
            "source_row_index": src_idx,
            "source_row_id": row_id(row, src_idx),
        }
    )


def build_current_only(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, source in enumerate(rows):
        row = dict(source)
        row["prior_image_paths"] = []
        row["prior_image_path"] = ""
        row["prior_study_id"] = ""
        row["prior_study_datetime"] = ""
        row["temporal_gap_days"] = None
        row["prior_report"] = ""
        row["context"] = ""
        annotate_common(row, "current_only", idx)
        out.append(row)
    return out


def build_real_prior(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, source in enumerate(rows):
        row = dict(source)
        annotate_common(row, "real_prior", idx)
        out.append(row)
    return out


def build_prior_report_only(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, source in enumerate(rows):
        row = dict(source)
        report = prior_report_text(source)
        row["prior_image_paths"] = []
        row["prior_image_path"] = ""
        row["prior_report"] = report
        row["context"] = report
        annotate_common(row, "prior_report_only", idx)
        out.append(row)
    return out


def build_shuffled_prior(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    match_split: bool = True,
) -> List[Dict[str, Any]]:
    donor_map = choose_shuffled_donors(rows, seed, match_split=match_split)
    out: List[Dict[str, Any]] = []
    for idx, source in enumerate(rows):
        donor_idx = donor_map[idx]
        donor = rows[donor_idx]
        row = dict(source)
        row["prior_study_id"] = str(donor.get("prior_study_id") or "")
        row["prior_image_paths"] = as_list(donor.get("prior_image_paths") or donor.get("prior_image_path"))
        row["prior_image_path"] = row["prior_image_paths"][0] if row["prior_image_paths"] else ""
        row["prior_report"] = prior_report_text(donor)
        row["context"] = row["prior_report"]
        row["prior_study_datetime"] = str(donor.get("prior_study_datetime") or "")
        row["temporal_gap_days"] = None
        annotate_common(row, "shuffled_prior", idx)
        row["counterfactual_source"].update(
            {
                "shuffled_from_row_index": donor_idx,
                "shuffled_from_row_id": row_id(donor, donor_idx),
                "shuffled_from_subject_id": subject_id(donor),
                "shuffled_from_split": row_split(donor),
                "shuffled_from_prior_study_id": str(donor.get("prior_study_id") or ""),
                "shuffled_from_prior_study_datetime": str(donor.get("prior_study_datetime") or ""),
                "target_subject_id": subject_id(source),
                "target_split": row_split(source),
            }
        )
        out.append(row)
    return out


def build_matched_shuffled_prior(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    match_split: bool = True,
) -> List[Dict[str, Any]]:
    donor_map = choose_matched_shuffled_donors(rows, seed, match_split=match_split)
    out: List[Dict[str, Any]] = []
    for idx, source in enumerate(rows):
        donor_idx = donor_map[idx]
        donor = rows[donor_idx]
        target_findings = reference_positive_findings(source)
        donor_findings = reference_positive_findings(donor)
        row = dict(source)
        row["prior_study_id"] = str(donor.get("prior_study_id") or "")
        row["prior_image_paths"] = as_list(donor.get("prior_image_paths") or donor.get("prior_image_path"))
        row["prior_image_path"] = row["prior_image_paths"][0] if row["prior_image_paths"] else ""
        row["prior_report"] = prior_report_text(donor)
        row["context"] = row["prior_report"]
        row["prior_study_datetime"] = str(donor.get("prior_study_datetime") or "")
        row["temporal_gap_days"] = None
        annotate_common(row, "matched_shuffled_prior", idx)
        row["counterfactual_source"].update(
            {
                "match_strategy": "same_split_reference_finding_signature_prior_length",
                "reference_conditioned_control": True,
                "analysis_only_control": True,
                "matched_from_row_index": donor_idx,
                "matched_from_row_id": row_id(donor, donor_idx),
                "matched_from_subject_id": subject_id(donor),
                "matched_from_split": row_split(donor),
                "matched_from_prior_study_id": str(donor.get("prior_study_id") or ""),
                "matched_from_prior_study_datetime": str(donor.get("prior_study_datetime") or ""),
                "target_subject_id": subject_id(source),
                "target_split": row_split(source),
                "target_reference_findings": target_findings,
                "donor_reference_findings": donor_findings,
                "reference_finding_overlap": len(set(target_findings) & set(donor_findings)),
                "reference_finding_symmetric_difference": len(set(target_findings) ^ set(donor_findings)),
                "prior_report_token_delta": abs(
                    len(tokenize(prior_report_text(source))) - len(tokenize(prior_report_text(donor)))
                ),
            }
        )
        out.append(row)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "with_prior_images": sum(1 for row in rows if as_list(row.get("prior_image_paths") or row.get("prior_image_path"))),
        "with_prior_report": sum(1 for row in rows if prior_report_text(row)),
        "with_context": sum(1 for row in rows if str(row.get("context") or "").strip()),
    }


def summarize_shuffled(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mismatched_subjects = 0
    matched_splits = 0
    cross_split = 0
    donor_prior_datetime_matches = 0
    temporal_gap_null_rows = 0
    for row in rows:
        info = row.get("counterfactual_source") if isinstance(row, dict) else None
        target = str((info or {}).get("target_subject_id") or "")
        donor = str((info or {}).get("shuffled_from_subject_id") or "")
        target_split = str((info or {}).get("target_split") or "")
        donor_split = str((info or {}).get("shuffled_from_split") or "")
        if target and donor and target != donor:
            mismatched_subjects += 1
        if target_split and donor_split and target_split == donor_split:
            matched_splits += 1
        if target_split and donor_split and target_split != donor_split:
            cross_split += 1
        if str(row.get("prior_study_datetime") or "") == str(
            (info or {}).get("shuffled_from_prior_study_datetime") or ""
        ):
            donor_prior_datetime_matches += 1
        if row.get("temporal_gap_days") is None:
            temporal_gap_null_rows += 1
    return {
        "subject_mismatch_rows": mismatched_subjects,
        "split_match_rows": matched_splits,
        "cross_split_rows": cross_split,
        "donor_prior_datetime_match_rows": donor_prior_datetime_matches,
        "temporal_gap_null_rows": temporal_gap_null_rows,
    }


def summarize_matched_shuffled(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mismatched_subjects = 0
    matched_splits = 0
    cross_split = 0
    donor_prior_datetime_matches = 0
    temporal_gap_null_rows = 0
    diffs: List[int] = []
    overlaps: List[int] = []
    length_deltas: List[int] = []
    for row in rows:
        info = row.get("counterfactual_source") if isinstance(row, dict) else {}
        target = str((info or {}).get("target_subject_id") or "")
        donor = str((info or {}).get("matched_from_subject_id") or "")
        target_split = str((info or {}).get("target_split") or "")
        donor_split = str((info or {}).get("matched_from_split") or "")
        if target and donor and target != donor:
            mismatched_subjects += 1
        if target_split and donor_split and target_split == donor_split:
            matched_splits += 1
        if target_split and donor_split and target_split != donor_split:
            cross_split += 1
        if str(row.get("prior_study_datetime") or "") == str(
            (info or {}).get("matched_from_prior_study_datetime") or ""
        ):
            donor_prior_datetime_matches += 1
        if row.get("temporal_gap_days") is None:
            temporal_gap_null_rows += 1
        diffs.append(int((info or {}).get("reference_finding_symmetric_difference") or 0))
        overlaps.append(int((info or {}).get("reference_finding_overlap") or 0))
        length_deltas.append(int((info or {}).get("prior_report_token_delta") or 0))
    base = {
        "subject_mismatch_rows": mismatched_subjects,
        "split_match_rows": matched_splits,
        "cross_split_rows": cross_split,
        "donor_prior_datetime_match_rows": donor_prior_datetime_matches,
        "temporal_gap_null_rows": temporal_gap_null_rows,
    }
    if rows:
        base.update(
            {
                "mean_reference_finding_symmetric_difference": sum(diffs) / len(rows),
                "mean_reference_finding_overlap": sum(overlaps) / len(rows),
                "mean_prior_report_token_delta": sum(length_deltas) / len(rows),
            }
        )
    return base


def materialize_arms(
    rows: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    seed: int,
    match_shuffled_split: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for arm in arms:
        if arm == "current_only":
            out[arm] = build_current_only(rows)
        elif arm == "real_prior":
            out[arm] = build_real_prior(rows)
        elif arm == "shuffled_prior":
            out[arm] = build_shuffled_prior(rows, seed=seed, match_split=match_shuffled_split)
        elif arm == "matched_shuffled_prior":
            out[arm] = build_matched_shuffled_prior(rows, seed=seed, match_split=match_shuffled_split)
        elif arm == "prior_report_only":
            out[arm] = build_prior_report_only(rows)
        else:
            raise RuntimeError(f"Unsupported arm: {arm}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help=(
            "Comma-separated arm list. Supported: "
            "current_only,real_prior,shuffled_prior,matched_shuffled_prior,prior_report_only"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cross-split-shuffle",
        action="store_true",
        help="Allow shuffled prior donors from a different official split. Disabled by default.",
    )
    parser.add_argument(
        "--allow-reference-conditioned-matching",
        action="store_true",
        help=(
            "Required for matched_shuffled_prior. This arm uses target reference findings "
            "to choose a wrong-patient donor, so it is an analysis-only stress test."
        ),
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.source_manifest.is_file():
        raise SystemExit(f"ERROR: source manifest not found: {args.source_manifest}")

    rows = load_rows(args.source_manifest)
    requested_arms = [item.strip() for item in str(args.arms).split(",") if item.strip()]
    if not requested_arms:
        raise SystemExit("ERROR: no arms requested")
    invalid = [arm for arm in requested_arms if arm not in ARM_NAMES]
    if invalid:
        raise SystemExit(f"ERROR: unsupported arms requested: {','.join(invalid)}")
    if "matched_shuffled_prior" in requested_arms and not args.allow_reference_conditioned_matching:
        raise SystemExit(
            "ERROR: matched_shuffled_prior is reference-conditioned and analysis-only. "
            "Pass --allow-reference-conditioned-matching to materialize it explicitly."
        )

    outputs = materialize_arms(
        rows,
        requested_arms,
        seed=args.seed,
        match_shuffled_split=not args.allow_cross_split_shuffle,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix.strip() or args.source_manifest.stem
    source_sha = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()

    arm_files: Dict[str, str] = {}
    arm_stats: Dict[str, Any] = {}
    for arm, arm_rows in outputs.items():
        out_path = args.output_dir / f"{prefix}_{arm}.json"
        out_path.write_text(json.dumps(arm_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        arm_files[arm] = str(out_path)
        stats = summarize(arm_rows)
        if arm == "shuffled_prior":
            stats.update(summarize_shuffled(arm_rows))
        if arm == "matched_shuffled_prior":
            stats.update(summarize_matched_shuffled(arm_rows))
        arm_stats[arm] = stats

    summary = {
        "generated_utc": utc_now_iso(),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": source_sha,
        "seed": int(args.seed),
        "match_shuffled_split": not args.allow_cross_split_shuffle,
        "reference_conditioned_matching": bool("matched_shuffled_prior" in requested_arms),
        "analysis_only_arms": [
            arm for arm in requested_arms if arm == "matched_shuffled_prior"
        ],
        "arms": requested_arms,
        "arm_files": arm_files,
        "arm_stats": arm_stats,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    summary_path = args.summary_output or (args.output_dir / f"{prefix}_counterfactual_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

