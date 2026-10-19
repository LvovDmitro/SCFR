#!/usr/bin/env python3
"""Build a leakage-audited train/eval pool for source-causal meta-controllers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object row {idx}: {path}")
        rows.append(dict(row))
    return rows


def as_text(value: Any) -> str:
    return str(value or "").strip()


def sample_key(row: Mapping[str, Any], *, idx: int, source: str) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = as_text(row.get(key))
        if value:
            return value
    raise RuntimeError(f"{source}: missing sample key at row {idx}")


def subjects_from(paths: Iterable[Path]) -> Set[str]:
    out: Set[str] = set()
    for path in paths:
        for idx, row in enumerate(load_rows(path)):
            subject_id = as_text(row.get("subject_id"))
            if not subject_id:
                raise RuntimeError(f"{path}: missing subject_id in exclusion source at row {idx}")
            out.add(subject_id)
    return out


def split_allowed(row: Mapping[str, Any], allowed: Set[str]) -> bool:
    if not allowed:
        return True
    return as_text(row.get("split")) in allowed


def has_required_arms(row: Mapping[str, Any], arms: Sequence[str]) -> bool:
    return all(as_text(row.get(arm)) for arm in arms)


def add_rows(
    out: List[Dict[str, Any]],
    *,
    path: Path,
    target_split: str,
    original_splits: Set[str],
    candidate_arms: Sequence[str],
    exclude_subjects: Set[str],
    apply_subject_exclusion: bool,
    require_subject_id: bool,
) -> Dict[str, Any]:
    rows = load_rows(path)
    summary: Dict[str, Any] = {
        "path": str(path),
        "input_rows": len(rows),
        "kept_rows": 0,
        "dropped_split": 0,
        "dropped_subject_overlap": 0,
        "missing_subject_id": 0,
        "dropped_missing_candidate": 0,
    }
    for idx, row in enumerate(rows):
        if not split_allowed(row, original_splits):
            summary["dropped_split"] += 1
            continue
        subject_id = as_text(row.get("subject_id"))
        if require_subject_id and not subject_id:
            summary["missing_subject_id"] += 1
            raise RuntimeError(f"{path}: missing subject_id at row {idx}")
        if apply_subject_exclusion and subject_id in exclude_subjects:
            summary["dropped_subject_overlap"] += 1
            continue
        if not has_required_arms(row, candidate_arms):
            summary["dropped_missing_candidate"] += 1
            continue
        item = dict(row)
        item["split"] = target_split
        item["source_pool_path"] = str(path)
        out.append(item)
        summary["kept_rows"] += 1
    return summary


def duplicate_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for idx, row in enumerate(rows):
        key = sample_key(row, idx=idx, source="output")
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def build_pool(
    *,
    train_inputs: Sequence[Path],
    eval_inputs: Sequence[Path],
    exclude_subjects_from: Sequence[Path],
    train_original_splits: Set[str],
    eval_original_splits: Set[str],
    train_split_name: str,
    eval_split_name: str,
    candidate_arms: Sequence[str],
    require_subject_id: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    excluded_subjects = subjects_from(exclude_subjects_from)
    rows: List[Dict[str, Any]] = []
    train_summaries = [
        add_rows(
            rows,
            path=path,
            target_split=train_split_name,
            original_splits=train_original_splits,
            candidate_arms=candidate_arms,
            exclude_subjects=excluded_subjects,
            apply_subject_exclusion=True,
            require_subject_id=require_subject_id,
        )
        for path in train_inputs
    ]
    eval_summaries = [
        add_rows(
            rows,
            path=path,
            target_split=eval_split_name,
            original_splits=eval_original_splits,
            candidate_arms=candidate_arms,
            exclude_subjects=excluded_subjects,
            apply_subject_exclusion=False,
            require_subject_id=require_subject_id,
        )
        for path in eval_inputs
    ]
    duplicates = duplicate_keys(rows)
    if duplicates:
        preview = ",".join(duplicates[:10])
        raise RuntimeError(f"duplicate output sample keys: {preview}")
    split_counts: Dict[str, int] = {}
    subject_counts: Dict[str, Set[str]] = {}
    for row in rows:
        split = as_text(row.get("split")) or "unspecified"
        split_counts[split] = split_counts.get(split, 0) + 1
        subject_counts.setdefault(split, set()).add(as_text(row.get("subject_id")))
    summary = {
        "output_rows": len(rows),
        "candidate_arms": list(candidate_arms),
        "excluded_subjects": len(excluded_subjects),
        "train_inputs": train_summaries,
        "eval_inputs": eval_summaries,
        "split_counts": split_counts,
        "subject_counts": {split: len(items) for split, items in subject_counts.items()},
        "require_subject_id": require_subject_id,
    }
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", action="append", required=True, type=Path)
    parser.add_argument("--eval-input", action="append", required=True, type=Path)
    parser.add_argument("--exclude-subjects-from", action="append", default=[], type=Path)
    parser.add_argument("--train-original-splits", default="train,validate")
    parser.add_argument("--eval-original-splits", default="validate")
    parser.add_argument("--train-split-name", default="train")
    parser.add_argument("--eval-split-name", default="validate")
    parser.add_argument(
        "--candidate-arms",
        default="current_only,cand_fp02,cand_fp08,verifier_t0,verifier_t2",
    )
    parser.add_argument(
        "--allow-missing-subject-id",
        action="store_true",
        help="Diagnostic override only. By default, missing subject_id hard-fails leakage-audited pools.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows, summary = build_pool(
        train_inputs=args.train_input,
        eval_inputs=args.eval_input,
        exclude_subjects_from=args.exclude_subjects_from,
        train_original_splits=set(parse_csv(args.train_original_splits)),
        eval_original_splits=set(parse_csv(args.eval_original_splits)),
        train_split_name=args.train_split_name,
        eval_split_name=args.eval_split_name,
        candidate_arms=parse_csv(args.candidate_arms),
        require_subject_id=not bool(args.allow_missing_subject_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

