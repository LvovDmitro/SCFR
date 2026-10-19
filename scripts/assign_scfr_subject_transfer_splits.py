#!/usr/bin/env python3
"""Assign deterministic, subject-disjoint component/meta/test transfer splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _fold(subject_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{subject_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--test-fold",
        type=int,
        choices=range(5),
        default=4,
        help="Outer fold reserved for test (default reproduces the original split).",
    )
    parser.add_argument(
        "--meta-fold",
        type=int,
        choices=range(5),
        default=3,
        help="Outer fold reserved for meta-controller/gating (default reproduces the original split).",
    )
    args = parser.parse_args()
    if args.test_fold == args.meta_fold:
        parser.error("--test-fold and --meta-fold must differ")

    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise RuntimeError("Expected a JSON list")
    split_subjects: dict[str, set[str]] = defaultdict(set)
    split_rows: Counter[str] = Counter()
    output = []
    for index, source_row in enumerate(rows):
        if not isinstance(source_row, dict):
            raise RuntimeError(f"Row {index} is not an object")
        subject_id = str(source_row.get("subject_id") or "").strip()
        if not subject_id:
            raise RuntimeError(f"Row {index} has no subject_id")
        fold = _fold(subject_id, args.seed)
        if fold == args.test_fold:
            split = "test"
        elif fold == args.meta_fold:
            split = "meta_train"
        else:
            split = "component_train"
        row = dict(source_row)
        row["source_split"] = row.get("split")
        row["split"] = split
        row["cv_fold"] = fold
        output.append(row)
        split_rows[split] += 1
        split_subjects[split].add(subject_id)

    names = sorted(split_subjects)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = split_subjects[left] & split_subjects[right]
            if overlap:
                raise RuntimeError(f"Subject overlap between {left} and {right}: {len(overlap)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(output),
                "seed": args.seed,
                "test_fold": args.test_fold,
                "meta_fold": args.meta_fold,
                "split_rows": dict(sorted(split_rows.items())),
                "split_subjects": {
                    split: len(subjects)
                    for split, subjects in sorted(split_subjects.items())
                },
                "subject_overlap": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

