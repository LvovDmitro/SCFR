#!/usr/bin/env python3
"""Audit CheXpert-style finding coverage for a report-generation manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple


def open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def clean_study_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.lstrip("s")


def canonical_label(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"1", "1.0"}:
        return "positive"
    if text in {"-1", "-1.0"}:
        return "uncertain"
    if text in {"0", "0.0"}:
        return "negative"
    if not text:
        return "not_mentioned"
    return f"other:{text}"


def parse_key_value_counts(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected FINDING=COUNT, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty FINDING in: {raw}")
        out[key] = int(value)
    return out


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


def load_label_index(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    labels: Dict[str, Dict[str, str]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Label CSV has no header: {path}")
        findings = [name for name in reader.fieldnames if name not in {"subject_id", "study_id"}]
        for row in reader:
            study_id = clean_study_id(row.get("study_id"))
            if study_id:
                labels[study_id] = dict(row)
    return findings, labels


def gap_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "missing"
    if value <= 30:
        return "<=30d"
    if value <= 180:
        return "31-180d"
    if value <= 365:
        return "181-365d"
    if value <= 1095:
        return "1-3y"
    return ">3y"


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    findings: Sequence[str],
    labels: Mapping[str, Mapping[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    missing_current = 0
    missing_prior = 0
    split_counts: Dict[str, int] = defaultdict(int)
    blockers: List[str] = []

    for row in rows:
        split = str(row.get(args.split_field) or "missing").strip().lower() or "missing"
        split_counts[split] += 1
        if clean_study_id(row.get(args.current_study_id_field)) not in labels:
            missing_current += 1
        if clean_study_id(row.get(args.prior_study_id_field)) not in labels:
            missing_prior += 1

    finding_payload: Dict[str, Any] = {}
    for finding in findings:
        current = Counter()
        prior = Counter()
        per_split_positive: Dict[str, int] = defaultdict(int)
        per_split_positive_or_uncertain: Dict[str, int] = defaultdict(int)
        per_split_new_positive: Dict[str, int] = defaultdict(int)
        gap_positive: Dict[str, int] = defaultdict(int)

        for row in rows:
            split = str(row.get(args.split_field) or "missing").strip().lower() or "missing"
            current_label = labels.get(clean_study_id(row.get(args.current_study_id_field)), {})
            prior_label = labels.get(clean_study_id(row.get(args.prior_study_id_field)), {})
            cur = canonical_label(current_label.get(finding, ""))
            prev = canonical_label(prior_label.get(finding, ""))
            current[cur] += 1
            prior[prev] += 1
            if cur == "positive":
                per_split_positive[split] += 1
                gap_positive[gap_bucket(row.get("temporal_gap_days"))] += 1
                if prev != "positive":
                    per_split_new_positive[split] += 1
            if cur in {"positive", "uncertain"}:
                per_split_positive_or_uncertain[split] += 1

        finding_payload[finding] = {
            "current": dict(current),
            "prior": dict(prior),
            "current_positive": int(current["positive"]),
            "current_positive_or_uncertain": int(current["positive"] + current["uncertain"]),
            "prior_positive": int(prior["positive"]),
            "new_or_reappearing_positive_from_nonpositive_prior": int(sum(per_split_new_positive.values())),
            "per_split_current_positive": dict(sorted(per_split_positive.items())),
            "per_split_current_positive_or_uncertain": dict(
                sorted(per_split_positive_or_uncertain.items())
            ),
            "per_split_new_or_reappearing_positive_from_nonpositive_prior": dict(
                sorted(per_split_new_positive.items())
            ),
            "positive_temporal_gap_buckets": dict(sorted(gap_positive.items())),
        }

    if args.require_label_coverage:
        if missing_current:
            blockers.append("missing_current_label_rows")
        if missing_prior:
            blockers.append("missing_prior_label_rows")

    for finding, required in parse_key_value_counts(args.min_current_positive).items():
        actual = int(finding_payload.get(finding, {}).get("current_positive", 0))
        if actual < required:
            blockers.append(f"finding_{finding}_current_positive_below_min")
    for finding, required in parse_key_value_counts(args.min_current_positive_or_uncertain).items():
        actual = int(finding_payload.get(finding, {}).get("current_positive_or_uncertain", 0))
        if actual < required:
            blockers.append(f"finding_{finding}_current_positive_or_uncertain_below_min")

    return {
        "rows": len(rows),
        "missing_current_label_rows": missing_current,
        "missing_prior_label_rows": missing_prior,
        "split_counts": dict(sorted(split_counts.items())),
        "findings": finding_payload,
        "gate": {
            "blocked": bool(blockers),
            "blockers": blockers,
            "require_label_coverage": bool(args.require_label_coverage),
            "min_current_positive": parse_key_value_counts(args.min_current_positive),
            "min_current_positive_or_uncertain": parse_key_value_counts(
                args.min_current_positive_or_uncertain
            ),
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--current-study-id-field", default="study_id")
    parser.add_argument("--prior-study-id-field", default="prior_study_id")
    parser.add_argument("--require-label-coverage", action="store_true")
    parser.add_argument("--min-current-positive", action="append", default=[])
    parser.add_argument("--min-current-positive-or-uncertain", action="append", default=[])
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        raise SystemExit(f"ERROR: manifest not found: {args.manifest}")
    if not args.label_csv.is_file():
        raise SystemExit(f"ERROR: label CSV not found: {args.label_csv}")
    try:
        parse_key_value_counts(args.min_current_positive)
        parse_key_value_counts(args.min_current_positive_or_uncertain)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    rows = load_rows(args.manifest)
    findings, labels = load_label_index(args.label_csv)
    payload = {
        "manifest": str(args.manifest),
        "label_csv": str(args.label_csv),
        **audit_rows(rows, findings, labels, args),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_blocker and payload["gate"]["blocked"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

