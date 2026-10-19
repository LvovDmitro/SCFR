#!/usr/bin/env python3
"""Decode structured reports from counterfactual source-causal agreement patterns."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from scripts.analyze_source_causal_arms import FINDING_PATTERNS, finding_vector, row_scores
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import FINDING_PATTERNS, finding_vector, row_scores


DEFAULT_ARMS = (
    "current_only",
    "real_prior",
    "shuffled_prior",
    "matched_shuffled_prior",
    "prior_report_only",
)

FINDING_PHRASES = {
    "atelectasis": "Atelectasis is present.",
    "cardiomegaly": "Cardiomegaly is present.",
    "consolidation": "Consolidation is present.",
    "edema": "Pulmonary edema is present.",
    "fracture": "Fracture is present.",
    "mass_or_nodule": "A pulmonary nodule or mass is present.",
    "opacity": "Pulmonary opacity is present.",
    "pleural_effusion": "Pleural effusion is present.",
    "pneumonia": "Pneumonia or aspiration is present.",
    "pneumothorax": "Pneumothorax is present.",
    "support_device": "A support device such as a central venous catheter or tube is present.",
}


def as_text(value: Any) -> str:
    return str(value or "").strip()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        out.append(dict(row))
    return out


def positive_set(text: str) -> Set[str]:
    return {finding for finding, status in finding_vector(text).items() if status == "positive"}


def arm_positive_sets(row: Mapping[str, Any], arms: Sequence[str]) -> Dict[str, Set[str]]:
    return {arm: positive_set(as_text(row.get(arm))) for arm in arms}


def decode_findings(row: Mapping[str, Any], policy: str, arms: Sequence[str]) -> Set[str]:
    positives = arm_positive_sets(row, arms)
    current = positives.get("current_only", set())
    real = positives.get("real_prior", set())
    shuffled = positives.get("shuffled_prior", set())
    matched = positives.get("matched_shuffled_prior", set())
    prior_only = positives.get("prior_report_only", set())
    wrong_history = shuffled | matched
    if policy == "current_structured":
        return set(current)
    if policy == "real_structured":
        return set(real)
    if policy == "causal_addition":
        safe_real_additions = {
            finding
            for finding in real - current
            if finding not in wrong_history and finding not in prior_only
        }
        return set(current | safe_real_additions)
    if policy == "wrong_history_veto":
        risky_real_additions = (wrong_history | prior_only) - current
        return set(real - risky_real_additions)
    if policy == "real_current_agreement":
        return set(real & current)
    if policy == "source_majority":
        selected: Set[str] = set()
        for finding in FINDING_PATTERNS:
            votes = int(finding in current) + int(finding in real) + int(finding in prior_only)
            wrong_votes = int(finding in shuffled) + int(finding in matched)
            if votes >= 2 and wrong_votes < 2:
                selected.add(finding)
        return selected
    raise ValueError(f"Unknown policy: {policy}")


def findings_to_report(findings: Set[str]) -> str:
    if not findings:
        return "No source-consensus positive findings."
    return " ".join(FINDING_PHRASES[finding] for finding in sorted(findings))


def mean_metrics(scores: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    return {
        metric: statistics.mean(float(row.get(metric, 0.0)) for row in scores)
        for metric in metrics
    }


def evaluate_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    arms: Sequence[str],
    emit_rows: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    decoded_rows: List[Dict[str, Any]] = []
    for row in rows:
        findings = decode_findings(row, policy, arms)
        prediction = findings_to_report(findings)
        selected_counts.update(findings)
        scores.append(row_scores({"prediction": prediction, "reference": as_text(row.get("reference"))}))
        if emit_rows:
            decoded_rows.append(
                {
                    **dict(row),
                    "consensus_policy": policy,
                    "consensus_findings": sorted(findings),
                    "prediction": prediction,
                }
            )
    return (
        {
            "rows": len(rows),
            "positive_finding_counts": dict(selected_counts),
            "metrics": mean_metrics(scores),
        },
        decoded_rows,
    )


def evaluate_arm(rows: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    scores = [
        row_scores({"prediction": as_text(row.get(arm)), "reference": as_text(row.get("reference"))})
        for row in rows
    ]
    return {"rows": len(rows), "metrics": mean_metrics(scores)}


def split_rows(rows: Sequence[Mapping[str, Any]], split_field: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        split = as_text(row.get(split_field)) or "unspecified"
        out.setdefault(split, []).append(row)
    out["all"] = list(rows)
    return out


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    policies: Sequence[str],
    split_field: str,
    eval_splits: Sequence[str],
    min_samples: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows_by_split = split_rows(rows, split_field)
    metrics_by_split: Dict[str, Any] = {}
    decoded_all: List[Dict[str, Any]] = []
    for split in eval_splits:
        split_rows_subset = rows_by_split.get(split, [])
        if len(split_rows_subset) < min_samples:
            raise RuntimeError(f"split rows below min-samples gate: {split} {len(split_rows_subset)} < {min_samples}")
        payload: Dict[str, Any] = {}
        for arm in arms:
            if any(as_text(row.get(arm)) for row in split_rows_subset):
                payload[arm] = evaluate_arm(split_rows_subset, arm)
        for policy in policies:
            metrics, decoded = evaluate_policy(
                split_rows_subset,
                policy=policy,
                arms=arms,
                emit_rows=True,
            )
            payload[policy] = metrics
            decoded_all.extend(decoded)
        metrics_by_split[split] = payload
    return (
        {
            "rows": len(rows),
            "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
            "arms": list(arms),
            "policies": list(policies),
            "eval_splits": list(eval_splits),
            "min_samples": min_samples,
            "metrics_by_split": metrics_by_split,
        },
        decoded_all,
    )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Consensus Decode",
        "",
        f"Rows: `{report['rows']}`",
        f"Eval splits: `{', '.join(report['eval_splits'])}`",
        "",
        "| Split | Arm/Policy | Token-F1 | Finding-F1 | Status Acc | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        for name, result in payload.items():
            metrics = result["metrics"]
            lines.append(
                f"| `{split}` | `{name}` | {fmt(metrics.get('token_f1'))} | "
                f"{fmt(metrics.get('finding_f1'))} | {fmt(metrics.get('finding_status_accuracy'))} | "
                f"{fmt(metrics.get('finding_fp_count'))} | {fmt(metrics.get('finding_fn_count'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument(
        "--policies",
        default="current_structured,real_structured,causal_addition,wrong_history_veto,real_current_agreement,source_majority",
    )
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument("--min-samples", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report, decoded_rows = build_report(
        load_rows(args.predictions),
        arms=parse_csv(args.arms),
        policies=parse_csv(args.policies),
        split_field=args.split_field,
        eval_splits=parse_csv(args.eval_splits),
        min_samples=args.min_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(decoded_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

