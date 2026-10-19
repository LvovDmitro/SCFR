#!/usr/bin/env python3
"""Gate source-causal corrector outputs against uncorrected draft arms."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import (
        bootstrap_delta,
        index_arm_rows,
        load_rows,
        parse_arm,
        row_scores,
        sample_key,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from analyze_source_causal_arms import (  # type: ignore
        bootstrap_delta,
        index_arm_rows,
        load_rows,
        parse_arm,
        row_scores,
        sample_key,
    )


METRICS = (
    "objective",
    "finding_f1",
    "finding_status_accuracy",
    "finding_fp_count",
    "finding_fn_count",
    "token_f1",
    "prediction_tokens",
)


def objective(scores: Mapping[str, float], *, status_weight: float, fp_weight: float, fn_weight: float) -> float:
    return (
        float(scores.get("finding_f1", 0.0))
        + status_weight * float(scores.get("finding_status_accuracy", 0.0))
        - fp_weight * float(scores.get("finding_fp_count", 0.0))
        - fn_weight * float(scores.get("finding_fn_count", 0.0))
    )


def scored_row(
    row: Mapping[str, Any],
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, float]:
    scores = dict(row_scores(row))
    scores["objective"] = objective(
        scores,
        status_weight=status_weight,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    return scores


def mean_metric(scores: Sequence[Mapping[str, float]], metric: str) -> float:
    return statistics.mean(float(row.get(metric, 0.0)) for row in scores) if scores else 0.0


def reference_mismatches(
    baseline_rows: Mapping[str, Mapping[str, Any]],
    corrected_rows: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> int:
    mismatches = 0
    for sample_id in ids:
        left = " ".join(str(baseline_rows[sample_id].get("reference") or "").split())
        right = " ".join(str(corrected_rows[sample_id].get("reference") or "").split())
        mismatches += int(left != right)
    return mismatches


def arm_report(
    *,
    arm: str,
    baseline_rows: Mapping[str, Mapping[str, Any]],
    corrected_rows: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    baseline_ids = sorted(baseline_rows)
    corrected_ids = sorted(corrected_rows)
    ids = sorted(set(baseline_ids) & set(corrected_ids))
    corrected_texts = [str(corrected_rows[i].get("prediction") or "").strip() for i in ids]
    text_counts = Counter(text for text in corrected_texts if text)
    max_identical = max(text_counts.values()) if text_counts else 0

    baseline_scores = [
        scored_row(
            baseline_rows[i],
            status_weight=args.status_weight,
            fp_weight=args.fp_weight,
            fn_weight=args.fn_weight,
        )
        for i in ids
    ]
    corrected_scores = [
        scored_row(
            corrected_rows[i],
            status_weight=args.status_weight,
            fp_weight=args.fp_weight,
            fn_weight=args.fn_weight,
        )
        for i in ids
    ]
    metrics = {
        metric: {
            "baseline": mean_metric(baseline_scores, metric),
            "corrected": mean_metric(corrected_scores, metric),
            "delta": mean_metric(corrected_scores, metric) - mean_metric(baseline_scores, metric),
        }
        for metric in METRICS
    }

    cluster_labels: Optional[List[str]] = None
    if args.n_bootstrap > 0 and args.cluster_key:
        cluster_labels = []
        for sample_id in ids:
            value = corrected_rows[sample_id].get(args.cluster_key) or baseline_rows[sample_id].get(args.cluster_key)
            cluster_labels.append(str(value or sample_id))

    bootstrap = {}
    if args.n_bootstrap > 0:
        for metric in METRICS:
            bootstrap[metric] = bootstrap_delta(
                [row[metric] for row in corrected_scores],
                [row[metric] for row in baseline_scores],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                clusters=cluster_labels,
            )

    return {
        "arm": arm,
        "baseline_rows": len(baseline_ids),
        "corrected_rows": len(corrected_ids),
        "paired_rows": len(ids),
        "processed_rate": len(ids) / len(baseline_ids) if baseline_ids else 0.0,
        "nonempty_corrected_rows": sum(bool(text) for text in corrected_texts),
        "empty_corrected_rows": sum(not text for text in corrected_texts),
        "unique_corrected_predictions": len(set(corrected_texts)),
        "max_identical_corrected": max_identical,
        "max_identical_fraction": max_identical / len(ids) if ids else 0.0,
        "reference_mismatch_rows": reference_mismatches(baseline_rows, corrected_rows, ids),
        "metrics": metrics,
        "bootstrap": bootstrap,
    }


def build_gate(report: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    reasons: List[str] = []
    arm_reports = report["arms"]
    for arm, payload in arm_reports.items():
        if payload["paired_rows"] < args.min_samples:
            reasons.append(f"{arm}:paired_rows_below_min:{payload['paired_rows']}<{args.min_samples}")
        if payload["processed_rate"] < args.min_processed_rate:
            reasons.append(f"{arm}:processed_rate_below_min:{payload['processed_rate']:.4f}")
        if payload["empty_corrected_rows"] > args.max_empty_rows:
            reasons.append(f"{arm}:empty_corrected_rows:{payload['empty_corrected_rows']}")
        if payload["max_identical_fraction"] > args.max_identical_fraction:
            reasons.append(f"{arm}:collapse_fraction:{payload['max_identical_fraction']:.4f}")
        if payload["reference_mismatch_rows"]:
            reasons.append(f"{arm}:reference_mismatch:{payload['reference_mismatch_rows']}")

    for arm in args.wrong_arm:
        payload = arm_reports.get(arm)
        if payload is None:
            reasons.append(f"{arm}:missing_wrong_arm")
            continue
        delta = payload["metrics"]["objective"]["delta"]
        if delta <= args.min_wrong_objective_delta:
            reasons.append(f"{arm}:wrong_objective_delta_not_positive:{delta:.4f}")
        fp_delta = payload["metrics"]["finding_fp_count"]["delta"]
        if fp_delta >= args.max_wrong_fp_delta:
            reasons.append(f"{arm}:wrong_fp_delta_not_reduced:{fp_delta:.4f}")
        if args.require_ci_positive and payload.get("bootstrap"):
            ci = payload["bootstrap"].get("objective", {}).get("ci95", [None, None])
            if ci[0] is None or ci[0] <= args.min_wrong_objective_delta:
                reasons.append(f"{arm}:wrong_objective_ci_not_positive:{ci}")

    for arm in args.deployable_arm:
        payload = arm_reports.get(arm)
        if payload is None:
            reasons.append(f"{arm}:missing_deployable_arm")
            continue
        metrics = payload["metrics"]
        if metrics["objective"]["delta"] < -args.max_deployable_objective_drop:
            reasons.append(f"{arm}:deployable_objective_drop:{metrics['objective']['delta']:.4f}")
        if metrics["finding_f1"]["delta"] < -args.max_deployable_finding_f1_drop:
            reasons.append(f"{arm}:deployable_finding_f1_drop:{metrics['finding_f1']['delta']:.4f}")
        if metrics["token_f1"]["delta"] < -args.max_deployable_token_f1_drop:
            reasons.append(f"{arm}:deployable_token_f1_drop:{metrics['token_f1']['delta']:.4f}")
        if metrics["finding_fn_count"]["delta"] > args.max_deployable_fn_increase:
            reasons.append(f"{arm}:deployable_fn_increase:{metrics['finding_fn_count']['delta']:.4f}")

    return {
        "verdict": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "policy": {
            "min_samples": args.min_samples,
            "min_processed_rate": args.min_processed_rate,
            "max_empty_rows": args.max_empty_rows,
            "max_identical_fraction": args.max_identical_fraction,
            "min_wrong_objective_delta": args.min_wrong_objective_delta,
            "max_wrong_fp_delta": args.max_wrong_fp_delta,
            "max_deployable_objective_drop": args.max_deployable_objective_drop,
            "max_deployable_finding_f1_drop": args.max_deployable_finding_f1_drop,
            "max_deployable_token_f1_drop": args.max_deployable_token_f1_drop,
            "max_deployable_fn_increase": args.max_deployable_fn_increase,
            "require_ci_positive": args.require_ci_positive,
        },
    }


def build_report(
    baselines: Sequence[Tuple[str, Path]],
    corrected: Sequence[Tuple[str, Path]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    baseline_rows = {name: index_arm_rows(name, load_rows(path)) for name, path in baselines}
    corrected_rows = {name: index_arm_rows(name, load_rows(path)) for name, path in corrected}
    missing = sorted(set(baseline_rows) ^ set(corrected_rows))
    if missing:
        raise RuntimeError(f"Baseline/corrected arm mismatch: {missing}")
    arms = {
        arm: arm_report(
            arm=arm,
            baseline_rows=baseline_rows[arm],
            corrected_rows=corrected_rows[arm],
            args=args,
        )
        for arm in sorted(baseline_rows)
    }
    report: Dict[str, Any] = {
        "arms": arms,
        "objective": {
            "formula": "finding_f1 + status_weight*status_accuracy - fp_weight*FP - fn_weight*FN",
            "status_weight": args.status_weight,
            "fp_weight": args.fp_weight,
            "fn_weight": args.fn_weight,
        },
        "wrong_arms": list(args.wrong_arm),
        "deployable_arms": list(args.deployable_arm),
    }
    report["gate"] = build_gate(report, args)
    return report


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Corrector Gate",
        "",
        f"Verdict: `{report['gate']['verdict']}`",
        "",
        "## Arm Metrics",
        "",
        "| Arm | Rows | Empty | Collapse | Obj delta | Finding-F1 delta | FP delta | FN delta | Token-F1 delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, payload in report["arms"].items():
        metrics = payload["metrics"]
        lines.append(
            f"| `{arm}` | {payload['paired_rows']} | {payload['empty_corrected_rows']} | "
            f"{payload['max_identical_fraction']:.4f} | "
            f"{metrics['objective']['delta']:.4f} | "
            f"{metrics['finding_f1']['delta']:.4f} | "
            f"{metrics['finding_fp_count']['delta']:.4f} | "
            f"{metrics['finding_fn_count']['delta']:.4f} | "
            f"{metrics['token_f1']['delta']:.4f} |"
        )
    reasons = report["gate"].get("reasons", [])
    lines.extend(["", "## Gate Reasons", ""])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", type=parse_arm, required=True)
    parser.add_argument("--corrected", action="append", type=parse_arm, required=True)
    parser.add_argument("--wrong-arm", action="append", default=[])
    parser.add_argument("--deployable-arm", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--status-weight", type=float, default=0.25)
    parser.add_argument("--fp-weight", type=float, default=0.5)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-processed-rate", type=float, default=0.9)
    parser.add_argument("--max-empty-rows", type=int, default=0)
    parser.add_argument("--max-identical-fraction", type=float, default=0.05)
    parser.add_argument("--min-wrong-objective-delta", type=float, default=0.0)
    parser.add_argument("--max-wrong-fp-delta", type=float, default=0.0)
    parser.add_argument("--max-deployable-objective-drop", type=float, default=0.005)
    parser.add_argument("--max-deployable-finding-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-deployable-token-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-deployable-fn-increase", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cluster-key", default="subject_id")
    parser.add_argument("--require-ci-positive", action="store_true")
    args = parser.parse_args(argv)
    if not args.wrong_arm:
        args.wrong_arm = ["shuffled_prior", "matched_shuffled_prior"]
    if not args.deployable_arm:
        args.deployable_arm = ["current_only", "real_prior", "prior_report_only"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args.baseline, args.corrected, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    print(json.dumps({"gate": report["gate"]}, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

