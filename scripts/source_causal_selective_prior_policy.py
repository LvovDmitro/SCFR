#!/usr/bin/env python3
"""Evaluate selective prior use from a trained source-causal reranker.

The policy starts from a strong base arm (usually real_prior) and falls back to
current_only for the highest predicted prior-risk rows. The fallback fraction is
selected on a validation split and then reported on test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import row_scores
    from scripts.source_causal_candidate_reranker import (
        arm_prediction,
        candidate_features,
        dot,
        feature_names,
        load_rows,
        objective,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import row_scores
    from scripts.source_causal_candidate_reranker import (
        arm_prediction,
        candidate_features,
        dot,
        feature_names,
        load_rows,
        objective,
    )


def as_text(value: Any) -> str:
    return str(value or "").strip()


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def mean_metrics(scores: Sequence[Mapping[str, float]], *, fp_weight: float, fn_weight: float) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    out = {
        metric: statistics.mean(float(row.get(metric, 0.0)) for row in scores)
        for metric in metrics
    }
    out["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight) for score in scores
    )
    return out


def load_weights(reranker_json: Path, candidate_arms: Sequence[str]) -> List[float]:
    payload = json.loads(reranker_json.read_text(encoding="utf-8-sig"))
    weights = payload.get("feature_weights") or {}
    names = feature_names(candidate_arms)
    missing = [name for name in names if name not in weights]
    if missing:
        raise RuntimeError(f"Reranker weights missing features: {missing[:5]}")
    return [float(weights[name]) for name in names]


def split_rows(rows: Sequence[Mapping[str, Any]], split_field: str) -> Dict[str, List[Mapping[str, Any]]]:
    fixed: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        fixed.setdefault(as_text(row.get(split_field)) or "unspecified", []).append(row)
    fixed["all"] = list(rows)
    return fixed


def risk_score(
    row: Mapping[str, Any],
    *,
    weights: Sequence[float],
    candidate_arms: Sequence[str],
    base_arm: str,
    fallback_arm: str,
) -> float:
    base_features = candidate_features(row, base_arm, candidate_arms)
    fallback_features = candidate_features(row, fallback_arm, candidate_arms)
    return dot(weights, fallback_features) - dot(weights, base_features)


def evaluate_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float],
    candidate_arms: Sequence[str],
    base_arm: str,
    fallback_arm: str,
    fallback_fraction: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scored_rows: List[Tuple[float, Mapping[str, Any]]] = [
        (
            risk_score(
                row,
                weights=weights,
                candidate_arms=candidate_arms,
                base_arm=base_arm,
                fallback_arm=fallback_arm,
            ),
            row,
        )
        for row in rows
    ]
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    fallback_n = int(round(len(scored_rows) * fallback_fraction))
    fallback_ids = {id(row) for _score, row in scored_rows[:fallback_n]}
    scores: List[Dict[str, float]] = []
    selected_counts = {base_arm: 0, fallback_arm: 0}
    for _risk, row in scored_rows:
        arm = fallback_arm if id(row) in fallback_ids else base_arm
        selected_counts[arm] += 1
        scores.append(row_scores({"prediction": arm_prediction(row, arm), "reference": as_text(row.get("reference"))}))
    return {
        "rows": len(rows),
        "fallback_fraction": fallback_fraction,
        "fallback_n": fallback_n,
        "selected_counts": selected_counts,
        "metrics": mean_metrics(scores, fp_weight=fp_weight, fn_weight=fn_weight),
    }


def evaluate_fixed_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores = [
        row_scores({"prediction": arm_prediction(row, arm), "reference": as_text(row.get("reference"))})
        for row in rows
    ]
    return {
        "rows": len(rows),
        "selected_counts": {arm: len(rows)},
        "metrics": mean_metrics(scores, fp_weight=fp_weight, fn_weight=fn_weight),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Sequence[float],
    candidate_arms: Sequence[str],
    split_field: str,
    tune_split: str,
    eval_split: str,
    base_arm: str,
    fallback_arm: str,
    fractions: Sequence[float],
    fp_weight: float,
    fn_weight: float,
    min_samples: int,
) -> Dict[str, Any]:
    rows_by_split = split_rows(rows, split_field)
    tune_rows = rows_by_split.get(tune_split, [])
    eval_rows = rows_by_split.get(eval_split, [])
    if len(tune_rows) < min_samples:
        raise RuntimeError(f"tune split rows below min-samples gate: {len(tune_rows)} < {min_samples}")
    if len(eval_rows) < min_samples:
        raise RuntimeError(f"eval split rows below min-samples gate: {len(eval_rows)} < {min_samples}")

    tune_curve = [
        evaluate_fraction(
            tune_rows,
            weights=weights,
            candidate_arms=candidate_arms,
            base_arm=base_arm,
            fallback_arm=fallback_arm,
            fallback_fraction=fraction,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        for fraction in fractions
    ]
    best_tune = max(
        tune_curve,
        key=lambda item: (
            float(item["metrics"].get("objective", 0.0)),
            float(item["metrics"].get("finding_f1", 0.0)),
            -float(item["metrics"].get("finding_fp_count", 0.0)),
        ),
    )
    chosen_fraction = float(best_tune["fallback_fraction"])
    eval_curve = [
        evaluate_fraction(
            eval_rows,
            weights=weights,
            candidate_arms=candidate_arms,
            base_arm=base_arm,
            fallback_arm=fallback_arm,
            fallback_fraction=fraction,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        for fraction in fractions
    ]
    eval_selected = evaluate_fraction(
        eval_rows,
        weights=weights,
        candidate_arms=candidate_arms,
        base_arm=base_arm,
        fallback_arm=fallback_arm,
        fallback_fraction=chosen_fraction,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    return {
        "rows": len(rows),
        "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
        "candidate_arms": list(candidate_arms),
        "base_arm": base_arm,
        "fallback_arm": fallback_arm,
        "tune_split": tune_split,
        "eval_split": eval_split,
        "fractions": list(fractions),
        "objective": {
            "metric": "finding_f1 - fp_weight*finding_fp_count - fn_weight*finding_fn_count",
            "fp_weight": fp_weight,
            "fn_weight": fn_weight,
        },
        "baselines": {
            tune_split: {
                base_arm: evaluate_fixed_arm(tune_rows, arm=base_arm, fp_weight=fp_weight, fn_weight=fn_weight),
                fallback_arm: evaluate_fixed_arm(tune_rows, arm=fallback_arm, fp_weight=fp_weight, fn_weight=fn_weight),
            },
            eval_split: {
                base_arm: evaluate_fixed_arm(eval_rows, arm=base_arm, fp_weight=fp_weight, fn_weight=fn_weight),
                fallback_arm: evaluate_fixed_arm(eval_rows, arm=fallback_arm, fp_weight=fp_weight, fn_weight=fn_weight),
            },
        },
        "tune_curve": tune_curve,
        "best_tune": best_tune,
        "eval_curve": eval_curve,
        "eval_selected": eval_selected,
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Selective Prior Policy",
        "",
        f"Base arm: `{report['base_arm']}`",
        f"Fallback arm: `{report['fallback_arm']}`",
        f"Tune split: `{report['tune_split']}`",
        f"Eval split: `{report['eval_split']}`",
        f"Chosen fallback fraction: `{fmt(report['best_tune']['fallback_fraction'])}`",
        "",
        "## Baselines and Selected Policy",
        "",
        "| Split | Arm/Policy | Fallback Fraction | Token-F1 | Finding-F1 | Status Acc | FP | FN | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in (report["tune_split"], report["eval_split"]):
        for arm, payload in report["baselines"][split].items():
            metrics = payload["metrics"]
            lines.append(
                f"| `{split}` | `{arm}` | n/a | {fmt(metrics.get('token_f1'))} | "
                f"{fmt(metrics.get('finding_f1'))} | {fmt(metrics.get('finding_status_accuracy'))} | "
                f"{fmt(metrics.get('finding_fp_count'))} | {fmt(metrics.get('finding_fn_count'))} | "
                f"{fmt(metrics.get('objective'))} |"
            )
    selected_metrics = report["eval_selected"]["metrics"]
    lines.append(
        f"| `{report['eval_split']}` | `selective_policy` | {fmt(report['eval_selected']['fallback_fraction'])} | "
        f"{fmt(selected_metrics.get('token_f1'))} | {fmt(selected_metrics.get('finding_f1'))} | "
        f"{fmt(selected_metrics.get('finding_status_accuracy'))} | {fmt(selected_metrics.get('finding_fp_count'))} | "
        f"{fmt(selected_metrics.get('finding_fn_count'))} | {fmt(selected_metrics.get('objective'))} |"
    )
    lines.extend(["", "## Eval Curve", "", "| Fallback Fraction | Finding-F1 | Status Acc | FP | FN | Objective |", "|---:|---:|---:|---:|---:|---:|"])
    for payload in report["eval_curve"]:
        metrics = payload["metrics"]
        lines.append(
            f"| {fmt(payload['fallback_fraction'])} | {fmt(metrics.get('finding_f1'))} | "
            f"{fmt(metrics.get('finding_status_accuracy'))} | {fmt(metrics.get('finding_fp_count'))} | "
            f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('objective'))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reranker-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--candidate-arms", default="current_only,real_prior,prior_report_only")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--tune-split", default="validate")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--base-arm", default="real_prior")
    parser.add_argument("--fallback-arm", default="current_only")
    parser.add_argument("--fractions", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0")
    parser.add_argument("--fp-weight", type=float, default=0.1)
    parser.add_argument("--fn-weight", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    candidate_arms = parse_csv(args.candidate_arms)
    report = build_report(
        load_rows(args.predictions),
        weights=load_weights(args.reranker_json, candidate_arms),
        candidate_arms=candidate_arms,
        split_field=args.split_field,
        tune_split=args.tune_split,
        eval_split=args.eval_split,
        base_arm=args.base_arm,
        fallback_arm=args.fallback_arm,
        fractions=[float(item) for item in parse_csv(args.fractions)],
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        min_samples=args.min_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

