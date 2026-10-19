#!/usr/bin/env python3
"""Train/evaluate a per-finding prior-validity gate from source-causal arms."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import (
        FINDING_PATTERNS,
        finding_vector,
        reference_text,
        row_scores,
        token_f1,
        tokenize,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import (
        FINDING_PATTERNS,
        finding_vector,
        reference_text,
        row_scores,
        token_f1,
        tokenize,
    )


DEFAULT_ALL_ARMS = (
    "current_only",
    "real_prior",
    "prior_report_only",
    "shuffled_prior",
    "matched_shuffled_prior",
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


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_grid(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_grid(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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


def arm_text(row: Mapping[str, Any], arm: str) -> str:
    value = row.get(arm)
    return value if isinstance(value, str) else ""


def positive_vector(text: str) -> Dict[str, bool]:
    return {
        finding: status == "positive"
        for finding, status in finding_vector(text).items()
    }


def prepare_row(row: Mapping[str, Any], arms: Sequence[str]) -> Dict[str, Any]:
    vectors = {arm: positive_vector(arm_text(row, arm)) for arm in arms}
    reference = reference_text(row)
    reference_status = finding_vector(reference)
    vectors["reference"] = {
        finding: status == "positive"
        for finding, status in reference_status.items()
    }
    return {
        "row": row,
        "split": as_text(row.get("split")) or "unspecified",
        "reference_text": reference,
        "vectors": vectors,
        "reference_status": reference_status,
    }


def split_prepared(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(as_text(row.get("split")) or "unspecified", []).append(row)
    out["all"] = list(rows)
    return out


def pattern_for(vectors: Mapping[str, Mapping[str, bool]], finding: str, feature_arms: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(bool(vectors.get(arm, {}).get(finding, False))) for arm in feature_arms)


def fit_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_arms: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    table: Dict[str, Dict[str, Any]] = {}
    for prepared in rows:
        vectors = prepared["vectors"]
        for finding in FINDING_PATTERNS:
            pattern = pattern_for(vectors, finding, feature_arms)
            key = json.dumps(pattern)
            bucket = table.setdefault(finding, {}).setdefault(key, {"total": 0, "positive": 0})
            bucket["total"] += 1
            bucket["positive"] += int(bool(vectors["reference"].get(finding, False)))
    return table


def fallback_decision(
    vectors: Mapping[str, Mapping[str, bool]],
    finding: str,
    *,
    fallback: str,
    wrong_arms: Sequence[str],
) -> bool:
    current = bool(vectors.get("current_only", {}).get(finding, False))
    real = bool(vectors.get("real_prior", {}).get(finding, False))
    prior_only = bool(vectors.get("prior_report_only", {}).get(finding, False))
    wrong_count = sum(int(bool(vectors.get(arm, {}).get(finding, False))) for arm in wrong_arms)
    if fallback == "real":
        return real
    if fallback == "current":
        return current
    if fallback == "current_or_real":
        return current or real
    if fallback == "real_without_wrong":
        return real and wrong_count == 0
    if fallback == "current_or_real_without_wrong":
        return current or (real and wrong_count == 0)
    if fallback == "current_or_clean_real":
        return current or (real and wrong_count == 0 and not prior_only)
    if fallback == "source_majority":
        votes = int(current) + int(real) + int(prior_only)
        return votes >= 2 and wrong_count < 2
    raise ValueError(f"Unknown fallback: {fallback}")


def predict_findings(
    vectors: Mapping[str, Mapping[str, bool]],
    *,
    table: Mapping[str, Mapping[str, Any]],
    feature_arms: Sequence[str],
    threshold: float,
    min_support: int,
    fallback: str,
    wrong_arms: Sequence[str],
) -> List[str]:
    selected: List[str] = []
    for finding in FINDING_PATTERNS:
        pattern = pattern_for(vectors, finding, feature_arms)
        bucket = table.get(finding, {}).get(json.dumps(pattern))
        if bucket and int(bucket.get("total", 0)) >= min_support:
            rate = float(bucket.get("positive", 0)) / float(bucket.get("total", 1))
            use_positive = rate >= threshold
        else:
            use_positive = fallback_decision(
                vectors,
                finding,
                fallback=fallback,
                wrong_arms=wrong_arms,
            )
        if use_positive:
            selected.append(finding)
    return selected


def findings_to_report(findings: Sequence[str]) -> str:
    if not findings:
        return "No source-valid positive findings."
    return " ".join(FINDING_PHRASES[finding] for finding in sorted(findings))


def objective(score: Mapping[str, float], *, fp_weight: float, fn_weight: float) -> float:
    return (
        float(score.get("finding_f1", 0.0))
        - fp_weight * float(score.get("finding_fp_count", 0.0))
        - fn_weight * float(score.get("finding_fn_count", 0.0))
    )


def mean_metrics(scores: Sequence[Mapping[str, float]], *, fp_weight: float, fn_weight: float) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    out = {
        metric: statistics.mean(float(score.get(metric, 0.0)) for score in scores)
        for metric in metrics
    }
    out["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight)
        for score in scores
    )
    return out


def score_findings(
    findings: Sequence[str],
    *,
    reference_text: str,
    reference_status: Mapping[str, str],
) -> Dict[str, float]:
    selected = set(findings)
    ref_pos = {
        finding
        for finding, status in reference_status.items()
        if status == "positive"
    }
    if not selected and not ref_pos:
        finding_f1 = 1.0
    elif not selected or not ref_pos:
        finding_f1 = 0.0
    else:
        overlap = len(selected & ref_pos)
        precision = overlap / len(selected)
        recall = overlap / len(ref_pos)
        finding_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    status_correct = 0
    for finding in FINDING_PATTERNS:
        pred_status = "positive" if finding in selected else "absent"
        if pred_status == reference_status.get(finding, "absent"):
            status_correct += 1
    prediction = findings_to_report(findings)
    return {
        "token_f1": token_f1(prediction, reference_text),
        "finding_f1": finding_f1,
        "finding_status_accuracy": status_correct / len(FINDING_PATTERNS),
        "finding_fp_count": float(len(selected - ref_pos)),
        "finding_fn_count": float(len(ref_pos - selected)),
        "prediction_tokens": float(len(tokenize(prediction))),
    }


def evaluate_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: Mapping[str, Mapping[str, Any]],
    feature_arms: Sequence[str],
    threshold: float,
    min_support: int,
    fallback: str,
    wrong_arms: Sequence[str],
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    selected_rows: List[Dict[str, Any]] = []
    for prepared in rows:
        findings = predict_findings(
            prepared["vectors"],
            table=table,
            feature_arms=feature_arms,
            threshold=threshold,
            min_support=min_support,
            fallback=fallback,
            wrong_arms=wrong_arms,
        )
        selected_counts.update(findings)
        prediction = findings_to_report(findings)
        scores.append(
            score_findings(
                findings,
                reference_text=as_text(prepared.get("reference_text")),
                reference_status=prepared["reference_status"],
            )
        )
        if emit_rows:
            selected_rows.append(
                {
                    **dict(prepared["row"]),
                    "selected_findings": sorted(findings),
                    "prediction": prediction,
                    "per_finding_gate": {
                        "threshold": threshold,
                        "min_support": min_support,
                        "fallback": fallback,
                        "feature_arms": list(feature_arms),
                    },
                }
            )
    return (
        {
            "rows": len(rows),
            "selected_finding_counts": dict(selected_counts),
            "metrics": mean_metrics(scores, fp_weight=fp_weight, fn_weight=fn_weight),
        },
        selected_rows,
    )


def evaluate_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores = [
        row_scores(
            {
                "prediction": arm_text(prepared["row"], arm),
                "reference": as_text(prepared.get("reference_text")),
            }
        )
        for prepared in rows
    ]
    return {
        "rows": len(rows),
        "metrics": mean_metrics(scores, fp_weight=fp_weight, fn_weight=fn_weight),
    }


def tune_params(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: Mapping[str, Mapping[str, Any]],
    feature_arms: Sequence[str],
    threshold_grid: Sequence[float],
    min_support_grid: Sequence[int],
    fallbacks: Sequence[str],
    wrong_arms: Sequence[str],
    fp_weight: float,
    fn_weight: float,
    baseline_metrics: Optional[Mapping[str, float]] = None,
    max_fp_delta: Optional[float] = None,
    max_fn_delta: Optional[float] = None,
    min_finding_f1_delta: Optional[float] = None,
    min_status_accuracy_delta: Optional[float] = None,
    min_objective_delta: Optional[float] = None,
) -> Dict[str, Any]:
    best_key: Optional[Tuple[int, float, float, float, float]] = None
    best: Optional[Dict[str, Any]] = None
    trials: List[Dict[str, Any]] = []
    satisfying_trials = 0
    for threshold in threshold_grid:
        for min_support in min_support_grid:
            for fallback in fallbacks:
                metrics, _rows = evaluate_gate(
                    rows,
                    table=table,
                    feature_arms=feature_arms,
                    threshold=threshold,
                    min_support=min_support,
                    fallback=fallback,
                    wrong_arms=wrong_arms,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                    emit_rows=False,
                )
                score = metrics["metrics"]
                constraint = constraint_status(
                    score,
                    baseline_metrics=baseline_metrics,
                    max_fp_delta=max_fp_delta,
                    max_fn_delta=max_fn_delta,
                    min_finding_f1_delta=min_finding_f1_delta,
                    min_status_accuracy_delta=min_status_accuracy_delta,
                    min_objective_delta=min_objective_delta,
                )
                satisfying_trials += int(bool(constraint["satisfied"]))
                key = (
                    int(bool(constraint["satisfied"])),
                    float(score.get("objective", 0.0)),
                    float(score.get("finding_f1", 0.0)),
                    float(score.get("finding_status_accuracy", 0.0)),
                    -float(score.get("finding_fp_count", 0.0)),
                )
                trial = {
                    "threshold": threshold,
                    "min_support": min_support,
                    "fallback": fallback,
                    "constraint": constraint,
                    **metrics,
                }
                trials.append(trial)
                if best_key is None or key > best_key:
                    best_key = key
                    best = trial
    if best is None:
        raise RuntimeError("No tuning candidates evaluated")
    return {
        "best": best,
        "n_trials": len(trials),
        "satisfying_trials": satisfying_trials,
        "constraints": {
            "baseline_metrics": dict(baseline_metrics or {}),
            "max_fp_delta": max_fp_delta,
            "max_fn_delta": max_fn_delta,
            "min_finding_f1_delta": min_finding_f1_delta,
            "min_status_accuracy_delta": min_status_accuracy_delta,
            "min_objective_delta": min_objective_delta,
        },
    }


def constraint_status(
    metrics: Mapping[str, float],
    *,
    baseline_metrics: Optional[Mapping[str, float]],
    max_fp_delta: Optional[float],
    max_fn_delta: Optional[float],
    min_finding_f1_delta: Optional[float],
    min_status_accuracy_delta: Optional[float],
    min_objective_delta: Optional[float],
) -> Dict[str, Any]:
    if not baseline_metrics:
        return {"satisfied": True, "reasons": [], "deltas": {}}
    deltas = {
        "finding_f1": float(metrics.get("finding_f1", 0.0)) - float(baseline_metrics.get("finding_f1", 0.0)),
        "finding_status_accuracy": float(metrics.get("finding_status_accuracy", 0.0))
        - float(baseline_metrics.get("finding_status_accuracy", 0.0)),
        "finding_fp_count": float(metrics.get("finding_fp_count", 0.0))
        - float(baseline_metrics.get("finding_fp_count", 0.0)),
        "finding_fn_count": float(metrics.get("finding_fn_count", 0.0))
        - float(baseline_metrics.get("finding_fn_count", 0.0)),
        "objective": float(metrics.get("objective", 0.0)) - float(baseline_metrics.get("objective", 0.0)),
    }
    reasons: List[str] = []
    if max_fp_delta is not None and deltas["finding_fp_count"] > max_fp_delta:
        reasons.append("fp_delta_above_max")
    if max_fn_delta is not None and deltas["finding_fn_count"] > max_fn_delta:
        reasons.append("fn_delta_above_max")
    if min_finding_f1_delta is not None and deltas["finding_f1"] < min_finding_f1_delta:
        reasons.append("finding_f1_delta_below_min")
    if min_status_accuracy_delta is not None and deltas["finding_status_accuracy"] < min_status_accuracy_delta:
        reasons.append("status_accuracy_delta_below_min")
    if min_objective_delta is not None and deltas["objective"] < min_objective_delta:
        reasons.append("objective_delta_below_min")
    return {
        "satisfied": not reasons,
        "reasons": reasons,
        "deltas": deltas,
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_arms: Sequence[str],
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    train_split: str,
    tune_split: str,
    eval_splits: Sequence[str],
    threshold_grid: Sequence[float],
    min_support_grid: Sequence[int],
    fallbacks: Sequence[str],
    fp_weight: float,
    fn_weight: float,
    min_samples: int,
    constraint_baseline_arm: Optional[str] = None,
    max_fp_delta: Optional[float] = None,
    max_fn_delta: Optional[float] = None,
    min_finding_f1_delta: Optional[float] = None,
    min_status_accuracy_delta: Optional[float] = None,
    min_objective_delta: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prepared = [prepare_row(row, all_arms) for row in rows]
    rows_by_split = split_prepared(prepared)
    train_rows = rows_by_split.get(train_split, [])
    tune_rows = rows_by_split.get(tune_split, [])
    if len(train_rows) < min_samples:
        raise RuntimeError(f"train split rows below min-samples gate: {len(train_rows)} < {min_samples}")
    if len(tune_rows) < min_samples:
        raise RuntimeError(f"tune split rows below min-samples gate: {len(tune_rows)} < {min_samples}")
    table = fit_table(train_rows, feature_arms=feature_arms)
    baseline_metrics = None
    if constraint_baseline_arm:
        baseline_metrics = evaluate_arm(
            tune_rows,
            arm=constraint_baseline_arm,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )["metrics"]
    tuning = tune_params(
        tune_rows,
        table=table,
        feature_arms=feature_arms,
        threshold_grid=threshold_grid,
        min_support_grid=min_support_grid,
        fallbacks=fallbacks,
        wrong_arms=wrong_arms,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
        baseline_metrics=baseline_metrics,
        max_fp_delta=max_fp_delta,
        max_fn_delta=max_fn_delta,
        min_finding_f1_delta=min_finding_f1_delta,
        min_status_accuracy_delta=min_status_accuracy_delta,
        min_objective_delta=min_objective_delta,
    )
    best = tuning["best"]

    metrics_by_split: Dict[str, Any] = {}
    selected_rows: List[Dict[str, Any]] = []
    for split in eval_splits:
        split_rows = rows_by_split.get(split, [])
        if len(split_rows) < min_samples:
            raise RuntimeError(f"eval split rows below min-samples gate: {split} {len(split_rows)} < {min_samples}")
        payload: Dict[str, Any] = {}
        for arm in all_arms:
            if any(arm_text(prepared_row["row"], arm) for prepared_row in split_rows):
                payload[arm] = evaluate_arm(
                    split_rows,
                    arm=arm,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                )
        gate_metrics, emitted = evaluate_gate(
            split_rows,
            table=table,
            feature_arms=feature_arms,
            threshold=float(best["threshold"]),
            min_support=int(best["min_support"]),
            fallback=str(best["fallback"]),
            wrong_arms=wrong_arms,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=True,
        )
        payload["per_finding_gate"] = gate_metrics
        metrics_by_split[split] = payload
        selected_rows.extend(emitted)

    return (
        {
            "rows": len(rows),
            "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
            "all_arms": list(all_arms),
            "feature_arms": list(feature_arms),
            "wrong_arms": list(wrong_arms),
            "train_split": train_split,
            "tune_split": tune_split,
            "eval_splits": list(eval_splits),
            "min_samples": min_samples,
            "objective": {
                "metric": "finding_f1 - fp_weight*finding_fp_count - fn_weight*finding_fn_count",
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "constraint_baseline_arm": constraint_baseline_arm,
            "tuning": tuning,
            "table_summary": {
                finding: {
                    "patterns": len(patterns),
                    "rows": sum(int(bucket["total"]) for bucket in patterns.values()),
                }
                for finding, patterns in table.items()
            },
            "metrics_by_split": metrics_by_split,
        },
        selected_rows,
    )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    best = report["tuning"]["best"]
    best_constraint = best.get("constraint") or {}
    lines = [
        "# Source-Causal Per-Finding Prior-Validity Gate",
        "",
        f"Rows: `{report['rows']}`",
        f"Train split: `{report['train_split']}`",
        f"Tune split: `{report['tune_split']}`",
        f"Feature arms: `{', '.join(report['feature_arms'])}`",
        (
            "Best params: "
            f"`threshold={fmt(best['threshold'])}`, "
            f"`min_support={best['min_support']}`, "
            f"`fallback={best['fallback']}`"
        ),
        f"Constraint baseline: `{report.get('constraint_baseline_arm') or 'none'}`",
        f"Best satisfies constraints: `{best_constraint.get('satisfied', True)}`",
        "",
        "| Split | Arm/Policy | Finding-F1 | Status Acc | FP | FN | Token-F1 | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        for name, result in payload.items():
            metrics = result["metrics"]
            lines.append(
                f"| `{split}` | `{name}` | {fmt(metrics.get('finding_f1'))} | "
                f"{fmt(metrics.get('finding_status_accuracy'))} | {fmt(metrics.get('finding_fp_count'))} | "
                f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('token_f1'))} | "
                f"{fmt(metrics.get('objective'))} |"
            )
    lines.extend(
        [
            "",
            "## Tuning",
            "",
            f"Trials: `{report['tuning']['n_trials']}`",
            f"Satisfying trials: `{report['tuning'].get('satisfying_trials', 0)}`",
            f"Best constraint reasons: `{', '.join(best_constraint.get('reasons') or []) or 'none'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--all-arms", default=",".join(DEFAULT_ALL_ARMS))
    parser.add_argument("--feature-arms", default="current_only,real_prior,prior_report_only,shuffled_prior,matched_shuffled_prior")
    parser.add_argument("--wrong-arms", default="shuffled_prior,matched_shuffled_prior")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="validate")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument("--threshold-grid", default="0.1,0.25,0.4,0.5,0.6,0.75,0.9")
    parser.add_argument("--min-support-grid", default="1,2,3,5,10")
    parser.add_argument(
        "--fallbacks",
        default="real,current,current_or_real,real_without_wrong,current_or_real_without_wrong,current_or_clean_real,source_majority",
    )
    parser.add_argument("--fp-weight", type=float, default=0.1)
    parser.add_argument("--fn-weight", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--constraint-baseline-arm", default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--max-fn-delta", type=float, default=None)
    parser.add_argument("--min-finding-f1-delta", type=float, default=None)
    parser.add_argument("--min-status-accuracy-delta", type=float, default=None)
    parser.add_argument("--min-objective-delta", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report, selected_rows = build_report(
        load_rows(args.predictions),
        all_arms=parse_csv(args.all_arms),
        feature_arms=parse_csv(args.feature_arms),
        wrong_arms=parse_csv(args.wrong_arms),
        train_split=args.train_split,
        tune_split=args.tune_split,
        eval_splits=parse_csv(args.eval_splits),
        threshold_grid=parse_float_grid(args.threshold_grid),
        min_support_grid=parse_int_grid(args.min_support_grid),
        fallbacks=parse_csv(args.fallbacks),
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        min_samples=args.min_samples,
        constraint_baseline_arm=args.constraint_baseline_arm,
        max_fp_delta=args.max_fp_delta,
        max_fn_delta=args.max_fn_delta,
        min_finding_f1_delta=args.min_finding_f1_delta,
        min_status_accuracy_delta=args.min_status_accuracy_delta,
        min_objective_delta=args.min_objective_delta,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

