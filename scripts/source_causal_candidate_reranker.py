#!/usr/bin/env python3
"""Train/evaluate a lightweight source-causal candidate reranker.

The reranker selects among already generated candidate arms without using the
reference report at inference time. Training uses source-valid held-out splits:
train is used for a structured perceptron update, validate/test are reported.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import finding_vector, reference_text, row_scores, tokenize
    from scripts.audit_prior_copy_temporal import extract_temporal_terms, prior_copy_precision
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_vector, reference_text, row_scores, tokenize
    from scripts.audit_prior_copy_temporal import extract_temporal_terms, prior_copy_precision


DEFAULT_CANDIDATE_ARMS = ("current_only", "real_prior", "prior_report_only")


@dataclass(frozen=True)
class Candidate:
    arm: str
    prediction: str
    features: List[float]
    true_score: Dict[str, float]
    utility: float


@dataclass(frozen=True)
class PreparedRow:
    row: Mapping[str, Any]
    split: str
    candidates: List[Candidate]


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


def arm_prediction(row: Mapping[str, Any], arm: str) -> str:
    value = row.get(arm)
    return value if isinstance(value, str) else ""


def positive_findings(text: str) -> set[str]:
    return {name for name, status in finding_vector(text).items() if status == "positive"}


def normalized_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left.lower(), right.lower()).ratio())


def signed_count(value: int, scale: float = 10.0) -> float:
    return float(value) / scale


def feature_names(candidate_arms: Sequence[str]) -> List[str]:
    names = [
        "bias",
        "prediction_tokens_log",
        "positive_findings",
        "temporal_terms",
        "prior_copy_precision",
        "similarity_to_prior_report",
        "similarity_to_current_prediction",
        "similarity_to_real_prediction",
        "similarity_to_prior_report_only_prediction",
        "added_positive_vs_current",
        "removed_positive_vs_current",
        "added_positive_vs_prior_report_text",
        "removed_positive_vs_prior_report_text",
        "positive_overlap_with_prior_report_text",
    ]
    names.extend(f"arm={arm}" for arm in candidate_arms)
    return names


def candidate_features(row: Mapping[str, Any], arm: str, candidate_arms: Sequence[str]) -> List[float]:
    prediction = arm_prediction(row, arm)
    prior_report = as_text(row.get("prior_report") or row.get("context"))
    current_prediction = arm_prediction(row, "current_only")
    real_prediction = arm_prediction(row, "real_prior")
    prior_report_only_prediction = arm_prediction(row, "prior_report_only")

    candidate_pos = positive_findings(prediction)
    current_pos = positive_findings(current_prediction)
    prior_report_pos = positive_findings(prior_report)
    token_count = len(tokenize(prediction))
    temporal_count = len(extract_temporal_terms(prediction))

    overlap_prior = len(candidate_pos & prior_report_pos)
    base = [
        1.0,
        math.log1p(token_count) / 5.0,
        signed_count(len(candidate_pos)),
        signed_count(temporal_count, scale=5.0),
        prior_copy_precision(prediction, prior_report),
        normalized_similarity(prediction, prior_report),
        normalized_similarity(prediction, current_prediction),
        normalized_similarity(prediction, real_prediction),
        normalized_similarity(prediction, prior_report_only_prediction),
        signed_count(len(candidate_pos - current_pos)),
        signed_count(len(current_pos - candidate_pos)),
        signed_count(len(candidate_pos - prior_report_pos)),
        signed_count(len(prior_report_pos - candidate_pos)),
        signed_count(overlap_prior),
    ]
    base.extend(1.0 if arm == item else 0.0 for item in candidate_arms)
    return base


def objective(score: Mapping[str, float], *, fp_weight: float, fn_weight: float) -> float:
    return (
        float(score.get("finding_f1", 0.0))
        - fp_weight * float(score.get("finding_fp_count", 0.0))
        - fn_weight * float(score.get("finding_fn_count", 0.0))
    )


def prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_arms: Sequence[str],
    split_field: str,
    fp_weight: float,
    fn_weight: float,
) -> List[PreparedRow]:
    prepared: List[PreparedRow] = []
    for row in rows:
        reference = reference_text(row)
        candidates: List[Candidate] = []
        for arm in candidate_arms:
            prediction = arm_prediction(row, arm)
            if not prediction:
                continue
            score = row_scores({"prediction": prediction, "reference": reference})
            candidates.append(
                Candidate(
                    arm=arm,
                    prediction=prediction,
                    features=candidate_features(row, arm, candidate_arms),
                    true_score=score,
                    utility=objective(score, fp_weight=fp_weight, fn_weight=fn_weight),
                )
            )
        if candidates:
            prepared.append(
                PreparedRow(
                    row=row,
                    split=as_text(row.get(split_field)) or "unspecified",
                    candidates=candidates,
                )
            )
    return prepared


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def argmax_index(values: Sequence[float]) -> int:
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def true_best_index(candidates: Sequence[Candidate]) -> int:
    return argmax_index([candidate.utility for candidate in candidates])


def predicted_best_index(candidates: Sequence[Candidate], weights: Sequence[float]) -> int:
    return argmax_index([dot(weights, candidate.features) for candidate in candidates])


def train_structured_perceptron(
    rows: Sequence[PreparedRow],
    *,
    feature_count: int,
    epochs: int,
    learning_rate: float,
    average_weights: bool,
) -> Dict[str, Any]:
    weights = [0.0] * feature_count
    summed = [0.0] * feature_count
    updates = 0
    steps = 0
    for _epoch in range(epochs):
        for prepared in rows:
            if len(prepared.candidates) < 2:
                continue
            truth_idx = true_best_index(prepared.candidates)
            pred_idx = predicted_best_index(prepared.candidates, weights)
            if pred_idx != truth_idx:
                truth_features = prepared.candidates[truth_idx].features
                pred_features = prepared.candidates[pred_idx].features
                for idx in range(feature_count):
                    weights[idx] += learning_rate * (truth_features[idx] - pred_features[idx])
                updates += 1
            if average_weights:
                for idx in range(feature_count):
                    summed[idx] += weights[idx]
                steps += 1
    final_weights = [value / steps for value in summed] if average_weights and steps else weights
    return {
        "weights": final_weights,
        "raw_weights": weights,
        "updates": updates,
        "steps": steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "average_weights": average_weights,
    }


def mean_metrics(scores: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    out = {
        metric: statistics.mean(float(row.get(metric, 0.0)) for row in scores)
        for metric in metrics
    }
    return out


def evaluate_fixed_arm(
    rows: Sequence[PreparedRow],
    *,
    arm: str,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores: List[Dict[str, float]] = []
    for prepared in rows:
        for candidate in prepared.candidates:
            if candidate.arm == arm:
                scores.append(candidate.true_score)
                break
    metrics = mean_metrics(scores)
    metrics["objective"] = (
        statistics.mean(objective(score, fp_weight=fp_weight, fn_weight=fn_weight) for score in scores)
        if scores
        else 0.0
    )
    return {
        "rows": len(scores),
        "selected_counts": {arm: len(scores)},
        "metrics": metrics,
    }


def evaluate_oracle(
    rows: Sequence[PreparedRow],
    *,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    selected_scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    for prepared in rows:
        candidate = prepared.candidates[true_best_index(prepared.candidates)]
        selected_scores.append(candidate.true_score)
        selected_counts[candidate.arm] += 1
    metrics = mean_metrics(selected_scores)
    metrics["objective"] = (
        statistics.mean(objective(score, fp_weight=fp_weight, fn_weight=fn_weight) for score in selected_scores)
        if selected_scores
        else 0.0
    )
    return {
        "rows": len(selected_scores),
        "selected_counts": dict(selected_counts),
        "metrics": metrics,
        "oracle_uses_reference": True,
    }


def evaluate_reranker(
    rows: Sequence[PreparedRow],
    *,
    weights: Sequence[float],
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    selected_scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    selected_rows: List[Dict[str, Any]] = []
    for prepared in rows:
        candidate = prepared.candidates[predicted_best_index(prepared.candidates, weights)]
        selected_scores.append(candidate.true_score)
        selected_counts[candidate.arm] += 1
        if emit_rows:
            selected_rows.append(
                {
                    **dict(prepared.row),
                    "selected_arm": candidate.arm,
                    "prediction": candidate.prediction,
                }
            )
    metrics = mean_metrics(selected_scores)
    metrics["objective"] = (
        statistics.mean(objective(score, fp_weight=fp_weight, fn_weight=fn_weight) for score in selected_scores)
        if selected_scores
        else 0.0
    )
    return (
        {
            "rows": len(selected_scores),
            "selected_counts": dict(selected_counts),
            "metrics": metrics,
        },
        selected_rows,
    )


def split_prepared(rows: Sequence[PreparedRow]) -> Dict[str, List[PreparedRow]]:
    out: Dict[str, List[PreparedRow]] = {}
    for row in rows:
        out.setdefault(row.split, []).append(row)
    out["all"] = list(rows)
    return out


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def eval_split_rows(rows_by_split: Mapping[str, List[PreparedRow]], split: str) -> List[PreparedRow]:
    if "+" not in split:
        return list(rows_by_split.get(split, []))
    rows: List[PreparedRow] = []
    for part in parse_csv(split.replace("+", ",")):
        rows.extend(rows_by_split.get(part, []))
    return rows


def eval_split_parts(split: str) -> List[str]:
    if "+" not in split:
        return [split]
    return parse_csv(split.replace("+", ","))


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_arms: Sequence[str],
    split_field: str,
    tune_split: str,
    eval_splits: Sequence[str],
    fp_weight: float,
    fn_weight: float,
    min_samples: int,
    epochs: int,
    learning_rate: float,
    average_weights: bool,
    allow_leaky_eval: bool,
    allow_underpowered_eval: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prepared = prepare_rows(
        rows,
        candidate_arms=candidate_arms,
        split_field=split_field,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    if len(prepared) < min_samples:
        raise RuntimeError(f"rows below min-samples gate: {len(prepared)} < {min_samples}")
    rows_by_split = split_prepared(prepared)
    train_rows = rows_by_split.get(tune_split, [])
    if len(train_rows) < min_samples:
        raise RuntimeError(f"tune split rows below min-samples gate: {len(train_rows)} < {min_samples}")

    names = feature_names(candidate_arms)
    training = train_structured_perceptron(
        train_rows,
        feature_count=len(names),
        epochs=epochs,
        learning_rate=learning_rate,
        average_weights=average_weights,
    )
    weights = training["weights"]

    metrics_by_split: Dict[str, Any] = {}
    split_warnings: Dict[str, List[str]] = {}
    selected_predictions: List[Dict[str, Any]] = []
    for split in eval_splits:
        warnings: List[str] = []
        parts = eval_split_parts(split)
        if split == "all":
            warnings.append("contains_tune_split")
        elif split == tune_split or tune_split in parts:
            warnings.append("is_tune_split")
        if warnings and not allow_leaky_eval:
            raise RuntimeError(
                f"leaky eval split requested without --allow-leaky-eval: {split} ({','.join(warnings)})"
            )
        split_rows = eval_split_rows(rows_by_split, split)
        if len(split_rows) < min_samples:
            warnings.append("below_min_samples")
            if not allow_underpowered_eval:
                raise RuntimeError(
                    f"eval split rows below min-samples gate: {split} {len(split_rows)} < {min_samples}"
                )
        payload: Dict[str, Any] = {}
        for arm in candidate_arms:
            payload[arm] = evaluate_fixed_arm(
                split_rows,
                arm=arm,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
        reranker_metrics, selected_rows = evaluate_reranker(
            split_rows,
            weights=weights,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=True,
        )
        payload["candidate_reranker"] = reranker_metrics
        payload["oracle"] = evaluate_oracle(
            split_rows,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        if split == "all":
            selected_predictions = selected_rows
        elif "all" not in eval_splits:
            selected_predictions.extend(selected_rows)
        metrics_by_split[split] = payload
        split_warnings[split] = warnings

    return (
        {
            "rows": len(prepared),
            "split_field": split_field,
            "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
            "candidate_arms": list(candidate_arms),
            "tune_split": tune_split,
            "eval_splits": list(eval_splits),
            "allow_leaky_eval": allow_leaky_eval,
            "allow_underpowered_eval": allow_underpowered_eval,
            "min_samples": min_samples,
            "objective": {
                "metric": "finding_f1 - fp_weight*finding_fp_count - fn_weight*finding_fn_count",
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "feature_names": names,
            "training": training,
            "feature_weights": dict(zip(names, weights)),
            "metrics_by_split": metrics_by_split,
            "split_warnings": split_warnings,
        },
        selected_predictions,
    )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Candidate Reranker",
        "",
        f"Rows: `{report['rows']}`",
        f"Candidate arms: `{', '.join(report['candidate_arms'])}`",
        f"Tune split: `{report['tune_split']}`",
        f"Training updates: `{report['training']['updates']}`",
        "",
        "| Split | Arm | Rows | Selected Counts | Token-F1 | Finding-F1 | Status Acc | FP | FN | Objective |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        warnings = report.get("split_warnings", {}).get(split) or []
        if warnings:
            lines.append(f"`{split}` warnings: {', '.join(warnings)}")
        for arm, result in payload.items():
            metrics = result.get("metrics") or {}
            selected_counts = result.get("selected_counts") or {}
            lines.append(
                f"| `{split}` | `{arm}` | {result.get('rows', 0)} | `{json.dumps(selected_counts, sort_keys=True)}` | "
                f"{fmt(metrics.get('token_f1'))} | {fmt(metrics.get('finding_f1'))} | "
                f"{fmt(metrics.get('finding_status_accuracy'))} | {fmt(metrics.get('finding_fp_count'))} | "
                f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('objective'))} |"
            )
    lines.extend(["", "## Top Feature Weights", ""])
    weights = sorted(
        report["feature_weights"].items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    lines.extend(f"- `{name}`: {fmt(weight)}" for name, weight in weights[:12])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--candidate-arms", default=",".join(DEFAULT_CANDIDATE_ARMS))
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--tune-split", default="train")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument("--allow-leaky-eval", action="store_true")
    parser.add_argument(
        "--allow-underpowered-eval",
        action="store_true",
        help="Allow eval splits below --min-samples. Use only for diagnostics, not stage-gate promotion.",
    )
    parser.add_argument("--fp-weight", type=float, default=0.1)
    parser.add_argument("--fn-weight", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--no-average-weights", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.predictions)
    report, selected_rows = build_report(
        rows,
        candidate_arms=parse_csv(args.candidate_arms),
        split_field=args.split_field,
        tune_split=args.tune_split,
        eval_splits=parse_csv(args.eval_splits),
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        min_samples=args.min_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        average_weights=not args.no_average_weights,
        allow_leaky_eval=bool(args.allow_leaky_eval),
        allow_underpowered_eval=bool(args.allow_underpowered_eval),
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

