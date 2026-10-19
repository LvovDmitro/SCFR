#!/usr/bin/env python3
"""Train/evaluate a source-causal per-finding verifier.

The verifier uses generated source arms as evidence and predicts which findings
are source-valid for the final report. It trains only on the train split, tunes
model/thresholds on validate, and reports held-out splits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from sklearn.linear_model import LogisticRegression
except ModuleNotFoundError as exc:  # pragma: no cover - environment contract
    raise SystemExit("scikit-learn is required for source_causal_verifier.py") from exc

try:
    from scripts.source_causal_per_finding_gate import (
        DEFAULT_ALL_ARMS,
        FINDING_PATTERNS,
        FINDING_PHRASES,
        arm_text,
        as_text,
        evaluate_arm,
        load_rows,
        parse_csv,
        parse_float_grid,
        prepare_row,
        score_findings,
    )
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, split_sentences, tokenize
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.source_causal_per_finding_gate import (
        DEFAULT_ALL_ARMS,
        FINDING_PATTERNS,
        FINDING_PHRASES,
        arm_text,
        as_text,
        evaluate_arm,
        load_rows,
        parse_csv,
        parse_float_grid,
        prepare_row,
        score_findings,
    )
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, split_sentences, tokenize


DEPLOYABLE_FEATURE_ARMS = ("current_only", "real_prior", "prior_report_only")
ANALYSIS_ONLY_FEATURE_ARMS = {"shuffled_prior", "matched_shuffled_prior"}
DEFAULT_FEATURE_ARMS = DEPLOYABLE_FEATURE_ARMS
DEFAULT_CLASS_WEIGHTS = ("none", "balanced")


def parse_class_weights(raw: str) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for item in parse_csv(raw):
        out.append(None if item.lower() == "none" else item)
    return out


def validate_inference_arms(
    *,
    feature_arms: Sequence[str],
    natural_base_arm: str,
    natural_append_arms: Sequence[str],
    allow_analysis_only_features: bool,
) -> None:
    if allow_analysis_only_features:
        return
    inference_arms = set(feature_arms) | {natural_base_arm} | set(natural_append_arms)
    blocked = sorted(inference_arms & ANALYSIS_ONLY_FEATURE_ARMS)
    if blocked:
        raise RuntimeError(
            "Analysis-only arms are not allowed for deployable verifier inference: "
            f"{','.join(blocked)}. Pass --allow-analysis-only-features only for "
            "analysis/stress-test runs, not deployable method claims."
        )


def findings_to_report(findings: Sequence[str]) -> str:
    if not findings:
        return "No source-valid positive findings."
    return " ".join(FINDING_PHRASES[finding] for finding in sorted(findings))


def clean_join(sentences: Sequence[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()


def positive_findings(text: str) -> set[str]:
    return {finding for finding, status in finding_vector(text).items() if status == "positive"}


def first_positive_sentence_for_finding(
    prepared: Mapping[str, Any],
    finding: str,
    *,
    candidate_arms: Sequence[str],
    allowed_findings: Sequence[str],
) -> str:
    allowed = set(allowed_findings)
    row = prepared["row"]
    for arm in candidate_arms:
        for sentence in split_sentences(arm_text(row, arm)):
            sentence_findings = positive_findings(sentence)
            if finding in sentence_findings and sentence_findings <= allowed:
                return sentence.strip()
    return FINDING_PHRASES[finding]


def natural_corrected_report(
    prepared: Mapping[str, Any],
    findings: Sequence[str],
    *,
    base_arm: str,
    append_arms: Sequence[str],
    append_missing: bool,
    max_token_delta: Optional[int] = None,
    prefer_compact_append: bool = False,
) -> str:
    selected = set(findings)
    kept: List[str] = []
    base_text = arm_text(prepared["row"], base_arm)
    max_tokens = None
    if max_token_delta is not None:
        max_tokens = len(tokenize(base_text)) + int(max_token_delta)
    for sentence in split_sentences(base_text):
        sentence_findings = positive_findings(sentence)
        if not sentence_findings or sentence_findings <= selected:
            kept.append(sentence)
    if append_missing:
        present = positive_findings(clean_join(kept))
        for finding in sorted(selected - present):
            candidate = first_positive_sentence_for_finding(
                prepared,
                finding,
                candidate_arms=append_arms,
                allowed_findings=selected,
            )
            compact = FINDING_PHRASES[finding]
            if prefer_compact_append:
                candidates = [compact]
            else:
                candidates = [candidate]
                if compact != candidate:
                    candidates.append(compact)
            for item in candidates:
                proposed = clean_join([*kept, item])
                if max_tokens is not None and len(tokenize(proposed)) > max_tokens:
                    continue
                kept.append(item)
                break
    text = clean_join(kept)
    if text:
        return text
    return findings_to_report(findings)


def prediction_for_findings(
    prepared: Mapping[str, Any],
    findings: Sequence[str],
    *,
    prediction_mode: str,
    natural_base_arm: str,
    natural_append_arms: Sequence[str],
    natural_max_token_delta: Optional[int],
) -> str:
    if prediction_mode == "structured":
        return findings_to_report(findings)
    if prediction_mode == "natural_filter":
        return natural_corrected_report(
            prepared,
            findings,
            base_arm=natural_base_arm,
            append_arms=natural_append_arms,
            append_missing=False,
            max_token_delta=natural_max_token_delta,
        )
    if prediction_mode == "natural_filter_append":
        return natural_corrected_report(
            prepared,
            findings,
            base_arm=natural_base_arm,
            append_arms=natural_append_arms,
            append_missing=True,
            max_token_delta=natural_max_token_delta,
        )
    if prediction_mode == "natural_filter_compact_append":
        return natural_corrected_report(
            prepared,
            findings,
            base_arm=natural_base_arm,
            append_arms=natural_append_arms,
            append_missing=True,
            max_token_delta=natural_max_token_delta,
            prefer_compact_append=True,
        )
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def split_prepared(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    by_split: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(as_text(row.get("split")) or "unspecified", []).append(row)
    by_split["all"] = list(rows)
    return by_split


def feature_names(feature_arms: Sequence[str], *, use_finding_identity: bool) -> List[str]:
    names = ["bias"]
    names.extend(f"source_positive={arm}" for arm in feature_arms)
    names.extend(
        [
            "current_and_real",
            "real_and_prior_report",
            "current_and_prior_report",
            "real_without_wrong_history",
            "real_with_wrong_history",
            "current_or_real",
            "current_or_clean_real",
            "wrong_history_count",
            "real_current_disagree",
            "real_prior_report_disagree",
        ]
    )
    if use_finding_identity:
        names.extend(f"finding={finding}" for finding in FINDING_PATTERNS)
    return names


def source_bool(prepared: Mapping[str, Any], arm: str, finding: str) -> int:
    return int(bool(prepared["vectors"].get(arm, {}).get(finding, False)))


def row_features(
    prepared: Mapping[str, Any],
    finding: str,
    *,
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    use_finding_identity: bool,
) -> List[float]:
    source_values = [source_bool(prepared, arm, finding) for arm in feature_arms]
    current = source_bool(prepared, "current_only", finding)
    real = source_bool(prepared, "real_prior", finding)
    prior_report = source_bool(prepared, "prior_report_only", finding)
    wrong_count = sum(source_bool(prepared, arm, finding) for arm in wrong_arms if arm in feature_arms)
    features: List[float] = [1.0]
    features.extend(float(value) for value in source_values)
    features.extend(
        [
            float(current and real),
            float(real and prior_report),
            float(current and prior_report),
            float(real and wrong_count == 0),
            float(real and wrong_count > 0),
            float(current or real),
            float(current or (real and wrong_count == 0)),
            float(wrong_count),
            float(abs(real - current)),
            float(abs(real - prior_report)),
        ]
    )
    if use_finding_identity:
        features.extend(1.0 if item == finding else 0.0 for item in FINDING_PATTERNS)
    return features


def make_xy(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    use_finding_identity: bool,
) -> Tuple[List[List[float]], List[int], List[Tuple[Mapping[str, Any], str]]]:
    x_rows: List[List[float]] = []
    labels: List[int] = []
    refs: List[Tuple[Mapping[str, Any], str]] = []
    for prepared in rows:
        for finding in FINDING_PATTERNS:
            x_rows.append(
                row_features(
                    prepared,
                    finding,
                    feature_arms=feature_arms,
                    wrong_arms=wrong_arms,
                    use_finding_identity=use_finding_identity,
                )
            )
            labels.append(int(bool(prepared["vectors"]["reference"].get(finding, False))))
            refs.append((prepared, finding))
    return x_rows, labels, refs


def objective(
    score: Mapping[str, float],
    *,
    fp_weight: float,
    fn_weight: float,
    token_weight: float = 0.0,
) -> float:
    return (
        float(score.get("finding_f1", 0.0))
        - fp_weight * float(score.get("finding_fp_count", 0.0))
        - fn_weight * float(score.get("finding_fn_count", 0.0))
        - token_weight * float(score.get("prediction_tokens", 0.0))
    )


def mean_metrics(
    scores: Sequence[Mapping[str, float]],
    *,
    fp_weight: float,
    fn_weight: float,
    token_weight: float = 0.0,
) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    out = {
        metric: statistics.mean(float(score.get(metric, 0.0)) for score in scores)
        for metric in metrics
    }
    out["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight, token_weight=token_weight)
        for score in scores
    )
    return out


def evaluate_source_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    fp_weight: float,
    fn_weight: float,
    token_weight: float = 0.0,
) -> Dict[str, Any]:
    payload = evaluate_arm(rows, arm=arm, fp_weight=fp_weight, fn_weight=fn_weight)
    metrics = dict(payload.get("metrics") or {})
    metrics["objective"] = objective(
        metrics,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
        token_weight=token_weight,
    )
    return {**payload, "metrics": metrics}


def evaluate_verifier(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: LogisticRegression,
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    use_finding_identity: bool,
    threshold: float,
    thresholds_by_finding: Optional[Mapping[str, float]],
    fp_weight: float,
    fn_weight: float,
    token_weight: float,
    prediction_mode: str,
    natural_base_arm: str,
    natural_append_arms: Sequence[str],
    natural_max_token_delta: Optional[int],
    emit_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    x_rows, _labels, refs = make_xy(
        rows,
        feature_arms=feature_arms,
        wrong_arms=wrong_arms,
        use_finding_identity=use_finding_identity,
    )
    probabilities = model.predict_proba(x_rows)[:, 1] if x_rows else []
    by_row: Dict[int, List[str]] = {id(row): [] for row in rows}
    for probability, (prepared, finding) in zip(probabilities, refs):
        cutoff = float((thresholds_by_finding or {}).get(finding, threshold))
        if float(probability) >= cutoff:
            by_row[id(prepared)].append(finding)

    scores: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for prepared in rows:
        findings = sorted(by_row[id(prepared)])
        counts.update(findings)
        prediction = prediction_for_findings(
            prepared,
            findings,
            prediction_mode=prediction_mode,
            natural_base_arm=natural_base_arm,
            natural_append_arms=natural_append_arms,
            natural_max_token_delta=natural_max_token_delta,
        )
        if prediction_mode == "structured":
            scores.append(
                score_findings(
                    findings,
                    reference_text=as_text(prepared.get("reference_text")),
                    reference_status=prepared["reference_status"],
                )
            )
        else:
            scores.append(
                row_scores(
                    {
                        "prediction": prediction,
                        "reference": as_text(prepared.get("reference_text")),
                    }
                )
            )
        if emit_rows:
            selected_rows.append(
                {
                    **dict(prepared["row"]),
                    "selected_findings": findings,
                    "prediction": prediction,
                    "source_causal_verifier": {
                        "feature_arms": list(feature_arms),
                        "wrong_arms": list(wrong_arms),
                        "threshold": threshold,
                        "thresholds_by_finding": dict(thresholds_by_finding or {}),
                        "prediction_mode": prediction_mode,
                        "natural_base_arm": natural_base_arm,
                        "natural_append_arms": list(natural_append_arms),
                        "natural_max_token_delta": natural_max_token_delta,
                    },
                }
            )
    return (
        {
            "rows": len(rows),
            "selected_finding_counts": dict(counts),
            "metrics": mean_metrics(
                scores,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
                token_weight=token_weight,
            ),
        },
        selected_rows,
    )


def per_finding_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: LogisticRegression,
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    use_finding_identity: bool,
    threshold_grid: Sequence[float],
    threshold_status_weight: float,
    threshold_fp_weight: float,
    threshold_fn_weight: float,
) -> Dict[str, float]:
    x_rows, labels, refs = make_xy(
        rows,
        feature_arms=feature_arms,
        wrong_arms=wrong_arms,
        use_finding_identity=use_finding_identity,
    )
    probabilities = model.predict_proba(x_rows)[:, 1] if x_rows else []
    thresholds: Dict[str, float] = {}
    for finding in FINDING_PATTERNS:
        best_key: Optional[Tuple[float, float, float, int, int]] = None
        best_threshold = float(threshold_grid[0])
        for threshold in threshold_grid:
            tp = fp = fn = tn = 0
            for probability, label, (_prepared, ref_finding) in zip(probabilities, labels, refs):
                if ref_finding != finding:
                    continue
                predicted = float(probability) >= float(threshold)
                if predicted and label:
                    tp += 1
                elif predicted:
                    fp += 1
                elif label:
                    fn += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
            recall = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            total = max(1, tp + fp + fn + tn)
            accuracy = (tp + tn) / total
            score = (
                f1
                + threshold_status_weight * accuracy
                - threshold_fp_weight * (fp / total)
                - threshold_fn_weight * (fn / total)
            )
            key = (score, f1, accuracy, -fp, -fn)
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
        thresholds[finding] = best_threshold
    return thresholds


def constraint_status(
    metrics: Mapping[str, float],
    *,
    baseline_metrics: Optional[Mapping[str, float]],
    max_fp_delta: Optional[float],
    max_fn_delta: Optional[float],
    max_prediction_tokens_delta: Optional[float],
    min_finding_f1_delta: Optional[float],
    min_status_accuracy_delta: Optional[float],
    min_token_f1_delta: Optional[float],
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
        "prediction_tokens": float(metrics.get("prediction_tokens", 0.0))
        - float(baseline_metrics.get("prediction_tokens", 0.0)),
        "token_f1": float(metrics.get("token_f1", 0.0)) - float(baseline_metrics.get("token_f1", 0.0)),
        "objective": float(metrics.get("objective", 0.0)) - float(baseline_metrics.get("objective", 0.0)),
    }
    reasons: List[str] = []
    if max_fp_delta is not None and deltas["finding_fp_count"] > max_fp_delta:
        reasons.append("fp_delta_above_max")
    if max_fn_delta is not None and deltas["finding_fn_count"] > max_fn_delta:
        reasons.append("fn_delta_above_max")
    if max_prediction_tokens_delta is not None and deltas["prediction_tokens"] > max_prediction_tokens_delta:
        reasons.append("prediction_tokens_delta_above_max")
    if min_finding_f1_delta is not None and deltas["finding_f1"] < min_finding_f1_delta:
        reasons.append("finding_f1_delta_below_min")
    if min_status_accuracy_delta is not None and deltas["finding_status_accuracy"] < min_status_accuracy_delta:
        reasons.append("status_accuracy_delta_below_min")
    if min_token_f1_delta is not None and deltas["token_f1"] < min_token_f1_delta:
        reasons.append("token_f1_delta_below_min")
    if min_objective_delta is not None and deltas["objective"] < min_objective_delta:
        reasons.append("objective_delta_below_min")
    return {"satisfied": not reasons, "reasons": reasons, "deltas": deltas}


def train_and_tune(
    train_rows: Sequence[Mapping[str, Any]],
    tune_rows: Sequence[Mapping[str, Any]],
    *,
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    use_finding_identity: bool,
    c_grid: Sequence[float],
    class_weights: Sequence[Optional[str]],
    threshold_grid: Sequence[float],
    threshold_mode: str,
    threshold_status_weight: float,
    threshold_fp_weight: float,
    threshold_fn_weight: float,
    fp_weight: float,
    fn_weight: float,
    token_weight: float,
    prediction_mode: str,
    natural_base_arm: str,
    natural_append_arms: Sequence[str],
    natural_max_token_delta: Optional[int],
    baseline_metrics: Optional[Mapping[str, float]],
    max_fp_delta: Optional[float],
    max_fn_delta: Optional[float],
    max_prediction_tokens_delta: Optional[float],
    min_finding_f1_delta: Optional[float],
    min_status_accuracy_delta: Optional[float],
    min_token_f1_delta: Optional[float],
    min_objective_delta: Optional[float],
) -> Dict[str, Any]:
    x_train, y_train, _refs = make_xy(
        train_rows,
        feature_arms=feature_arms,
        wrong_arms=wrong_arms,
        use_finding_identity=use_finding_identity,
    )
    trials: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[int, float, float, float, float]] = None
    best: Optional[Dict[str, Any]] = None
    satisfying_trials = 0
    for c_value in c_grid:
        for class_weight in class_weights:
            model = LogisticRegression(
                C=float(c_value),
                class_weight=class_weight,
                solver="liblinear",
                max_iter=1000,
                random_state=0,
            )
            model.fit(x_train, y_train)
            if threshold_mode == "per_finding":
                thresholds_by_finding = per_finding_thresholds(
                    tune_rows,
                    model=model,
                    feature_arms=feature_arms,
                    wrong_arms=wrong_arms,
                    use_finding_identity=use_finding_identity,
                    threshold_grid=threshold_grid,
                    threshold_status_weight=threshold_status_weight,
                    threshold_fp_weight=threshold_fp_weight,
                    threshold_fn_weight=threshold_fn_weight,
                )
                threshold_values = [0.5]
            else:
                thresholds_by_finding = None
                threshold_values = [float(value) for value in threshold_grid]
            for threshold in threshold_values:
                result, _rows = evaluate_verifier(
                    tune_rows,
                    model=model,
                    feature_arms=feature_arms,
                    wrong_arms=wrong_arms,
                    use_finding_identity=use_finding_identity,
                    threshold=float(threshold),
                    thresholds_by_finding=thresholds_by_finding,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                    token_weight=token_weight,
                    prediction_mode=prediction_mode,
                    natural_base_arm=natural_base_arm,
                    natural_append_arms=natural_append_arms,
                    natural_max_token_delta=natural_max_token_delta,
                    emit_rows=False,
                )
                metrics = result["metrics"]
                constraint = constraint_status(
                    metrics,
                    baseline_metrics=baseline_metrics,
                    max_fp_delta=max_fp_delta,
                    max_fn_delta=max_fn_delta,
                    max_prediction_tokens_delta=max_prediction_tokens_delta,
                    min_finding_f1_delta=min_finding_f1_delta,
                    min_status_accuracy_delta=min_status_accuracy_delta,
                    min_token_f1_delta=min_token_f1_delta,
                    min_objective_delta=min_objective_delta,
                )
                satisfying_trials += int(bool(constraint["satisfied"]))
                key = (
                    int(bool(constraint["satisfied"])),
                    float(metrics.get("objective", 0.0)),
                    float(metrics.get("finding_f1", 0.0)),
                    float(metrics.get("finding_status_accuracy", 0.0)),
                    -float(metrics.get("finding_fp_count", 0.0)),
                    -float(metrics.get("prediction_tokens", 0.0)),
                )
                trial = {
                    "C": float(c_value),
                    "class_weight": class_weight or "none",
                    "threshold": float(threshold),
                    "threshold_mode": threshold_mode,
                    "prediction_mode": prediction_mode,
                    "natural_base_arm": natural_base_arm,
                    "natural_append_arms": list(natural_append_arms),
                    "natural_max_token_delta": natural_max_token_delta,
                    "thresholds_by_finding": dict(thresholds_by_finding or {}),
                    "constraint": constraint,
                    **result,
                }
                trials.append(trial)
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        **trial,
                        "model": model,
                    }
    if best is None:
        raise RuntimeError("No verifier tuning candidates evaluated")
    model = best.pop("model")
    return {
        "model": model,
        "best": best,
        "n_trials": len(trials),
        "satisfying_trials": satisfying_trials,
        "constraints": {
            "baseline_metrics": dict(baseline_metrics or {}),
            "max_fp_delta": max_fp_delta,
            "max_fn_delta": max_fn_delta,
            "max_prediction_tokens_delta": max_prediction_tokens_delta,
            "min_finding_f1_delta": min_finding_f1_delta,
            "min_status_accuracy_delta": min_status_accuracy_delta,
            "min_token_f1_delta": min_token_f1_delta,
            "min_objective_delta": min_objective_delta,
        },
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
    c_grid: Sequence[float],
    class_weights: Sequence[Optional[str]],
    threshold_grid: Sequence[float],
    threshold_mode: str,
    threshold_status_weight: float,
    threshold_fp_weight: float,
    threshold_fn_weight: float,
    fp_weight: float,
    fn_weight: float,
    token_weight: float,
    min_samples: int,
    constraint_baseline_arm: Optional[str],
    max_fp_delta: Optional[float],
    max_fn_delta: Optional[float],
    max_prediction_tokens_delta: Optional[float],
    min_finding_f1_delta: Optional[float],
    min_status_accuracy_delta: Optional[float],
    min_token_f1_delta: Optional[float],
    min_objective_delta: Optional[float],
    use_finding_identity: bool,
    prediction_mode: str,
    natural_base_arm: str,
    natural_append_arms: Sequence[str],
    natural_max_token_delta: Optional[int],
    allow_analysis_only_features: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validate_inference_arms(
        feature_arms=feature_arms,
        natural_base_arm=natural_base_arm,
        natural_append_arms=natural_append_arms,
        allow_analysis_only_features=allow_analysis_only_features,
    )
    prepared = [prepare_row(row, all_arms) for row in rows]
    rows_by_split = split_prepared(prepared)
    train_rows = rows_by_split.get(train_split, [])
    tune_rows = rows_by_split.get(tune_split, [])
    if len(train_rows) < min_samples:
        raise RuntimeError(f"train split rows below min-samples gate: {len(train_rows)} < {min_samples}")
    if len(tune_rows) < min_samples:
        raise RuntimeError(f"tune split rows below min-samples gate: {len(tune_rows)} < {min_samples}")

    baseline_metrics = None
    if constraint_baseline_arm:
        baseline_metrics = evaluate_source_arm(
            tune_rows,
            arm=constraint_baseline_arm,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            token_weight=token_weight,
        )["metrics"]
    tuning = train_and_tune(
        train_rows,
        tune_rows,
        feature_arms=feature_arms,
        wrong_arms=wrong_arms,
        use_finding_identity=use_finding_identity,
        c_grid=c_grid,
        class_weights=class_weights,
        threshold_grid=threshold_grid,
        threshold_mode=threshold_mode,
        threshold_status_weight=threshold_status_weight,
        threshold_fp_weight=threshold_fp_weight,
        threshold_fn_weight=threshold_fn_weight,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
        token_weight=token_weight,
        prediction_mode=prediction_mode,
        natural_base_arm=natural_base_arm,
        natural_append_arms=natural_append_arms,
        natural_max_token_delta=natural_max_token_delta,
        baseline_metrics=baseline_metrics,
        max_fp_delta=max_fp_delta,
        max_fn_delta=max_fn_delta,
        max_prediction_tokens_delta=max_prediction_tokens_delta,
        min_finding_f1_delta=min_finding_f1_delta,
        min_status_accuracy_delta=min_status_accuracy_delta,
        min_token_f1_delta=min_token_f1_delta,
        min_objective_delta=min_objective_delta,
    )
    model = tuning.pop("model")
    best = tuning["best"]

    metrics_by_split: Dict[str, Any] = {}
    selected_rows: List[Dict[str, Any]] = []
    for split in eval_splits:
        split_rows = rows_by_split.get(split, [])
        if len(split_rows) < min_samples:
            raise RuntimeError(f"eval split rows below min-samples gate: {split} {len(split_rows)} < {min_samples}")
        split_metrics: Dict[str, Any] = {}
        for arm in all_arms:
            split_metrics[arm] = evaluate_source_arm(
                split_rows,
                arm=arm,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
                token_weight=token_weight,
            )
        verifier_metrics, emitted = evaluate_verifier(
            split_rows,
            model=model,
            feature_arms=feature_arms,
            wrong_arms=wrong_arms,
            use_finding_identity=use_finding_identity,
            threshold=float(best["threshold"]),
            thresholds_by_finding=best.get("thresholds_by_finding") or None,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            token_weight=token_weight,
            prediction_mode=prediction_mode,
            natural_base_arm=natural_base_arm,
            natural_append_arms=natural_append_arms,
            natural_max_token_delta=natural_max_token_delta,
            emit_rows=True,
        )
        split_metrics["source_causal_verifier"] = verifier_metrics
        metrics_by_split[split] = split_metrics
        selected_rows.extend(emitted)

    return (
        {
            "rows": len(rows),
            "splits": {split: len(items) for split, items in sorted(rows_by_split.items())},
            "all_arms": list(all_arms),
            "feature_arms": list(feature_arms),
            "wrong_arms": list(wrong_arms),
            "train_split": train_split,
            "tune_split": tune_split,
            "eval_splits": list(eval_splits),
            "min_samples": min_samples,
            "feature_names": feature_names(feature_arms, use_finding_identity=use_finding_identity),
            "objective": {
                "metric": (
                    "finding_f1 - fp_weight*finding_fp_count - "
                    "fn_weight*finding_fn_count - token_weight*prediction_tokens"
                ),
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
                "token_weight": token_weight,
            },
            "threshold_objective": {
                "metric": "per-finding f1 + status_weight*accuracy - fp_weight*fp_rate - fn_weight*fn_rate",
                "status_weight": threshold_status_weight,
                "fp_weight": threshold_fp_weight,
                "fn_weight": threshold_fn_weight,
            },
            "prediction_config": {
                "prediction_mode": prediction_mode,
                "natural_base_arm": natural_base_arm,
                "natural_append_arms": list(natural_append_arms),
                "natural_max_token_delta": natural_max_token_delta,
                "allow_analysis_only_features": allow_analysis_only_features,
            },
            "constraint_baseline_arm": constraint_baseline_arm,
            "tuning": tuning,
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
        "# Source-Causal Verifier",
        "",
        f"Rows: `{report['rows']}`",
        f"Train split: `{report['train_split']}`",
        f"Tune split: `{report['tune_split']}`",
        f"Feature arms: `{', '.join(report['feature_arms'])}`",
        f"Wrong-history arms: `{', '.join(report['wrong_arms'])}`",
        f"Prediction mode: `{report.get('prediction_config', {}).get('prediction_mode', 'structured')}`",
        f"Natural base arm: `{report.get('prediction_config', {}).get('natural_base_arm', 'real_prior')}`",
        f"Natural max token delta: `{report.get('prediction_config', {}).get('natural_max_token_delta', 'none')}`",
        f"Analysis-only inference features allowed: `{report.get('prediction_config', {}).get('allow_analysis_only_features', False)}`",
        (
            "Best params: "
            f"`C={fmt(best['C'])}`, "
            f"`class_weight={best['class_weight']}`, "
            f"`threshold_mode={best['threshold_mode']}`"
        ),
        f"Constraint baseline: `{report.get('constraint_baseline_arm') or 'none'}`",
        f"Best satisfies constraints: `{best_constraint.get('satisfied', True)}`",
        "",
        "| Split | Arm/Policy | Finding-F1 | Status Acc | FP | FN | Token-F1 | Pred Tokens | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        for name, result in payload.items():
            metrics = result["metrics"]
            lines.append(
                f"| `{split}` | `{name}` | {fmt(metrics.get('finding_f1'))} | "
                f"{fmt(metrics.get('finding_status_accuracy'))} | {fmt(metrics.get('finding_fp_count'))} | "
                f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('token_f1'))} | "
                f"{fmt(metrics.get('prediction_tokens'))} | "
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
            "",
            "## Per-Finding Thresholds",
            "",
            "| Finding | Threshold |",
            "|---|---:|",
        ]
    )
    for finding, threshold in sorted((best.get("thresholds_by_finding") or {}).items()):
        lines.append(f"| `{finding}` | {fmt(threshold)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--all-arms", default=",".join(DEFAULT_ALL_ARMS))
    parser.add_argument("--feature-arms", default=",".join(DEFAULT_FEATURE_ARMS))
    parser.add_argument("--wrong-arms", default="shuffled_prior,matched_shuffled_prior")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="validate")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument("--c-grid", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--class-weights", default="none,balanced")
    parser.add_argument("--threshold-grid", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95")
    parser.add_argument("--threshold-mode", choices=("global", "per_finding"), default="per_finding")
    parser.add_argument("--threshold-status-weight", type=float, default=0.25)
    parser.add_argument("--threshold-fp-weight", type=float, default=0.5)
    parser.add_argument("--threshold-fn-weight", type=float, default=0.02)
    parser.add_argument("--fp-weight", type=float, default=0.1)
    parser.add_argument("--fn-weight", type=float, default=0.0)
    parser.add_argument("--token-weight", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--constraint-baseline-arm", default=None)
    parser.add_argument("--max-fp-delta", type=float, default=None)
    parser.add_argument("--max-fn-delta", type=float, default=None)
    parser.add_argument("--max-prediction-token-delta", type=float, default=None)
    parser.add_argument("--min-finding-f1-delta", type=float, default=None)
    parser.add_argument("--min-status-accuracy-delta", type=float, default=None)
    parser.add_argument("--min-token-f1-delta", type=float, default=None)
    parser.add_argument("--min-objective-delta", type=float, default=None)
    parser.add_argument("--no-finding-identity", action="store_true")
    parser.add_argument(
        "--prediction-mode",
        choices=("structured", "natural_filter", "natural_filter_append", "natural_filter_compact_append"),
        default="structured",
    )
    parser.add_argument("--natural-base-arm", default="real_prior")
    parser.add_argument("--natural-append-arms", default="real_prior,current_only,prior_report_only")
    parser.add_argument(
        "--natural-max-token-delta",
        type=int,
        default=None,
        help=(
            "Optional per-row token cap for natural_filter[_append] output relative "
            "to the natural base arm. A value of 0 prevents append from making a "
            "row longer than the selected base-arm report."
        ),
    )
    parser.add_argument(
        "--allow-analysis-only-features",
        action="store_true",
        help=(
            "Allow shuffled/matched_shuffled arms as inference features. "
            "Use only for analysis/stress tests, not deployable method claims."
        ),
    )
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
        c_grid=parse_float_grid(args.c_grid),
        class_weights=parse_class_weights(args.class_weights),
        threshold_grid=parse_float_grid(args.threshold_grid),
        threshold_mode=args.threshold_mode,
        threshold_status_weight=args.threshold_status_weight,
        threshold_fp_weight=args.threshold_fp_weight,
        threshold_fn_weight=args.threshold_fn_weight,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        token_weight=args.token_weight,
        min_samples=args.min_samples,
        constraint_baseline_arm=args.constraint_baseline_arm,
        max_fp_delta=args.max_fp_delta,
        max_fn_delta=args.max_fn_delta,
        max_prediction_tokens_delta=args.max_prediction_token_delta,
        min_finding_f1_delta=args.min_finding_f1_delta,
        min_status_accuracy_delta=args.min_status_accuracy_delta,
        min_token_f1_delta=args.min_token_f1_delta,
        min_objective_delta=args.min_objective_delta,
        use_finding_identity=not args.no_finding_identity,
        prediction_mode=args.prediction_mode,
        natural_base_arm=args.natural_base_arm,
        natural_append_arms=parse_csv(args.natural_append_arms),
        natural_max_token_delta=args.natural_max_token_delta,
        allow_analysis_only_features=args.allow_analysis_only_features,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")
    best_constraint = report["tuning"]["best"].get("constraint") or {}
    return 0 if best_constraint.get("satisfied", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())

