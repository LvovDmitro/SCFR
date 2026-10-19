#!/usr/bin/env python3
"""Train/evaluate a source-causal pairwise reranker on preference pairs.

This is a SMOKE/PILOT diagnostic for the source-causal preference direction.
It trains only from train pairs and reports validation pair accuracy. The
features intentionally avoid the reference report; labels/rewards are used only
to form chosen/rejected supervision.
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from sklearn.linear_model import LogisticRegression

    from scripts.analyze_source_causal_arms import finding_vector, reference_text, row_scores, split_sentences, tokenize
    from scripts.audit_prior_copy_temporal import extract_temporal_terms, prior_copy_precision
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sklearn.linear_model import LogisticRegression

    from scripts.analyze_source_causal_arms import finding_vector, reference_text, row_scores, split_sentences, tokenize
    from scripts.audit_prior_copy_temporal import extract_temporal_terms, prior_copy_precision


DEPLOYABLE_SOURCES = {"current_only", "real_prior", "prior_report_only"}
WRONG_HISTORY_SOURCES = {"shuffled_prior", "matched_shuffled_prior"}


@dataclass(frozen=True)
class PairRow:
    sample_id: str
    split: str
    context: str
    chosen: str
    rejected: str
    chosen_source: str
    rejected_source: str
    reward_margin: float
    raw: Mapping[str, Any]


def as_text(value: Any) -> str:
    return str(value or "").strip()


def load_pairs(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            chosen = as_text(payload.get("chosen"))
            rejected = as_text(payload.get("rejected"))
            if not chosen or not rejected:
                continue
            rows.append(
                PairRow(
                    sample_id=as_text(payload.get("sample_id") or payload.get("id") or line_no),
                    split=as_text(payload.get("split") or "unspecified"),
                    context=as_text(payload.get("context") or payload.get("prior_report")),
                    chosen=chosen,
                    rejected=rejected,
                    chosen_source=as_text(payload.get("chosen_source") or "unknown"),
                    rejected_source=as_text(payload.get("rejected_source") or "unknown"),
                    reward_margin=float(payload.get("reward_margin", 1.0) or 0.0),
                    raw=payload,
                )
            )
    return rows


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        out.append(dict(row))
    return out


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def positive_findings(text: str) -> set[str]:
    return {name for name, status in finding_vector(text).items() if status == "positive"}


def negative_findings(text: str) -> set[str]:
    return {name for name, status in finding_vector(text).items() if status == "negative"}


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def normalized_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left.lower(), right.lower()).ratio())


def scaled_count(value: int, scale: float = 10.0) -> float:
    return float(value) / scale


def feature_names(source_names: Sequence[str], *, include_source_identity: bool) -> List[str]:
    names = [
        "bias",
        "prediction_tokens_log",
        "sentence_count",
        "positive_findings",
        "negative_findings",
        "temporal_terms",
        "prior_copy_precision",
        "token_jaccard_with_prior",
        "similarity_to_prior",
        "added_positive_vs_prior",
        "removed_positive_vs_prior",
        "positive_overlap_with_prior",
        "support_device_positive",
        "has_temporal_language",
    ]
    if include_source_identity:
        names.extend(f"source={source}" for source in source_names)
    return names


def candidate_features(
    *,
    text: str,
    source: str,
    context: str,
    source_names: Sequence[str],
    include_source_identity: bool,
) -> List[float]:
    candidate_pos = positive_findings(text)
    candidate_neg = negative_findings(text)
    prior_pos = positive_findings(context)
    token_count = len(tokenize(text))
    temporal_terms = extract_temporal_terms(text)
    overlap_prior = len(candidate_pos & prior_pos)
    base = [
        1.0,
        math.log1p(token_count) / 5.0,
        scaled_count(len(split_sentences(text))),
        scaled_count(len(candidate_pos)),
        scaled_count(len(candidate_neg)),
        scaled_count(len(temporal_terms), scale=5.0),
        prior_copy_precision(text, context),
        token_jaccard(text, context),
        normalized_similarity(text, context),
        scaled_count(len(candidate_pos - prior_pos)),
        scaled_count(len(prior_pos - candidate_pos)),
        scaled_count(overlap_prior),
        1.0 if "support_device" in candidate_pos else 0.0,
        1.0 if temporal_terms else 0.0,
    ]
    if include_source_identity:
        base.extend(1.0 if source == item else 0.0 for item in source_names)
    return base


def pair_diff(
    pair: PairRow,
    *,
    source_names: Sequence[str],
    include_source_identity: bool,
) -> List[float]:
    chosen_features = candidate_features(
        text=pair.chosen,
        source=pair.chosen_source,
        context=pair.context,
        source_names=source_names,
        include_source_identity=include_source_identity,
    )
    rejected_features = candidate_features(
        text=pair.rejected,
        source=pair.rejected_source,
        context=pair.context,
        source_names=source_names,
        include_source_identity=include_source_identity,
    )
    return [left - right for left, right in zip(chosen_features, rejected_features)]


def dot(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(float(w) * float(x) for w, x in zip(weights, features))


def arm_prediction(row: Mapping[str, Any], arm: str) -> str:
    value = row.get(arm)
    return value if isinstance(value, str) else ""


def split_pairs(rows: Sequence[PairRow]) -> Dict[str, List[PairRow]]:
    out: Dict[str, List[PairRow]] = {}
    for row in rows:
        out.setdefault(row.split, []).append(row)
    out["all"] = list(rows)
    return out


def train_model(
    train_pairs: Sequence[PairRow],
    *,
    source_names: Sequence[str],
    include_source_identity: bool,
    c_value: float,
    max_iter: int,
    sample_weight_mode: str,
    max_sample_weight: float,
) -> Tuple[List[float], Dict[str, Any]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    weights: List[float] = []
    for pair in train_pairs:
        diff = pair_diff(
            pair,
            source_names=source_names,
            include_source_identity=include_source_identity,
        )
        margin_weight = min(max(float(pair.reward_margin), 0.01), max_sample_weight)
        sample_weight = margin_weight if sample_weight_mode == "margin" else 1.0
        x_rows.append(diff)
        y_rows.append(1)
        weights.append(sample_weight)
        x_rows.append([-value for value in diff])
        y_rows.append(0)
        weights.append(sample_weight)

    if not x_rows:
        raise RuntimeError("no train examples")
    model = LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="liblinear",
        max_iter=int(max_iter),
        class_weight=None,
        random_state=0,
    )
    model.fit(x_rows, y_rows, sample_weight=weights)
    learned = [float(value) for value in model.coef_[0]]
    return (
        learned,
        {
            "examples": len(x_rows),
            "pairs": len(train_pairs),
            "c_value": float(c_value),
            "max_iter": int(max_iter),
            "sample_weight_mode": sample_weight_mode,
            "max_sample_weight": float(max_sample_weight),
            "n_iter": [int(value) for value in getattr(model, "n_iter_", [])],
        },
    )


def summarize_margins(margins: Sequence[float]) -> Dict[str, Any]:
    if not margins:
        return {
            "accuracy": 0.0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "min_margin": 0.0,
            "max_margin": 0.0,
            "nonpositive_margins": 0,
        }
    correct = sum(1 for value in margins if value > 0.0)
    return {
        "accuracy": correct / len(margins),
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "nonpositive_margins": len(margins) - correct,
    }


def evaluate_model(
    pairs: Sequence[PairRow],
    *,
    weights: Sequence[float],
    source_names: Sequence[str],
    include_source_identity: bool,
) -> Dict[str, Any]:
    margins = [
        dot(
            weights,
            pair_diff(
                pair,
                source_names=source_names,
                include_source_identity=include_source_identity,
            ),
        )
        for pair in pairs
    ]
    source_pair_counts: Counter[str] = Counter(
        f"{pair.chosen_source}>{pair.rejected_source}" for pair in pairs
    )
    source_pair_correct: Counter[str] = Counter()
    for pair, margin in zip(pairs, margins):
        if margin > 0.0:
            source_pair_correct[f"{pair.chosen_source}>{pair.rejected_source}"] += 1
    return {
        "pairs": len(pairs),
        **summarize_margins(margins),
        "source_pair_counts": dict(source_pair_counts),
        "source_pair_accuracy": {
            key: source_pair_correct[key] / count
            for key, count in sorted(source_pair_counts.items())
            if count > 0
        },
    }


def evaluate_baseline(
    pairs: Sequence[PairRow],
    *,
    scorer: Callable[[PairRow, str, str], float],
) -> Dict[str, Any]:
    margins = [
        scorer(pair, pair.chosen, pair.chosen_source)
        - scorer(pair, pair.rejected, pair.rejected_source)
        for pair in pairs
    ]
    return {"pairs": len(pairs), **summarize_margins(margins)}


def baseline_reports(pairs: Sequence[PairRow]) -> Dict[str, Any]:
    def source_family(_pair: PairRow, _text: str, source: str) -> float:
        if source in DEPLOYABLE_SOURCES:
            return 1.0
        if source in WRONG_HISTORY_SOURCES:
            return 0.0
        return 0.5

    def lower_prior_copy(pair: PairRow, text: str, _source: str) -> float:
        return -prior_copy_precision(text, pair.context)

    def shorter_text(_pair: PairRow, text: str, _source: str) -> float:
        return -float(len(tokenize(text)))

    def fewer_positive_findings(_pair: PairRow, text: str, _source: str) -> float:
        return -float(len(positive_findings(text)))

    return {
        "source_family": evaluate_baseline(pairs, scorer=source_family),
        "lower_prior_copy": evaluate_baseline(pairs, scorer=lower_prior_copy),
        "shorter_text": evaluate_baseline(pairs, scorer=shorter_text),
        "fewer_positive_findings": evaluate_baseline(pairs, scorer=fewer_positive_findings),
    }


def preference_objective(
    metrics: Mapping[str, float],
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> float:
    return (
        float(metrics.get("finding_f1", 0.0))
        + status_weight * float(metrics.get("finding_status_accuracy", 0.0))
        - fp_weight * float(metrics.get("finding_fp_count", 0.0))
        - fn_weight * float(metrics.get("finding_fn_count", 0.0))
    )


def mean_metric_rows(
    scores: Sequence[Mapping[str, float]],
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, float]:
    if not scores:
        return {
            "token_f1": 0.0,
            "finding_f1": 0.0,
            "finding_status_accuracy": 0.0,
            "finding_fp_count": 0.0,
            "finding_fn_count": 0.0,
            "prediction_tokens": 0.0,
            "objective": 0.0,
        }
    metric_names = sorted(scores[0])
    out = {
        name: statistics.mean(float(row.get(name, 0.0)) for row in scores)
        for name in metric_names
    }
    out["objective"] = statistics.mean(
        preference_objective(
            row,
            status_weight=status_weight,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        for row in scores
    )
    return out


def evaluate_fixed_arm_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores: List[Dict[str, float]] = []
    for row in rows:
        prediction = arm_prediction(row, arm)
        if not prediction:
            continue
        scores.append(row_scores({"prediction": prediction, "reference": reference_text(row)}))
    return {
        "rows": len(scores),
        "selected_counts": {arm: len(scores)},
        "metrics": mean_metric_rows(
            scores,
            status_weight=status_weight,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        ),
    }


def evaluate_candidate_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_arms: Sequence[str],
    weights: Sequence[float],
    source_names: Sequence[str],
    include_source_identity: bool,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    selected_rows: List[Dict[str, Any]] = []
    for row in rows:
        context = as_text(row.get("context") or row.get("prior_report"))
        candidates: List[Tuple[str, str, float]] = []
        for arm in candidate_arms:
            prediction = arm_prediction(row, arm)
            if not prediction:
                continue
            features = candidate_features(
                text=prediction,
                source=arm,
                context=context,
                source_names=source_names,
                include_source_identity=include_source_identity,
            )
            candidates.append((arm, prediction, dot(weights, features)))
        if not candidates:
            continue
        selected_arm, selected_prediction, selected_score = max(candidates, key=lambda item: item[2])
        selected_counts[selected_arm] += 1
        scores.append(
            row_scores({"prediction": selected_prediction, "reference": reference_text(row)})
        )
        if emit_rows:
            selected_rows.append(
                {
                    **dict(row),
                    "selected_arm": selected_arm,
                    "selected_score": selected_score,
                    "prediction": selected_prediction,
                }
            )
    return (
        {
            "rows": len(scores),
            "selected_counts": dict(selected_counts),
            "metrics": mean_metric_rows(
                scores,
                status_weight=status_weight,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            ),
        },
        selected_rows,
    )


def evaluate_candidate_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_arms: Sequence[str],
    split_field: str,
    eval_splits: Sequence[str],
    weights: Sequence[float],
    source_names: Sequence[str],
    include_source_identity: bool,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
    min_candidate_rows: int,
    max_finding_f1_drop: float,
    max_token_f1_drop: float,
    max_fn_increase: float,
    max_fp_increase: float,
    output_split_rows: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows_by_split: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        split = as_text(row.get(split_field)) or "unspecified"
        rows_by_split.setdefault(split, []).append(row)
    rows_by_split["all"] = list(rows)

    metrics_by_split: Dict[str, Any] = {}
    split_warnings: Dict[str, List[str]] = {}
    selected_predictions: List[Dict[str, Any]] = []
    for split in eval_splits:
        split_rows = rows_by_split.get(split, [])
        warnings: List[str] = []
        if len(split_rows) < min_candidate_rows:
            warnings.append(f"below_min_candidate_rows:{len(split_rows)}<{min_candidate_rows}")
        payload: Dict[str, Any] = {}
        for arm in candidate_arms:
            payload[arm] = evaluate_fixed_arm_predictions(
                split_rows,
                arm=arm,
                status_weight=status_weight,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
        reranker_metrics, selected_rows = evaluate_candidate_selection(
            split_rows,
            candidate_arms=candidate_arms,
            weights=weights,
            source_names=source_names,
            include_source_identity=include_source_identity,
            status_weight=status_weight,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=output_split_rows,
        )
        payload["pairwise_reranker"] = reranker_metrics
        if output_split_rows:
            selected_predictions.extend(selected_rows)
        metrics_by_split[split] = payload
        split_warnings[split] = warnings

    gate_reasons: List[str] = []
    analysis_only_arms = sorted(set(candidate_arms) & WRONG_HISTORY_SOURCES)
    if analysis_only_arms:
        gate_reasons.append(f"analysis_only_candidate_arms_present:{','.join(analysis_only_arms)}")
    validate_payload = metrics_by_split.get("validate") or {}
    validate_rows = int((validate_payload.get("pairwise_reranker") or {}).get("rows", 0) or 0)
    if "validate" in eval_splits and validate_rows < min_candidate_rows:
        gate_reasons.append(f"validate_candidate_rows_below_min:{validate_rows}<{min_candidate_rows}")
    if "validate" in eval_splits:
        reranker_metrics = (validate_payload.get("pairwise_reranker") or {}).get("metrics") or {}
        reranker_obj = float(reranker_metrics.get("objective", 0.0))
        for baseline_name in ("current_only", "real_prior"):
            baseline_payload = validate_payload.get(baseline_name) or {}
            if int(baseline_payload.get("rows", 0) or 0) <= 0:
                continue
            baseline_metrics = baseline_payload.get("metrics") or {}
            baseline_obj = float(baseline_metrics.get("objective", 0.0))
            if reranker_obj <= baseline_obj:
                gate_reasons.append(f"validate_not_above_{baseline_name}")
            if (
                float(reranker_metrics.get("finding_f1", 0.0)) + max_finding_f1_drop
                < float(baseline_metrics.get("finding_f1", 0.0))
            ):
                gate_reasons.append(f"validate_finding_f1_drop_vs_{baseline_name}")
            if (
                float(reranker_metrics.get("token_f1", 0.0)) + max_token_f1_drop
                < float(baseline_metrics.get("token_f1", 0.0))
            ):
                gate_reasons.append(f"validate_token_f1_drop_vs_{baseline_name}")
            if (
                float(reranker_metrics.get("finding_fn_count", 0.0))
                > float(baseline_metrics.get("finding_fn_count", 0.0)) + max_fn_increase
            ):
                gate_reasons.append(f"validate_fn_increase_vs_{baseline_name}")
            if (
                float(reranker_metrics.get("finding_fp_count", 0.0))
                > float(baseline_metrics.get("finding_fp_count", 0.0)) + max_fp_increase
            ):
                gate_reasons.append(f"validate_fp_increase_vs_{baseline_name}")

    return (
        {
            "candidate_arms": list(candidate_arms),
            "split_field": split_field,
            "eval_splits": list(eval_splits),
            "objective": {
                "metric": "finding_f1 + status_weight*finding_status_accuracy - fp_weight*fp - fn_weight*fn",
                "status_weight": status_weight,
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "metrics_by_split": metrics_by_split,
            "split_warnings": split_warnings,
            "gate": {
                "verdict": "GO" if not gate_reasons else "NO-GO",
                "reasons": sorted(set(gate_reasons)),
                "min_candidate_rows": min_candidate_rows,
                "max_finding_f1_drop": max_finding_f1_drop,
                "max_token_f1_drop": max_token_f1_drop,
                "max_fn_increase": max_fn_increase,
                "max_fp_increase": max_fp_increase,
            },
        },
        selected_predictions,
    )


def source_summary(pairs: Sequence[PairRow]) -> Dict[str, Any]:
    return {
        "split_counts": dict(Counter(pair.split for pair in pairs)),
        "chosen_source_counts": dict(Counter(pair.chosen_source for pair in pairs)),
        "rejected_source_counts": dict(Counter(pair.rejected_source for pair in pairs)),
        "unique_sample_ids": len({pair.sample_id for pair in pairs}),
    }


def build_report(
    pairs: Sequence[PairRow],
    *,
    train_split: str,
    eval_splits: Sequence[str],
    include_source_identity: bool,
    min_train_pairs: int,
    min_eval_pairs: int,
    min_validate_accuracy: float,
    c_value: float,
    max_iter: int,
    sample_weight_mode: str,
    max_sample_weight: float,
    allow_test_pairs: bool,
) -> Dict[str, Any]:
    if not allow_test_pairs:
        test_pairs = [pair for pair in pairs if pair.split == "test"]
        if test_pairs:
            raise RuntimeError(f"test pairs are forbidden for this stage: {len(test_pairs)}")

    pairs_by_split = split_pairs(pairs)
    train_pairs = pairs_by_split.get(train_split, [])
    if len(train_pairs) < min_train_pairs:
        raise RuntimeError(f"train pairs below min gate: {len(train_pairs)} < {min_train_pairs}")

    source_names = sorted(
        {
            source
            for pair in pairs
            for source in (pair.chosen_source, pair.rejected_source)
            if source
        }
    )
    names = feature_names(source_names, include_source_identity=include_source_identity)
    weights, training = train_model(
        train_pairs,
        source_names=source_names,
        include_source_identity=include_source_identity,
        c_value=c_value,
        max_iter=max_iter,
        sample_weight_mode=sample_weight_mode,
        max_sample_weight=max_sample_weight,
    )

    metrics_by_split: Dict[str, Any] = {}
    baseline_by_split: Dict[str, Any] = {}
    split_warnings: Dict[str, List[str]] = {}
    for split in eval_splits:
        split_rows = pairs_by_split.get(split, [])
        warnings: List[str] = []
        if split == train_split:
            warnings.append("is_train_split")
        if len(split_rows) < min_eval_pairs:
            warnings.append(f"below_min_eval_pairs:{len(split_rows)}<{min_eval_pairs}")
        metrics_by_split[split] = evaluate_model(
            split_rows,
            weights=weights,
            source_names=source_names,
            include_source_identity=include_source_identity,
        )
        baseline_by_split[split] = baseline_reports(split_rows)
        split_warnings[split] = warnings

    validate_metrics = metrics_by_split.get("validate") or {}
    validate_baselines = baseline_by_split.get("validate") or {}
    validate_accuracy = float(validate_metrics.get("accuracy", 0.0) or 0.0)
    validate_rows = int(validate_metrics.get("pairs", 0) or 0)
    source_baseline_accuracy = float(
        (validate_baselines.get("source_family") or {}).get("accuracy", 0.0) or 0.0
    )
    reasons: List[str] = []
    warnings: List[str] = []
    if validate_rows < min_eval_pairs:
        reasons.append(f"validate_pairs_below_min:{validate_rows}<{min_eval_pairs}")
    if validate_accuracy < min_validate_accuracy:
        reasons.append(f"validate_accuracy_below_target:{validate_accuracy:.4f}<{min_validate_accuracy:.4f}")
    if include_source_identity:
        warnings.append("source_identity_features_enabled")
    if source_baseline_accuracy >= 0.95:
        warnings.append("source_family_baseline_saturates")
    if include_source_identity and validate_accuracy <= source_baseline_accuracy + 1e-9:
        warnings.append("model_not_above_source_family_baseline")

    interpretation = (
        "diagnostic_only_source_identity"
        if include_source_identity
        else ("content_signal_candidate" if not reasons else "no_content_signal_yet")
    )
    gate = {
        "verdict": "GO" if not reasons else "NO-GO",
        "interpretation": interpretation,
        "reasons": reasons,
        "warnings": warnings,
        "min_train_pairs": min_train_pairs,
        "min_eval_pairs": min_eval_pairs,
        "min_validate_accuracy": min_validate_accuracy,
    }
    return {
        "rows": len(pairs),
        "summary": source_summary(pairs),
        "train_split": train_split,
        "eval_splits": list(eval_splits),
        "include_source_identity": include_source_identity,
        "source_names": source_names,
        "feature_names": names,
        "feature_weights": dict(zip(names, weights)),
        "training": training,
        "metrics_by_split": metrics_by_split,
        "baseline_by_split": baseline_by_split,
        "split_warnings": split_warnings,
        "gate": gate,
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Pairwise Reranker",
        "",
        f"Rows: `{report['rows']}`",
        f"Train split: `{report['train_split']}`",
        f"Eval splits: `{', '.join(report['eval_splits'])}`",
        f"Include source identity: `{report['include_source_identity']}`",
        f"Gate: `{report['gate']['verdict']}`",
        f"Interpretation: `{report['gate']['interpretation']}`",
        f"Gate reasons: `{', '.join(report['gate']['reasons']) or 'none'}`",
        f"Warnings: `{', '.join(report['gate']['warnings']) or 'none'}`",
        "",
        "## Pair Summary",
        "",
        f"- Split counts: `{report['summary']['split_counts']}`",
        f"- Chosen sources: `{report['summary']['chosen_source_counts']}`",
        f"- Rejected sources: `{report['summary']['rejected_source_counts']}`",
        f"- Unique sample IDs: `{report['summary']['unique_sample_ids']}`",
        "",
        "## Learned Model",
        "",
        "| Split | Pairs | Accuracy | Mean Margin | Nonpositive Margins |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, metrics in report["metrics_by_split"].items():
        lines.append(
            f"| `{split}` | {metrics.get('pairs', 0)} | {fmt(metrics.get('accuracy'))} | "
            f"{fmt(metrics.get('mean_margin'))} | {metrics.get('nonpositive_margins', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Split | Baseline | Accuracy | Mean Margin |",
            "|---|---|---:|---:|",
        ]
    )
    for split, baselines in report["baseline_by_split"].items():
        for name, metrics in baselines.items():
            lines.append(
                f"| `{split}` | `{name}` | {fmt(metrics.get('accuracy'))} | "
                f"{fmt(metrics.get('mean_margin'))} |"
            )
    lines.extend(["", "## Top Feature Weights", ""])
    weights = sorted(
        report["feature_weights"].items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    lines.extend(f"- `{name}`: {fmt(weight)}" for name, weight in weights[:14])
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            (
                "This diagnostic does not establish report-generation improvement. "
                "It only tests whether train-only preference pairs contain a validation "
                "pairwise signal under features that do not inspect the reference report."
            ),
        ]
    )
    candidate_eval = report.get("candidate_prediction_eval")
    if candidate_eval:
        lines.extend(
            [
                "",
                "## Candidate Prediction Eval",
                "",
                f"Candidate gate: `{candidate_eval['gate']['verdict']}`",
                f"Candidate gate reasons: `{', '.join(candidate_eval['gate']['reasons']) or 'none'}`",
                "",
                "| Split | Arm | Rows | Selected Counts | Token-F1 | Finding-F1 | Status Acc | FP | FN | Objective |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split, payload in candidate_eval["metrics_by_split"].items():
            for arm, result in payload.items():
                metrics = result.get("metrics") or {}
                selected_counts = result.get("selected_counts") or {}
                lines.append(
                    f"| `{split}` | `{arm}` | {result.get('rows', 0)} | "
                    f"`{json.dumps(selected_counts, sort_keys=True)}` | "
                    f"{fmt(metrics.get('token_f1'))} | {fmt(metrics.get('finding_f1'))} | "
                    f"{fmt(metrics.get('finding_status_accuracy'))} | "
                    f"{fmt(metrics.get('finding_fp_count'))} | "
                    f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('objective'))} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", default="train,validate")
    parser.add_argument("--include-source-identity", action="store_true")
    parser.add_argument("--allow-test-pairs", action="store_true")
    parser.add_argument("--min-train-pairs", type=int, default=50)
    parser.add_argument("--min-eval-pairs", type=int, default=30)
    parser.add_argument("--min-validate-accuracy", type=float, default=0.65)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--sample-weight-mode", choices=("none", "margin"), default="margin")
    parser.add_argument("--max-sample-weight", type=float, default=3.0)
    parser.add_argument("--candidate-predictions", type=Path, default=None)
    parser.add_argument(
        "--candidate-arms",
        default="current_only,real_prior,prior_report_only",
    )
    parser.add_argument("--candidate-split-field", default="split")
    parser.add_argument("--candidate-eval-splits", default="validate")
    parser.add_argument("--min-candidate-rows", type=int, default=30)
    parser.add_argument("--status-weight", type=float, default=0.25)
    parser.add_argument("--fp-weight", type=float, default=0.5)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    parser.add_argument("--max-finding-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-token-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-fn-increase", type=float, default=0.05)
    parser.add_argument("--max-fp-increase", type=float, default=0.0)
    parser.add_argument("--fail-on-no-go", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pairs = load_pairs(args.pairs)
    report = build_report(
        pairs,
        train_split=args.train_split,
        eval_splits=parse_csv(args.eval_splits),
        include_source_identity=bool(args.include_source_identity),
        min_train_pairs=int(args.min_train_pairs),
        min_eval_pairs=int(args.min_eval_pairs),
        min_validate_accuracy=float(args.min_validate_accuracy),
        c_value=float(args.c_value),
        max_iter=int(args.max_iter),
        sample_weight_mode=str(args.sample_weight_mode),
        max_sample_weight=float(args.max_sample_weight),
        allow_test_pairs=bool(args.allow_test_pairs),
    )
    selected_predictions: List[Dict[str, Any]] = []
    if args.candidate_predictions:
        candidate_rows = load_prediction_rows(args.candidate_predictions)
        candidate_eval, selected_predictions = evaluate_candidate_predictions(
            candidate_rows,
            candidate_arms=parse_csv(args.candidate_arms),
            split_field=str(args.candidate_split_field),
            eval_splits=parse_csv(args.candidate_eval_splits),
            weights=list(report["feature_weights"].values()),
            source_names=list(report["source_names"]),
            include_source_identity=bool(args.include_source_identity),
            status_weight=float(args.status_weight),
            fp_weight=float(args.fp_weight),
            fn_weight=float(args.fn_weight),
            min_candidate_rows=int(args.min_candidate_rows),
            max_finding_f1_drop=float(args.max_finding_f1_drop),
            max_token_f1_drop=float(args.max_token_f1_drop),
            max_fn_increase=float(args.max_fn_increase),
            max_fp_increase=float(args.max_fp_increase),
            output_split_rows=bool(args.output_predictions),
        )
        report["candidate_prediction_eval"] = candidate_eval
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(
            json.dumps(selected_predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        write_markdown(report, args.output_md)
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    if args.fail_on_no_go and report["gate"]["verdict"] != "GO":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

