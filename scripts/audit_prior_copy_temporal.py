#!/usr/bin/env python3
"""Audit prior-copy and temporal-use risks across counterfactual longitudinal arms."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

RE_WORD = re.compile(r"\w+")
DEFAULT_ARMS: Sequence[str] = (
    "current_only",
    "real_prior",
    "shuffled_prior",
    "prior_report_only",
)
TEMPORAL_TERMS: Sequence[str] = (
    "new",
    "unchanged",
    "stable",
    "improved",
    "improving",
    "worse",
    "worsened",
    "worsening",
    "increased",
    "decreased",
    "resolved",
    "persistent",
    "again",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_text(value: Any) -> str:
    return str(value or "").strip()


def tokenize(text: str) -> List[str]:
    return RE_WORD.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> Set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def prior_copy_precision(prediction: str, prior_report: str, ngram_n: int = 3) -> float:
    pred_t = tokenize(prediction)
    prior_t = tokenize(prior_report)
    pred_ng = ngrams(pred_t, ngram_n)
    prior_ng = ngrams(prior_t, ngram_n)
    if not pred_ng or not prior_ng:
        return 0.0
    return len(pred_ng & prior_ng) / len(pred_ng)


def extract_temporal_terms(text: str) -> Set[str]:
    words = set(tokenize(text))
    found: Set[str] = set()
    for term in TEMPORAL_TERMS:
        if term in words:
            found.add(term)
    return found


def iter_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        rows = payload.get("samples") or []
    else:
        raise RuntimeError("Expected JSON list rows or {'samples': [...]} payload")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}")
        out.append(row)
    return out


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return iter_rows(payload)


def metric_summary(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return round(float(sum(vals) / len(vals)), 6)


def evaluate_arm(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    copy_threshold: float,
    min_tokens_for_copy: int,
) -> Dict[str, Any]:
    arm_rows = 0
    nonempty = 0
    copy_hits = 0
    copy_scores: List[float] = []
    copy_denominator = 0
    temporal_mentions = 0
    unsupported_temporal = 0

    for row in rows:
        if arm not in row:
            continue
        arm_rows += 1
        pred = as_text(row.get(arm))
        prior = as_text(row.get("prior_report") or row.get("context"))
        reference = as_text(row.get("reference"))
        if not pred:
            continue
        nonempty += 1

        if prior:
            copy_denominator += 1
            score = prior_copy_precision(pred, prior)
            copy_scores.append(score)
            if len(tokenize(pred)) >= min_tokens_for_copy and score >= copy_threshold:
                copy_hits += 1

        pred_terms = extract_temporal_terms(pred)
        ref_terms = extract_temporal_terms(reference)
        if pred_terms:
            temporal_mentions += 1
            if pred_terms - ref_terms:
                unsupported_temporal += 1

    return {
        "arm": arm,
        "rows_with_arm": arm_rows,
        "nonempty_predictions": nonempty,
        "copy_denominator": copy_denominator,
        "prior_copy_rate": round(copy_hits / copy_denominator, 6) if copy_denominator > 0 else 0.0,
        "prior_copy_precision_mean": metric_summary(copy_scores),
        "temporal_mention_rate": round(temporal_mentions / nonempty, 6) if nonempty > 0 else 0.0,
        "unsupported_temporal_rate": round(unsupported_temporal / nonempty, 6) if nonempty > 0 else 0.0,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--copy-threshold", type=float, default=0.6)
    parser.add_argument("--min-tokens-for-copy", type=int, default=6)
    parser.add_argument("--max-real-copy-rate", type=float, default=0.35)
    parser.add_argument("--max-prior-only-copy-gap", type=float, default=0.05)
    parser.add_argument("--min-copy-rate-for-prior-only-gap", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.predictions.is_file():
        raise SystemExit(f"ERROR: predictions file not found: {args.predictions}")

    rows = load_rows(args.predictions)
    arms = [item.strip() for item in str(args.arms).split(",") if item.strip()]
    required = {"current_only", "real_prior", "shuffled_prior", "prior_report_only"}

    arm_metrics: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        arm_metrics[arm] = evaluate_arm(
            rows,
            arm,
            copy_threshold=float(args.copy_threshold),
            min_tokens_for_copy=int(args.min_tokens_for_copy),
        )

    operational_reasons: List[str] = []
    scientific_reasons: List[str] = []

    missing_required = sorted(required - set(arms))
    if missing_required:
        operational_reasons.append("missing_required_arms")

    sample_floor = int(args.min_samples)
    for arm in set(required) & set(arms):
        if int(arm_metrics[arm].get("nonempty_predictions", 0)) < sample_floor:
            operational_reasons.append(f"{arm}_below_min_samples")

    real_metrics = arm_metrics.get("real_prior", {})
    shuffled_metrics = arm_metrics.get("shuffled_prior", {})
    prior_only_metrics = arm_metrics.get("prior_report_only", {})

    real_copy_rate = float(real_metrics.get("prior_copy_rate", 0.0) or 0.0)
    shuffled_unsupported = float(shuffled_metrics.get("unsupported_temporal_rate", 0.0) or 0.0)
    real_unsupported = float(real_metrics.get("unsupported_temporal_rate", 0.0) or 0.0)
    prior_only_copy_rate = float(prior_only_metrics.get("prior_copy_rate", 0.0) or 0.0)

    if real_copy_rate > float(args.max_real_copy_rate):
        scientific_reasons.append("real_prior_copy_rate_high")
    if real_unsupported >= shuffled_unsupported:
        scientific_reasons.append("real_prior_not_better_than_shuffled_temporal")
    copy_signal_floor = float(args.min_copy_rate_for_prior_only_gap)
    if (
        max(prior_only_copy_rate, real_copy_rate) >= copy_signal_floor
        and abs(prior_only_copy_rate - real_copy_rate) <= float(args.max_prior_only_copy_gap)
    ):
        scientific_reasons.append("real_prior_copy_signal_matches_prior_report_only")

    causal_gap = {
        "unsupported_temporal_gap_shuffled_minus_real": round(shuffled_unsupported - real_unsupported, 6),
        "copy_rate_gap_shuffled_minus_real": round(
            float(shuffled_metrics.get("prior_copy_rate", 0.0) or 0.0) - real_copy_rate,
            6,
        ),
    }

    gate = {
        "operational_verdict": "GO" if not operational_reasons else "NO-GO",
        "scientific_screen": "GO" if not scientific_reasons else "NO-GO",
        "recommendation": (
            "NO-GO_OPERATIONAL"
            if operational_reasons
            else ("KILL_OR_REDESIGN" if scientific_reasons else "REVIEW_BEFORE_PROMOTION")
        ),
        "operational_reasons": sorted(set(operational_reasons)),
        "scientific_reasons": sorted(set(scientific_reasons)),
    }

    report = {
        "generated_utc": utc_now_iso(),
        "predictions": str(args.predictions),
        "rows": len(rows),
        "arms": arms,
        "arm_metrics": arm_metrics,
        "causal_gap": causal_gap,
        "config": {
            "min_samples": int(args.min_samples),
            "copy_threshold": float(args.copy_threshold),
            "min_tokens_for_copy": int(args.min_tokens_for_copy),
            "max_real_copy_rate": float(args.max_real_copy_rate),
            "max_prior_only_copy_gap": float(args.max_prior_only_copy_gap),
            "min_copy_rate_for_prior_only_gap": float(args.min_copy_rate_for_prior_only_gap),
        },
        "gate": gate,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

