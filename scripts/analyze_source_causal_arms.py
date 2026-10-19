#!/usr/bin/env python3
"""Analyze source-causal report-generation arms with paired bootstrap gates."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FINDING_PATTERNS: Dict[str, Sequence[str]] = {
    "atelectasis": (r"\batelecta(?:sis|tic)\b",),
    "cardiomegaly": (
        r"\bcardiomegaly\b",
        r"\bcardiac silhouette (?:is )?(?:mildly |moderately |markedly )?enlarged\b",
        r"\bheart (?:size )?(?:is )?(?:mildly |moderately |markedly )?enlarged\b",
    ),
    "consolidation": (r"\bconsolidat(?:ion|ive)\b",),
    "edema": (r"\bedema\b", r"\bvascular congestion\b", r"\bpulmonary vascular congestion\b"),
    "fracture": (r"\bfracture\b",),
    "mass_or_nodule": (r"\bnodule\b", r"\bnodular\b", r"\bmass(?:es)?\b"),
    "opacity": (r"\bopacit(?:y|ies)\b", r"\binfiltrat(?:e|es|ion)\b"),
    "pleural_effusion": (r"\bpleural effusion(?:s)?\b", r"\beffusion(?:s)?\b"),
    "pneumonia": (r"\bpneumonia\b", r"\binfection\b", r"\baspiration\b"),
    "pneumothorax": (r"\bpneumothorax\b",),
    "support_device": (
        r"\bpicc\b",
        r"\bcentral venous catheter\b",
        r"\bendotracheal tube\b",
        r"\benteric tube\b",
        r"\bchest tube\b",
        r"\bline tip\b",
    ),
}

NEGATION_RE = re.compile(
    r"\b("
    r"no|without|negative for|free of|absence of|absent|"
    r"no evidence of|no definite|no focal|no convincing|not seen|"
    r"not identified|not present|resolved|clear of"
    r")\b",
    re.IGNORECASE,
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        out.append(row)
    return out


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.;:])\s+|\n+", str(text or "")) if part.strip()]


def mention_is_negated(sentence: str, start: int) -> bool:
    prefix = sentence[max(0, start - 80) : start]
    if not NEGATION_RE.search(prefix):
        return False
    tail = sentence[start : start + 80].lower()
    if "cannot be excluded" in tail or "not excluded" in tail:
        return False
    return True


def finding_status(text: str, patterns: Sequence[str]) -> str:
    saw_negative = False
    for sentence in split_sentences(text):
        for pattern in patterns:
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                if mention_is_negated(sentence, match.start()):
                    saw_negative = True
                else:
                    return "positive"
    return "negative" if saw_negative else "absent"


def finding_vector(text: str) -> Dict[str, str]:
    return {
        finding: finding_status(text, patterns)
        for finding, patterns in FINDING_PATTERNS.items()
    }


def positive_finding_f1(prediction: str, reference: str) -> float:
    pred_vec = finding_vector(prediction)
    ref_vec = finding_vector(reference)
    pred_pos = {k for k, v in pred_vec.items() if v == "positive"}
    ref_pos = {k for k, v in ref_vec.items() if v == "positive"}
    if not pred_pos and not ref_pos:
        return 1.0
    if not pred_pos or not ref_pos:
        return 0.0
    overlap = len(pred_pos & ref_pos)
    precision = overlap / len(pred_pos)
    recall = overlap / len(ref_pos)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def finding_status_accuracy(prediction: str, reference: str) -> float:
    pred_vec = finding_vector(prediction)
    ref_vec = finding_vector(reference)
    return sum(pred_vec[k] == ref_vec[k] for k in FINDING_PATTERNS) / len(FINDING_PATTERNS)


def finding_error_counts(prediction: str, reference: str) -> Tuple[int, int]:
    pred_vec = finding_vector(prediction)
    ref_vec = finding_vector(reference)
    pred_pos = {k for k, v in pred_vec.items() if v == "positive"}
    ref_pos = {k for k, v in ref_vec.items() if v == "positive"}
    return len(pred_pos - ref_pos), len(ref_pos - pred_pos)


def reference_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get("reference") or "").split())


def row_with_reference(row: Mapping[str, Any], reference: str) -> Dict[str, Any]:
    patched = dict(row)
    patched["reference"] = reference
    return patched


def per_finding_summary(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    reference_by_id: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = {
        finding: {
            "reference_positive": 0,
            "prediction_positive": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "status_correct": 0,
        }
        for finding in FINDING_PATTERNS
    }
    for sample_id in ids:
        row = rows_by_id[sample_id]
        pred_vec = finding_vector(str(row.get("prediction") or ""))
        reference = (
            reference_by_id[sample_id]
            if reference_by_id is not None
            else str(row.get("reference") or "")
        )
        ref_vec = finding_vector(reference)
        for finding in FINDING_PATTERNS:
            pred_pos = pred_vec[finding] == "positive"
            ref_pos = ref_vec[finding] == "positive"
            bucket = counts[finding]
            bucket["reference_positive"] += int(ref_pos)
            bucket["prediction_positive"] += int(pred_pos)
            bucket["true_positive"] += int(pred_pos and ref_pos)
            bucket["false_positive"] += int(pred_pos and not ref_pos)
            bucket["false_negative"] += int((not pred_pos) and ref_pos)
            bucket["status_correct"] += int(pred_vec[finding] == ref_vec[finding])

    out: Dict[str, Any] = {}
    n_rows = len(ids)
    for finding, bucket in counts.items():
        tp = bucket["true_positive"]
        fp = bucket["false_positive"]
        fn = bucket["false_negative"]
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[finding] = {
            **bucket,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "status_accuracy": bucket["status_correct"] / n_rows if n_rows else 0.0,
        }
    return out


def bootstrap_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
    clusters: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if len(left) != len(right):
        raise ValueError("bootstrap vectors must have equal length")
    if clusters is not None and len(clusters) != len(left):
        raise ValueError("cluster labels must match bootstrap vector length")
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    observed = statistics.mean(deltas) if deltas else 0.0
    if not deltas or n_bootstrap <= 0:
        return {
            "observed_delta": observed,
            "ci95": [None, None],
            "p_two_sided": None,
            "n_bootstrap": n_bootstrap,
            "bootstrap_unit": "cluster" if clusters is not None else "row",
            "n_clusters": len(set(clusters or [])) if clusters is not None else None,
        }
    rng = random.Random(seed)
    samples: List[float] = []
    if clusters is None:
        sample_size = len(deltas)
        for _ in range(n_bootstrap):
            total = 0.0
            for _idx in range(sample_size):
                total += deltas[rng.randrange(sample_size)]
            samples.append(total / sample_size)
        n_clusters = None
    else:
        buckets: Dict[str, List[float]] = {}
        for delta, cluster in zip(deltas, clusters):
            buckets.setdefault(str(cluster), []).append(delta)
        cluster_keys = sorted(buckets)
        n_clusters = len(cluster_keys)
        for _ in range(n_bootstrap):
            total = 0.0
            count = 0
            for _idx in range(n_clusters):
                key = cluster_keys[rng.randrange(n_clusters)]
                values = buckets[key]
                total += sum(values)
                count += len(values)
            samples.append(total / count if count else 0.0)
    samples.sort()
    lower = samples[int(0.025 * len(samples))]
    upper = samples[int(0.975 * len(samples))]
    p_left = (sum(value <= 0 for value in samples) + 1) / (len(samples) + 1)
    p_right = (sum(value >= 0 for value in samples) + 1) / (len(samples) + 1)
    return {
        "observed_delta": observed,
        "ci95": [lower, upper],
        "p_two_sided": min(1.0, 2.0 * min(p_left, p_right)),
        "n_bootstrap": n_bootstrap,
        "bootstrap_unit": "cluster" if clusters is not None else "row",
        "n_clusters": n_clusters,
    }


def parse_arm(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected ARM=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("ARM name is empty")
    return name, Path(path.strip())


def sample_key(row: Mapping[str, Any], *, arm: str, idx: int) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise RuntimeError(f"{arm}: missing sample key at row {idx}; expected sample_id/id/study_id")


def index_arm_rows(arm: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for idx, row in enumerate(rows):
        key = sample_key(row, arm=arm, idx=idx)
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
    if duplicates:
        preview = ",".join(sorted(set(duplicates))[:10])
        raise RuntimeError(f"{arm}: duplicate sample keys: {preview}")
    return indexed


def align_arms(arms: Mapping[str, List[Dict[str, Any]]]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
    indexed = {name: index_arm_rows(name, rows) for name, rows in arms.items()}
    common = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    return common, indexed


def row_scores(row: Mapping[str, Any]) -> Dict[str, float]:
    prediction = str(row.get("prediction") or "")
    reference = str(row.get("reference") or "")
    fp, fn = finding_error_counts(prediction, reference)
    return {
        "token_f1": token_f1(prediction, reference),
        "finding_f1": positive_finding_f1(prediction, reference),
        "finding_status_accuracy": finding_status_accuracy(prediction, reference),
        "finding_fp_count": float(fp),
        "finding_fn_count": float(fn),
        "prediction_tokens": float(len(tokenize(prediction))),
    }


def summarize_arm(
    name: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    scores_by_id: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    predictions = [str(rows_by_id[i].get("prediction") or "").strip() for i in ids]
    scores = [
        scores_by_id[i] if scores_by_id is not None else row_scores(rows_by_id[i])
        for i in ids
    ]
    metric_names = list(scores[0]) if scores else []
    return {
        "rows": len(ids),
        "nonempty_predictions": sum(bool(text) for text in predictions),
        "unique_predictions": len(set(predictions)),
        "prior_image_rows": sum(bool(rows_by_id[i].get("prior_image_paths")) for i in ids),
        "prior_report_rows": sum(bool(str(rows_by_id[i].get("prior_report") or "").strip()) for i in ids),
        "metrics": {
            metric: statistics.mean(float(row[metric]) for row in scores) if scores else 0.0
            for metric in metric_names
        },
    }


def pairwise_similarity(
    left_rows: Mapping[str, Mapping[str, Any]],
    right_rows: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
) -> Dict[str, float]:
    ratios = [
        difflib.SequenceMatcher(
            None,
            str(left_rows[i].get("prediction") or ""),
            str(right_rows[i].get("prediction") or ""),
        ).ratio()
        for i in ids
    ]
    exact = sum(
        str(left_rows[i].get("prediction") or "").strip()
        == str(right_rows[i].get("prediction") or "").strip()
        for i in ids
    )
    return {
        "exact_same": exact,
        "mean_sequence_ratio": statistics.mean(ratios) if ratios else 0.0,
        "median_sequence_ratio": statistics.median(ratios) if ratios else 0.0,
        "min_sequence_ratio": min(ratios) if ratios else 0.0,
        "max_sequence_ratio": max(ratios) if ratios else 0.0,
    }


def build_analysis(
    arm_paths: Sequence[Tuple[str, Path]],
    *,
    baseline: str,
    n_bootstrap: int,
    seed: int,
    min_samples: int = 1,
    cluster_by: str = "",
) -> Dict[str, Any]:
    rows = {name: load_rows(path) for name, path in arm_paths}
    ids, indexed = align_arms(rows)
    if baseline not in indexed:
        raise RuntimeError(f"Baseline arm not found: {baseline}")

    baseline_reference_by_id = {
        sample_id: reference_text(indexed[baseline][sample_id])
        for sample_id in ids
    }
    reference_mismatch_rows = 0
    reference_mismatch_examples: List[Dict[str, Any]] = []
    for sample_id in ids:
        baseline_ref = baseline_reference_by_id[sample_id]
        mismatched_arms = [
            name
            for name, _path in arm_paths
            if reference_text(indexed[name][sample_id]) != baseline_ref
        ]
        if mismatched_arms:
            reference_mismatch_rows += 1
            if len(reference_mismatch_examples) < 10:
                reference_mismatch_examples.append(
                    {
                        "sample_id": sample_id,
                        "baseline": baseline,
                        "mismatched_arms": mismatched_arms,
                    }
                )

    score_cache = {
        name: {
            sample_id: row_scores(
                row_with_reference(indexed[name][sample_id], baseline_reference_by_id[sample_id])
            )
            for sample_id in ids
        }
        for name, _path in arm_paths
    }
    arm_summaries = {
        name: summarize_arm(name, indexed[name], ids, score_cache[name])
        for name, _path in arm_paths
    }
    finding_summaries = {
        name: per_finding_summary(indexed[name], ids, baseline_reference_by_id)
        for name, _path in arm_paths
    }
    per_arm_scores = {
        name: {
            metric: [score_cache[name][sample_id][metric] for sample_id in ids]
            for metric in arm_summaries[name]["metrics"]
        }
        for name, _path in arm_paths
    }
    cluster_labels: Optional[List[str]] = None
    missing_cluster_rows = 0
    if cluster_by:
        cluster_labels = []
        for sample_id in ids:
            value = str(indexed[baseline][sample_id].get(cluster_by) or "").strip()
            if not value:
                missing_cluster_rows += 1
                value = f"__missing_cluster__:{sample_id}"
            cluster_labels.append(value)

    metric_deltas: Dict[str, Any] = {}
    for name, _path in arm_paths:
        if name == baseline:
            continue
        metric_deltas[name] = {}
        for metric, values in per_arm_scores[name].items():
            metric_deltas[name][metric] = bootstrap_delta(
                values,
                per_arm_scores[baseline][metric],
                n_bootstrap=n_bootstrap,
                seed=seed,
                clusters=cluster_labels,
            )

    pairwise: Dict[str, Any] = {}
    for idx, (left, _left_path) in enumerate(arm_paths):
        for right, _right_path in arm_paths[idx + 1 :]:
            key = f"{left}__vs__{right}"
            pairwise[key] = pairwise_similarity(indexed[left], indexed[right], ids)

    hard_invalid = []
    if len(ids) < min_samples:
        hard_invalid.append(f"paired_rows_below_min:{len(ids)}<{min_samples}")
    if reference_mismatch_rows:
        hard_invalid.append(f"cross_arm_reference_mismatch:{reference_mismatch_rows}")
    for name, summary in arm_summaries.items():
        if summary["rows"] == 0:
            hard_invalid.append(f"{name}:zero_rows")
        if summary["nonempty_predictions"] == 0:
            hard_invalid.append(f"{name}:zero_nonempty_predictions")
        if summary["unique_predictions"] <= 1 and summary["rows"] > 1:
            hard_invalid.append(f"{name}:prediction_collapse")

    return {
        "sample_count": len(ids),
        "min_samples": int(min_samples),
        "baseline": baseline,
        "bootstrap": {
            "unit": "cluster" if cluster_labels is not None else "row",
            "cluster_by": cluster_by,
            "clusters": len(set(cluster_labels or [])) if cluster_labels is not None else None,
            "missing_cluster_rows": missing_cluster_rows if cluster_labels is not None else None,
        },
        "reference_consistency": {
            "baseline": baseline,
            "mismatch_rows": reference_mismatch_rows,
            "mismatch_examples": reference_mismatch_examples,
        },
        "alignment": {
            name: {
                "input_rows": len(rows[name]),
                "paired_rows": len(ids),
                "paired_coverage": len(ids) / len(rows[name]) if rows[name] else 0.0,
            }
            for name, _path in arm_paths
        },
        "arms": arm_summaries,
        "per_finding": finding_summaries,
        "deltas_vs_baseline": metric_deltas,
        "pairwise_prediction_similarity": pairwise,
        "gate": {
            "hard_invalid": bool(hard_invalid),
            "hard_invalid_reasons": hard_invalid,
        },
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Arm Analysis",
        "",
        f"Common paired samples: {report['sample_count']}",
        f"Baseline: `{report['baseline']}`",
        f"Bootstrap unit: `{report.get('bootstrap', {}).get('unit', 'row')}`",
        f"Cluster field: `{report.get('bootstrap', {}).get('cluster_by') or 'none'}`",
        f"Clusters: `{report.get('bootstrap', {}).get('clusters') or 'n/a'}`",
        f"Reference mismatch rows: `{report.get('reference_consistency', {}).get('mismatch_rows', 0)}`",
        "",
        "## Arm Summary",
        "",
        "| Arm | Rows | Non-empty | Unique | Prior Images | Prior Reports | Token-F1 | Finding-F1 | Finding Acc | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, summary in report["arms"].items():
        metrics = summary["metrics"]
        lines.append(
            f"| `{arm}` | {summary['rows']} | {summary['nonempty_predictions']} | "
            f"{summary['unique_predictions']} | {summary['prior_image_rows']} | "
            f"{summary['prior_report_rows']} | {metrics['token_f1']:.4f} | "
            f"{metrics['finding_f1']:.4f} | {metrics['finding_status_accuracy']:.4f} | "
            f"{metrics['finding_fp_count']:.4f} | {metrics['finding_fn_count']:.4f} |"
        )
    lines.extend(["", "## Deltas vs Baseline", ""])
    for arm, metrics in report["deltas_vs_baseline"].items():
        lines.append(f"### `{arm}`")
        lines.append("")
        lines.append("| Metric | Delta | 95% CI | p(two-sided) |")
        lines.append("|---|---:|---:|---:|")
        for metric, payload in metrics.items():
            ci = payload["ci95"]
            ci_text = "n/a" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            p_value = payload["p_two_sided"]
            p_text = "n/a" if p_value is None else f"{p_value:.4f}"
            lines.append(
                f"| `{metric}` | {payload['observed_delta']:.4f} | {ci_text} | {p_text} |"
            )
        lines.append("")
    lines.extend(["## Per-Finding Summary", ""])
    for arm, finding_payload in report["per_finding"].items():
        lines.append(f"### `{arm}`")
        lines.append("")
        lines.append("| Finding | Ref+ | Pred+ | TP | FP | FN | F1 | Status Acc |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for finding, metrics in finding_payload.items():
            lines.append(
                f"| `{finding}` | {metrics['reference_positive']} | "
                f"{metrics['prediction_positive']} | {metrics['true_positive']} | "
                f"{metrics['false_positive']} | {metrics['false_negative']} | "
                f"{metrics['f1']:.4f} | {metrics['status_accuracy']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Gate", ""])
    if report["gate"]["hard_invalid"]:
        lines.append("Hard invalidity: FAIL")
        for reason in report["gate"]["hard_invalid_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("Hard invalidity: PASS")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--baseline", default="current_only")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--cluster-by",
        default="",
        help="Optional row field used as bootstrap cluster unit, e.g. subject_id.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_analysis(
        args.arm,
        baseline=args.baseline,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        min_samples=args.min_samples,
        cluster_by=args.cluster_by,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_md:
        write_markdown(report, args.output_md)
    return 2 if report["gate"]["hard_invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

