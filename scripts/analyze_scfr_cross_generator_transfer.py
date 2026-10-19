#!/usr/bin/env python3
"""Analyze non-oracle SCFR transfer against paired current and wrong-history arms."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_source_causal_arms import reference_text, row_scores


METRICS = (
    "token_f1",
    "finding_f1",
    "finding_status_accuracy",
    "finding_fp_count",
    "finding_fn_count",
)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(rows: Sequence[Mapping[str, float]], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    per_policy: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    left: str,
    right: str,
    metric: str,
    cluster_field: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        cluster = str(row.get(cluster_field) or "").strip()
        if not cluster:
            raise RuntimeError(f"Missing cluster field {cluster_field} at row {index}")
        clusters[cluster].append(index)
    cluster_ids = sorted(clusters)
    point = _mean(per_policy[left], metric) - _mean(per_policy[right], metric)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_indexes = [
            index
            for cluster in sampled_clusters
            for index in clusters[cluster]
        ]
        left_mean = statistics.mean(
            float(per_policy[left][index][metric]) for index in sampled_indexes
        )
        right_mean = statistics.mean(
            float(per_policy[right][index][metric]) for index in sampled_indexes
        )
        draws.append(left_mean - right_mean)
    nonpositive = sum(value <= 0.0 for value in draws) / len(draws)
    nonnegative = sum(value >= 0.0 for value in draws) / len(draws)
    return {
        "estimate": point,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "p_two_sided_bootstrap": min(1.0, 2.0 * min(nonpositive, nonnegative)),
        "clusters": len(cluster_ids),
        "bootstrap_replicates": n_bootstrap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cluster-field", default="subject_id")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Expected a non-empty selected-prediction JSON list")
    per_policy: dict[str, list[dict[str, float]]] = {
        "current_only": [],
        "matched_wrong_history": [],
        "scfr": [],
    }
    routes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Row {index} is not an object")
        reference = reference_text(row)
        policies = {
            "current_only": str(row.get("current_only") or ""),
            "matched_wrong_history": str(row.get("matched_shuffled_prior") or ""),
            "scfr": str(row.get("prediction") or ""),
        }
        if not reference or any(not prediction for prediction in policies.values()):
            raise RuntimeError(f"Incomplete paired row at index {index}")
        for name, prediction in policies.items():
            score = row_scores({"prediction": prediction, "reference": reference})
            per_policy[name].append(
                {metric: float(score[metric]) for metric in METRICS}
            )
        routes[str(row.get("selected_arm") or "unknown")] += 1

    absolute = {
        policy: {
            metric: _mean(scores, metric)
            for metric in METRICS
        }
        for policy, scores in per_policy.items()
    }
    contrasts = {}
    for left, right in (
        ("scfr", "current_only"),
        ("scfr", "matched_wrong_history"),
        ("matched_wrong_history", "current_only"),
    ):
        contrast_name = f"{left}_minus_{right}"
        contrasts[contrast_name] = {
            metric: _paired_bootstrap(
                rows,
                per_policy,
                left=left,
                right=right,
                metric=metric,
                cluster_field=args.cluster_field,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    current_route_count = sum(
        count for route, count in routes.items() if route.startswith("current_only")
    )
    report = {
        "schema_version": "scfr-cross-generator-transfer-analysis-v1",
        "label": args.label,
        "rows": len(rows),
        "subjects": len({str(row[args.cluster_field]) for row in rows}),
        "route_counts": dict(sorted(routes.items())),
        "non_current_route_rate": 1.0 - current_route_count / len(rows),
        "absolute": absolute,
        "contrasts": contrasts,
        "oracle_selection_used": False,
        "reference_available_to_controller_at_inference": False,
        "bootstrap_cluster": args.cluster_field,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        lines = [
            f"# {args.label}",
            "",
            f"Paired cases: **{report['rows']}**; subjects: **{report['subjects']}**.",
            f"Non-current route rate: **{100 * report['non_current_route_rate']:.1f}%**.",
            "",
            "| Policy | Finding-F1 | Status acc. | FP/report | FN/report | Token-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for policy in ("current_only", "matched_wrong_history", "scfr"):
            value = absolute[policy]
            lines.append(
                f"| {policy} | {value['finding_f1']:.4f} | "
                f"{value['finding_status_accuracy']:.4f} | "
                f"{value['finding_fp_count']:.4f} | "
                f"{value['finding_fn_count']:.4f} | "
                f"{value['token_f1']:.4f} |"
            )
        lines.extend(["", "No oracle or reference-conditioned route selection was used."])
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "absolute": absolute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

