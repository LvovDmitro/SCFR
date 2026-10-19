#!/usr/bin/env python3
"""Build source-causal chosen/rejected pairs from counterfactual arm outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from scripts.analyze_source_causal_arms import row_scores
    from scripts.source_causal_per_finding_gate import as_text, parse_csv
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import row_scores
    from scripts.source_causal_per_finding_gate import as_text, parse_csv


DEFAULT_DEPLOYABLE_ARMS = ("current_only", "real_prior", "prior_report_only")
DEFAULT_REJECTED_ARMS = ("shuffled_prior", "matched_shuffled_prior")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def objective(
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


def arm_score(
    row: Mapping[str, Any],
    arm: str,
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Optional[Dict[str, Any]]:
    text = as_text(row.get(arm))
    if not text.strip():
        return None
    metrics = row_scores({"prediction": text, "reference": as_text(row.get("reference"))})
    score = objective(
        metrics,
        status_weight=status_weight,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    return {
        "arm": arm,
        "text": text,
        "score": score,
        "metrics": metrics,
    }


def split_allowed(row: Mapping[str, Any], train_splits: Sequence[str]) -> bool:
    return as_text(row.get("split") or "unspecified") in set(train_splits)


def build_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    deployable_arms: Sequence[str],
    rejected_arms: Sequence[str],
    train_splits: Sequence[str],
    min_margin: float,
    max_pairs: int,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for row in rows:
        if not split_allowed(row, train_splits):
            continue
        deployable = [
            item
            for arm in deployable_arms
            for item in [arm_score(
                row,
                arm,
                status_weight=status_weight,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )]
            if item is not None
        ]
        rejected = [
            item
            for arm in rejected_arms
            for item in [arm_score(
                row,
                arm,
                status_weight=status_weight,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )]
            if item is not None
        ]
        if not deployable or not rejected:
            continue

        chosen = max(deployable, key=lambda item: float(item["score"]))
        loser = min(rejected, key=lambda item: float(item["score"]))
        if norm_text(chosen["text"]) == norm_text(loser["text"]):
            continue
        margin = float(chosen["score"]) - float(loser["score"])
        if margin < min_margin:
            continue

        pairs.append(
            {
                "sample_id": row.get("sample_id") or row.get("id") or row.get("study_id"),
                "split": row.get("split"),
                "reference": row.get("reference", ""),
                "image_paths": as_list(row.get("image_paths")),
                "prior_image_paths": as_list(row.get("prior_image_paths")),
                "context": row.get("context"),
                "chosen": chosen["text"],
                "rejected": loser["text"],
                "chosen_source": chosen["arm"],
                "rejected_source": loser["arm"],
                "chosen_reward": round(float(chosen["score"]), 6),
                "rejected_reward": round(float(loser["score"]), 6),
                "reward_margin": round(margin, 6),
                "reward_name": "source_causal_objective",
                "metadata": {
                    "objective": {
                        "finding_f1": 1.0,
                        "status_weight": status_weight,
                        "fp_weight": fp_weight,
                        "fn_weight": fn_weight,
                    },
                    "chosen_metrics": chosen["metrics"],
                    "rejected_metrics": loser["metrics"],
                    "candidate_scores": [
                        {
                            "arm": item["arm"],
                            "score": round(float(item["score"]), 6),
                        }
                        for item in sorted(
                            deployable + rejected,
                            key=lambda item: float(item["score"]),
                            reverse=True,
                        )
                    ],
                },
            }
        )
        if max_pairs > 0 and len(pairs) >= max_pairs:
            break
    return pairs


def summarize(
    pairs: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    train_splits: Sequence[str],
    min_pairs: int,
) -> Dict[str, Any]:
    split_counts = Counter(as_text(pair.get("split")) for pair in pairs)
    chosen_counts = Counter(as_text(pair.get("chosen_source")) for pair in pairs)
    rejected_counts = Counter(as_text(pair.get("rejected_source")) for pair in pairs)
    margins = [float(pair.get("reward_margin", 0.0) or 0.0) for pair in pairs]
    test_pairs = split_counts.get("test", 0)
    image_rows = sum(bool(pair.get("image_paths")) for pair in pairs)
    prior_image_rows = sum(bool(pair.get("prior_image_paths")) for pair in pairs)
    source_rows = sum(1 for row in rows if split_allowed(row, train_splits))
    reasons = []
    if len(pairs) < min_pairs:
        reasons.append(f"pairs_below_min:{len(pairs)}<{min_pairs}")
    if test_pairs:
        reasons.append(f"heldout_test_pairs_present:{test_pairs}")
    if image_rows < len(pairs):
        reasons.append("missing_current_images")
    return {
        "pairs": len(pairs),
        "source_rows_in_train_splits": source_rows,
        "train_splits": list(train_splits),
        "split_counts": dict(split_counts),
        "chosen_source_counts": dict(chosen_counts),
        "rejected_source_counts": dict(rejected_counts),
        "image_rows": image_rows,
        "prior_image_rows": prior_image_rows,
        "reward_margin": {
            "mean": statistics.mean(margins) if margins else 0.0,
            "median": statistics.median(margins) if margins else 0.0,
            "min": min(margins) if margins else 0.0,
            "max": max(margins) if margins else 0.0,
        },
        "gate": {
            "verdict": "GO" if not reasons else "NO-GO",
            "reasons": reasons,
            "min_pairs": min_pairs,
        },
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Source-Causal Preference Pair Gate",
        "",
        f"Pairs: `{summary['pairs']}`",
        f"Source train/validate rows: `{summary['source_rows_in_train_splits']}`",
        f"Gate: `{summary['gate']['verdict']}`",
        f"Gate reasons: `{', '.join(summary['gate']['reasons']) or 'none'}`",
        "",
        "## Sources",
        "",
        "| Type | Counts |",
        "|---|---|",
        f"| Split | `{summary['split_counts']}` |",
        f"| Chosen | `{summary['chosen_source_counts']}` |",
        f"| Rejected | `{summary['rejected_source_counts']}` |",
        "",
        "## Reward Margin",
        "",
        "| Mean | Median | Min | Max |",
        "|---:|---:|---:|---:|",
        (
            f"| {summary['reward_margin']['mean']:.4f} | "
            f"{summary['reward_margin']['median']:.4f} | "
            f"{summary['reward_margin']['min']:.4f} | "
            f"{summary['reward_margin']['max']:.4f} |"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--deployable-arms", default=",".join(DEFAULT_DEPLOYABLE_ARMS))
    parser.add_argument("--rejected-arms", default=",".join(DEFAULT_REJECTED_ARMS))
    parser.add_argument("--train-splits", default="train,validate")
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--status-weight", type=float, default=0.25)
    parser.add_argument("--fp-weight", type=float, default=0.5)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.predictions)
    train_splits = parse_csv(args.train_splits)
    pairs = build_pairs(
        rows,
        deployable_arms=parse_csv(args.deployable_arms),
        rejected_arms=parse_csv(args.rejected_arms),
        train_splits=train_splits,
        min_margin=args.min_margin,
        max_pairs=args.max_pairs,
        status_weight=args.status_weight,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
    )
    summary = summarize(
        pairs,
        rows=rows,
        train_splits=train_splits,
        min_pairs=args.min_pairs,
    )
    write_jsonl(args.output_jsonl, pairs)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(summary, args.output_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate"]["verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

