#!/usr/bin/env python3
"""Build denser source-causal self-correcting preference pairs.

The conservative pair builder emits one pair per row. This miner emits multiple
explicit-reward pairs when arm deltas are clinically separable and keeps test
rows untouched. The output is intended for DPO/corrector staging, not for direct
claim summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import finding_error_counts, finding_vector, row_scores
    from scripts.source_causal_per_finding_gate import as_text, parse_csv
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_error_counts, finding_vector, row_scores
    from scripts.source_causal_per_finding_gate import as_text, parse_csv


DEFAULT_ARMS = (
    "current_only",
    "real_prior",
    "prior_report_only",
    "matched_shuffled_prior",
    "shuffled_prior",
)
DEFAULT_DEPLOYABLE_ARMS = ("current_only", "real_prior", "prior_report_only")
DEFAULT_WRONG_HISTORY_ARMS = ("matched_shuffled_prior", "shuffled_prior")
PAIR_POLICIES = ("deployable_main", "exploratory")


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
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
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


def arm_payload(
    row: Mapping[str, Any],
    arm: str,
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Optional[Dict[str, Any]]:
    text = as_text(row.get(arm))
    if not text:
        return None
    reference = as_text(row.get("reference"))
    metrics = row_scores({"prediction": text, "reference": reference})
    score = objective(metrics, status_weight=status_weight, fp_weight=fp_weight, fn_weight=fn_weight)
    fp, fn = finding_error_counts(text, reference)
    return {
        "arm": arm,
        "text": text,
        "metrics": metrics,
        "score": score,
        "fp": fp,
        "fn": fn,
        "positive_findings": sorted(
            finding for finding, status in finding_vector(text).items() if status == "positive"
        ),
    }


def pair_group(chosen: Mapping[str, Any], rejected: Mapping[str, Any], wrong_history_arms: Sequence[str]) -> str:
    rejected_arm = str(rejected["arm"])
    chosen_arm = str(chosen["arm"])
    fp_delta = float(rejected["metrics"].get("finding_fp_count", 0.0)) - float(chosen["metrics"].get("finding_fp_count", 0.0))
    fn_delta = float(rejected["metrics"].get("finding_fn_count", 0.0)) - float(chosen["metrics"].get("finding_fn_count", 0.0))
    if rejected_arm in wrong_history_arms and fp_delta > 0:
        return "source_invalid_fp"
    if fn_delta > 0:
        return "missed_reference_finding"
    if rejected_arm in wrong_history_arms:
        return "source_history_quality"
    if chosen_arm == "real_prior" and rejected_arm == "current_only":
        return "valid_prior_recall_gain"
    return "clinical_preference"


def split_allowed(row: Mapping[str, Any], train_splits: Sequence[str]) -> bool:
    return as_text(row.get("split") or "unspecified") in set(train_splits)


def build_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    arms: Sequence[str],
    deployable_arms: Sequence[str],
    wrong_history_arms: Sequence[str],
    pair_policy: str,
    train_splits: Sequence[str],
    min_margin: float,
    max_pairs: int,
    max_pairs_per_row: int,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> List[Dict[str, Any]]:
    if pair_policy not in PAIR_POLICIES:
        raise ValueError(f"Unknown pair_policy={pair_policy!r}; expected one of {PAIR_POLICIES}")
    pairs: List[Dict[str, Any]] = []
    deployable_set = set(deployable_arms)
    wrong_history_set = set(wrong_history_arms)
    for row in rows:
        if not split_allowed(row, train_splits):
            continue
        scored = [
            item
            for arm in arms
            for item in [
                arm_payload(
                    row,
                    arm,
                    status_weight=status_weight,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                )
            ]
            if item is not None
        ]
        row_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any], str]] = []
        for chosen in scored:
            for rejected in scored:
                if chosen["arm"] == rejected["arm"]:
                    continue
                if norm_text(chosen["text"]) == norm_text(rejected["text"]):
                    continue
                margin = float(chosen["score"]) - float(rejected["score"])
                if margin < min_margin:
                    continue
                if pair_policy == "deployable_main":
                    if chosen["arm"] not in deployable_set or rejected["arm"] not in wrong_history_set:
                        continue
                else:
                    # At least one side must make this source-causal or deployable-safety relevant.
                    if not (
                        rejected["arm"] in wrong_history_set
                        or chosen["arm"] in deployable_set
                        or rejected["arm"] in deployable_set
                    ):
                        continue
                group = pair_group(chosen, rejected, wrong_history_arms)
                row_pairs.append((margin, chosen, rejected, group))

        row_pairs.sort(key=lambda item: item[0], reverse=True)
        if max_pairs_per_row > 0:
            row_pairs = row_pairs[:max_pairs_per_row]

        for margin, chosen, rejected, group in row_pairs:
            pair = {
                "sample_id": row.get("sample_id") or row.get("id") or row.get("study_id"),
                "subject_id": row.get("subject_id"),
                "study_id": row.get("study_id"),
                "split": row.get("split"),
                "reference": row.get("reference", ""),
                "image_paths": as_list(row.get("image_paths")),
                "prior_image_paths": as_list(row.get("prior_image_paths")),
                "context": row.get("context") or row.get("prior_report"),
                "chosen": chosen["text"],
                "rejected": rejected["text"],
                "chosen_source": chosen["arm"],
                "rejected_source": rejected["arm"],
                "chosen_reward": round(float(chosen["score"]), 6),
                "rejected_reward": round(float(rejected["score"]), 6),
                "reward_margin": round(float(margin), 6),
                "reward_name": "source_causal_self_correction_objective",
                "error_group": group,
                "metadata": {
                    "pair_policy": pair_policy,
                    "objective": {
                        "finding_f1": 1.0,
                        "status_weight": status_weight,
                        "fp_weight": fp_weight,
                        "fn_weight": fn_weight,
                    },
                    "chosen_metrics": chosen["metrics"],
                    "rejected_metrics": rejected["metrics"],
                    "chosen_positive_findings": chosen["positive_findings"],
                    "rejected_positive_findings": rejected["positive_findings"],
                },
            }
            pairs.append(pair)
            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs
    return pairs


def summarize(
    pairs: Sequence[Mapping[str, Any]],
    *,
    rows: Sequence[Mapping[str, Any]],
    train_splits: Sequence[str],
    deployable_arms: Sequence[str],
    wrong_history_arms: Sequence[str],
    pair_policy: str,
    min_pairs: int,
    max_group_fraction: float,
) -> Dict[str, Any]:
    deployable_set = set(deployable_arms)
    wrong_history_set = set(wrong_history_arms)
    split_counts = Counter(as_text(pair.get("split")) for pair in pairs)
    chosen_counts = Counter(as_text(pair.get("chosen_source")) for pair in pairs)
    rejected_counts = Counter(as_text(pair.get("rejected_source")) for pair in pairs)
    group_counts = Counter(as_text(pair.get("error_group")) for pair in pairs)
    margins = [float(pair.get("reward_margin", 0.0) or 0.0) for pair in pairs]
    image_rows = sum(bool(pair.get("image_paths")) for pair in pairs)
    source_rows = sum(1 for row in rows if split_allowed(row, train_splits))
    unique_sample_ids = {as_text(pair.get("sample_id")) for pair in pairs if as_text(pair.get("sample_id"))}
    unique_subject_ids = {as_text(pair.get("subject_id")) for pair in pairs if as_text(pair.get("subject_id"))}
    chosen_wrong = sum(as_text(pair.get("chosen_source")) in wrong_history_set for pair in pairs)
    rejected_not_wrong = sum(as_text(pair.get("rejected_source")) not in wrong_history_set for pair in pairs)
    chosen_not_deployable = sum(as_text(pair.get("chosen_source")) not in deployable_set for pair in pairs)
    wrong_beats_deployable = sum(
        as_text(pair.get("chosen_source")) in wrong_history_set
        and as_text(pair.get("rejected_source")) in deployable_set
        for pair in pairs
    )
    max_group = max(group_counts.values()) if group_counts else 0
    reasons: List[str] = []
    if len(pairs) < min_pairs:
        reasons.append(f"pairs_below_min:{len(pairs)}<{min_pairs}")
    if split_counts.get("test", 0):
        reasons.append(f"heldout_test_pairs_present:{split_counts['test']}")
    if image_rows < len(pairs):
        reasons.append("missing_current_images")
    if pairs and max_group / len(pairs) > max_group_fraction:
        reasons.append(f"dominant_error_group:{max_group / len(pairs):.4f}>{max_group_fraction}")
    if margins and min(margins) <= 0:
        reasons.append("non_positive_margin")
    if pair_policy == "deployable_main":
        if chosen_wrong:
            reasons.append(f"chosen_wrong_history_rows:{chosen_wrong}")
        if chosen_not_deployable:
            reasons.append(f"chosen_not_deployable_rows:{chosen_not_deployable}")
        if rejected_not_wrong:
            reasons.append(f"rejected_not_wrong_history_rows:{rejected_not_wrong}")
        if wrong_beats_deployable:
            reasons.append(f"wrong_history_beats_deployable_rows:{wrong_beats_deployable}")
    return {
        "pairs": len(pairs),
        "source_rows_in_train_splits": source_rows,
        "unique_sample_id": len(unique_sample_ids),
        "unique_subject_id": len(unique_subject_ids),
        "pair_policy": pair_policy,
        "train_splits": list(train_splits),
        "deployable_arms": list(deployable_arms),
        "wrong_history_arms": list(wrong_history_arms),
        "split_counts": dict(split_counts),
        "chosen_source_counts": dict(chosen_counts),
        "rejected_source_counts": dict(rejected_counts),
        "error_group_counts": dict(group_counts),
        "image_rows": image_rows,
        "orientation_checks": {
            "chosen_wrong_history_rows": chosen_wrong,
            "chosen_not_deployable_rows": chosen_not_deployable,
            "rejected_not_wrong_history_rows": rejected_not_wrong,
            "wrong_history_beats_deployable_rows": wrong_beats_deployable,
        },
        "reward_margin": {
            "mean": statistics.mean(margins) if margins else 0.0,
            "median": statistics.median(margins) if margins else 0.0,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "gate": {
            "verdict": "GO" if not reasons else "NO-GO",
            "reasons": reasons,
            "policy": {
                "min_pairs": min_pairs,
                "max_group_fraction": max_group_fraction,
                "pair_policy": pair_policy,
            },
        },
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    margin = summary["reward_margin"]
    lines = [
        "# Source-Causal Self-Correcting Preference Pair Gate",
        "",
        f"Pairs: `{summary['pairs']}`",
        f"Source rows: `{summary['source_rows_in_train_splits']}`",
        f"Unique sample IDs: `{summary['unique_sample_id']}`",
        f"Unique subject IDs: `{summary['unique_subject_id']}`",
        f"Pair policy: `{summary['pair_policy']}`",
        f"Gate: `{summary['gate']['verdict']}`",
        f"Gate reasons: `{', '.join(summary['gate']['reasons']) or 'none'}`",
        "",
        "## Counts",
        "",
        "| Type | Counts |",
        "|---|---|",
        f"| Split | `{summary['split_counts']}` |",
        f"| Chosen source | `{summary['chosen_source_counts']}` |",
        f"| Rejected source | `{summary['rejected_source_counts']}` |",
        f"| Error group | `{summary['error_group_counts']}` |",
        f"| Orientation checks | `{summary['orientation_checks']}` |",
        "",
        "## Reward Margin",
        "",
        "| Mean | Median | Min | Max |",
        "|---:|---:|---:|---:|",
        f"| {margin['mean']:.4f} | {margin['median']:.4f} | {margin['min']:.4f} | {margin['max']:.4f} |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--deployable-arms", default=",".join(DEFAULT_DEPLOYABLE_ARMS))
    parser.add_argument("--wrong-history-arms", default=",".join(DEFAULT_WRONG_HISTORY_ARMS))
    parser.add_argument("--pair-policy", choices=PAIR_POLICIES, default="deployable_main")
    parser.add_argument("--train-splits", default="train,validate")
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-pairs-per-row", type=int, default=3)
    parser.add_argument("--max-group-fraction", type=float, default=0.75)
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
        arms=parse_csv(args.arms),
        deployable_arms=parse_csv(args.deployable_arms),
        wrong_history_arms=parse_csv(args.wrong_history_arms),
        pair_policy=args.pair_policy,
        train_splits=train_splits,
        min_margin=args.min_margin,
        max_pairs=args.max_pairs,
        max_pairs_per_row=args.max_pairs_per_row,
        status_weight=args.status_weight,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
    )
    summary = summarize(
        pairs,
        rows=rows,
        train_splits=train_splits,
        deployable_arms=parse_csv(args.deployable_arms),
        wrong_history_arms=parse_csv(args.wrong_history_arms),
        pair_policy=args.pair_policy,
        min_pairs=args.min_pairs,
        max_group_fraction=args.max_group_fraction,
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

