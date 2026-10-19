#!/usr/bin/env python3
"""Train/evaluate a meta-controller over frozen source-causal candidates.

The script materializes prediction files from earlier controllers as candidate
arms and then reuses the train-only candidate reranker. It is intended for
PILOT/FULL-gate diagnostics over already generated arm outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.source_causal_candidate_reranker import (
        build_report as build_reranker_report,
        load_rows,
        parse_csv,
        write_markdown as write_reranker_markdown,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.source_causal_candidate_reranker import (
        build_report as build_reranker_report,
        load_rows,
        parse_csv,
        write_markdown as write_reranker_markdown,
    )


def parse_candidate(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    return name, Path(path.strip())


def sample_key(row: Mapping[str, Any], *, idx: int, source: str) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise RuntimeError(f"{source}: missing sample key at row {idx}")


def prediction_index(path: Path) -> Dict[str, str]:
    rows = load_rows(path)
    out: Dict[str, str] = {}
    duplicates: List[str] = []
    for idx, row in enumerate(rows):
        key = sample_key(row, idx=idx, source=str(path))
        if key in out:
            duplicates.append(key)
            continue
        prediction = str(row.get("prediction") or "").strip()
        if prediction:
            out[key] = prediction
    if duplicates:
        preview = ",".join(sorted(set(duplicates))[:10])
        raise RuntimeError(f"{path}: duplicate sample keys: {preview}")
    return out


def materialize_candidates(
    base_rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[Tuple[str, Path]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    indexes = {name: prediction_index(path) for name, path in candidates}
    rows: List[Dict[str, Any]] = []
    missing_counts = {name: 0 for name, _path in candidates}
    for idx, row in enumerate(base_rows):
        key = sample_key(row, idx=idx, source="base")
        out = dict(row)
        for name, _path in candidates:
            prediction = indexes[name].get(key, "")
            if not prediction:
                missing_counts[name] += 1
            out[name] = prediction
        rows.append(out)
    summary = {
        "base_rows": len(base_rows),
        "candidates": {
            name: {
                "path": str(path),
                "indexed_predictions": len(indexes[name]),
                "missing_base_rows": missing_counts[name],
            }
            for name, path in candidates
        },
    }
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=parse_candidate)
    parser.add_argument("--base-arm", default="current_only")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--output-candidates", type=Path, default=None)
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--tune-split", default="train")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument("--allow-leaky-eval", action="store_true")
    parser.add_argument(
        "--allow-underpowered-eval",
        action="store_true",
        help="Allow eval splits below --min-samples. Use only for diagnostics, not stage-gate promotion.",
    )
    parser.add_argument("--fp-weight", type=float, default=0.4)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--no-average-weights", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    materialized, summary = materialize_candidates(
        load_rows(args.base_predictions),
        candidates=args.candidate,
    )
    candidate_arms = [args.base_arm] + [name for name, _path in args.candidate]
    report, selected_rows = build_reranker_report(
        materialized,
        candidate_arms=candidate_arms,
        split_field=args.split_field,
        tune_split=args.tune_split,
        eval_splits=parse_csv(args.eval_splits),
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        min_samples=args.min_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        average_weights=not args.no_average_weights,
        allow_leaky_eval=args.allow_leaky_eval,
        allow_underpowered_eval=args.allow_underpowered_eval,
    )
    report["controller"] = "source_causal_meta_controller"
    report["candidate_materialization"] = summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_reranker_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_candidates:
        args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
        args.output_candidates.write_text(json.dumps(materialized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

