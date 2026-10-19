#!/usr/bin/env python3
"""Apply one archived SCFR weight vector without target-generator refitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.source_causal_candidate_reranker import (
    evaluate_reranker,
    feature_names,
    prepare_rows,
    split_prepared,
)
from scripts.source_causal_meta_controller import (
    load_rows,
    materialize_candidates,
    parse_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=parse_candidate)
    parser.add_argument("--weights-receipt", required=True, type=Path)
    parser.add_argument("--weight-vector", default="locked1000")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--fp-weight", type=float, default=0.2)
    parser.add_argument("--fn-weight", type=float, default=0.4)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-predictions", required=True, type=Path)
    args = parser.parse_args()

    candidate_arms = ["current_only"] + [name for name, _path in args.candidate]
    base_rows = load_rows(args.base_predictions)
    materialized, materialization = materialize_candidates(
        base_rows,
        candidates=args.candidate,
    )
    receipt = json.loads(args.weights_receipt.read_text(encoding="utf-8-sig"))
    archived_names = list(receipt["feature_names"])
    expected_names = feature_names(candidate_arms)
    if archived_names != expected_names:
        raise RuntimeError(
            "Frozen feature contract mismatch: "
            f"expected={expected_names}, archived={archived_names}"
        )
    panel = receipt["panels"].get(args.weight_vector)
    if not isinstance(panel, dict):
        raise RuntimeError(f"Unknown frozen weight vector: {args.weight_vector}")
    weights = [float(value) for value in panel["weights"]]
    if len(weights) != len(expected_names):
        raise RuntimeError("Frozen coefficient count does not match feature contract")

    prepared = prepare_rows(
        materialized,
        candidate_arms=candidate_arms,
        split_field=args.split_field,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
    )
    split_rows = split_prepared(prepared).get(args.eval_split, [])
    if not split_rows:
        raise RuntimeError(f"No rows in eval split {args.eval_split}")
    evaluation, selected = evaluate_reranker(
        split_rows,
        weights=weights,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        emit_rows=True,
    )
    report = {
        "schema_version": "scfr-frozen-cross-generator-application-v1",
        "controller": "source_causal_meta_controller",
        "adaptation": "frozen_final_ranker_generator_specific_components",
        "weight_vector": args.weight_vector,
        "weights_source": str(args.weights_receipt),
        "feature_names": expected_names,
        "weights": weights,
        "eval_split": args.eval_split,
        "candidate_arms": candidate_arms,
        "candidate_materialization": materialization,
        "evaluation": evaluation,
        "oracle_selection_used": False,
        "reference_available_to_controller_at_inference": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

