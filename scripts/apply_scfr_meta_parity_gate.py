#!/usr/bin/env python3
"""Apply the paper's fixed point-estimate parity gate before test deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-analysis", required=True, type=Path)
    parser.add_argument("--scfr-test-predictions", required=True, type=Path)
    parser.add_argument("--output-predictions", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--min-finding-f1-delta", type=float, default=-0.02)
    parser.add_argument("--min-status-delta", type=float, default=-0.02)
    parser.add_argument("--max-fp-delta", type=float, default=0.10)
    parser.add_argument("--max-fn-delta", type=float, default=0.10)
    args = parser.parse_args()

    meta = json.loads(args.meta_analysis.read_text(encoding="utf-8-sig"))
    contrast = meta["contrasts"]["scfr_minus_current_only"]
    observed = {
        "finding_f1_delta": float(contrast["finding_f1"]["estimate"]),
        "status_delta": float(contrast["finding_status_accuracy"]["estimate"]),
        "fp_delta": float(contrast["finding_fp_count"]["estimate"]),
        "fn_delta": float(contrast["finding_fn_count"]["estimate"]),
    }
    margins = {
        "min_finding_f1_delta": args.min_finding_f1_delta,
        "min_status_delta": args.min_status_delta,
        "max_fp_delta": args.max_fp_delta,
        "max_fn_delta": args.max_fn_delta,
    }
    checks = {
        "finding_f1": observed["finding_f1_delta"] >= args.min_finding_f1_delta,
        "status": observed["status_delta"] >= args.min_status_delta,
        "fp": observed["fp_delta"] <= args.max_fp_delta,
        "fn": observed["fn_delta"] <= args.max_fn_delta,
    }
    parity_passed = all(checks.values())
    repair = meta["contrasts"]["scfr_minus_matched_wrong_history"][
        "finding_fp_count"
    ]
    repair_direction = {
        "scfr_minus_matched_wrong_fp_delta": float(repair["estimate"]),
        "ci95": [float(value) for value in repair["ci95"]],
        "point_direction_favorable": float(repair["estimate"]) <= 0.0,
        "ci_excludes_zero_in_favorable_direction": float(repair["ci95"][1]) < 0.0,
    }

    rows = json.loads(args.scfr_test_predictions.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Expected non-empty SCFR test predictions")
    deployed = []
    for source_row in rows:
        row = dict(source_row)
        if not parity_passed:
            current = str(row.get("current_only") or "")
            if not current:
                raise RuntimeError("Gate fallback row has no current_only prediction")
            row["prediction"] = current
            row["selected_arm"] = "current_only_gate_fallback"
        row["deployment_gate_decision"] = (
            "deploy_scfr" if parity_passed else "fallback_current_only"
        )
        deployed.append(row)

    receipt = {
        "schema_version": "scfr-meta-parity-deployment-gate-v1",
        "gate_source_split": "meta_train",
        "test_outcomes_used_for_gate": False,
        "gate_type": "paper_fixed_point_estimate_parity_margins",
        "margins": margins,
        "observed_meta_deltas": observed,
        "checks": checks,
        "parity_passed": parity_passed,
        "repair_direction": repair_direction,
        "full_parity_and_statistically_confirmed_repair_claimed": False,
        "decision": (
            "deploy_scfr" if parity_passed else "fallback_current_only"
        ),
        "meta_rows": int(meta["rows"]),
        "meta_subjects": int(meta["subjects"]),
        "test_rows": len(deployed),
    }
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.write_text(
        json.dumps(deployed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

