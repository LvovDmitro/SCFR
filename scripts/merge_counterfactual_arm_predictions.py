#!/usr/bin/env python3
"""Merge per-arm manifests and per-arm predictions into one audit payload."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_kv_pairs(items: Sequence[str], field_name: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        raw = str(item or "").strip()
        if not raw or "=" not in raw:
            raise SystemExit(f"ERROR: invalid {field_name} entry '{item}', expected arm=path")
        arm, raw_path = raw.split("=", 1)
        arm = arm.strip()
        raw_path = raw_path.strip()
        if not arm or not raw_path:
            raise SystemExit(f"ERROR: invalid {field_name} entry '{item}', expected arm=path")
        if arm in out:
            raise SystemExit(f"ERROR: duplicate arm in {field_name}: {arm}")
        out[arm] = Path(raw_path)
    return out


def as_rows(payload: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        rows = payload["samples"]
    else:
        raise RuntimeError(f"Expected list JSON or {{'samples': [...]}} in {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object row at index {idx} in {path}")
        out.append(dict(row))
    return out


def sample_id(row: Mapping[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row_{idx}"


def prediction_text(row: Mapping[str, Any], arm: str) -> str:
    direct = row.get("prediction")
    if isinstance(direct, str):
        return direct
    keyed = row.get(arm)
    if isinstance(keyed, str):
        return keyed
    return ""


def build_prediction_index(rows: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for idx, row in enumerate(rows):
        sid = sample_id(row, idx)
        if sid in out:
            continue
        out[sid] = prediction_text(row, arm)
    return out


def id_set(rows: Sequence[Mapping[str, Any]]) -> set:
    return {sample_id(row, idx) for idx, row in enumerate(rows)}


def duplicate_ids(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    seen = set()
    duplicates = []
    for idx, row in enumerate(rows):
        sid = sample_id(row, idx)
        if sid in seen and sid not in duplicates:
            duplicates.append(sid)
        seen.add(sid)
    return duplicates


def choose_base_arm(arms: Sequence[str]) -> str:
    for preferred in ("real_prior", "current_only"):
        if preferred in arms:
            return preferred
    return sorted(arms)[0]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-manifest",
        action="append",
        default=[],
        help="Repeat as arm=path to materialized arm JSON.",
    )
    parser.add_argument(
        "--arm-results",
        action="append",
        default=[],
        help=(
            "Repeat as arm=path to eval predictions JSON "
            "(e.g. src/eval/results/results_<generator>_<suffix>.json)."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any arm has missing predictions for at least one manifest row.",
    )
    parser.add_argument(
        "--only-complete-results",
        action="store_true",
        help="Write only rows with non-empty predictions for every requested arm.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_map = parse_kv_pairs(args.arm_manifest, "arm-manifest")
    results_map = parse_kv_pairs(args.arm_results, "arm-results")
    if not manifest_map:
        raise SystemExit("ERROR: at least one --arm-manifest arm=path is required")
    if not results_map:
        raise SystemExit("ERROR: at least one --arm-results arm=path is required")

    arms = sorted(set(manifest_map) | set(results_map))
    missing_manifests = sorted(set(arms) - set(manifest_map))
    missing_results = sorted(set(arms) - set(results_map))
    if missing_manifests:
        raise SystemExit(f"ERROR: missing --arm-manifest for: {','.join(missing_manifests)}")
    if missing_results:
        raise SystemExit(f"ERROR: missing --arm-results for: {','.join(missing_results)}")

    merged_by_id: Dict[str, Dict[str, Any]] = {}
    per_arm_stats: Dict[str, Dict[str, Any]] = {}
    incomplete_arms: List[str] = []
    mismatched_manifest_arms: List[str] = []
    duplicate_id_arms: List[str] = []
    base_arm = choose_base_arm(arms)
    base_manifest_ids: Optional[set] = None

    ordered_arms = [base_arm] + [arm for arm in arms if arm != base_arm]
    for arm in ordered_arms:
        manifest_path = manifest_map[arm]
        result_path = results_map[arm]
        if not manifest_path.is_file():
            raise SystemExit(f"ERROR: manifest not found for {arm}: {manifest_path}")
        if not result_path.is_file():
            raise SystemExit(f"ERROR: results not found for {arm}: {result_path}")

        manifest_rows = as_rows(load_json(manifest_path), manifest_path)
        result_rows = as_rows(load_json(result_path), result_path)
        if duplicate_ids(manifest_rows) or duplicate_ids(result_rows):
            duplicate_id_arms.append(arm)
        pred_index = build_prediction_index(result_rows, arm)
        manifest_ids = id_set(manifest_rows)
        if arm == base_arm:
            base_manifest_ids = manifest_ids
        elif base_manifest_ids is not None and manifest_ids != base_manifest_ids:
            mismatched_manifest_arms.append(arm)

        n_missing = 0
        for idx, row in enumerate(manifest_rows):
            sid = sample_id(row, idx)
            if sid not in merged_by_id or arm == base_arm:
                enriched = dict(row)
                enriched["sample_id"] = sid
                merged_by_id[sid] = enriched
            else:
                enriched = merged_by_id[sid]
            pred = pred_index.get(sid, "")
            if not pred:
                n_missing += 1
            enriched[arm] = pred

        if n_missing > 0:
            incomplete_arms.append(arm)
        per_arm_stats[arm] = {
            "manifest_path": str(manifest_path),
            "results_path": str(result_path),
            "manifest_rows": len(manifest_rows),
            "result_rows": len(result_rows),
            "manifest_unique_ids": len(manifest_ids),
            "manifest_duplicate_ids": duplicate_ids(manifest_rows),
            "result_duplicate_ids": duplicate_ids(result_rows),
            "matched_predictions": len(manifest_rows) - n_missing,
            "missing_predictions": n_missing,
            "manifest_id_set_matches_base": bool(
                arm == base_arm or (base_manifest_ids is not None and manifest_ids == base_manifest_ids)
            ),
        }

    if args.require_complete and duplicate_id_arms:
        joined = ",".join(sorted(duplicate_id_arms))
        raise SystemExit(f"ERROR: duplicate sample ids detected for arms: {joined}")
    if args.require_complete and mismatched_manifest_arms:
        joined = ",".join(sorted(mismatched_manifest_arms))
        raise SystemExit(f"ERROR: manifest sample-set mismatch for arms: {joined}")
    if args.require_complete and incomplete_arms:
        joined = ",".join(sorted(incomplete_arms))
        raise SystemExit(f"ERROR: missing predictions detected for arms: {joined}")

    merged_rows = [merged_by_id[sid] for sid in sorted(merged_by_id)]
    complete_prediction_rows = [
        row for row in merged_rows if all(str(row.get(arm) or "").strip() for arm in arms)
    ]
    if args.only_complete_results:
        merged_rows = complete_prediction_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "generated_utc": utc_now_iso(),
        "output": str(args.output),
        "rows": len(merged_rows),
        "arms": arms,
        "base_arm": base_arm,
        "require_complete": bool(args.require_complete),
        "only_complete_results": bool(args.only_complete_results),
        "output_row_filter": "complete_results" if args.only_complete_results else "all_manifest_rows",
        "complete_prediction_rows": len(complete_prediction_rows),
        "incomplete_arms": sorted(incomplete_arms),
        "mismatched_manifest_arms": sorted(mismatched_manifest_arms),
        "duplicate_id_arms": sorted(duplicate_id_arms),
        "per_arm": per_arm_stats,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

