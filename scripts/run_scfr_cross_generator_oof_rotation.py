#!/usr/bin/env python3
"""Run additive five-fold OOF validation of the frozen SCFR ranker.

Row-level reports stay under --protected-root.  Only aggregate analyses and a
validation receipt are emitted under rebuttal/results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


SEED = 20260728
GENERATORS = {
    "radinfer": "radinfer_pilot400_frozen_4arm_merged.json",
    "libra": "libra_pilot400_frozen_4arm_completecase_merged.json",
}
COMPONENTS = ("fp02", "fp08", "t0", "t2")
CANDIDATES = ("current_only", "cand_fp02", "cand_fp08", "verifier_t0", "verifier_t2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload]


def sample_key(row: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise RuntimeError("Row has no sample key")


def subject(row: dict[str, Any]) -> str:
    value = str(row.get("subject_id") or "").strip()
    if not value:
        raise RuntimeError("Row has no subject_id")
    return value


def write_status(path: Path, stage: str, **extra: Any) -> None:
    payload = {"stage": stage, "updated_unix": time.time(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_logged(
    command: Sequence[str],
    *,
    log_base: Path,
    cwd: Path,
    acceptable_returncodes: Sequence[int] = (0,),
) -> None:
    started = time.time()
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_base.parent.mkdir(parents=True, exist_ok=True)
    log_base.with_suffix(".out.log").write_text(result.stdout, encoding="utf-8")
    log_base.with_suffix(".err.log").write_text(result.stderr, encoding="utf-8")
    log_base.with_suffix(".receipt.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "elapsed_sec": time.time() - started,
                "argv_sha256": hashlib.sha256(
                    json.dumps(list(command), separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if result.returncode not in acceptable_returncodes:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log_base.name}.err.log"
        )


def component_command(
    *,
    python: str,
    repo: Path,
    split_path: Path,
    fold_dir: Path,
    component: str,
) -> list[str]:
    output_json = fold_dir / f"{component}_component_eval.json"
    output_predictions = fold_dir / f"{component}_component_predictions.json"
    if component in {"fp02", "fp08"}:
        fp_weight = "0.2" if component == "fp02" else "0.8"
        return [
            python,
            str(repo / "scripts" / "source_causal_candidate_reranker.py"),
            "--predictions",
            str(split_path),
            "--output-json",
            str(output_json),
            "--output-predictions",
            str(output_predictions),
            "--candidate-arms",
            "current_only,real_prior,prior_report_only",
            "--split-field",
            "split",
            "--tune-split",
            "component_train",
            "--eval-splits",
            "meta_train,test",
            "--fp-weight",
            fp_weight,
            "--fn-weight",
            "0.1",
            "--min-samples",
            "20",
            "--epochs",
            "50",
            "--learning-rate",
            "0.05",
        ]
    token_delta = "0" if component == "t0" else "2"
    return [
        python,
        str(repo / "scripts" / "source_causal_verifier.py"),
        "--predictions",
        str(split_path),
        "--output-json",
        str(output_json),
        "--output-predictions",
        str(output_predictions),
        "--all-arms",
        "current_only,real_prior,prior_report_only,shuffled_prior,matched_shuffled_prior",
        "--feature-arms",
        "current_only,real_prior,prior_report_only",
        "--wrong-arms",
        "none",
        "--train-split",
        "component_train",
        "--tune-split",
        "component_train",
        "--eval-splits",
        "meta_train,test",
        "--threshold-status-weight",
        "0.25",
        "--threshold-fp-weight",
        "1.0",
        "--threshold-fn-weight",
        "0.1",
        "--fp-weight",
        "0.4",
        "--fn-weight",
        "0.1",
        "--token-weight",
        "0.05",
        "--min-samples",
        "20",
        "--constraint-baseline-arm",
        "current_only",
        "--max-fp-delta",
        "0",
        "--max-fn-delta",
        "0.05",
        "--max-prediction-token-delta",
        "0",
        "--min-finding-f1-delta",
        "-0.02",
        "--min-token-f1-delta",
        "-0.01",
        "--prediction-mode",
        "natural_filter_compact_append",
        "--natural-base-arm",
        "current_only",
        "--natural-append-arms",
        "real_prior,current_only,prior_report_only",
        "--natural-max-token-delta",
        token_delta,
    ]


def base_python_command(python: str, script: Path, *args: str) -> list[str]:
    return [python, str(script), *args]


def validate_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_rows = Counter(str(row["split"]) for row in rows)
    split_subjects: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_subjects[str(row["split"])].add(subject(row))
    names = ("component_train", "meta_train", "test")
    overlaps = {
        f"{left}_vs_{right}": len(split_subjects[left] & split_subjects[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Subject overlap: {overlaps}")
    return {
        "rows": dict(split_rows),
        "subjects": {name: len(split_subjects[name]) for name in names},
        "pairwise_subject_overlaps": overlaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-root", required=True, type=Path)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timebox-min", type=float, default=25.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    results_dir = repo / "rebuttal" / "results"
    work_root = args.protected_root / "oof5"
    status_path = work_root / "status.json"
    logs_dir = work_root / "logs"
    weights = repo / "rebuttal" / "scfr_controller_fitted_vectors_recovered_2026_07.json"
    start = time.time()
    deadline = start + args.timebox_min * 60.0
    work_root.mkdir(parents=True, exist_ok=True)
    write_status(status_path, "preparing_splits", timebox_min=args.timebox_min)

    jobs: list[tuple[list[str], Path]] = []
    split_receipts: dict[str, dict[str, Any]] = {}
    for generator, merged_name in GENERATORS.items():
        source = args.protected_root / "work" / merged_name
        if not source.is_file():
            raise RuntimeError(f"Missing protected merged input for {generator}")
        source_rows = load_rows(source)
        source_keys = [sample_key(row) for row in source_rows]
        if len(source_keys) != len(set(source_keys)):
            raise RuntimeError(f"Duplicate source sample key for {generator}")
        split_receipts[generator] = {
            "complete_case_rows": len(source_rows),
            "subjects": len({subject(row) for row in source_rows}),
            "folds": {},
        }
        for test_fold in range(5):
            meta_fold = (test_fold + 1) % 5
            fold_dir = work_root / generator / f"fold{test_fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            split_path = fold_dir / "split.json"
            command = base_python_command(
                args.python_exe,
                repo / "scripts" / "assign_scfr_subject_transfer_splits.py",
                "--input",
                str(source),
                "--output",
                str(split_path),
                "--seed",
                str(SEED),
                "--test-fold",
                str(test_fold),
                "--meta-fold",
                str(meta_fold),
            )
            run_logged(
                command,
                log_base=logs_dir / f"{generator}_fold{test_fold}_split",
                cwd=repo,
            )
            split_receipts[generator]["folds"][str(test_fold)] = {
                "test_fold": test_fold,
                "meta_fold": meta_fold,
                **validate_split(load_rows(split_path)),
            }
            for component in COMPONENTS:
                component_eval = fold_dir / f"{component}_component_eval.json"
                component_predictions = (
                    fold_dir / f"{component}_component_predictions.json"
                )
                if component_eval.is_file() and component_predictions.is_file():
                    continue
                jobs.append(
                    (
                        component_command(
                            python=args.python_exe,
                            repo=repo,
                            split_path=split_path,
                            fold_dir=fold_dir,
                            component=component,
                        ),
                        logs_dir / f"{generator}_fold{test_fold}_{component}",
                    )
                )

    write_status(
        status_path,
        "component_fits_running",
        component_jobs=len(jobs),
        max_workers=args.max_workers,
        elapsed_sec=time.time() - start,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                run_logged,
                command,
                log_base=log,
                cwd=repo,
                acceptable_returncodes=(
                    (0, 2)
                    if Path(command[1]).name == "source_causal_verifier.py"
                    else (0,)
                ),
            )
            for command, log in jobs
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            write_status(
                status_path,
                "component_fits_running",
                component_jobs=len(jobs),
                component_jobs_completed=completed,
                max_workers=args.max_workers,
                elapsed_sec=time.time() - start,
            )
            if time.time() > deadline:
                raise TimeoutError("OOF timebox exceeded during component fitting")

    write_status(status_path, "fold_gating_and_analysis", elapsed_sec=time.time() - start)
    fold_gate_receipts: dict[str, dict[str, Any]] = defaultdict(dict)
    combined_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generator in GENERATORS:
        for test_fold in range(5):
            fold_dir = work_root / generator / f"fold{test_fold}"
            split_path = fold_dir / "split.json"
            candidate_args: list[str] = []
            for name, component in (
                ("cand_fp02", "fp02"),
                ("cand_fp08", "fp08"),
                ("verifier_t0", "t0"),
                ("verifier_t2", "t2"),
            ):
                candidate_args.extend(
                    [
                        "--candidate",
                        f"{name}={fold_dir / f'{component}_component_predictions.json'}",
                    ]
                )
            for eval_split in ("meta_train", "test"):
                command = base_python_command(
                    args.python_exe,
                    repo / "scripts" / "apply_frozen_scfr_controller.py",
                    "--base-predictions",
                    str(split_path),
                    *candidate_args,
                    "--weights-receipt",
                    str(weights),
                    "--weight-vector",
                    "locked1000",
                    "--eval-split",
                    eval_split,
                    "--fp-weight",
                    "0.2",
                    "--fn-weight",
                    "0.4",
                    "--output-json",
                    str(fold_dir / f"frozen_{eval_split}_eval.json"),
                    "--output-predictions",
                    str(fold_dir / f"frozen_{eval_split}_predictions.json"),
                )
                run_logged(
                    command,
                    log_base=logs_dir / f"{generator}_fold{test_fold}_frozen_{eval_split}",
                    cwd=repo,
                )
            run_logged(
                base_python_command(
                    args.python_exe,
                    repo / "scripts" / "analyze_scfr_cross_generator_transfer.py",
                    "--predictions",
                    str(fold_dir / "frozen_meta_train_predictions.json"),
                    "--label",
                    f"{generator} fold {test_fold} frozen meta",
                    "--n-bootstrap",
                    "5000",
                    "--seed",
                    str(SEED + test_fold),
                    "--output-json",
                    str(fold_dir / "meta_analysis.json"),
                ),
                log_base=logs_dir / f"{generator}_fold{test_fold}_meta_analysis",
                cwd=repo,
            )
            run_logged(
                base_python_command(
                    args.python_exe,
                    repo / "scripts" / "apply_scfr_meta_parity_gate.py",
                    "--meta-analysis",
                    str(fold_dir / "meta_analysis.json"),
                    "--scfr-test-predictions",
                    str(fold_dir / "frozen_test_predictions.json"),
                    "--output-predictions",
                    str(fold_dir / "gated_test_predictions.json"),
                    "--output-receipt",
                    str(fold_dir / "gate_receipt.json"),
                ),
                log_base=logs_dir / f"{generator}_fold{test_fold}_gate",
                cwd=repo,
            )
            run_logged(
                base_python_command(
                    args.python_exe,
                    repo / "scripts" / "analyze_scfr_cross_generator_transfer.py",
                    "--predictions",
                    str(fold_dir / "gated_test_predictions.json"),
                    "--label",
                    f"{generator} fold {test_fold} gated test",
                    "--n-bootstrap",
                    "5000",
                    "--seed",
                    str(SEED + 100 + test_fold),
                    "--output-json",
                    str(fold_dir / "gated_test_analysis.json"),
                ),
                log_base=logs_dir / f"{generator}_fold{test_fold}_test_analysis",
                cwd=repo,
            )
            gate = json.loads((fold_dir / "gate_receipt.json").read_text(encoding="utf-8"))
            fold_gate_receipts[generator][str(test_fold)] = gate
            fold_rows = load_rows(fold_dir / "gated_test_predictions.json")
            for row in fold_rows:
                row["_oof_test_fold"] = test_fold
            combined_rows[generator].extend(fold_rows)
            if time.time() > deadline:
                raise TimeoutError("OOF timebox exceeded during fold analysis")

    aggregate_outputs: dict[str, dict[str, Any]] = {}
    for generator, rows in combined_rows.items():
        source_rows = load_rows(
            args.protected_root / "work" / GENERATORS[generator]
        )
        source_keys = {sample_key(row) for row in source_rows}
        test_keys = [sample_key(row) for row in rows]
        if len(test_keys) != len(set(test_keys)):
            raise RuntimeError(f"OOF duplicate test row for {generator}")
        if set(test_keys) != source_keys:
            raise RuntimeError(f"OOF test coverage mismatch for {generator}")
        subject_folds: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            subject_folds[subject(row)].add(int(row["_oof_test_fold"]))
            route = str(row.get("selected_arm") or "")
            if "matched" in route or "shuffled" in route:
                raise RuntimeError(f"Wrong-history route selected for {generator}")
        if any(len(folds) != 1 for folds in subject_folds.values()):
            raise RuntimeError(f"Subject tested in multiple outer folds for {generator}")
        combined_path = work_root / generator / "oof_gated_test_predictions.json"
        combined_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        aggregate_json = (
            results_dir
            / f"scfr_cross_generator_{generator}_frozen_locked1000_gated_deployment_oof5_2026_07_28.json"
        )
        aggregate_md = aggregate_json.with_suffix(".md")
        run_logged(
            base_python_command(
                args.python_exe,
                repo / "scripts" / "analyze_scfr_cross_generator_transfer.py",
                "--predictions",
                str(combined_path),
                "--label",
                f"{generator} frozen locked1000 gated OOF5",
                "--n-bootstrap",
                "5000",
                "--seed",
                str(SEED + 500),
                "--output-json",
                str(aggregate_json),
                "--output-md",
                str(aggregate_md),
            ),
            log_base=logs_dir / f"{generator}_oof_aggregate_analysis",
            cwd=repo,
        )
        analysis = json.loads(aggregate_json.read_text(encoding="utf-8"))
        aggregate_outputs[generator] = {
            "rows": len(rows),
            "subjects": len(subject_folds),
            "each_subject_in_exactly_one_test_fold": True,
            "all_complete_case_rows_tested_exactly_once": True,
            "analysis_json": aggregate_json.name,
            "analysis_json_sha256": sha256(aggregate_json),
            "analysis_md": aggregate_md.name,
            "analysis_md_sha256": sha256(aggregate_md),
            "route_counts": analysis["route_counts"],
            "contrasts": analysis["contrasts"],
        }

    receipt = {
        "schema_version": "scfr-cross-generator-frozen-ranker-oof5-validation-v1",
        "date_utc": "2026-07-28",
        "status": "fully_validated",
        "seed": SEED,
        "outer_folds": 5,
        "fold_roles": {
            "test": "outer fold k",
            "meta_gate": "outer fold (k+1) modulo 5",
            "component_fit": "remaining three outer folds",
        },
        "candidate_names": list(CANDIDATES),
        "final_ranker": "frozen_locked1000",
        "target_generator_final_ranker_refit": False,
        "oracle_selection_used": False,
        "reference_available_to_controller_at_inference": False,
        "matched_wrong_history_is_candidate": False,
        "split_validation": split_receipts,
        "fold_gate_receipts": fold_gate_receipts,
        "aggregate_outputs": aggregate_outputs,
        "privacy": {
            "receipt_contains_row_level_text": False,
            "receipt_contains_patient_or_study_ids": False,
            "receipt_contains_protected_local_paths": False,
            "row_level_outputs_remain_in_protected_non_synced_storage": True,
            "repository_outputs_are_aggregate_only": True,
        },
        "elapsed_sec": time.time() - start,
        "timebox_min": args.timebox_min,
    }
    receipt_path = (
        results_dir
        / "scfr_cross_generator_frozen_locked1000_gated_oof5_validation_receipt_2026_07_28.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_status(
        status_path,
        "completed",
        elapsed_sec=time.time() - start,
        aggregate_receipt_sha256=sha256(receipt_path),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "elapsed_sec": time.time() - start,
                "receipt_sha256": sha256(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

