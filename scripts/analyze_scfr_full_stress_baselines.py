#!/usr/bin/env python3
"""Analyze full SCFR source-stress arms directly from a local cache inventory.

Only aggregate numeric results are written.  Report text, row IDs, subject IDs,
and image paths remain in the source cache and are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.analyze_scfr_selector_baselines import (
        as_text,
        mean_scores,
        percentile,
        positive_findings,
    )
    from scripts.analyze_source_causal_arms import finding_vector, token_f1, tokenize
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from analyze_scfr_selector_baselines import (
        as_text,
        mean_scores,
        percentile,
        positive_findings,
    )
    from analyze_source_causal_arms import finding_vector, token_f1, tokenize


SUBJECT_PATH_RE = re.compile(
    r"(?:^|[\\/])p\d{2}[\\/]p(\d+)[\\/]s\d+(?:[\\/]|$)",
    flags=re.IGNORECASE,
)
SUBJECT_SHARD_RE = re.compile(
    r"(?:^|[\\/])files[\\/](p\d{2})[\\/]p\d+[\\/]s\d+(?:[\\/]|$)",
    flags=re.IGNORECASE,
)
REQUIRED_ARMS = ("current_only", "real_prior")
GATE_NAMES = (
    "gate_status_agreement",
    "gate_no_positive_additions",
    "gate_block_prior_only_additions",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_index(
    inventory: Mapping[str, Any],
    *,
    arms: Sequence[str] = REQUIRED_ARMS,
) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    full = inventory.get("full18716")
    if not isinstance(full, dict) or not full.get("all_five_arms_complete_and_aligned"):
        raise RuntimeError("Inventory does not certify complete aligned full18716 arms")
    expected_chunks = {int(value) for value in full.get("expected_chunks") or []}
    if not expected_chunks:
        raise RuntimeError("Inventory has no expected full18716 chunks")
    indexed: Dict[str, Dict[int, Mapping[str, Any]]] = {arm: {} for arm in arms}
    for artifact in inventory.get("artifacts") or []:
        marker = artifact.get("full18716")
        if not isinstance(marker, dict):
            continue
        arm = as_text(marker.get("arm"))
        if arm not in indexed:
            continue
        chunk = int(marker.get("chunk"))
        if chunk in indexed[arm]:
            raise RuntimeError(f"Duplicate inventory artifact for {arm}/chunk{chunk:03d}")
        indexed[arm][chunk] = artifact
    for arm in arms:
        chunks = set(indexed[arm])
        if chunks != expected_chunks:
            raise RuntimeError(
                f"{arm}: inventory chunks {sorted(chunks)} != expected {sorted(expected_chunks)}"
            )
        summary = full.get("arms", {}).get(arm, {})
        if not summary.get("expected_complete"):
            raise RuntimeError(f"{arm}: inventory does not mark the arm complete")
    return indexed


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        rows.append(dict(row))
    return rows


def index_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for idx, row in enumerate(rows):
        sample_id = as_text(row.get("id"))
        if not sample_id:
            raise RuntimeError(f"{label}: row {idx} has no id")
        if sample_id in indexed:
            raise RuntimeError(f"{label}: duplicate id")
        indexed[sample_id] = row
    return indexed


def canonical_paths(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(as_text(item).replace("\\", "/") for item in value if as_text(item))


def subject_from_image_paths(value: Any) -> str:
    subjects = set()
    for path in canonical_paths(value):
        match = SUBJECT_PATH_RE.search(path)
        if match:
            subjects.add(match.group(1))
    if len(subjects) != 1:
        raise RuntimeError(
            "Could not derive exactly one subject from current image paths"
        )
    return next(iter(subjects))


def shard_from_image_paths(value: Any) -> str:
    shards = set()
    for path in canonical_paths(value):
        match = SUBJECT_SHARD_RE.search(path)
        if match:
            shards.add(match.group(1).lower())
    if len(shards) != 1:
        raise RuntimeError(
            "Could not derive exactly one source shard from current image paths"
        )
    return next(iter(shards))


def gate_choices(
    current_vector: Mapping[str, str],
    real_vector: Mapping[str, str],
    prior_positive: set[str],
) -> Dict[str, str]:
    current_positive = {
        finding for finding, status in current_vector.items() if status == "positive"
    }
    real_positive = {
        finding for finding, status in real_vector.items() if status == "positive"
    }
    return {
        "gate_status_agreement": (
            "real" if dict(real_vector) == dict(current_vector) else "current"
        ),
        "gate_no_positive_additions": (
            "real" if real_positive.issubset(current_positive) else "current"
        ),
        "gate_block_prior_only_additions": (
            "current"
            if (real_positive - current_positive) & prior_positive
            else "real"
        ),
    }


def score_from_vectors(
    prediction: str,
    reference: str,
    prediction_vector: Mapping[str, str],
    reference_vector: Mapping[str, str],
    *,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, float]:
    pred_positive = {
        finding for finding, status in prediction_vector.items() if status == "positive"
    }
    ref_positive = {
        finding for finding, status in reference_vector.items() if status == "positive"
    }
    if not pred_positive and not ref_positive:
        finding_f1 = 1.0
    elif not pred_positive or not ref_positive:
        finding_f1 = 0.0
    else:
        overlap = len(pred_positive & ref_positive)
        precision = overlap / len(pred_positive)
        recall = overlap / len(ref_positive)
        finding_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    fp_count = float(len(pred_positive - ref_positive))
    fn_count = float(len(ref_positive - pred_positive))
    score = {
        "token_f1": token_f1(prediction, reference),
        "finding_f1": finding_f1,
        "finding_status_accuracy": (
            sum(
                prediction_vector[finding] == reference_vector[finding]
                for finding in reference_vector
            )
            / len(reference_vector)
        ),
        "finding_fp_count": fp_count,
        "finding_fn_count": fn_count,
        "prediction_tokens": float(len(tokenize(prediction))),
    }
    score["objective"] = finding_f1 - fp_weight * fp_count - fn_weight * fn_count
    return score


def prepare_cache_rows(
    inventory_path: Path,
    *,
    fp_weight: float,
    fn_weight: float,
    verify_sha256: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    inventory = load_json(inventory_path)
    if not isinstance(inventory, dict):
        raise RuntimeError(f"Expected object inventory: {inventory_path}")
    indexed_artifacts = artifact_index(inventory)
    expected_rows = int(inventory["full18716"]["expected_rows_per_arm"])
    prepared: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    safe_inputs: List[Dict[str, Any]] = []
    shard_counts: Counter[str] = Counter()
    for chunk in sorted(indexed_artifacts["current_only"]):
        artifacts = {
            arm: indexed_artifacts[arm][chunk]
            for arm in REQUIRED_ARMS
        }
        paths = {arm: Path(str(artifact["source_path"])) for arm, artifact in artifacts.items()}
        for arm, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"{arm}/chunk{chunk:03d}: cache file missing")
            actual_sha = sha256_file(path) if verify_sha256 else None
            expected_sha = as_text(artifacts[arm].get("sha256"))
            if verify_sha256 and actual_sha != expected_sha:
                raise RuntimeError(f"{arm}/chunk{chunk:03d}: SHA-256 mismatch")
            safe_inputs.append(
                {
                    "arm": arm,
                    "chunk": chunk,
                    "logical_name": as_text(artifacts[arm].get("logical_name")),
                    "sha256": expected_sha,
                    "rows": int(artifacts[arm].get("json_list_rows") or 0),
                    "sha256_verified": bool(verify_sha256),
                }
            )
        arm_rows = {
            arm: index_rows(load_rows(path), label=f"{arm}/chunk{chunk:03d}")
            for arm, path in paths.items()
        }
        id_sets = {arm: set(rows) for arm, rows in arm_rows.items()}
        if id_sets["current_only"] != id_sets["real_prior"]:
            raise RuntimeError(f"chunk{chunk:03d}: current/real ID set mismatch")
        for sample_id in sorted(id_sets["current_only"]):
            if sample_id in seen_ids:
                raise RuntimeError("Duplicate ID across chunks")
            seen_ids.add(sample_id)
            current_row = arm_rows["current_only"][sample_id]
            real_row = arm_rows["real_prior"][sample_id]
            current_reference = as_text(current_row.get("reference"))
            real_reference = as_text(real_row.get("reference"))
            if not current_reference or current_reference != real_reference:
                raise RuntimeError(f"chunk{chunk:03d}: reference mismatch or empty reference")
            current_paths = canonical_paths(current_row.get("image_paths"))
            real_paths = canonical_paths(real_row.get("image_paths"))
            if not current_paths or current_paths != real_paths:
                raise RuntimeError(f"chunk{chunk:03d}: current image path mismatch")
            subject_id = subject_from_image_paths(current_paths)
            shard_counts[shard_from_image_paths(current_paths)] += 1
            current_prediction = as_text(current_row.get("prediction"))
            real_prediction = as_text(real_row.get("prediction"))
            prior_report = as_text(real_row.get("prior_report") or real_row.get("context"))
            if not current_prediction or not real_prediction or not prior_report:
                raise RuntimeError(
                    f"chunk{chunk:03d}: empty current/real prediction or prior report"
                )
            reference_vector = finding_vector(current_reference)
            current_vector = finding_vector(current_prediction)
            real_vector = finding_vector(real_prediction)
            current_score = score_from_vectors(
                current_prediction,
                current_reference,
                current_vector,
                reference_vector,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            real_score = score_from_vectors(
                real_prediction,
                current_reference,
                real_vector,
                reference_vector,
                fp_weight=fp_weight,
                fn_weight=fn_weight,
            )
            choices = gate_choices(
                current_vector,
                real_vector,
                positive_findings(prior_report),
            )
            prepared.append(
                {
                    "subject_id": subject_id,
                    "scores": {
                        "current": current_score,
                        "real": real_score,
                    },
                    "gate_choices": choices,
                }
            )
        del arm_rows
    if len(prepared) != expected_rows:
        raise RuntimeError(
            f"Prepared rows {len(prepared)} != inventory expected {expected_rows}"
        )
    return prepared, {
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "cache_inputs": safe_inputs,
        "rows": len(prepared),
        "unique_rows": len(seen_ids),
        "subjects": len({row["subject_id"] for row in prepared}),
        "source_shard_counts": dict(sorted(shard_counts.items())),
        "row_ids_or_text_emitted": False,
    }


def materialize_methods(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    methods: Dict[str, Dict[str, Any]] = {
        "always_current_only": {
            "classification": "deployable_reference_free_fixed_baseline",
            "choices": ["current"] * len(rows),
            "scores": [row["scores"]["current"] for row in rows],
        },
        "always_real_history": {
            "classification": "deployable_reference_free_fixed_baseline",
            "choices": ["real"] * len(rows),
            "scores": [row["scores"]["real"] for row in rows],
        },
    }
    for gate in GATE_NAMES:
        choices = [row["gate_choices"][gate] for row in rows]
        methods[gate] = {
            "classification": "post_hoc_reference_free_nonlearned_heuristic",
            "choices": choices,
            "scores": [
                row["scores"][choice]
                for row, choice in zip(rows, choices)
            ],
        }
    oracle_choices: List[str] = []
    oracle_scores: List[Mapping[str, float]] = []
    for row in rows:
        choice = (
            "real"
            if (
                row["scores"]["real"]["objective"]
                - row["scores"]["current"]["objective"]
            )
            > 1e-12
            else "current"
        )
        oracle_choices.append(choice)
        oracle_scores.append(row["scores"][choice])
    methods["oracle_current_vs_real_reference_proxy"] = {
        "classification": "analysis_only_outcome_conditioned_reference_leakage",
        "choices": oracle_choices,
        "scores": oracle_scores,
        "uses_reference_at_selection": True,
        "restriction": "current-versus-real scalar proxy objective only",
    }
    return methods


def paired_patient_bootstrap_matrix(
    methods: Mapping[str, Mapping[str, Any]],
    subject_ids: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    baseline = methods["always_current_only"]["scores"]
    compared_methods = [
        name for name in methods if name != "always_current_only"
    ]
    metrics = sorted(baseline[0])
    columns = [
        (method, metric)
        for method in compared_methods
        for metric in metrics
    ]
    delta_matrix = np.asarray(
        [
            [
                float(methods[method]["scores"][row_idx][metric])
                - float(baseline[row_idx][metric])
                for method, metric in columns
            ]
            for row_idx in range(len(subject_ids))
        ],
        dtype=np.float64,
    )
    normalized_subjects = [as_text(value) for value in subject_ids]
    if any(not value for value in normalized_subjects):
        raise ValueError("subject IDs must be non-empty")
    subjects = sorted(set(normalized_subjects))
    subject_index = {subject: idx for idx, subject in enumerate(subjects)}
    row_subject_indices = np.asarray(
        [subject_index[value] for value in normalized_subjects],
        dtype=np.int64,
    )
    subject_sums = np.zeros((len(subjects), len(columns)), dtype=np.float64)
    np.add.at(subject_sums, row_subject_indices, delta_matrix)
    subject_sizes = np.bincount(
        row_subject_indices,
        minlength=len(subjects),
    ).astype(np.float64)
    subject_means = subject_sums / subject_sizes[:, None]
    row_observed = delta_matrix.mean(axis=0)
    patient_observed = subject_means.mean(axis=0)
    row_draws = np.empty((n_bootstrap, len(columns)), dtype=np.float64)
    patient_draws = np.empty((n_bootstrap, len(columns)), dtype=np.float64)
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        probabilities = np.full(len(subjects), 1.0 / len(subjects), dtype=np.float64)
        for start in range(0, n_bootstrap, batch_size):
            stop = min(n_bootstrap, start + batch_size)
            counts = rng.multinomial(
                len(subjects),
                probabilities,
                size=stop - start,
            ).astype(np.int32, copy=False)
            row_draws[start:stop] = (
                counts @ subject_sums
            ) / (counts @ subject_sizes)[:, None]
            patient_draws[start:stop] = (
                counts @ subject_means
            ) / len(subjects)
    out: Dict[str, Dict[str, Any]] = {
        method: {} for method in compared_methods
    }
    for column_idx, (method, metric) in enumerate(columns):
        if n_bootstrap:
            row_ci: List[Optional[float]] = [
                percentile(row_draws[:, column_idx], 0.025),
                percentile(row_draws[:, column_idx], 0.975),
            ]
            patient_ci: List[Optional[float]] = [
                percentile(patient_draws[:, column_idx], 0.025),
                percentile(patient_draws[:, column_idx], 0.975),
            ]
        else:
            row_ci = [None, None]
            patient_ci = [None, None]
        out[method][metric] = {
            "paired_rows": len(subject_ids),
            "patients": len(subjects),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "row_weighted_patient_cluster": {
                "observed_delta": float(row_observed[column_idx]),
                "ci95_percentile": row_ci,
            },
            "patient_balanced_sensitivity": {
                "observed_delta": float(patient_observed[column_idx]),
                "ci95_percentile": patient_ci,
            },
        }
    return out


def reference_proxy_strata(
    rows: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    strata: List[str] = []
    for row in rows:
        delta = row["scores"]["real"]["objective"] - row["scores"]["current"]["objective"]
        if delta > 1e-12:
            strata.append("real_better")
        elif delta < -1e-12:
            strata.append("current_better")
        else:
            strata.append("tie")
    routes: Dict[str, Any] = {}
    for method_name, payload in methods.items():
        routes[method_name] = {}
        for stratum in ("real_better", "current_better", "tie"):
            indices = [idx for idx, value in enumerate(strata) if value == stratum]
            choices = Counter(payload["choices"][idx] for idx in indices)
            routes[method_name][stratum] = {
                "rows": len(indices),
                "choice_counts": dict(sorted(choices.items())),
                "real_choice_rate": (
                    sum(payload["choices"][idx] == "real" for idx in indices)
                    / len(indices)
                    if indices
                    else None
                ),
            }
    return {
        "classification": "retrospective_outcome_conditioned_diagnostic",
        "definition": (
            "The strata use the report-reference proxy objective and are not "
            "clinically valid/invalid-history labels."
        ),
        "counts": dict(sorted(Counter(strata).items())),
        "method_routes": routes,
    }


def build_report(
    inventory_path: Path,
    *,
    fp_weight: float = 0.2,
    fn_weight: float = 0.4,
    n_bootstrap: int = 5000,
    seed: int = 20260724,
    bootstrap_batch_size: int = 100,
    verify_sha256: bool = True,
) -> Dict[str, Any]:
    rows, provenance = prepare_cache_rows(
        inventory_path,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
        verify_sha256=verify_sha256,
    )
    methods = materialize_methods(rows)
    deltas = paired_patient_bootstrap_matrix(
        methods,
        [row["subject_id"] for row in rows],
        n_bootstrap=n_bootstrap,
        seed=seed,
        batch_size=bootstrap_batch_size,
    )
    method_summaries: Dict[str, Any] = {}
    for method_name, payload in methods.items():
        summary = {
            key: value
            for key, value in payload.items()
            if key not in {"scores", "choices"}
        }
        summary["rows"] = len(rows)
        summary["choice_counts"] = dict(sorted(Counter(payload["choices"]).items()))
        summary["metrics"] = mean_scores(payload["scores"])
        if method_name in deltas:
            summary["deltas_vs_current"] = deltas[method_name]
        method_summaries[method_name] = summary
    return {
        "analysis_stage": "FULL18716_SOURCE_STRESS_BASELINES_NO_SCFR",
        "dataset_role": (
            "complete source-condition stress set; no SCFR controller outputs are "
            "present, so this analysis cannot evaluate SCFR routing"
        ),
        "claim_boundary": {
            "metrics": "11-finding report-reference regex proxy, not image truth",
            "gates": "post-hoc fixed non-learned reference-free heuristics",
            "oracle": "analysis-only outcome-conditioned reference leakage",
            "history_strata": "proxy utility, not clinically valid history",
            "privacy": "only aggregate numeric results and artifact hashes are emitted",
            "coverage": (
                "scope is the cached subject-shard snapshot reported in provenance; "
                "do not broaden it to all eligible MIMIC-CXR without a separate census"
            ),
        },
        "config": {
            "objective": {
                "formula": "finding_f1 - fp_weight*fp_count - fn_weight*fn_count",
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "bootstrap_batch_size": bootstrap_batch_size,
            "subject_derivation": (
                "numeric subject component parsed from standard MIMIC image path; "
                "subject values are not emitted"
            ),
        },
        "provenance": provenance,
        "methods": method_summaries,
        "reference_proxy_utility_strata": reference_proxy_strata(rows, methods),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    def delta_text(payload: Mapping[str, Any], metric: str) -> str:
        delta = payload["deltas_vs_current"][metric][
            "row_weighted_patient_cluster"
        ]
        lower, upper = delta["ci95_percentile"]
        return (
            f"{fmt(delta['observed_delta'])} "
            f"[{fmt(lower)}, {fmt(upper)}]"
        )

    lines = [
        "# Full-18716 SCFR Source-Stress Baselines",
        "",
        "Stage: `FULL18716_SOURCE_STRESS_BASELINES_NO_SCFR`.",
        "",
        "This is a report-reference-proxy analysis of absolute source arms and",
        "post-hoc non-learned gates. No SCFR controller output exists on this set,",
        "so this table does not compare SCFR with current-only. The oracle is",
        "analysis-only and uses the reference at routing time.",
        "",
        f"Rows: `{report['provenance']['rows']}`; "
        f"patients: `{report['provenance']['subjects']}`.",
        f"Cached subject shards: "
        f"`{json.dumps(report['provenance']['source_shard_counts'], sort_keys=True)}`.",
        "",
        "| Method | Class | Current picks | Real picks | F1 | Status | FP | FN | Objective | dF1 (95% CI) | dFP (95% CI) | dFN (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for method_name, payload in report["methods"].items():
        choices = payload["choice_counts"]
        metrics = payload["metrics"]
        finding_delta = "baseline"
        fp_delta = "baseline"
        fn_delta = "baseline"
        if "deltas_vs_current" in payload:
            finding_delta = delta_text(payload, "finding_f1")
            fp_delta = delta_text(payload, "finding_fp_count")
            fn_delta = delta_text(payload, "finding_fn_count")
        lines.append(
            f"| `{method_name}` | `{payload['classification']}` | "
            f"{choices.get('current', 0)} | {choices.get('real', 0)} | "
            f"{fmt(metrics.get('finding_f1'))} | "
            f"{fmt(metrics.get('finding_status_accuracy'))} | "
            f"{fmt(metrics.get('finding_fp_count'))} | "
            f"{fmt(metrics.get('finding_fn_count'))} | "
            f"{fmt(metrics.get('objective'))} | {finding_delta} | "
            f"{fp_delta} | {fn_delta} |"
        )
    lines.extend(
        [
            "",
            "## Outcome-conditioned routing diagnostic",
            "",
            "These strata are defined by the reference-proxy objective and are not",
            "clinically valid/invalid-history labels.",
            "",
            "| Method | Real pick rate when real is proxy-better | Current pick rate when current is proxy-better |",
            "|---|---:|---:|",
        ]
    )
    route_payload = report["reference_proxy_utility_strata"]["method_routes"]
    for method_name in report["methods"]:
        real_rate = route_payload[method_name]["real_better"]["real_choice_rate"]
        harmful_real_rate = route_payload[method_name]["current_better"]["real_choice_rate"]
        current_rate = None if harmful_real_rate is None else 1.0 - harmful_real_rate
        lines.append(
            f"| `{method_name}` | {fmt(real_rate)} | {fmt(current_rate)} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Metrics are report-reference regex proxies, not image truth.",
            "- Gates are post-hoc diagnostics, not confirmatory baselines.",
            "- Oracle and proxy-utility strata are outcome-conditioned.",
            "- Proxy-beneficial history must not be called clinically valid history.",
            "- No row IDs, subject IDs, report text, or image paths are written.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--fp-weight", type=float, default=0.2)
    parser.add_argument("--fn-weight", type=float, default=0.4)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    parser.add_argument(
        "--skip-sha256-verification",
        action="store_true",
        help="Faster diagnostic only; final results should verify every cache file.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(
        args.inventory,
        fp_weight=float(args.fp_weight),
        fn_weight=float(args.fn_weight),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        bootstrap_batch_size=int(args.bootstrap_batch_size),
        verify_sha256=not bool(args.skip_sha256_verification),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(report), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

