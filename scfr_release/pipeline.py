"""Fit/export proposals and route reports with the original SCFR functions.

This wrapper separates fitting, reference-free inference, and evaluation.
It accepts generated reports and includes the archived final-router vectors.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from scripts import source_causal_candidate_reranker as ranker

BASE_ARMS = ("current_only", "real_prior", "prior_report_only")
FINAL_ARMS = ("current_only", "cand_fp02", "cand_fp08", "verifier_t0", "verifier_t2")
INFERENCE_FIELDS = {"id", "sample_id", "prior_report", "context", *BASE_ARMS}


def validate_inference_row(row: Mapping[str, Any]) -> None:
    if set(row) - INFERENCE_FIELDS:
        raise ValueError("inference accepts only IDs, valid prior text, and the three source-valid reports")
    if not str(row.get("id") or row.get("sample_id") or "").strip():
        raise ValueError("inference requires a nonempty local sample id")
    for field in ("id", "sample_id", "prior_report", "context"):
        if field in row and not isinstance(row[field], str):
            raise ValueError(f"inference requires a string {field}; structured payloads are forbidden")
    for arm in BASE_ARMS:
        if not isinstance(row.get(arm), str) or not row[arm].strip():
            raise ValueError(f"inference requires a nonempty {arm} report")


def validate_training_rows(rows: Sequence[Mapping[str, Any]], min_samples: int = 20) -> None:
    ids: set[str] = set()
    subject_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        sample = str(row.get("id") or row.get("sample_id") or "").strip()
        subject = str(row.get("subject_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if not sample or sample in ids:
            raise ValueError("training requires unique, nonempty sample ids")
        ids.add(sample)
        if not subject or not split:
            raise ValueError("training requires subject_id and split")
        if subject in subject_splits and subject_splits[subject] != split:
            raise ValueError("subject overlap across splits")
        subject_splits[subject] = split
        counts[split] += 1
        validate_inference_row({k: v for k, v in row.items() if k in INFERENCE_FIELDS})
        if not isinstance(row.get("reference"), str) or not row["reference"].strip():
            raise ValueError("training requires an explicit nonempty reference")
    if min_samples < 1 or not rows:
        raise ValueError("training requires positive min_samples and nonempty rows")


def export_ranker(report: Mapping[str, Any]) -> dict[str, Any]:
    return {"candidate_arms": list(report["candidate_arms"]), "feature_names": list(report["feature_names"]), "weights": list(report["training"]["weights"])}


def rank_report(row: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[str, str]:
    arms = list(model["candidate_arms"])
    if list(model["feature_names"]) != ranker.feature_names(arms):
        raise ValueError("ranker feature contract mismatch")
    weights = [float(value) for value in model["weights"]]
    if len(weights) != len(model["feature_names"]) or not all(math.isfinite(v) for v in weights):
        raise ValueError("ranker feature coefficient mismatch or nonfinite weight")
    available = [arm for arm in arms if isinstance(row.get(arm), str) and row[arm].strip()]
    if len(available) != len(arms):
        raise ValueError("ranker candidate coverage is incomplete")
    scores = [ranker.dot(weights, ranker.candidate_features(row, arm, arms)) for arm in available]
    arm = available[ranker.argmax_index(scores)]
    return arm, row[arm]


def route_candidates(rows: Sequence[Mapping[str, Any]], archive: Mapping[str, Any], panel: str = "locked1000") -> list[dict[str, Any]]:
    """Apply an archived final router to already constructed source-valid reports."""
    if panel not in archive["panels"]:
        raise ValueError(f"Unknown archived router panel: {panel}")
    model = {"candidate_arms": list(FINAL_ARMS), "feature_names": archive["feature_names"],
             "weights": archive["panels"][panel]["weights"]}
    outputs = []
    seen = set()
    for row in rows:
        if set(row) - (INFERENCE_FIELDS | set(FINAL_ARMS)):
            raise ValueError("routing accepts only IDs, valid prior text, and source-valid candidate reports")
        validate_inference_row({k: v for k, v in row.items() if k in INFERENCE_FIELDS})
        sample = row.get("id") or row["sample_id"]
        if sample in seen:
            raise ValueError("duplicate routing sample id")
        seen.add(sample)
        arm, report = rank_report(row, model)
        outputs.append({"id": sample, "prediction": report, "selected_arm": arm})
    return outputs


def _verifier_report(row: Mapping[str, Any], frozen: Mapping[str, Any], delta: int) -> str:
    from scripts import source_causal_verifier as verifier
    from scripts.source_causal_per_finding_gate import prepare_row

    names = verifier.feature_names(BASE_ARMS, use_finding_identity=True)
    if list(frozen["feature_names"]) != names:
        raise ValueError("verifier feature contract mismatch")
    weights = [float(x) for x in frozen["coefficients"]]
    intercept = float(frozen["intercept"])
    if len(weights) != len(names) or not all(math.isfinite(v) for v in [*weights, intercept]):
        raise ValueError("verifier coefficient mismatch or nonfinite weight")
    prepared = prepare_row(dict(row), BASE_ARMS)
    findings = []
    for finding in verifier.FINDING_PATTERNS:
        features = verifier.row_features(prepared, finding, feature_arms=BASE_ARMS, wrong_arms=(), use_finding_identity=True)
        logit = intercept + ranker.dot(weights, features)
        probability = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        cutoff = float(frozen["thresholds_by_finding"].get(finding, frozen["threshold"]))
        if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("verifier threshold must be finite and between zero and one")
        if probability >= cutoff:
            findings.append(finding)
    if frozen.get("realization_budget") != delta:
        raise ValueError("verifier realization budget mismatch")
    return verifier.prediction_for_findings(
        prepared, findings,
        prediction_mode="natural_filter_compact_append",
        natural_base_arm="current_only",
        natural_append_arms=("real_prior", "current_only", "prior_report_only"),
        natural_max_token_delta=delta,
    )


def _verifier_reports(row: Mapping[str, Any], frozen_models: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(
        _verifier_report(row, frozen_models[f"verifier_t{delta}"], delta)
        for delta in (0, 2)
    )


def proposal_row(row: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    validate_inference_row(row)
    out = dict(row)
    for key in ("cand_fp02", "cand_fp08"):
        out[key] = rank_report(row, bundle["proposals"][key])[1]
    out["verifier_t0"], out["verifier_t2"] = _verifier_reports(row, bundle["verifiers"])
    return out


def infer(rows: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("schema") != "scfr-portable-model-v2":
        raise ValueError("unsupported model schema")
    if list(bundle["router"]["candidate_arms"]) != list(FINAL_ARMS):
        raise ValueError("final candidate order mismatch")
    out = []
    seen = set()
    for row in rows:
        proposals = proposal_row(row, bundle)
        sample = str(row.get("id") or row.get("sample_id"))
        if sample in seen:
            raise ValueError("duplicate inference sample id")
        seen.add(sample)
        arm, report = rank_report(proposals, bundle["router"])
        out.append({"id": sample, "prediction": report, "selected_arm": arm, "proposals": {name: proposals[name] for name in FINAL_ARMS}})
    return out


def train_bundle(rows: Sequence[Mapping[str, Any]], min_samples: int = 20, diagnostic: bool = False, component_split: str = "train", meta_split: str = "validate") -> dict[str, Any]:
    from scripts import source_causal_verifier as verifier
    from scripts.source_causal_per_finding_gate import prepare_row

    validate_training_rows(rows, min_samples)
    if component_split == meta_split:
        raise ValueError("component and meta training splits must differ")
    components = [dict(row) for row in rows if row["split"] == component_split]
    meta = [dict(row) for row in rows if row["split"] == meta_split]
    if min(len(components), len(meta)) < min_samples:
        raise ValueError("component/meta training rows below min_samples")
    proposals = {}
    for name, fp in (("cand_fp02", .2), ("cand_fp08", .8)):
        prepared = ranker.prepare_rows(components, candidate_arms=BASE_ARMS, split_field="split", fp_weight=fp, fn_weight=.1)
        training = ranker.train_structured_perceptron(prepared, feature_count=17, epochs=50, learning_rate=.05, average_weights=True)
        proposals[name] = {"candidate_arms": list(BASE_ARMS), "feature_names": ranker.feature_names(BASE_ARMS), "weights": training["weights"]}
    prepared = [prepare_row(row, BASE_ARMS) for row in components]
    baseline = verifier.evaluate_source_arm(prepared, arm="current_only", fp_weight=.4, fn_weight=.1, token_weight=.05)["metrics"]
    verifier_models = {}
    for delta in (0, 2):
        tuning = verifier.train_and_tune(
            prepared, prepared,
            feature_arms=BASE_ARMS, wrong_arms=(), use_finding_identity=True,
            c_grid=(.1,) if diagnostic else (.01, .03, .1, .3, 1., 3., 10.),
            class_weights=("balanced",) if diagnostic else (None, "balanced"),
            threshold_grid=(.5,) if diagnostic else tuple(x / 20 for x in range(1, 20)),
            threshold_mode="per_finding", threshold_status_weight=.25,
            threshold_fp_weight=1., threshold_fn_weight=.1,
            fp_weight=.4, fn_weight=.1, token_weight=.05,
            prediction_mode="natural_filter_compact_append",
            natural_base_arm="current_only",
            natural_append_arms=("real_prior", "current_only", "prior_report_only"),
            natural_max_token_delta=delta, baseline_metrics=baseline,
            max_fp_delta=0., max_fn_delta=.05, max_prediction_tokens_delta=0.,
            min_finding_f1_delta=-.02, min_status_accuracy_delta=None,
            min_token_f1_delta=-.01, min_objective_delta=None,
        )
        logistic, best = tuning["model"], tuning["best"]
        verifier_models[f"verifier_t{delta}"] = {
            "realization_budget": delta,
            "feature_names": verifier.feature_names(BASE_ARMS, use_finding_identity=True),
            "coefficients": logistic.coef_[0].tolist(),
            "intercept": float(logistic.intercept_[0]),
            "threshold": float(best["threshold"]),
            "thresholds_by_finding": best["thresholds_by_finding"],
            "C": best["C"], "class_weight": best["class_weight"],
            "constraint": best["constraint"],
        }
    bundle = {
        "schema": "scfr-portable-model-v2", "diagnostic_synthetic_fit": diagnostic,
        "component_split": component_split, "meta_split": meta_split,
        "proposals": proposals, "verifiers": verifier_models,
    }
    materialized = []
    for row in meta:
        source = {k: v for k, v in row.items() if k in INFERENCE_FIELDS}
        materialized.append({**proposal_row(source, bundle), "reference": row["reference"], "split": meta_split})
    final = ranker.prepare_rows(materialized, candidate_arms=FINAL_ARMS, split_field="split", fp_weight=.2, fn_weight=.4)
    training = ranker.train_structured_perceptron(final, feature_count=19, epochs=50, learning_rate=.05, average_weights=True)
    bundle["router"] = {"candidate_arms": list(FINAL_ARMS), "feature_names": ranker.feature_names(FINAL_ARMS), "weights": training["weights"]}
    return bundle


def synthetic_rows() -> list[dict[str, Any]]:
    """Invented text for software tests. No patients, images, or clinical claims."""
    cases = (("No pneumothorax.", "Pneumothorax is present.", "Pneumothorax is present."), ("Pleural effusion is present.", "No pleural effusion.", "Pleural effusion is present."), ("No edema.", "Pulmonary edema is present.", "No edema."), ("Cardiomegaly is present.", "No cardiomegaly.", "Cardiomegaly is present."))
    return [{"id": f"invented-{split}-{index}", "subject_id": f"invented-subject-{split}-{index}", "split": split, "current_only": current, "real_prior": history, "prior_report_only": history, "prior_report": history, "reference": reference} for split in ("train", "validate", "test") for index, (current, history, reference) in enumerate(cases)]
