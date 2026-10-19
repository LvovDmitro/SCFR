#!/usr/bin/env python3
"""Compose natural reports from a source-causal per-finding ledger.

The architecture is:
1. fit a source-causal finding ledger from train rows;
2. predict allowed positive findings without using the reference at inference;
3. filter a natural base report to remove sentences with non-ledger positives;
4. optionally append short ledger-approved finding statements.

This is a SMOKE/PILOT method gate, not a final clinical system.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, split_sentences
    from scripts.source_causal_per_finding_gate import (
        DEFAULT_ALL_ARMS,
        FINDING_PHRASES,
        arm_text,
        fit_table,
        load_rows,
        parse_csv,
        parse_float_grid,
        parse_int_grid,
        prepare_row,
        predict_findings,
        split_prepared,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, split_sentences
    from scripts.source_causal_per_finding_gate import (
        DEFAULT_ALL_ARMS,
        FINDING_PHRASES,
        arm_text,
        fit_table,
        load_rows,
        parse_csv,
        parse_float_grid,
        parse_int_grid,
        prepare_row,
        predict_findings,
        split_prepared,
    )


def as_text(value: Any) -> str:
    return str(value or "").strip()


def positive_findings(text: str) -> Set[str]:
    return {finding for finding, status in finding_vector(text).items() if status == "positive"}


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


def mean_metrics(
    scores: Sequence[Mapping[str, float]],
    *,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, float]:
    if not scores:
        return {}
    metric_names = sorted(scores[0])
    out = {
        metric: statistics.mean(float(row.get(metric, 0.0)) for row in scores)
        for metric in metric_names
    }
    out["objective"] = statistics.mean(
        objective(row, status_weight=status_weight, fp_weight=fp_weight, fn_weight=fn_weight)
        for row in scores
    )
    return out


def finding_phrase(finding: str) -> str:
    return FINDING_PHRASES.get(finding, f"{finding.replace('_', ' ')} is present.")


def sentence_score(sentence: str, selected: Set[str], covered: Set[str]) -> Tuple[int, int, int]:
    positives = positive_findings(sentence)
    newly_covered = len((positives & selected) - covered)
    extra = len(positives - selected)
    tokenish_len = len(sentence.split())
    return newly_covered, -extra, -tokenish_len


def compose_report(
    row: Mapping[str, Any],
    *,
    selected_findings: Sequence[str],
    base_arm: str,
    append_missing: bool,
    keep_neutral_sentences: bool,
    max_appended_findings: int,
) -> Dict[str, Any]:
    selected = set(selected_findings)
    base_text = arm_text(row, base_arm)
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    covered: Set[str] = set()

    for sentence in split_sentences(base_text):
        positives = positive_findings(sentence)
        if positives - selected:
            dropped.append({"sentence": sentence, "positive_findings": sorted(positives)})
            continue
        if positives or keep_neutral_sentences:
            kept.append(sentence)
            covered.update(positives & selected)

    appended: List[str] = []
    if append_missing:
        missing = sorted(selected - covered)
        evidence_sentences: List[Tuple[str, str, Tuple[int, int, int]]] = []
        for arm in ("current_only", "real_prior", "prior_report_only"):
            for sentence in split_sentences(arm_text(row, arm)):
                positives = positive_findings(sentence)
                if positives and positives <= selected:
                    evidence_sentences.append((arm, sentence, sentence_score(sentence, selected, covered)))
        evidence_sentences.sort(key=lambda item: item[2], reverse=True)
        for _arm, sentence, _score in evidence_sentences:
            positives = positive_findings(sentence)
            if not positives or positives <= covered:
                continue
            kept.append(sentence)
            covered.update(positives)
            missing = sorted(selected - covered)
            if not missing:
                break
        for finding in missing[:max(0, max_appended_findings)]:
            appended.append(finding_phrase(finding))
            covered.add(finding)

    if kept or appended:
        prediction = " ".join([*kept, *appended])
    else:
        prediction = "No source-valid positive findings."
    return {
        "prediction": prediction,
        "covered_findings": sorted(covered),
        "dropped_sentences": dropped,
        "appended_sentences": appended,
    }


def evaluate_fixed_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores = [
        row_scores(
            {
                "prediction": arm_text(prepared["row"], arm),
                "reference": as_text(prepared.get("reference_text")),
            }
        )
        for prepared in rows
    ]
    return {
        "rows": len(rows),
        "metrics": mean_metrics(scores, status_weight=status_weight, fp_weight=fp_weight, fn_weight=fn_weight),
    }


def evaluate_composer(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: Mapping[str, Mapping[str, Any]],
    feature_arms: Sequence[str],
    threshold: float,
    min_support: int,
    fallback: str,
    wrong_arms: Sequence[str],
    base_arm: str,
    append_missing: bool,
    keep_neutral_sentences: bool,
    max_appended_findings: int,
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scores: List[Dict[str, float]] = []
    selected_counts: Counter[str] = Counter()
    emitted: List[Dict[str, Any]] = []
    dropped_sentence_count = 0
    appended_sentence_count = 0
    for prepared in rows:
        findings = predict_findings(
            prepared["vectors"],
            table=table,
            feature_arms=feature_arms,
            threshold=threshold,
            min_support=min_support,
            fallback=fallback,
            wrong_arms=wrong_arms,
        )
        selected_counts.update(findings)
        composed = compose_report(
            prepared["row"],
            selected_findings=findings,
            base_arm=base_arm,
            append_missing=append_missing,
            keep_neutral_sentences=keep_neutral_sentences,
            max_appended_findings=max_appended_findings,
        )
        dropped_sentence_count += len(composed["dropped_sentences"])
        appended_sentence_count += len(composed["appended_sentences"])
        scores.append(
            row_scores(
                {
                    "prediction": composed["prediction"],
                    "reference": as_text(prepared.get("reference_text")),
                }
            )
        )
        if emit_rows:
            emitted.append(
                {
                    **dict(prepared["row"]),
                    "prediction": composed["prediction"],
                    "ledger_findings": sorted(findings),
                    "covered_findings": composed["covered_findings"],
                    "dropped_sentences": composed["dropped_sentences"],
                    "appended_sentences": composed["appended_sentences"],
                    "ledger_composer": {
                        "threshold": threshold,
                        "min_support": min_support,
                        "fallback": fallback,
                        "feature_arms": list(feature_arms),
                        "base_arm": base_arm,
                        "append_missing": append_missing,
                        "keep_neutral_sentences": keep_neutral_sentences,
                        "max_appended_findings": max_appended_findings,
                    },
                }
            )
    return (
        {
            "rows": len(rows),
            "selected_finding_counts": dict(selected_counts),
            "dropped_sentence_count": dropped_sentence_count,
            "appended_sentence_count": appended_sentence_count,
            "metrics": mean_metrics(scores, status_weight=status_weight, fp_weight=fp_weight, fn_weight=fn_weight),
        },
        emitted,
    )


def gate_against_baselines(
    method_metrics: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    required_baselines: Sequence[str],
    min_objective_delta: float,
    max_finding_f1_drop: float,
    max_token_f1_drop: float,
    max_fn_increase: float,
    max_fp_increase: float,
) -> Dict[str, Any]:
    reasons: List[str] = []
    deltas: Dict[str, Dict[str, float]] = {}
    for baseline_name in required_baselines:
        baseline_payload = baselines.get(baseline_name) or {}
        baseline_metrics = baseline_payload.get("metrics") or {}
        if not baseline_metrics:
            reasons.append(f"missing_baseline:{baseline_name}")
            continue
        delta = {
            "objective": float(method_metrics.get("objective", 0.0)) - float(baseline_metrics.get("objective", 0.0)),
            "finding_f1": float(method_metrics.get("finding_f1", 0.0)) - float(baseline_metrics.get("finding_f1", 0.0)),
            "token_f1": float(method_metrics.get("token_f1", 0.0)) - float(baseline_metrics.get("token_f1", 0.0)),
            "finding_fn_count": float(method_metrics.get("finding_fn_count", 0.0))
            - float(baseline_metrics.get("finding_fn_count", 0.0)),
            "finding_fp_count": float(method_metrics.get("finding_fp_count", 0.0))
            - float(baseline_metrics.get("finding_fp_count", 0.0)),
        }
        deltas[baseline_name] = delta
        if delta["objective"] < min_objective_delta:
            reasons.append(f"objective_delta_below_min_vs_{baseline_name}")
        if delta["finding_f1"] < -max_finding_f1_drop:
            reasons.append(f"finding_f1_drop_vs_{baseline_name}")
        if delta["token_f1"] < -max_token_f1_drop:
            reasons.append(f"token_f1_drop_vs_{baseline_name}")
        if delta["finding_fn_count"] > max_fn_increase:
            reasons.append(f"fn_increase_vs_{baseline_name}")
        if delta["finding_fp_count"] > max_fp_increase:
            reasons.append(f"fp_increase_vs_{baseline_name}")
    return {"verdict": "GO" if not reasons else "NO-GO", "reasons": sorted(set(reasons)), "deltas": deltas}


def tune_composer(
    rows: Sequence[Mapping[str, Any]],
    *,
    table: Mapping[str, Mapping[str, Any]],
    feature_arms: Sequence[str],
    threshold_grid: Sequence[float],
    min_support_grid: Sequence[int],
    fallbacks: Sequence[str],
    wrong_arms: Sequence[str],
    base_arms: Sequence[str],
    append_missing_options: Sequence[bool],
    keep_neutral_options: Sequence[bool],
    max_appended_findings_grid: Sequence[int],
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    best_key: Optional[Tuple[float, float, float, float, float]] = None
    best: Optional[Dict[str, Any]] = None
    trials = 0
    for threshold in threshold_grid:
        for min_support in min_support_grid:
            for fallback in fallbacks:
                for base_arm in base_arms:
                    for append_missing in append_missing_options:
                        for keep_neutral in keep_neutral_options:
                            for max_appended_findings in max_appended_findings_grid:
                                metrics, _rows = evaluate_composer(
                                    rows,
                                    table=table,
                                    feature_arms=feature_arms,
                                    threshold=threshold,
                                    min_support=min_support,
                                    fallback=fallback,
                                    wrong_arms=wrong_arms,
                                    base_arm=base_arm,
                                    append_missing=append_missing,
                                    keep_neutral_sentences=keep_neutral,
                                    max_appended_findings=max_appended_findings,
                                    status_weight=status_weight,
                                    fp_weight=fp_weight,
                                    fn_weight=fn_weight,
                                    emit_rows=False,
                                )
                                score = metrics["metrics"]
                                key = (
                                    float(score.get("objective", 0.0)),
                                    float(score.get("finding_f1", 0.0)),
                                    float(score.get("finding_status_accuracy", 0.0)),
                                    -float(score.get("finding_fp_count", 0.0)),
                                    float(score.get("token_f1", 0.0)),
                                )
                                trial = {
                                    "threshold": threshold,
                                    "min_support": min_support,
                                    "fallback": fallback,
                                    "base_arm": base_arm,
                                    "append_missing": append_missing,
                                    "keep_neutral_sentences": keep_neutral,
                                    "max_appended_findings": max_appended_findings,
                                    **metrics,
                                }
                                trials += 1
                                if best_key is None or key > best_key:
                                    best_key = key
                                    best = trial
    if best is None:
        raise RuntimeError("No composer trials evaluated")
    return {"best": best, "n_trials": trials}


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_arms: Sequence[str],
    feature_arms: Sequence[str],
    wrong_arms: Sequence[str],
    fit_split: str,
    tune_split: str,
    eval_splits: Sequence[str],
    threshold_grid: Sequence[float],
    min_support_grid: Sequence[int],
    fallbacks: Sequence[str],
    base_arms: Sequence[str],
    append_missing_options: Sequence[bool],
    keep_neutral_options: Sequence[bool],
    max_appended_findings_grid: Sequence[int],
    status_weight: float,
    fp_weight: float,
    fn_weight: float,
    min_samples: int,
    required_baselines: Sequence[str],
    min_objective_delta: float,
    max_finding_f1_drop: float,
    max_token_f1_drop: float,
    max_fn_increase: float,
    max_fp_increase: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prepared = [prepare_row(row, all_arms) for row in rows]
    rows_by_split = split_prepared(prepared)
    fit_rows = rows_by_split.get(fit_split, [])
    tune_rows = rows_by_split.get(tune_split, [])
    if len(fit_rows) < min_samples:
        raise RuntimeError(f"fit split rows below min-samples gate: {len(fit_rows)} < {min_samples}")
    if len(tune_rows) < min_samples:
        raise RuntimeError(f"tune split rows below min-samples gate: {len(tune_rows)} < {min_samples}")
    table = fit_table(fit_rows, feature_arms=feature_arms)
    tuning = tune_composer(
        tune_rows,
        table=table,
        feature_arms=feature_arms,
        threshold_grid=threshold_grid,
        min_support_grid=min_support_grid,
        fallbacks=fallbacks,
        wrong_arms=wrong_arms,
        base_arms=base_arms,
        append_missing_options=append_missing_options,
        keep_neutral_options=keep_neutral_options,
        max_appended_findings_grid=max_appended_findings_grid,
        status_weight=status_weight,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    best = tuning["best"]

    metrics_by_split: Dict[str, Any] = {}
    gates_by_split: Dict[str, Any] = {}
    emitted_rows: List[Dict[str, Any]] = []
    for split in eval_splits:
        split_rows = rows_by_split.get(split, [])
        if len(split_rows) < min_samples:
            raise RuntimeError(f"eval split rows below min-samples gate: {split} {len(split_rows)} < {min_samples}")
        payload: Dict[str, Any] = {}
        for arm in all_arms:
            if any(arm_text(prepared_row["row"], arm) for prepared_row in split_rows):
                payload[arm] = evaluate_fixed_arm(
                    split_rows,
                    arm=arm,
                    status_weight=status_weight,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                )
        composer_metrics, rows_out = evaluate_composer(
            split_rows,
            table=table,
            feature_arms=feature_arms,
            threshold=float(best["threshold"]),
            min_support=int(best["min_support"]),
            fallback=str(best["fallback"]),
            wrong_arms=wrong_arms,
            base_arm=str(best["base_arm"]),
            append_missing=bool(best["append_missing"]),
            keep_neutral_sentences=bool(best["keep_neutral_sentences"]),
            max_appended_findings=int(best["max_appended_findings"]),
            status_weight=status_weight,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=True,
        )
        payload["ledger_composer"] = composer_metrics
        metrics_by_split[split] = payload
        gates_by_split[split] = gate_against_baselines(
            composer_metrics["metrics"],
            payload,
            required_baselines=required_baselines,
            min_objective_delta=min_objective_delta,
            max_finding_f1_drop=max_finding_f1_drop,
            max_token_f1_drop=max_token_f1_drop,
            max_fn_increase=max_fn_increase,
            max_fp_increase=max_fp_increase,
        )
        emitted_rows.extend(rows_out)

    return (
        {
            "rows": len(rows),
            "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
            "all_arms": list(all_arms),
            "feature_arms": list(feature_arms),
            "wrong_arms": list(wrong_arms),
            "fit_split": fit_split,
            "tune_split": tune_split,
            "eval_splits": list(eval_splits),
            "min_samples": min_samples,
            "objective": {
                "metric": "finding_f1 + status_weight*status_accuracy - fp_weight*fp - fn_weight*fn",
                "status_weight": status_weight,
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "tuning": tuning,
            "gate_config": {
                "required_baselines": list(required_baselines),
                "min_objective_delta": min_objective_delta,
                "max_finding_f1_drop": max_finding_f1_drop,
                "max_token_f1_drop": max_token_f1_drop,
                "max_fn_increase": max_fn_increase,
                "max_fp_increase": max_fp_increase,
            },
            "metrics_by_split": metrics_by_split,
            "gates_by_split": gates_by_split,
        },
        emitted_rows,
    )


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    best = report["tuning"]["best"]
    lines = [
        "# Source-Causal Ledger Composer",
        "",
        f"Rows: `{report['rows']}`",
        f"Fit split: `{report['fit_split']}`",
        f"Tune split: `{report['tune_split']}`",
        f"Eval splits: `{', '.join(report['eval_splits'])}`",
        f"Feature arms: `{', '.join(report['feature_arms'])}`",
        (
            "Best params: "
            f"`threshold={fmt(best['threshold'])}`, "
            f"`min_support={best['min_support']}`, "
            f"`fallback={best['fallback']}`, "
            f"`base_arm={best['base_arm']}`, "
            f"`append_missing={best['append_missing']}`, "
            f"`keep_neutral={best['keep_neutral_sentences']}`, "
            f"`max_appended={best['max_appended_findings']}`"
        ),
        f"Trials: `{report['tuning']['n_trials']}`",
        "",
        "| Split | Arm/Policy | Finding-F1 | Status Acc | FP | FN | Token-F1 | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        for name, result in payload.items():
            metrics = result["metrics"]
            lines.append(
                f"| `{split}` | `{name}` | {fmt(metrics.get('finding_f1'))} | "
                f"{fmt(metrics.get('finding_status_accuracy'))} | {fmt(metrics.get('finding_fp_count'))} | "
                f"{fmt(metrics.get('finding_fn_count'))} | {fmt(metrics.get('token_f1'))} | "
                f"{fmt(metrics.get('objective'))} |"
            )
        gate = report["gates_by_split"].get(split) or {}
        lines.append(
            f"| `{split}` | `GATE={gate.get('verdict', 'n/a')}` |  |  |  |  |  |  |"
        )
        if gate.get("reasons"):
            lines.append(f"`{split}` gate reasons: {', '.join(gate['reasons'])}")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            (
                "This is an architecture SMOKE gate. It may support a source-causal "
                "plan-and-compose method direction only if held-out gates pass without "
                "using analysis-only wrong-history arms at inference."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_bool_grid(raw: str) -> List[bool]:
    out: List[bool] = []
    for item in parse_csv(raw):
        lower = item.lower()
        if lower in {"1", "true", "yes", "y"}:
            out.append(True)
        elif lower in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean grid value: {item}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--all-arms", default=",".join(DEFAULT_ALL_ARMS))
    parser.add_argument("--feature-arms", default="current_only,real_prior,prior_report_only")
    parser.add_argument("--wrong-arms", default="")
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--tune-split", default="train")
    parser.add_argument("--eval-splits", default="validate")
    parser.add_argument("--threshold-grid", default="0.1,0.25,0.4,0.5,0.6,0.75,0.9")
    parser.add_argument("--min-support-grid", default="1,2,3,5,10")
    parser.add_argument(
        "--fallbacks",
        default="real,current,current_or_real,real_without_wrong,current_or_real_without_wrong,current_or_clean_real,source_majority",
    )
    parser.add_argument("--base-arms", default="real_prior,current_only")
    parser.add_argument("--append-missing-options", default="true,false")
    parser.add_argument("--keep-neutral-options", default="true,false")
    parser.add_argument("--max-appended-findings-grid", default="0,1,2,3,5")
    parser.add_argument("--status-weight", type=float, default=0.25)
    parser.add_argument("--fp-weight", type=float, default=0.5)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--required-baselines", default="current_only,real_prior")
    parser.add_argument("--min-objective-delta", type=float, default=0.0)
    parser.add_argument("--max-finding-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-token-f1-drop", type=float, default=0.02)
    parser.add_argument("--max-fn-increase", type=float, default=0.05)
    parser.add_argument("--max-fp-increase", type=float, default=0.0)
    parser.add_argument("--fail-on-no-go", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report, emitted = build_report(
        load_rows(args.predictions),
        all_arms=parse_csv(args.all_arms),
        feature_arms=parse_csv(args.feature_arms),
        wrong_arms=parse_csv(args.wrong_arms),
        fit_split=args.fit_split,
        tune_split=args.tune_split,
        eval_splits=parse_csv(args.eval_splits),
        threshold_grid=parse_float_grid(args.threshold_grid),
        min_support_grid=parse_int_grid(args.min_support_grid),
        fallbacks=parse_csv(args.fallbacks),
        base_arms=parse_csv(args.base_arms),
        append_missing_options=parse_bool_grid(args.append_missing_options),
        keep_neutral_options=parse_bool_grid(args.keep_neutral_options),
        max_appended_findings_grid=parse_int_grid(args.max_appended_findings_grid),
        status_weight=float(args.status_weight),
        fp_weight=float(args.fp_weight),
        fn_weight=float(args.fn_weight),
        min_samples=int(args.min_samples),
        required_baselines=parse_csv(args.required_baselines),
        min_objective_delta=float(args.min_objective_delta),
        max_finding_f1_drop=float(args.max_finding_f1_drop),
        max_token_f1_drop=float(args.max_token_f1_drop),
        max_fn_increase=float(args.max_fn_increase),
        max_fp_increase=float(args.max_fp_increase),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(json.dumps(emitted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate_verdicts = {split: gate["verdict"] for split, gate in report["gates_by_split"].items()}
    print(json.dumps({"gates": gate_verdicts, "best": report["tuning"]["best"]}, ensure_ascii=False, indent=2))
    if args.fail_on_no_go and any(verdict != "GO" for verdict in gate_verdicts.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

