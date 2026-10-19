#!/usr/bin/env python3
"""Train/evaluate a simple source-causal risk selector over longitudinal arms."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, tokenize
    from scripts.audit_prior_copy_temporal import prior_copy_precision
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import finding_vector, row_scores, tokenize
    from scripts.audit_prior_copy_temporal import prior_copy_precision


@dataclass(frozen=True)
class SelectorParams:
    max_copy_precision: float
    max_added_findings: int
    max_token_delta: int


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        out.append(dict(row))
    return out


def as_text(value: Any) -> str:
    return str(value or "").strip()


def arm_prediction(row: Mapping[str, Any], arm: str) -> str:
    value = row.get(arm)
    if isinstance(value, str):
        return value
    if arm == "prediction":
        return as_text(row.get("prediction"))
    return ""


def positive_findings(text: str) -> set[str]:
    return {name for name, status in finding_vector(text).items() if status == "positive"}


def selector_features(row: Mapping[str, Any], current_arm: str, real_arm: str) -> Dict[str, float]:
    current = arm_prediction(row, current_arm)
    real = arm_prediction(row, real_arm)
    prior = as_text(row.get("prior_report") or row.get("context"))
    current_pos = positive_findings(current)
    real_pos = positive_findings(real)
    return {
        "real_prior_copy_precision": prior_copy_precision(real, prior),
        "added_positive_findings": float(len(real_pos - current_pos)),
        "removed_positive_findings": float(len(current_pos - real_pos)),
        "token_delta": float(len(tokenize(real)) - len(tokenize(current))),
    }


def select_arm(
    row: Mapping[str, Any],
    *,
    current_arm: str,
    real_arm: str,
    params: SelectorParams,
) -> str:
    features = selector_features(row, current_arm, real_arm)
    if features["real_prior_copy_precision"] > params.max_copy_precision:
        return current_arm
    if features["added_positive_findings"] > params.max_added_findings:
        return current_arm
    if features["token_delta"] > params.max_token_delta:
        return current_arm
    return real_arm


def select_from_features(features: Mapping[str, float], params: SelectorParams) -> bool:
    if features["real_prior_copy_precision"] > params.max_copy_precision:
        return False
    if features["added_positive_findings"] > params.max_added_findings:
        return False
    if features["token_delta"] > params.max_token_delta:
        return False
    return True


def score_prediction(prediction: str, reference: str) -> Dict[str, float]:
    return row_scores({"prediction": prediction, "reference": reference})


def objective(score: Mapping[str, float], *, fp_weight: float, fn_weight: float) -> float:
    return (
        float(score.get("finding_f1", 0.0))
        - fp_weight * float(score.get("finding_fp_count", 0.0))
        - fn_weight * float(score.get("finding_fn_count", 0.0))
    )


def mean_metrics(scores: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    metrics = sorted(scores[0])
    return {
        metric: statistics.mean(float(row.get(metric, 0.0)) for row in scores)
        for metric in metrics
    }


def evaluate_fixed_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores = [
        score_prediction(arm_prediction(row, arm), as_text(row.get("reference")))
        for row in rows
    ]
    metrics = mean_metrics(scores)
    metrics["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight)
        for score in scores
    ) if scores else 0.0
    return {
        "rows": len(rows),
        "selected_real": None,
        "selected_real_rate": None,
        "metrics": metrics,
    }


def evaluate_selector(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
    params: SelectorParams,
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prepared = prepare_selector_rows(rows, current_arm=current_arm, real_arm=real_arm)
    return evaluate_prepared_selector(
        prepared,
        current_arm=current_arm,
        real_arm=real_arm,
        params=params,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
        emit_rows=emit_rows,
    )


def prepare_selector_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for row in rows:
        reference = as_text(row.get("reference"))
        current_prediction = arm_prediction(row, current_arm)
        real_prediction = arm_prediction(row, real_arm)
        prepared.append(
            {
                "row": row,
                "features": selector_features(row, current_arm, real_arm),
                "predictions": {
                    current_arm: current_prediction,
                    real_arm: real_prediction,
                },
                "scores": {
                    current_arm: score_prediction(current_prediction, reference),
                    real_arm: score_prediction(real_prediction, reference),
                },
            }
        )
    return prepared


def evaluate_prepared_selector(
    prepared: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
    params: SelectorParams,
    fp_weight: float,
    fn_weight: float,
    emit_rows: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    selected_rows: List[Dict[str, Any]] = []
    scores: List[Dict[str, float]] = []
    real_count = 0
    for item in prepared:
        row = item["row"]
        selected = real_arm if select_from_features(item["features"], params) else current_arm
        real_count += int(selected == real_arm)
        prediction = item["predictions"][selected]
        score = item["scores"][selected]
        scores.append(score)
        if emit_rows:
            selected_rows.append(
                {
                    **dict(row),
                    "selected_arm": selected,
                    "prediction": prediction,
                    "selector_features": item["features"],
                }
            )
    metrics = mean_metrics(scores)
    metrics["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight)
        for score in scores
    ) if scores else 0.0
    return (
        {
            "rows": len(prepared),
            "selected_real": real_count,
            "selected_real_rate": real_count / len(prepared) if prepared else 0.0,
            "metrics": metrics,
        },
        selected_rows,
    )


def evaluate_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    scores: List[Dict[str, float]] = []
    real_count = 0
    for row in rows:
        reference = as_text(row.get("reference"))
        current_score = score_prediction(arm_prediction(row, current_arm), reference)
        real_score = score_prediction(arm_prediction(row, real_arm), reference)
        if objective(real_score, fp_weight=fp_weight, fn_weight=fn_weight) > objective(
            current_score,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        ):
            real_count += 1
            scores.append(real_score)
        else:
            scores.append(current_score)
    metrics = mean_metrics(scores)
    metrics["objective"] = statistics.mean(
        objective(score, fp_weight=fp_weight, fn_weight=fn_weight)
        for score in scores
    ) if scores else 0.0
    return {
        "rows": len(rows),
        "selected_real": real_count,
        "selected_real_rate": real_count / len(rows) if rows else 0.0,
        "metrics": metrics,
        "oracle_uses_reference": True,
    }


def split_rows(rows: Sequence[Mapping[str, Any]], split_field: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        split = as_text(row.get(split_field)) or "unspecified"
        out.setdefault(split, []).append(row)
    out["all"] = list(rows)
    return out


def parse_float_grid(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_grid(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def tune_params(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
    copy_grid: Sequence[float],
    added_grid: Sequence[int],
    token_delta_grid: Sequence[int],
    fp_weight: float,
    fn_weight: float,
) -> Dict[str, Any]:
    prepared = prepare_selector_rows(rows, current_arm=current_arm, real_arm=real_arm)
    best: Optional[Tuple[float, float, float, SelectorParams, Dict[str, Any]]] = None
    trials: List[Dict[str, Any]] = []
    for copy_threshold, max_added, max_token_delta in itertools.product(
        copy_grid,
        added_grid,
        token_delta_grid,
    ):
        params = SelectorParams(
            max_copy_precision=float(copy_threshold),
            max_added_findings=int(max_added),
            max_token_delta=int(max_token_delta),
        )
        metrics, _rows = evaluate_prepared_selector(
            prepared,
            current_arm=current_arm,
            real_arm=real_arm,
            params=params,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=False,
        )
        objective_value = float(metrics["metrics"].get("objective", 0.0))
        fp_value = float(metrics["metrics"].get("finding_fp_count", 0.0))
        selected_rate = float(metrics.get("selected_real_rate", 0.0) or 0.0)
        trials.append({"params": params.__dict__, **metrics})
        key = (objective_value, -fp_value, selected_rate)
        if best is None or key > best[:3]:
            best = (objective_value, -fp_value, selected_rate, params, metrics)
    if best is None:
        raise RuntimeError("No selector candidates evaluated")
    return {
        "best_params": best[3].__dict__,
        "best_train_metrics": best[4],
        "n_trials": len(trials),
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_arm: str,
    real_arm: str,
    optional_arms: Sequence[str],
    split_field: str,
    tune_split: str,
    eval_splits: Sequence[str],
    copy_grid: Sequence[float],
    added_grid: Sequence[int],
    token_delta_grid: Sequence[int],
    fp_weight: float,
    fn_weight: float,
    min_samples: int,
    allow_leaky_eval: bool = False,
    allow_underpowered_eval: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(rows) < min_samples:
        raise RuntimeError(f"rows below min-samples gate: {len(rows)} < {min_samples}")
    rows_by_split = split_rows(rows, split_field)
    tune_rows = rows_by_split.get(tune_split, [])
    if not tune_rows:
        raise RuntimeError(f"No rows found for tune split: {tune_split}")
    if len(tune_rows) < min_samples:
        raise RuntimeError(
            f"tune split rows below min-samples gate: {len(tune_rows)} < {min_samples}"
        )
    tuning = tune_params(
        tune_rows,
        current_arm=current_arm,
        real_arm=real_arm,
        copy_grid=copy_grid,
        added_grid=added_grid,
        token_delta_grid=token_delta_grid,
        fp_weight=fp_weight,
        fn_weight=fn_weight,
    )
    params = SelectorParams(**tuning["best_params"])

    selected_predictions: List[Dict[str, Any]] = []
    metrics_by_split: Dict[str, Any] = {}
    split_warnings: Dict[str, List[str]] = {}
    arms = [current_arm, real_arm] + [arm for arm in optional_arms if arm not in {current_arm, real_arm}]
    for split in eval_splits:
        split_payload: Dict[str, Any] = {}
        split_subset = rows_by_split.get(split, [])
        warnings: List[str] = []
        if split == "all":
            warnings.append("contains_tune_split")
        elif split == tune_split:
            warnings.append("is_tune_split")
        if warnings and not allow_leaky_eval:
            raise RuntimeError(
                f"leaky eval split requested without --allow-leaky-eval: {split} ({','.join(warnings)})"
            )
        if len(split_subset) < min_samples:
            warnings.append("below_min_samples")
            if not allow_underpowered_eval:
                raise RuntimeError(
                    f"eval split rows below min-samples gate: {len(split_subset)} < {min_samples} "
                    f"for split {split}; pass --allow-underpowered-eval for diagnostics only"
                )
        for arm in arms:
            if any(arm_prediction(row, arm) for row in split_subset):
                split_payload[arm] = evaluate_fixed_arm(
                    split_subset,
                    arm=arm,
                    fp_weight=fp_weight,
                    fn_weight=fn_weight,
                )
        selector_metrics, selector_rows = evaluate_selector(
            split_subset,
            current_arm=current_arm,
            real_arm=real_arm,
            params=params,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
            emit_rows=True,
        )
        if split == "all":
            selected_predictions = selector_rows
        elif "all" not in eval_splits:
            selected_predictions.extend(selector_rows)
        split_payload["risk_selector"] = selector_metrics
        split_payload["oracle_current_real"] = evaluate_oracle(
            split_subset,
            current_arm=current_arm,
            real_arm=real_arm,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        metrics_by_split[split] = split_payload
        split_warnings[split] = warnings

    return (
        {
            "rows": len(rows),
            "split_field": split_field,
            "splits": {split: len(rows_by_split.get(split, [])) for split in sorted(rows_by_split)},
            "current_arm": current_arm,
            "real_arm": real_arm,
            "optional_arms": list(optional_arms),
            "tune_split": tune_split,
            "eval_splits": list(eval_splits),
            "allow_leaky_eval": bool(allow_leaky_eval),
            "allow_underpowered_eval": bool(allow_underpowered_eval),
            "min_samples": min_samples,
            "split_warnings": split_warnings,
            "objective": {
                "metric": "finding_f1 - fp_weight*finding_fp_count - fn_weight*finding_fn_count",
                "fp_weight": fp_weight,
                "fn_weight": fn_weight,
            },
            "tuning": tuning,
            "metrics_by_split": metrics_by_split,
        },
        selected_predictions,
    )


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    params = report["tuning"]["best_params"]
    lines = [
        "# Source-Causal Risk Selector",
        "",
        f"Rows: {report['rows']}",
        f"Tune split: `{report['tune_split']}`",
        (
            "Best params: "
            f"`max_copy_precision={params['max_copy_precision']}`, "
            f"`max_added_findings={params['max_added_findings']}`, "
            f"`max_token_delta={params['max_token_delta']}`"
        ),
        "",
        "| Split | Arm | Rows | Real Pick Rate | Token-F1 | Finding-F1 | Status Acc | FP | FN | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in report["metrics_by_split"].items():
        warnings = report.get("split_warnings", {}).get(split) or []
        if warnings:
            lines.append(f"`{split}` warnings: {', '.join(warnings)}")
        for arm, metrics_payload in payload.items():
            metrics = metrics_payload["metrics"]
            selected_rate = metrics_payload.get("selected_real_rate")
            selected_text = "n/a" if selected_rate is None else f"{selected_rate:.4f}"
            lines.append(
                f"| `{split}` | `{arm}` | {metrics_payload['rows']} | {selected_text} | "
                f"{metrics.get('token_f1', 0.0):.4f} | {metrics.get('finding_f1', 0.0):.4f} | "
                f"{metrics.get('finding_status_accuracy', 0.0):.4f} | "
                f"{metrics.get('finding_fp_count', 0.0):.4f} | "
                f"{metrics.get('finding_fn_count', 0.0):.4f} | "
                f"{metrics.get('objective', 0.0):.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--current-arm", default="current_only")
    parser.add_argument("--real-arm", default="real_prior")
    parser.add_argument("--optional-arms", default="shuffled_prior,prior_report_only")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--tune-split", default="train")
    parser.add_argument("--eval-splits", default="validate,test")
    parser.add_argument(
        "--allow-leaky-eval",
        action="store_true",
        help="Allow eval splits that include the tune split, such as all or the tune split itself.",
    )
    parser.add_argument("--copy-grid", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--added-grid", default="0,1,2,3,4,99")
    parser.add_argument("--token-delta-grid", default="0,5,10,20,40,9999")
    parser.add_argument("--fp-weight", type=float, default=0.1)
    parser.add_argument("--fn-weight", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--allow-underpowered-eval",
        action="store_true",
        help="Diagnostic override only. By default, eval splits below --min-samples hard-fail.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.predictions)
    report, selected_rows = build_report(
        rows,
        current_arm=str(args.current_arm),
        real_arm=str(args.real_arm),
        optional_arms=[arm.strip() for arm in str(args.optional_arms).split(",") if arm.strip()],
        split_field=str(args.split_field),
        tune_split=str(args.tune_split),
        eval_splits=[split.strip() for split in str(args.eval_splits).split(",") if split.strip()],
        copy_grid=parse_float_grid(str(args.copy_grid)),
        added_grid=parse_int_grid(str(args.added_grid)),
        token_delta_grid=parse_int_grid(str(args.token_delta_grid)),
        fp_weight=float(args.fp_weight),
        fn_weight=float(args.fn_weight),
        min_samples=int(args.min_samples),
        allow_leaky_eval=bool(args.allow_leaky_eval),
        allow_underpowered_eval=bool(args.allow_underpowered_eval),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(report, args.output_md)
    if args.output_predictions:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.output_predictions.write_text(
            json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

