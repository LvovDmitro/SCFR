#!/usr/bin/env python3
"""Evaluate cheap SCFR report-realizer variants from saved verifier rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from scripts.analyze_source_causal_arms import (
        finding_vector,
        reference_text,
        row_scores,
        split_sentences,
        tokenize,
    )
    from scripts.source_causal_per_finding_gate import FINDING_PHRASES
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_source_causal_arms import (
        finding_vector,
        reference_text,
        row_scores,
        split_sentences,
        tokenize,
    )
    from scripts.source_causal_per_finding_gate import FINDING_PHRASES


DEFAULT_INPUT = Path("logs/source_causal_verifier_deployable_fn_guard_current_predictions.json")
DEFAULT_OUT_JSON = Path("logs/source_causal_composer_sweep.json")
DEFAULT_OUT_MD = Path("outputs/source_causal_composer_sweep.md")
DEFAULT_RADFACT_DIR = Path("logs/source_causal_composer_sweep_radfact_input")
DEFAULT_ARM_DIR = Path("logs/source_causal_composer_sweep_arms")
DEFAULT_CANDIDATE_RADFACT_DIR = Path("logs/source_causal_composer_current_compact_radfact_input")

METRICS = [
    "token_f1",
    "finding_f1",
    "finding_status_accuracy",
    "finding_fp_count",
    "finding_fn_count",
    "prediction_tokens",
]

SOURCE_ARMS = ["current_only", "real_prior", "prior_report_only"]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    return [dict(row) for row in payload]


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_join(sentences: Sequence[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()


def positive_findings(text: str) -> set[str]:
    return {finding for finding, status in finding_vector(text).items() if status == "positive"}


def source_count(row: Mapping[str, Any], finding: str) -> int:
    return sum(finding in positive_findings(str(row.get(arm) or "")) for arm in SOURCE_ARMS)


def selected_findings(row: Mapping[str, Any]) -> List[str]:
    return sorted(str(item) for item in (row.get("selected_findings") or []))


def filter_base(row: Mapping[str, Any], base_arm: str, selected: Sequence[str]) -> List[str]:
    allowed = set(selected)
    kept: List[str] = []
    for sentence in split_sentences(str(row.get(base_arm) or "")):
        findings = positive_findings(sentence)
        if not findings or findings <= allowed:
            kept.append(sentence)
    return kept


def append_compact(
    row: Mapping[str, Any],
    sentences: Sequence[str],
    selected: Sequence[str],
    *,
    min_source_count: int = 1,
    max_token_delta: Optional[int] = None,
    base_arm: str,
) -> str:
    kept = list(sentences)
    base_tokens = len(tokenize(str(row.get(base_arm) or "")))
    max_tokens = None if max_token_delta is None else base_tokens + int(max_token_delta)
    present = positive_findings(clean_join(kept))
    for finding in sorted(set(selected) - present):
        if source_count(row, finding) < min_source_count:
            continue
        phrase = FINDING_PHRASES[finding]
        proposed = clean_join([*kept, phrase])
        if max_tokens is not None and len(tokenize(proposed)) > max_tokens:
            continue
        kept.append(phrase)
    return clean_join(kept)


def fallback_report(selected: Sequence[str]) -> str:
    if not selected:
        return "No source-valid positive findings."
    return clean_join(FINDING_PHRASES[finding] for finding in sorted(selected))


def compose(row: Mapping[str, Any], variant: str) -> str:
    selected = selected_findings(row)
    if variant == "primary_current_sentence_append":
        return str(row.get("prediction") or "")
    if variant == "current_compact_append":
        text = append_compact(
            row,
            filter_base(row, "current_only", selected),
            selected,
            min_source_count=1,
            base_arm="current_only",
        )
        return text or fallback_report(selected)
    if variant == "current_compact_append_consensus2":
        text = append_compact(
            row,
            filter_base(row, "current_only", selected),
            selected,
            min_source_count=2,
            base_arm="current_only",
        )
        return text or fallback_report(selected)
    if variant == "real_filter_only":
        text = clean_join(filter_base(row, "real_prior", selected))
        return text or fallback_report(selected)
    if variant == "real_compact_append":
        text = append_compact(
            row,
            filter_base(row, "real_prior", selected),
            selected,
            min_source_count=1,
            base_arm="real_prior",
        )
        return text or fallback_report(selected)
    if variant == "real_compact_append_consensus2":
        text = append_compact(
            row,
            filter_base(row, "real_prior", selected),
            selected,
            min_source_count=2,
            base_arm="real_prior",
        )
        return text or fallback_report(selected)
    if variant == "real_preserve_compact_append":
        text = append_compact(
            row,
            split_sentences(str(row.get("real_prior") or "")),
            selected,
            min_source_count=1,
            base_arm="real_prior",
        )
        return text or str(row.get("real_prior") or "")
    if variant == "real_preserve_compact_append_consensus2":
        text = append_compact(
            row,
            split_sentences(str(row.get("real_prior") or "")),
            selected,
            min_source_count=2,
            base_arm="real_prior",
        )
        return text or str(row.get("real_prior") or "")
    if variant == "real_preserve_compact_append_cap1":
        text = append_compact(
            row,
            split_sentences(str(row.get("real_prior") or "")),
            selected,
            min_source_count=1,
            max_token_delta=1,
            base_arm="real_prior",
        )
        return text or str(row.get("real_prior") or "")
    raise ValueError(f"Unknown variant: {variant}")


VARIANTS = [
    "primary_current_sentence_append",
    "current_compact_append",
    "current_compact_append_consensus2",
    "real_filter_only",
    "real_compact_append",
    "real_compact_append_consensus2",
    "real_preserve_compact_append",
    "real_preserve_compact_append_consensus2",
    "real_preserve_compact_append_cap1",
]


BASELINES = ["real_prior", "current_only", "matched_shuffled_prior", "prior_report_only"]


def score_prediction(prediction: str, row: Mapping[str, Any]) -> Dict[str, float]:
    return row_scores({"prediction": prediction, "reference": reference_text(row)})


def mean_scores(scored: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not scored:
        return {metric: 0.0 for metric in METRICS}
    return {
        metric: sum(float(row.get(metric, 0.0)) for row in scored) / len(scored)
        for metric in METRICS
    }


def split_rows(rows: Sequence[Mapping[str, Any]], split: str) -> List[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("split") or "") == split]


def evaluate_variant(rows: Sequence[Mapping[str, Any]], variant: str, split: str) -> Dict[str, Any]:
    selected = split_rows(rows, split)
    scored = [score_prediction(compose(row, variant), row) for row in selected]
    return {"rows": len(selected), "metrics": mean_scores(scored)}


def evaluate_baseline(rows: Sequence[Mapping[str, Any]], arm: str, split: str) -> Dict[str, Any]:
    selected = split_rows(rows, split)
    scored = [score_prediction(str(row.get(arm) or ""), row) for row in selected]
    return {"rows": len(selected), "metrics": mean_scores(scored)}


def deltas(left: Mapping[str, float], right: Mapping[str, float]) -> Dict[str, float]:
    return {metric: float(left.get(metric, 0.0)) - float(right.get(metric, 0.0)) for metric in METRICS}


def build_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    splits = ["validate", "test"]
    baselines = {
        split: {arm: evaluate_baseline(rows, arm, split) for arm in BASELINES}
        for split in splits
    }
    variants = {
        split: {variant: evaluate_variant(rows, variant, split) for variant in VARIANTS}
        for split in splits
    }
    for split in splits:
        real = baselines[split]["real_prior"]["metrics"]
        matched = baselines[split]["matched_shuffled_prior"]["metrics"]
        for variant in VARIANTS:
            metrics = variants[split][variant]["metrics"]
            variants[split][variant]["delta_vs_real_prior"] = deltas(metrics, real)
            variants[split][variant]["delta_vs_matched_shuffled_prior"] = deltas(metrics, matched)
    return {
        "stage": "PILOT_ANALYSIS",
        "rows": len(rows),
        "variants": VARIANTS,
        "baselines": baselines,
        "results": variants,
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SCFR Composer Sweep",
        "",
        "Stage: PILOT_ANALYSIS. This sweep reuses saved verifier rows and does not run new generation or training.",
        "",
        "## Test Metrics",
        "",
        "| Variant | Finding-F1 | Status Acc | FP | FN | Token-F1 | Tokens | dF1 vs real | dStatus vs real | dFP vs real | dFN vs real | dTokens vs real |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        payload = report["results"]["test"][variant]
        metrics = payload["metrics"]
        delta = payload["delta_vs_real_prior"]
        lines.append(
            f"| `{variant}` | {fmt(metrics['finding_f1'])} | "
            f"{fmt(metrics['finding_status_accuracy'])} | {fmt(metrics['finding_fp_count'])} | "
            f"{fmt(metrics['finding_fn_count'])} | {fmt(metrics['token_f1'])} | "
            f"{fmt(metrics['prediction_tokens'])} | {fmt(delta['finding_f1'])} | "
            f"{fmt(delta['finding_status_accuracy'])} | {fmt(delta['finding_fp_count'])} | "
            f"{fmt(delta['finding_fn_count'])} | {fmt(delta['prediction_tokens'])} |"
        )
    lines.extend(["", "## Baselines", ""])
    for split in ["validate", "test"]:
        lines.extend(
            [
                f"### {split}",
                "",
                "| Arm | Finding-F1 | Status Acc | FP | FN | Token-F1 | Tokens |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in BASELINES:
            metrics = report["baselines"][split][arm]["metrics"]
            lines.append(
                f"| `{arm}` | {fmt(metrics['finding_f1'])} | "
                f"{fmt(metrics['finding_status_accuracy'])} | {fmt(metrics['finding_fp_count'])} | "
                f"{fmt(metrics['finding_fn_count'])} | {fmt(metrics['token_f1'])} | "
                f"{fmt(metrics['prediction_tokens'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gate Notes",
            "",
            "- A candidate is useful only if it keeps non-negative Finding-F1/status direction vs `real_prior`, does not increase FN, and materially lowers token delta vs `primary_current_sentence_append`.",
            "- These are proxy metrics; RadFact/GREEN should remain SMOKE-only until a proxy candidate clears the gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_radfact_input(rows: Sequence[Mapping[str, Any]], out_dir: Path, variants: Sequence[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for row in split_rows(rows, "test"):
        sample = {
            "sample_id": row.get("sample_id") or row.get("id"),
            "reference": reference_text(row),
            "baseline": str(row.get("real_prior") or ""),
        }
        for variant in variants:
            sample[variant] = compose(row, variant)
        samples.append(sample)
    out_path = out_dir / "steering_predictions_scfr_composer_sweep_test.json"
    dump_json(out_path, samples)
    return out_path


def write_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    split: str,
    variants: Sequence[str],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    selected_rows = split_rows(rows, split)
    for arm in [*BASELINES, *variants]:
        payload = []
        for row in selected_rows:
            prediction = compose(row, arm) if arm in variants else str(row.get(arm) or "")
            payload.append(
                {
                    "sample_id": row.get("sample_id") or row.get("id"),
                    "subject_id": row.get("subject_id"),
                    "study_id": row.get("study_id"),
                    "split": row.get("split"),
                    "reference": reference_text(row),
                    "prediction": prediction,
                }
            )
        out_path = out_dir / f"{arm}_{split}.json"
        dump_json(out_path, payload)
        paths[arm] = str(out_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--radfact-dir", type=Path, default=DEFAULT_RADFACT_DIR)
    parser.add_argument("--arm-dir", type=Path, default=DEFAULT_ARM_DIR)
    parser.add_argument("--candidate-radfact-dir", type=Path, default=DEFAULT_CANDIDATE_RADFACT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    report = build_report(rows)
    radfact_path = build_radfact_input(
        rows,
        args.radfact_dir,
        [
            "primary_current_sentence_append",
            "real_preserve_compact_append",
            "real_preserve_compact_append_consensus2",
            "real_preserve_compact_append_cap1",
        ],
    )
    arm_paths = write_arm_rows(rows, args.arm_dir, split="test", variants=VARIANTS)
    candidate_radfact_path = build_radfact_input(
        rows,
        args.candidate_radfact_dir,
        ["current_compact_append"],
    )
    report["radfact_green_input"] = str(radfact_path)
    report["candidate_radfact_green_input"] = str(candidate_radfact_path)
    report["arm_rows"] = arm_paths
    dump_json(args.out_json, report)
    write_markdown(args.out_md, report)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {radfact_path}")
    print(f"Wrote {candidate_radfact_path}")
    print(f"Wrote arm rows to {args.arm_dir}")


if __name__ == "__main__":
    main()

