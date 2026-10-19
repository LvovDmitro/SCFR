#!/usr/bin/env python3
"""Run a trained source-causal corrector on held-out counterfactual draft arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.source_causal_corrector_gate import build_report, write_markdown


DEFAULT_ARMS = (
    "current_only",
    "real_prior",
    "prior_report_only",
    "matched_shuffled_prior",
    "shuffled_prior",
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list: {path}")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected object at row {idx}: {path}")
        rows.append(dict(row))
    return rows


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(as_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def sample_id(row: Mapping[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "study_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"row_{idx}"


def selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        if split != "all" and str(row.get("split") or "") != split:
            continue
        selected.append(dict(row))
        if max_samples > 0 and len(selected) >= max_samples:
            break
    return selected


def resolve_image(paths: Iterable[str], image_root: Optional[Path]) -> Optional[str]:
    for raw in paths:
        raw_path = Path(raw)
        candidates = [raw_path]
        if image_root is not None:
            candidates.extend([image_root / raw, image_root / raw_path.name])
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def load_corrector(args: argparse.Namespace):
    copy_arms = set(args.copy_arm or [])
    if args.copy_baseline or set(args.arm).issubset(copy_arms):
        return None
    from train.train_corrector_medgemma_multimodal import MultimodalCorrector

    adapter_dir = args.adapter_dir
    if not adapter_dir:
        adapter_dir = args.checkpoint_dir / args.adapter_name
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    return MultimodalCorrector(
        model_name=args.model,
        use_qlora=not args.no_qlora,
        adapter_dir=str(adapter_dir),
        local_files_only=args.local_files_only,
    )


def prediction_for_arm(row: Mapping[str, Any], arm: str) -> str:
    value = row.get(arm)
    if isinstance(value, str):
        return value.strip()
    value = row.get("prediction")
    return value.strip() if isinstance(value, str) else ""


def build_prompt_error_type(arm: str) -> str:
    if arm in {"matched_shuffled_prior", "shuffled_prior"}:
        return "source_causal_wrong_history"
    return "clinical_preference"


def baseline_arm_rows(rows: Sequence[Mapping[str, Any]], arm: str) -> List[Dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        draft = prediction_for_arm(row, arm)
        if not draft:
            continue
        out.append(
            {
                **dict(row),
                "sample_id": sample_id(row, idx),
                "prediction": draft,
                "source_arm": arm,
            }
        )
    return out


def correct_arm_rows(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    model: Any,
    image_root: Optional[Path],
    image_ablation: str,
    max_new_tokens: int,
    copy_baseline: bool,
    progress_every: int,
) -> List[Dict[str, Any]]:
    corrected: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        draft = prediction_for_arm(row, arm)
        if not draft:
            continue
        image_path = None if image_ablation == "blank" else resolve_image(as_list(row.get("image_paths")), image_root)
        if copy_baseline:
            prediction = draft
        else:
            prediction = model.correct(
                draft=draft,
                image_path=image_path,
                error_type=build_prompt_error_type(arm),
                max_new_tokens=max_new_tokens,
            )
        corrected.append(
            {
                **dict(row),
                "sample_id": sample_id(row, idx),
                "prediction": prediction,
                "source_arm": arm,
                "baseline_prediction": draft,
                "corrector_image_path": image_path,
                "corrector_image_ablation": image_ablation,
                "corrector_error_type": build_prompt_error_type(arm),
            }
        )
        if progress_every > 0 and len(corrected) % progress_every == 0:
            print(
                f"[corrector_eval] arm={arm} corrected={len(corrected)}/{len(rows)} "
                f"image_ablation={image_ablation}",
                flush=True,
            )
    return corrected


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gate_args(args: argparse.Namespace):
    return argparse.Namespace(
        status_weight=args.status_weight,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
        min_samples=args.min_samples,
        min_processed_rate=args.min_processed_rate,
        max_empty_rows=args.max_empty_rows,
        max_identical_fraction=args.max_identical_fraction,
        min_wrong_objective_delta=args.min_wrong_objective_delta,
        max_wrong_fp_delta=args.max_wrong_fp_delta,
        max_deployable_objective_drop=args.max_deployable_objective_drop,
        max_deployable_finding_f1_drop=args.max_deployable_finding_f1_drop,
        max_deployable_token_f1_drop=args.max_deployable_token_f1_drop,
        max_deployable_fn_increase=args.max_deployable_fn_increase,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        cluster_key=args.cluster_key,
        require_ci_positive=args.require_ci_positive,
        wrong_arm=args.wrong_arm,
        deployable_arm=args.deployable_arm,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    all_rows = load_rows(args.input_merged)
    rows = selected_rows(all_rows, split=args.split, max_samples=args.max_samples)
    if len(rows) < args.min_samples:
        raise RuntimeError(f"selected rows below min_samples: {len(rows)}<{args.min_samples}")

    image_root = args.image_root
    if image_root and not image_root.is_absolute():
        image_root = PROJECT_ROOT / image_root
    model = load_corrector(args)

    baseline_specs = []
    corrected_specs = []
    outputs: Dict[str, Any] = {
        "input": str(args.input_merged),
        "split": args.split,
        "rows": len(rows),
        "arms": {},
        "copy_baseline": bool(args.copy_baseline),
        "copy_arm": list(args.copy_arm),
        "image_ablation": args.image_ablation,
    }
    for arm in args.arm:
        copy_arm = bool(args.copy_baseline or arm in args.copy_arm)
        baseline_rows = baseline_arm_rows(rows, arm)
        corrected_rows = correct_arm_rows(
            rows,
            arm,
            model=model,
            image_root=image_root,
            image_ablation=args.image_ablation,
            max_new_tokens=args.max_new_tokens,
            copy_baseline=copy_arm,
            progress_every=args.progress_every,
        )
        baseline_path = args.output_dir / f"baseline_{arm}.json"
        corrected_path = args.output_dir / f"corrected_{arm}.json"
        write_rows(baseline_path, baseline_rows)
        write_rows(corrected_path, corrected_rows)
        baseline_specs.append((arm, baseline_path))
        corrected_specs.append((arm, corrected_path))
        image_rows = sum(bool(row.get("corrector_image_path")) for row in corrected_rows)
        outputs["arms"][arm] = {
            "baseline_path": str(baseline_path),
            "corrected_path": str(corrected_path),
            "baseline_rows": len(baseline_rows),
            "corrected_rows": len(corrected_rows),
            "resolved_image_rows": image_rows,
            "image_resolution": image_rows / len(corrected_rows) if corrected_rows else 0.0,
            "copied_from_baseline": copy_arm,
        }

    report = build_report(baseline_specs, corrected_specs, gate_args(args))
    args.gate_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.gate_output_md:
        write_markdown(report, args.gate_output_md)
    outputs["gate"] = report["gate"]
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-merged", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/source_causal_corrector_smoke"))
    parser.add_argument("--adapter-name", default="lora_best")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--image-root", type=Path, default=Path("data/local_sources"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--gate-output-json", type=Path, required=True)
    parser.add_argument("--gate-output-md", type=Path, default=None)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--image-ablation",
        choices=["none", "blank"],
        default="none",
        help="Use blank images for source-condition ablation instead of resolved study images.",
    )
    parser.add_argument("--copy-baseline", action="store_true")
    parser.add_argument(
        "--copy-arm",
        action="append",
        default=[],
        help="Copy selected arms without model correction. Repeat for routed deployable arms.",
    )
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--wrong-arm", action="append", default=[])
    parser.add_argument("--deployable-arm", action="append", default=[])
    parser.add_argument("--status-weight", type=float, default=0.25)
    parser.add_argument("--fp-weight", type=float, default=0.5)
    parser.add_argument("--fn-weight", type=float, default=0.1)
    parser.add_argument("--min-processed-rate", type=float, default=0.9)
    parser.add_argument("--max-empty-rows", type=int, default=0)
    parser.add_argument("--max-identical-fraction", type=float, default=0.05)
    parser.add_argument("--min-wrong-objective-delta", type=float, default=0.0)
    parser.add_argument("--max-wrong-fp-delta", type=float, default=0.0)
    parser.add_argument("--max-deployable-objective-drop", type=float, default=0.005)
    parser.add_argument("--max-deployable-finding-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-deployable-token-f1-drop", type=float, default=0.01)
    parser.add_argument("--max-deployable-fn-increase", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cluster-key", default="subject_id")
    parser.add_argument("--require-ci-positive", action="store_true")
    args = parser.parse_args(argv)
    if not args.arm:
        args.arm = list(DEFAULT_ARMS)
    if not args.wrong_arm:
        args.wrong_arm = ["matched_shuffled_prior", "shuffled_prior"]
    if not args.deployable_arm:
        args.deployable_arm = ["current_only", "real_prior", "prior_report_only"]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps({"gate": summary["gate"], "rows": summary["rows"]}, ensure_ascii=False, indent=2))
    return 0 if summary["gate"]["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

