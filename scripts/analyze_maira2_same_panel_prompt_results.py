#!/usr/bin/env python3
"""Analyze paired same-panel MAIRA-2 default/source-verification outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.analyze_source_causal_arms import (
        bootstrap_delta,
        finding_vector,
        row_scores,
    )
except ImportError:
    from analyze_source_causal_arms import (
        bootstrap_delta,
        finding_vector,
        row_scores,
    )


METHODS = (
    "default",
    "scfr_safety_v1",
    "archived_current_only",
    "archived_history_output",
)
STANDARD_METRICS = (
    "finding_f1",
    "finding_status_accuracy",
    "proxy_fp_per_report",
    "proxy_fn_per_report",
)
ADDITION_METRICS = (
    "positive_additions_vs_current",
    "unsupported_positive_additions_vs_current",
    "history_only_unsupported_carryover",
    "history_supported_additions_vs_current",
)


@dataclass(frozen=True)
class Cell:
    label: str
    result: Path
    source_input: Path
    receipt: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def parse_cell(raw: str) -> Cell:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "--cell must be LABEL=RESULT,INPUT,RECEIPT"
        )
    label, paths = raw.split("=", 1)
    values = paths.split(",")
    if not label or len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--cell must be LABEL=RESULT,INPUT,RECEIPT"
        )
    return Cell(label, *(Path(value) for value in values))


def positive_set(text: str) -> set[str]:
    return {
        finding
        for finding, status in finding_vector(text).items()
        if status == "positive"
    }


def method_text(output_row: Mapping[str, Any], method: str) -> str:
    if method in {"default", "scfr_safety_v1"}:
        return str(output_row["generated"][method]["prediction"])
    return str(output_row[method])


def per_row_metrics(
    prediction: str,
    *,
    reference: str,
    current: str,
    prior: str,
) -> dict[str, float]:
    standard = row_scores(
        {"prediction": prediction, "reference": reference}
    )
    pred = positive_set(prediction)
    ref = positive_set(reference)
    cur = positive_set(current)
    hist = positive_set(prior)
    additions = pred - cur
    return {
        "finding_f1": standard["finding_f1"],
        "finding_status_accuracy": standard[
            "finding_status_accuracy"
        ],
        "proxy_fp_per_report": standard["finding_fp_count"],
        "proxy_fn_per_report": standard["finding_fn_count"],
        "positive_additions_vs_current": float(len(additions)),
        "unsupported_positive_additions_vs_current": float(
            len(additions - ref)
        ),
        "history_only_unsupported_carryover": float(
            len((pred & hist) - cur - ref)
        ),
        "history_supported_additions_vs_current": float(
            len((pred & hist & ref) - cur)
        ),
    }


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = STANDARD_METRICS + ADDITION_METRICS
    return {
        key: statistics.mean(float(row[key]) for row in rows)
        for key in keys
    }


def paired(
    left: Sequence[Mapping[str, float]],
    right: Sequence[Mapping[str, float]],
    clusters: Sequence[str],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    return {
        key: bootstrap_delta(
            [row[key] for row in left],
            [row[key] for row in right],
            n_bootstrap=n_bootstrap,
            seed=seed + index,
            clusters=clusters,
        )
        for index, key in enumerate(STANDARD_METRICS + ADDITION_METRICS)
    }


def analyze_cell(
    cell: Cell,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    result = load(cell.result)
    source = load(cell.source_input)
    receipt = load(cell.receipt)
    if (
        result.get("schema_version")
        != "maira2-same-panel-prompt-output-v1"
        or source.get("schema_version")
        != "maira2-same-panel-prompt-input-v1"
        or receipt.get("schema_version")
        != "maira2-same-panel-prompt-run-receipt-v1"
    ):
        raise RuntimeError(f"{cell.label}: schema mismatch")
    if (
        result["cohort"] != source["cohort"]
        or result["cohort"] != receipt["cohort"]
        or result["history_arm"] != source["history_arm"]
        or result["history_arm"] != receipt["history_arm"]
        or receipt["input_sha256"] != sha256_file(cell.source_input)
        or receipt["output_sha256"] != sha256_file(cell.result)
    ):
        raise RuntimeError(f"{cell.label}: frozen binding mismatch")
    source_by_token = {
        row["case_token"]: row for row in source["rows"]
    }
    output_by_token = {
        row["case_token"]: row for row in result["rows"]
    }
    if (
        len(source_by_token) != len(source["rows"])
        or len(output_by_token) != len(result["rows"])
        or set(source_by_token) != set(output_by_token)
        or len(output_by_token) != receipt["rows"]
    ):
        raise RuntimeError(f"{cell.label}: exact case alignment mismatch")
    tokens = sorted(source_by_token)
    scores: dict[str, list[dict[str, float]]] = {
        method: [] for method in METHODS
    }
    cap_hits = {style: 0 for style in ("default", "scfr_safety_v1")}
    for token in tokens:
        src = source_by_token[token]
        out = output_by_token[token]
        if (
            out["reference"] != src["reference"]
            or out["archived_current_only"]
            != src["archived_current_only"]
            or out["archived_history_output"]
            != src["archived_history_output"]
        ):
            raise RuntimeError(f"{cell.label}: output/source text drift")
        for style in ("default", "scfr_safety_v1"):
            cap_hits[style] += int(
                out["generated"][style]["generation_hit_max_new_tokens"]
            )
        for method in METHODS:
            scores[method].append(
                per_row_metrics(
                    method_text(out, method),
                    reference=src["reference"],
                    current=src["archived_current_only"],
                    prior=src["prior_report"],
                )
            )
    contrasts = {}
    for index, (left, right) in enumerate(
        (
            ("scfr_safety_v1", "default"),
            ("scfr_safety_v1", "archived_current_only"),
            ("scfr_safety_v1", "archived_history_output"),
        )
    ):
        contrasts[f"{left}_minus_{right}"] = paired(
            scores[left],
            scores[right],
            tokens,
            seed=seed + 100 * index,
            n_bootstrap=n_bootstrap,
        )
    return {
        "cohort": result["cohort"],
        "history_arm": result["history_arm"],
        "rows": len(tokens),
        "task_id": receipt["task_id"],
        "hashes": {
            "result_sha256": sha256_file(cell.result),
            "input_sha256": sha256_file(cell.source_input),
            "receipt_sha256": sha256_file(cell.receipt),
        },
        "absolute": {
            method: mean_metrics(scores[method]) for method in METHODS
        },
        "paired_contrasts": contrasts,
        "generation_diagnostics": {
            "cap_hits": cap_hits,
            "nonempty": {
                style: sum(
                    bool(
                        output_by_token[token]["generated"][style][
                            "prediction"
                        ].strip()
                    )
                    for token in tokens
                )
                for style in ("default", "scfr_safety_v1")
            },
        },
        "bootstrap": {
            "unit": "case_token",
            "case_is_unique_subject": True,
            "n_clusters": len(tokens),
            "replicates": n_bootstrap,
            "seed": seed,
        },
        "estimand": (
            "fresh paired MAIRA-2 source-verification instruction versus "
            "fresh default prompt on identical current images and exact prior "
            "report; prior image intentionally absent"
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# MAIRA-2 same-panel source-verification prompt baseline",
        "",
        (
            "All values are report-reference 11-finding proxies. "
            "They are not radiologist image-grounded correctness labels."
        ),
        "",
    ]
    for label, cell in payload["cells"].items():
        lines.extend(
            [
                f"## {label}",
                "",
                (
                    f"{cell['cohort']}; {cell['history_arm']}; "
                    f"n={cell['rows']}; Task `{cell['task_id']}`."
                ),
                "",
                "| Method | Finding F1 | Status acc. | FP/report | FN/report | "
                "Unsupported additions | History-only unsupported carryover |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            item = cell["absolute"][method]
            lines.append(
                f"| {method} | {item['finding_f1']:.4f} | "
                f"{item['finding_status_accuracy']:.4f} | "
                f"{item['proxy_fp_per_report']:.4f} | "
                f"{item['proxy_fn_per_report']:.4f} | "
                f"{item['unsupported_positive_additions_vs_current']:.4f} | "
                f"{item['history_only_unsupported_carryover']:.4f} |"
            )
        lines.extend(["", "Safety minus fresh default (95% cluster CI):", ""])
        contrast = cell["paired_contrasts"][
            "scfr_safety_v1_minus_default"
        ]
        for metric in STANDARD_METRICS + ADDITION_METRICS:
            item = contrast[metric]
            lines.append(
                f"- {metric}: {item['observed_delta']:+.4f} "
                f"[{item['ci95'][0]:+.4f}, {item['ci95'][1]:+.4f}]"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=parse_cell, action="append", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("analysis output already exists")
    labels = [cell.label for cell in args.cell]
    if len(set(labels)) != len(labels):
        raise RuntimeError("duplicate cell label")
    payload = {
        "schema_version": "maira2-same-panel-prompt-analysis-v1",
        "cells": {
            cell.label: analyze_cell(
                cell,
                seed=args.seed + index * 1000,
                n_bootstrap=args.n_bootstrap,
            )
            for index, cell in enumerate(args.cell)
        },
        "scope": {
            "clinical_truth": False,
            "direct_scfr_comparison": False,
            "reason": "v1d proxy-stress mapping contains no SCFR arm",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("xb") as handle:
        handle.write(canonical_bytes(payload))
    with args.output_md.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    print(
        json.dumps(
            {
                "cells": labels,
                "json_sha256": sha256_file(args.output_json),
                "markdown_sha256": sha256_file(args.output_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

