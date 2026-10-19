#!/usr/bin/env python3
"""Build LaTeX-ready SCFR main result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


TARGETS = ("seed1221", "seed911", "locked1000")
RAW_TARGET_LABELS = {
    "seed1221": "Cohort A",
    "seed911": "Cohort B",
    "locked1000": "Cohort C",
}


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def raw_targets(raw: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {target["name"]: target for target in raw["targets"]}


def raw_delta(raw: Mapping[str, Any], target: str, arm: str, metric: str) -> Mapping[str, Any]:
    return raw_targets(raw)[target]["deltas_vs_current_only"][arm][metric]


def confirm_item(confirm: Mapping[str, Any], target: str, family: str, metric: str) -> Optional[Mapping[str, Any]]:
    if family == "primary_holm":
        items = confirm["primary_holm"]
    else:
        items = confirm["secondary_holm"][family]
    for item in items:
        if item["target"] == target and item["metric"] == metric:
            return item
    return None


def parity_items(confirm: Mapping[str, Any], target: str) -> Sequence[Mapping[str, Any]]:
    return [item for item in confirm["current_only_parity"] if item["target"] == target]


def fmt(value: Any, digits: int = 3, signed: bool = True) -> str:
    number = float(value)
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def fmt_p(value: Any) -> str:
    number = float(value)
    if number < 0.001:
        return "<.001"
    return f"{number:.4f}"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def make_raw_table_tex(raw: Mapping[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Raw source-causal vulnerability across longitudinal MIMIC-CXR cohorts. Deltas are versus current-image generation; positive FP means more unsupported positive findings, while negative FN means fewer missed reference positives.}",
        "\\label{tab:scfr_raw_vulnerability}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Cohort & \\multicolumn{3}{c}{Real-history generation} & \\multicolumn{3}{c}{Matched wrong-history stress test} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        " & $\\Delta$FP & $\\Delta$FN & $\\Delta$Status & $\\Delta$FP & $\\Delta$FN & $\\Delta$Status \\\\",
        "\\midrule",
    ]
    for target in TARGETS:
        real_fp = raw_delta(raw, target, "real_prior", "finding_fp_count")
        real_fn = raw_delta(raw, target, "real_prior", "finding_fn_count")
        real_status = raw_delta(raw, target, "real_prior", "finding_status_accuracy")
        matched_fp = raw_delta(raw, target, "matched_shuffled_prior", "finding_fp_count")
        matched_fn = raw_delta(raw, target, "matched_shuffled_prior", "finding_fn_count")
        matched_status = raw_delta(raw, target, "matched_shuffled_prior", "finding_status_accuracy")
        lines.append(
            "{target} & {rfp} & {rfn} & {rst} & {mfp} & {mfn} & {mst} \\\\".format(
                target=tex_escape(RAW_TARGET_LABELS[target]),
                rfp=fmt(real_fp["delta"]),
                rfn=fmt(real_fn["delta"]),
                rst=fmt(real_status["delta"]),
                mfp=fmt(matched_fp["delta"]),
                mfn=fmt(matched_fn["delta"]),
                mst=fmt(matched_status["delta"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def make_confirm_table_tex(confirm: Mapping[str, Any]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Confirmatory SCFR gate with 5000 subject-cluster bootstrap resamples. The primary endpoint is matched wrong-history false-positive repair with Holm correction; current-image parity is checked under fixed margins for F1, status, FP, and FN.}",
        "\\label{tab:scfr_confirmatory_gate}",
        "\\begin{tabular}{lrrrrc}",
        "\\toprule",
        "Cohort & $\\Delta$FP vs matched & Holm $p$ & $\\Delta$Status vs matched & $\\Delta$Token-F1 vs matched & Current parity \\\\",
        "\\midrule",
    ]
    for target in TARGETS:
        primary = confirm_item(confirm, target, "primary_holm", "finding_fp_count")
        status = confirm_item(confirm, target, "secondary_matched_status", "finding_status_accuracy")
        token = confirm_item(confirm, target, "secondary_matched_token_f1", "token_f1")
        parity = all(bool(item["passed"]) for item in parity_items(confirm, target))
        lines.append(
            "{target} & {fp} & {pval} & {status} & {token} & {parity} \\\\".format(
                target=tex_escape(RAW_TARGET_LABELS[target]),
                fp=fmt(primary["observed_delta"]),
                pval=fmt_p(primary["holm_adjusted_p"]),
                status=fmt(status["observed_delta"]),
                token=fmt(token["observed_delta"]),
                parity="pass" if parity else "fail",
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def make_markdown(raw: Mapping[str, Any], confirm: Mapping[str, Any], tex_path: Path) -> str:
    lines = [
        "# SCFR Main Result Tables 2026-05-04",
        "",
        f"LaTeX output: `{tex_path}`",
        "",
        "## Table 1 Source",
        "",
        "| Cohort | Real-history dFP | Real-history dFN | Real-history dStatus | Matched wrong-history dFP | Matched wrong-history dFN | Matched wrong-history dStatus |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        lines.append(
            "| {target} | {rfp} | {rfn} | {rst} | {mfp} | {mfn} | {mst} |".format(
                target=RAW_TARGET_LABELS[target],
                rfp=fmt(raw_delta(raw, target, "real_prior", "finding_fp_count")["delta"]),
                rfn=fmt(raw_delta(raw, target, "real_prior", "finding_fn_count")["delta"]),
                rst=fmt(raw_delta(raw, target, "real_prior", "finding_status_accuracy")["delta"]),
                mfp=fmt(raw_delta(raw, target, "matched_shuffled_prior", "finding_fp_count")["delta"]),
                mfn=fmt(raw_delta(raw, target, "matched_shuffled_prior", "finding_fn_count")["delta"]),
                mst=fmt(raw_delta(raw, target, "matched_shuffled_prior", "finding_status_accuracy")["delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Table 2 Source",
            "",
            "| Cohort | dFP vs matched wrong-history | Holm p | dStatus vs matched wrong-history | dToken-F1 vs matched wrong-history | Current-image parity |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for target in TARGETS:
        primary = confirm_item(confirm, target, "primary_holm", "finding_fp_count")
        status = confirm_item(confirm, target, "secondary_matched_status", "finding_status_accuracy")
        token = confirm_item(confirm, target, "secondary_matched_token_f1", "token_f1")
        parity = all(bool(item["passed"]) for item in parity_items(confirm, target))
        lines.append(
            f"| {RAW_TARGET_LABELS[target]} | {fmt(primary['observed_delta'])} | {fmt_p(primary['holm_adjusted_p'])} | "
            f"{fmt(status['observed_delta'])} | {fmt(token['observed_delta'])} | {'pass' if parity else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Safe Notes",
            "",
            "- Table 1 shows the vulnerability: prior context is a causal source, not a monotonic quality signal.",
            "- Table 2 shows the main controller gate: matched wrong-history FP repair under current-only parity margins.",
            "- Mention the FN tradeoff in the surrounding result text; do not hide it.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, default=Path("logs/source_causal_p12_raw_vulnerability_consolidated_2026_05_04.json"))
    parser.add_argument("--confirm-json", type=Path, default=Path("logs/source_causal_confirmatory_gate_bootstrap5000_2026_05_04.json"))
    parser.add_argument("--output-tex", type=Path, default=Path("paper/main_track_scfr_2026/tables/scfr_main_tables_2026_05_04.tex"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/scfr_main_tables_2026_05_04.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_json(args.raw_json)
    confirm = load_json(args.confirm_json)
    tex = make_raw_table_tex(raw) + "\n\n" + make_confirm_table_tex(confirm) + "\n"
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex, encoding="utf-8")
    args.output_md.write_text(make_markdown(raw, confirm, args.output_tex), encoding="utf-8")
    print(f"Wrote {args.output_tex}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
