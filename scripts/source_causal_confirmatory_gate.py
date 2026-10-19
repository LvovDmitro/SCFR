#!/usr/bin/env python3
"""Apply the SCFR confirmatory gate to clustered result reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PARITY_MARGINS = {
    "finding_f1": -0.02,
    "finding_status_accuracy": -0.02,
    "finding_fp_count": 0.10,
    "finding_fn_count": 0.10,
}


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_delta(report: Mapping[str, Any], metric: str, arm: str = "crossseed_meta") -> Mapping[str, Any]:
    return report["deltas_vs_baseline"][arm][metric]


def holm(items: Sequence[Mapping[str, Any]], *, alpha: float) -> List[Dict[str, Any]]:
    ordered = sorted((dict(item) for item in items), key=lambda item: float(item["p_value"]))
    adjusted: List[Dict[str, Any]] = []
    running_max = 0.0
    m = len(ordered)
    for idx, item in enumerate(ordered):
        threshold = alpha / (m - idx)
        p_value = float(item["p_value"])
        adjusted_p = min(1.0, max(running_max, (m - idx) * p_value))
        running_max = adjusted_p
        item["holm_threshold"] = threshold
        item["holm_adjusted_p"] = adjusted_p
        # Holm is a step-down procedure: once an ordered hypothesis fails,
        # later hypotheses cannot be rejected merely because their individual
        # threshold is wider.  The monotone adjusted p-value encodes this rule.
        item["holm_reject"] = bool(
            adjusted_p <= alpha and item.get("direction_ok", True)
        )
        adjusted.append(item)
    return sorted(adjusted, key=lambda item: (str(item["target"]), str(item["metric"])))


def signed_direction_ok(delta: float, direction: str) -> bool:
    if direction == "negative":
        return delta < 0.0
    if direction == "positive":
        return delta > 0.0
    raise ValueError(f"Unknown direction: {direction}")


def collect_primary_fp(records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    items = []
    for target in sorted(records):
        payload = metric_delta(records[target]["matched"], "finding_fp_count")
        observed = float(payload["observed_delta"])
        items.append(
            {
                "family": "primary_matched_fp_repair",
                "target": target,
                "metric": "finding_fp_count",
                "baseline": "matched_shuffled_prior",
                "observed_delta": observed,
                "p_value": float(payload["p_two_sided"]),
                "direction": "negative",
                "direction_ok": signed_direction_ok(observed, "negative"),
            }
        )
    return items


def collect_secondary(records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    families = {
        "secondary_matched_status": ("finding_status_accuracy", "positive"),
        "secondary_matched_token_f1": ("token_f1", "positive"),
        "exploratory_matched_finding_f1": ("finding_f1", "positive"),
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for family, (metric, direction) in families.items():
        items = []
        for target in sorted(records):
            payload = metric_delta(records[target]["matched"], metric)
            observed = float(payload["observed_delta"])
            items.append(
                {
                    "family": family,
                    "target": target,
                    "metric": metric,
                    "baseline": "matched_shuffled_prior",
                    "observed_delta": observed,
                    "p_value": float(payload["p_two_sided"]),
                    "direction": direction,
                    "direction_ok": signed_direction_ok(observed, direction),
                }
            )
        out[family] = items
    return out


def parity_checks(records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    checks = []
    for target in sorted(records):
        for metric, margin in PARITY_MARGINS.items():
            payload = metric_delta(records[target]["current"], metric)
            observed = float(payload["observed_delta"])
            if metric in {"finding_f1", "finding_status_accuracy"}:
                passed = observed >= margin
                criterion = f"delta >= {margin:+.4f}"
            else:
                passed = observed <= margin
                criterion = f"delta <= {margin:+.4f}"
            checks.append(
                {
                    "target": target,
                    "metric": metric,
                    "baseline": "current_only",
                    "observed_delta": observed,
                    "criterion": criterion,
                    "passed": bool(passed),
                    "p_value": float(payload["p_two_sided"]),
                }
            )
    return checks


def load_records(reports: Mapping[str, Mapping[str, Path]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    records: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for target, paths in reports.items():
        records[target] = {}
        for name, path in paths.items():
            payload = dict(load_json(path))
            payload["_path"] = str(path)
            records[target][name] = payload
    return records


def build_payload(
    *,
    alpha: float,
    reports: Mapping[str, Mapping[str, Path]],
    report_set: str = "custom",
) -> Dict[str, Any]:
    records = load_records(reports)
    primary = holm(collect_primary_fp(records), alpha=alpha)
    secondary = {
        family: holm(items, alpha=alpha)
        for family, items in collect_secondary(records).items()
    }
    parity = parity_checks(records)
    return {
        "stage": "RETROSPECTIVE_CONFIRMATORY_GOVERNANCE",
        "report_set": report_set,
        "inputs": {
            target: {name: str(path) for name, path in paths.items()}
            for target, paths in reports.items()
        },
        "alpha": alpha,
        "primary_family": "matched wrong-history FP repair across supplied targets",
        "primary_holm": primary,
        "primary_pass": all(item["holm_reject"] for item in primary),
        "current_only_parity_margins": PARITY_MARGINS,
        "current_only_parity": parity,
        "current_only_parity_pass": all(item["passed"] for item in parity),
        "secondary_holm": secondary,
        "verdict": "PASS_CONSERVATIVE_SCFR_CLAIM"
        if all(item["holm_reject"] for item in primary) and all(item["passed"] for item in parity)
        else "FAIL_CONFIRMATORY_GATE",
        "claim_boundary": {
            "safe": [
                "matched wrong-history FP repair with Holm correction",
                "current-only-like proxy quality under fixed parity margins",
                "explicit FN tradeoff",
            ],
            "unsafe": [
                "generic RRG SOTA",
                "clinical safety",
                "current-only dominance",
                "post-hoc preregistration",
            ],
        },
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# SCFR Confirmatory Gate",
        "",
        "Stage: `RETROSPECTIVE_CONFIRMATORY_GOVERNANCE` on supplied clustered result artifacts.",
        f"Report set: `{payload.get('report_set', 'custom')}`.",
        "",
        "This gate specification is reusable for future locked holdouts. For",
        "existing artifacts it is retrospective governance, not post-hoc",
        "preregistration.",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        "## Primary Family",
        "",
        f"Primary endpoint family: {payload['primary_family']}.",
        f"Holm alpha: `{payload['alpha']}`.",
        "",
        "| Target | Delta FP vs matched | p | Holm adjusted p | Holm reject |",
        "|---|---:|---:|---:|---|",
    ]
    for item in payload["primary_holm"]:
        lines.append(
            f"| `{item['target']}` | {fmt(item['observed_delta'])} | {fmt(item['p_value'])} | "
            f"{fmt(item['holm_adjusted_p'])} | `{item['holm_reject']}` |"
        )
    lines.extend(
        [
            "",
            "## Current-Only Parity Margins",
            "",
            "| Target | Metric | Delta vs current-only | Criterion | Pass |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in payload["current_only_parity"]:
        lines.append(
            f"| `{item['target']}` | `{item['metric']}` | {fmt(item['observed_delta'])} | "
            f"{item['criterion']} | `{item['passed']}` |"
        )
    lines.extend(["", "## Secondary Families", ""])
    for family, items in payload["secondary_holm"].items():
        lines.extend([f"### `{family}`", "", "| Target | Delta | p | Holm adjusted p | Holm reject |", "|---|---:|---:|---:|---|"])
        for item in items:
            lines.append(
                f"| `{item['target']}` | {fmt(item['observed_delta'])} | {fmt(item['p_value'])} | "
                f"{fmt(item['holm_adjusted_p'])} | `{item['holm_reject']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- Safe: conservative SCFR claim if primary FP repair and current-only",
            "  parity margins pass.",
            "- Required caveat: FN increases versus matched wrong-history are a tradeoff.",
            "- Unsafe: generic RRG SOTA, clinical safety, current-only dominance, or",
            "  calling this retrospective gate a preregistered analysis.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_target_spec(spec: str) -> tuple[str, Dict[str, Path]]:
    parts = spec.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise argparse.ArgumentTypeError(
            "Target specs must use NAME=current.json,matched.json"
        )
    target = parts[0].strip()
    paths = [item.strip() for item in parts[1].split(",") if item.strip()]
    if len(paths) != 2:
        raise argparse.ArgumentTypeError(
            "Target specs must provide exactly two paths: current,matched"
        )
    return target, {"current": Path(paths[0]), "matched": Path(paths[1])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target_spec,
        default=[],
        metavar="NAME=CURRENT_JSON,MATCHED_JSON",
        help="Clustered reports for one evaluation target. Repeat for multiple targets.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("logs/source_causal_confirmatory_gate.json"))
    parser.add_argument("--output-md", type=Path, default=Path("outputs/source_causal_confirmatory_gate.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {target: paths for target, paths in args.target}
    if not reports:
        raise SystemExit("At least one --target NAME=CURRENT_JSON,MATCHED_JSON is required.")
    payload = build_payload(
        alpha=float(args.alpha),
        reports=reports,
        report_set="custom",
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()

