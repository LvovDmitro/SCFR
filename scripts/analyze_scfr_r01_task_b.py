#!/usr/bin/env python3
"""Decode and aggregate the locked R01 Task-B comparison.

Only aggregate statistics are written.  Case tokens, source identifiers,
report text, free-text reasons, and finding strings never leave the private
root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.manage_scfr_human_adjudication_release import (
        _load_state,
        _read_secret,
    )
    from scripts.prepare_scfr_human_adjudication_packet import (
        sha256_canonical_json,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.manage_scfr_human_adjudication_release import (
        _load_state,
        _read_secret,
    )
    from scripts.prepare_scfr_human_adjudication_packet import (
        sha256_canonical_json,
        sha256_file,
    )


ARMS = ("current_only", "scfr")
LIST_FIELDS = (
    "supported_findings_recovered",
    "supported_findings_lost",
    "unsupported_additions",
    "important_omissions",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError("Expected JSON object")
    return value


def _fraction(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("Cannot calculate an empty percentile")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    differences: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not differences:
        raise RuntimeError("Paired bootstrap requires observations")
    rng = random.Random(seed)
    n = len(differences)
    samples = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    ]
    return {
        "estimate": statistics.fmean(differences),
        "ci95_low": _percentile(samples, 0.025),
        "ci95_high": _percentile(samples, 0.975),
        "replicates": replicates,
        "seed": seed,
    }


def _exact_sign_p(left: int, right: int) -> float | None:
    n = left + right
    if n == 0:
        return None
    tail = sum(math.comb(n, index) for index in range(min(left, right) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def _arm_summary(assessments: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(assessments)}
    for field in LIST_FIELDS:
        counts = [len(item[field]) for item in assessments]
        output[field] = {
            "cases_with_any": sum(value > 0 for value in counts),
            "case_rate": _fraction(sum(value > 0 for value in counts), len(counts)),
            "item_count": sum(counts),
            "mean_items_per_case": (
                statistics.fmean(counts) if counts else None
            ),
        }
    temporal = Counter(item["temporal_correctness"] for item in assessments)
    output["temporal_correctness"] = dict(sorted(temporal.items()))
    return output


def _group_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    choices = Counter(row["preference"] for row in rows)
    return {
        "n": len(rows),
        "preference": dict(sorted(choices.items())),
        "arm_metrics": {
            arm: _arm_summary([row[arm] for row in rows]) for arm in ARMS
        },
    }


def aggregate(
    *,
    private_root: Path,
    bootstrap_replicates: int = 100_000,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    private_root = private_root.resolve()
    public = private_root / "prepared_public"
    state_path = private_root / "control" / "reader_R01_task_b_state.json"
    secret_path = private_root / "secrets" / "release_signing.secret"
    private_key_path = (
        private_root / "prepared_private" / "private_answer_key.json"
    )
    locked_root = private_root / "control" / "locked_responses" / "R01"
    history_path = locked_root / "stage_02_response.json"
    reports_path = locked_root / "stage_03_response.json"

    secret = _read_secret(secret_path)
    _signed, unsigned, _receipt = _load_state(
        state_path, secret, public_packet_dir=public
    )
    if (
        unsigned["stages"]["2"]["status"] != "locked"
        or unsigned["stages"]["3"]["status"] != "locked"
    ):
        raise RuntimeError("Both Task-B stages must be locked before analysis")
    private_key = _load(private_key_path)
    history = _load(history_path)
    reports = _load(reports_path)
    if (
        private_key.get("packet_token") != history.get("packet_token")
        or private_key.get("packet_token") != reports.get("packet_token")
    ):
        raise RuntimeError("Task-B packet binding mismatch")
    history_cases = history.get("cases")
    report_cases = reports.get("cases")
    if not isinstance(history_cases, list) or not isinstance(report_cases, list):
        raise RuntimeError("Task-B cases are missing")
    history_order = [case.get("case_token") for case in history_cases]
    report_order = [case.get("case_token") for case in report_cases]
    if (
        history_order != report_order
        or sha256_canonical_json(history_order)
        != private_key.get("task_b_case_order_sha256")
    ):
        raise RuntimeError("Task-B case order commitment mismatch")
    key_by_token = {
        case["case_token"]: case
        for case in private_key.get("cases", [])
        if isinstance(case, dict)
        and isinstance(case.get("task_b"), dict)
        and not case["task_b"].get("excluded")
    }
    if set(key_by_token) != set(history_order):
        raise RuntimeError("Private Task-B decoding key is incomplete")

    preference = Counter()
    history_counts = Counter()
    assessments: dict[str, list[Mapping[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    paired_rows: list[dict[str, Any]] = []
    panels: dict[str, list[Mapping[str, Any]]] = {}
    for history_case, report_case in zip(history_cases, report_cases):
        token = history_case["case_token"]
        key = key_by_token[token]["task_b"]
        side_to_arm = {
            "A": key["report_a"]["arm"],
            "B": key["report_b"]["arm"],
        }
        if set(side_to_arm.values()) != set(ARMS):
            raise RuntimeError("Anonymous A/B mapping is not current-only vs SCFR")
        history_status = history_case["response"]["history_informativeness"]
        history_counts[history_status] += 1
        choice = report_case["response"]["overall_preference"]
        decoded_choice = side_to_arm.get(choice, choice)
        preference[decoded_choice] += 1
        arm_assessments: dict[str, Mapping[str, Any]] = {}
        for side, arm in side_to_arm.items():
            assessment = report_case["response"][
                f"report_{side.lower()}_assessment"
            ]
            for field in LIST_FIELDS:
                if not isinstance(assessment.get(field), list):
                    raise RuntimeError("Task-B assessment list schema changed")
            assessments[arm].append(assessment)
            arm_assessments[arm] = assessment
        paired_row = {
                "history_status": history_status,
                "panel": key_by_token[token]["panel"],
                "preference": decoded_choice,
                "current_only": arm_assessments["current_only"],
                "scfr": arm_assessments["scfr"],
            }
        paired_rows.append(paired_row)
        panels.setdefault(str(paired_row["panel"]), []).append(paired_row)

    n = len(paired_rows)
    scfr_wins = preference["scfr"]
    current_wins = preference["current_only"]
    decisive = scfr_wins + current_wins
    preference_scores = [
        1.0 if row["preference"] == "scfr" else
        -1.0 if row["preference"] == "current_only" else 0.0
        for row in paired_rows
        if row["preference"] != "unassessable"
    ]
    preference_bootstrap = (
        _paired_bootstrap(
            preference_scores,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        if preference_scores
        else None
    )
    arm_output = {arm: _arm_summary(assessments[arm]) for arm in ARMS}

    paired: dict[str, Any] = {}
    for field in LIST_FIELDS:
        current_any = [
            bool(row["current_only"][field]) for row in paired_rows
        ]
        scfr_any = [bool(row["scfr"][field]) for row in paired_rows]
        differences = [
            float(scfr) - float(current)
            for current, scfr in zip(current_any, scfr_any)
        ]
        current_only_bad = sum(
            current and not scfr
            for current, scfr in zip(current_any, scfr_any)
        )
        scfr_only_bad = sum(
            scfr and not current
            for current, scfr in zip(current_any, scfr_any)
        )
        bootstrap = _paired_bootstrap(
            differences,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        item_differences = [
            float(len(row["scfr"][field]) - len(row["current_only"][field]))
            for row in paired_rows
        ]
        paired[field] = {
            "delta_scfr_minus_current_only": bootstrap,
            "item_count_delta_scfr_minus_current_only": _paired_bootstrap(
                item_differences,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            "current_only_only_event": current_only_bad,
            "scfr_only_event": scfr_only_bad,
            "both_event": sum(
                current and scfr
                for current, scfr in zip(current_any, scfr_any)
            ),
            "neither_event": sum(
                not current and not scfr
                for current, scfr in zip(current_any, scfr_any)
            ),
            "exact_mcnemar_p_two_sided": _exact_sign_p(
                current_only_bad, scfr_only_bad
            ),
        }

    scfr_net = [
        len(row["scfr"]["supported_findings_recovered"])
        - len(row["scfr"]["supported_findings_lost"])
        for row in paired_rows
    ]
    paired["supported_recovered_minus_lost"] = {
        "definition": (
            "SCFR perspective only: per-case "
            "count(SCFR supported_findings_recovered) minus "
            "count(SCFR supported_findings_lost); no cross-arm "
            "double-difference"
        ),
        "scfr_mean": statistics.fmean(scfr_net),
        "scfr_bootstrap": _paired_bootstrap(
            [float(value) for value in scfr_net],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "raw_item_counts": {
            "current_only_recovered": arm_output["current_only"][
                "supported_findings_recovered"
            ]["item_count"],
            "current_only_lost": arm_output["current_only"][
                "supported_findings_lost"
            ]["item_count"],
            "scfr_recovered": arm_output["scfr"][
                "supported_findings_recovered"
            ]["item_count"],
            "scfr_lost": arm_output["scfr"][
                "supported_findings_lost"
            ]["item_count"],
        },
    }

    temporal_order = {"incorrect": 0, "partially_correct": 1, "correct": 2}
    temporal_pairs = [
        (
            temporal_order[row["current_only"]["temporal_correctness"]],
            temporal_order[row["scfr"]["temporal_correctness"]],
        )
        for row in paired_rows
        if row["scfr"]["temporal_correctness"] in temporal_order
        and row["current_only"]["temporal_correctness"] in temporal_order
    ]
    scfr_temporal_better = sum(scfr > current for current, scfr in temporal_pairs)
    current_temporal_better = sum(
        current > scfr for current, scfr in temporal_pairs
    )
    paired["temporal_correctness"] = {
        "descriptive_ordinal_order": [
            "incorrect",
            "partially_correct",
            "correct",
        ],
        "handling": {
            "not_applicable": "excluded",
            "unassessable": "excluded",
        },
        "paired_assessable_n": len(temporal_pairs),
        "scfr_better": scfr_temporal_better,
        "current_only_better": current_temporal_better,
        "tie": sum(current == scfr for current, scfr in temporal_pairs),
        "excluded": n - len(temporal_pairs),
        "exact_sign_p_two_sided": _exact_sign_p(
            scfr_temporal_better, current_temporal_better
        ),
    }

    stratified: dict[str, Any] = {}
    for status in sorted(history_counts):
        rows = [row for row in paired_rows if row["history_status"] == status]
        stratified[status] = _group_summary(rows)

    return {
        "analysis": "R01 Task-B valid-history blinded current-only vs SCFR",
        "reader_code": "R01",
        "n_cases": n,
        "history_informativeness": {
            "counts": dict(sorted(history_counts.items())),
            "clinically_informative_rate": _fraction(
                history_counts["clinically_informative"], n
            ),
        },
        "overall_preference": {
            "counts": dict(sorted(preference.items())),
            "scfr_win_rate_all_cases": _fraction(scfr_wins, n),
            "current_only_win_rate_all_cases": _fraction(current_wins, n),
            "scfr_decisive_win_rate": _fraction(scfr_wins, decisive),
            "decisive_n": decisive,
            "exact_sign_p_two_sided": _exact_sign_p(scfr_wins, current_wins),
            "net_scfr_preference_rate_all_cases": (
                (scfr_wins - current_wins) / n if n else None
            ),
            "net_preference_score": {
                "definition": (
                    "+1 SCFR, -1 current-only, 0 tie; unassessable excluded"
                ),
                "denominator": len(preference_scores),
                "paired_bootstrap": preference_bootstrap,
            },
        },
        "arm_metrics": arm_output,
        "paired_case_metrics": paired,
        "stratified_by_history_informativeness": stratified,
        "stratified_by_source_panel": {
            panel: _group_summary(rows)
            for panel, rows in sorted(panels.items())
        },
        "inference_scope": (
            "Descriptive paired expert adjudication of the frozen, balanced "
            "active-route (SCFR-different) 30-case Task-B cohort; not a "
            "population prevalence or population-average effect estimate."
        ),
        "provenance": {
            "private_answer_key_sha256": sha256_file(private_key_path),
            "locked_history_response_sha256": sha256_file(history_path),
            "locked_reports_response_sha256": sha256_file(reports_path),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
    }


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def render_markdown(result: Mapping[str, Any]) -> str:
    pref = result["overall_preference"]
    history = result["history_informativeness"]
    lines = [
        "# R01 Task B: blinded valid-history comparison",
        "",
        f"- Cases: {result['n_cases']}.",
        (
            "- Clinically informative history: "
            f"{history['counts'].get('clinically_informative', 0)}/"
            f"{result['n_cases']} "
            f"({_pct(history['clinically_informative_rate'])})."
        ),
        (
            "- Preference: SCFR "
            f"{pref['counts'].get('scfr', 0)}, current-only "
            f"{pref['counts'].get('current_only', 0)}, tie "
            f"{pref['counts'].get('tie', 0)}, unassessable "
            f"{pref['counts'].get('unassessable', 0)}."
        ),
        (
            "- SCFR decisive win rate: "
            f"{_pct(pref['scfr_decisive_win_rate'])} "
            f"(n={pref['decisive_n']}; exact two-sided "
            f"p={pref['exact_sign_p_two_sided']})."
        ),
        (
            "- Net preference score (+1 SCFR, -1 current-only, 0 tie; "
            "unassessable excluded): "
            + (
                f"{pref['net_preference_score']['paired_bootstrap']['estimate']:.3f} "
                f"[{pref['net_preference_score']['paired_bootstrap']['ci95_low']:.3f}, "
                f"{pref['net_preference_score']['paired_bootstrap']['ci95_high']:.3f}], "
                f"n={pref['net_preference_score']['denominator']}."
                if pref["net_preference_score"]["paired_bootstrap"] is not None
                else "NA."
            )
        ),
        "",
        "| Outcome | Current-only | SCFR | Δ SCFR-current (95% paired bootstrap CI) | Exact paired p |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "supported_findings_recovered": "Any supported finding recovered",
        "supported_findings_lost": "Any supported finding lost",
        "unsupported_additions": "Any unsupported addition",
        "important_omissions": "Any important omission",
    }
    for field, label in labels.items():
        current_arm = result["arm_metrics"]["current_only"]
        scfr_arm = result["arm_metrics"]["scfr"]
        current = current_arm[field]
        scfr = scfr_arm[field]
        paired = result["paired_case_metrics"][field]
        delta = paired["delta_scfr_minus_current_only"]
        lines.append(
            f"| {label} | {current['cases_with_any']}/{current_arm['n']}"
            f" ({_pct(current['case_rate'])}) | "
            f"{scfr['cases_with_any']}/{scfr_arm['n']} "
            f"({_pct(scfr['case_rate'])}) | "
            f"{_pct(delta['estimate'])} "
            f"[{_pct(delta['ci95_low'])}, {_pct(delta['ci95_high'])}] | "
            f"{paired['exact_mcnemar_p_two_sided']} |"
        )
    lines.extend(
        [
            "",
            "| Outcome items | Current-only total | SCFR total | Δ mean items/case (95% paired bootstrap CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    item_labels = {
        "supported_findings_recovered": "Supported findings recovered",
        "supported_findings_lost": "Supported findings lost",
        "unsupported_additions": "Unsupported additions",
        "important_omissions": "Important omissions",
    }
    for field, label in item_labels.items():
        current = result["arm_metrics"]["current_only"][field]
        scfr = result["arm_metrics"]["scfr"][field]
        delta = result["paired_case_metrics"][field][
            "item_count_delta_scfr_minus_current_only"
        ]
        lines.append(
            f"| {label} | {current['item_count']} | {scfr['item_count']} | "
            f"{delta['estimate']:.3f} "
            f"[{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}] |"
        )
    net = result["paired_case_metrics"]["supported_recovered_minus_lost"]
    net_bootstrap = net["scfr_bootstrap"]
    lines.extend(
        [
            (
                "| SCFR recovered-minus-lost supported-item balance "
                "(SCFR perspective only; no cross-arm double-difference) | "
                "— | "
                f"{net['scfr_mean']:.3f} mean | "
                f"{net_bootstrap['estimate']:.3f} "
                f"[{net_bootstrap['ci95_low']:.3f}, "
                f"{net_bootstrap['ci95_high']:.3f}] |"
            ),
            "",
            "| Temporal correctness | Current-only | SCFR |",
            "|---|---:|---:|",
        ]
    )
    temporal_values = (
        "correct",
        "partially_correct",
        "incorrect",
        "not_applicable",
        "unassessable",
    )
    for value in temporal_values:
        lines.append(
            f"| {value} | "
            f"{result['arm_metrics']['current_only']['temporal_correctness'].get(value, 0)} | "
            f"{result['arm_metrics']['scfr']['temporal_correctness'].get(value, 0)} |"
        )
    temporal_pair = result["paired_case_metrics"]["temporal_correctness"]
    lines.extend(
        [
            "",
            (
                "Descriptive paired temporal ordering "
                "(incorrect < partially_correct < correct; N/A and "
                "unassessable excluded): SCFR better "
                f"{temporal_pair['scfr_better']}, current-only better "
                f"{temporal_pair['current_only_better']}, tie "
                f"{temporal_pair['tie']}, paired n="
                f"{temporal_pair['paired_assessable_n']}, exact two-sided "
                f"p={temporal_pair['exact_sign_p_two_sided']}."
            ),
        ]
    )
    lines.extend(["", "## History-informativeness strata", ""])
    for status, stratum in result[
        "stratified_by_history_informativeness"
    ].items():
        current = stratum["arm_metrics"]["current_only"]
        scfr = stratum["arm_metrics"]["scfr"]
        lines.append(
            f"- {status} (n={stratum['n']}): preference "
            f"{stratum['preference']}; unsupported-addition cases "
            f"current-only {current['unsupported_additions']['cases_with_any']}, "
            f"SCFR {scfr['unsupported_additions']['cases_with_any']}; "
            f"recovered items current-only "
            f"{current['supported_findings_recovered']['item_count']}, "
            f"SCFR {scfr['supported_findings_recovered']['item_count']}."
        )
    lines.extend(
        [
            "",
            (
                "Scope: " + str(result["inference_scope"])
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.write_bytes(data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate(
            private_root=args.private_root,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        json_bytes = (
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        markdown_bytes = render_markdown(result).encode("utf-8")
        _write_new(args.output_json, json_bytes)
        try:
            _write_new(args.output_md, markdown_bytes)
        except BaseException:
            args.output_json.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "aggregated",
                "n_cases": result["n_cases"],
                "output_json_sha256": hashlib.sha256(json_bytes).hexdigest(),
                "output_md_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

