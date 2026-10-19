import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_source_causal_arms import tokenize
from scripts.source_causal_per_finding_gate import DEFAULT_ALL_ARMS, prepare_row
from scripts.source_causal_verifier import constraint_status, main, natural_corrected_report


def _toy_rows():
    rows = []
    for split in ("train", "validate", "test"):
        for idx in range(4):
            rows.append(
                {
                    "id": f"{split}_valid_prior_{idx}",
                    "split": split,
                    "reference": "Pleural effusion is present.",
                    "current_only": "No pleural effusion.",
                    "real_prior": "Pleural effusion is present.",
                    "prior_report_only": "Pleural effusion is present.",
                    "shuffled_prior": "No pleural effusion.",
                    "matched_shuffled_prior": "No pleural effusion.",
                }
            )
            rows.append(
                {
                    "id": f"{split}_wrong_history_{idx}",
                    "split": split,
                    "reference": "No pneumothorax.",
                    "current_only": "No pneumothorax.",
                    "real_prior": "Pneumothorax is present.",
                    "prior_report_only": "Pneumothorax is present.",
                    "shuffled_prior": "Pneumothorax is present.",
                    "matched_shuffled_prior": "Pneumothorax is present.",
                }
            )
    return rows


class SourceCausalVerifierTests(unittest.TestCase):
    def test_verifier_trains_and_writes_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "verifier.json"
            out_md = root / "verifier.md"
            out_predictions = root / "selected.json"
            predictions.write_text(json.dumps(_toy_rows()), encoding="utf-8")

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--output-predictions",
                    str(out_predictions),
                    "--min-samples",
                    "1",
                    "--constraint-baseline-arm",
                    "real_prior",
                    "--max-fp-delta",
                    "0",
                    "--min-finding-f1-delta",
                    "0",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected_rows = json.loads(out_predictions.read_text(encoding="utf-8"))
            markdown = out_md.read_text(encoding="utf-8")

        self.assertEqual(0, rc)
        self.assertIn("source_causal_verifier", report["metrics_by_split"]["test"])
        self.assertGreater(report["tuning"]["satisfying_trials"], 0)
        self.assertTrue(report["tuning"]["best"]["constraint"]["satisfied"])
        self.assertEqual(8, len([row for row in selected_rows if row["split"] == "test"]))
        self.assertIn("Constraint baseline: `real_prior`", markdown)

    def test_natural_filter_append_mode_preserves_report_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "verifier.json"
            out_predictions = root / "selected.json"
            predictions.write_text(json.dumps(_toy_rows()), encoding="utf-8")

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--output-predictions",
                    str(out_predictions),
                    "--min-samples",
                    "1",
                    "--prediction-mode",
                    "natural_filter_append",
                    "--natural-base-arm",
                    "current_only",
                    "--natural-append-arms",
                    "real_prior,current_only,prior_report_only",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected_rows = json.loads(out_predictions.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual("natural_filter_append", report["prediction_config"]["prediction_mode"])
        test_rows = [row for row in selected_rows if row["split"] == "test"]
        self.assertEqual(8, len(test_rows))
        self.assertTrue(any("Pleural effusion is present." in row["prediction"] for row in test_rows))
        self.assertTrue(all(row["prediction"].strip() for row in test_rows))
        self.assertTrue(
            all(
                row["source_causal_verifier"]["prediction_mode"] == "natural_filter_append"
                for row in test_rows
            )
        )

    def test_natural_filter_append_drops_mixed_unsupported_positive_sentence(self):
        row = {
            "id": "mixed",
            "split": "test",
            "reference": "Pleural effusion is present.",
            "current_only": (
                "Pleural effusion is present and pneumothorax is present. "
                "Heart size is unchanged."
            ),
            "real_prior": "Pleural effusion is present.",
            "prior_report_only": "Pleural effusion is present.",
            "shuffled_prior": "Pneumothorax is present.",
            "matched_shuffled_prior": "Pneumothorax is present.",
        }
        prepared = prepare_row(row, DEFAULT_ALL_ARMS)

        prediction = natural_corrected_report(
            prepared,
            ["pleural_effusion"],
            base_arm="current_only",
            append_arms=["current_only", "real_prior", "prior_report_only"],
            append_missing=True,
        )

        self.assertIn("Pleural effusion is present.", prediction)
        self.assertNotIn("pneumothorax is present", prediction)

    def test_length_capped_append_uses_compact_fallback(self):
        row = {
            "id": "compact",
            "split": "test",
            "reference": "Pleural effusion is present.",
            "current_only": "Pleural effusion is present with extensive basilar opacity.",
            "real_prior": (
                "Pleural effusion is present and pneumothorax is present. "
                "Heart size is unchanged."
            ),
            "prior_report_only": "Pleural effusion is present with extensive basilar opacity.",
            "shuffled_prior": "Pneumothorax is present.",
            "matched_shuffled_prior": "Pneumothorax is present.",
        }
        prepared = prepare_row(row, DEFAULT_ALL_ARMS)

        prediction = natural_corrected_report(
            prepared,
            ["pleural_effusion"],
            base_arm="real_prior",
            append_arms=["real_prior", "current_only", "prior_report_only"],
            append_missing=True,
            max_token_delta=0,
        )

        self.assertIn("Pleural effusion is present.", prediction)
        self.assertNotIn("pneumothorax is present", prediction)
        self.assertLessEqual(len(tokenize(prediction)), len(tokenize(row["real_prior"])))

    def test_compact_append_mode_prefers_standard_phrase(self):
        row = {
            "id": "compact_mode",
            "split": "test",
            "reference": "Pleural effusion is present.",
            "current_only": "No pneumothorax.",
            "real_prior": "Pleural effusion is present with extensive right basilar opacity.",
            "prior_report_only": "Pleural effusion is present with extensive right basilar opacity.",
            "shuffled_prior": "No pleural effusion.",
            "matched_shuffled_prior": "No pleural effusion.",
        }
        prepared = prepare_row(row, DEFAULT_ALL_ARMS)

        prediction = natural_corrected_report(
            prepared,
            ["pleural_effusion"],
            base_arm="current_only",
            append_arms=["real_prior", "current_only", "prior_report_only"],
            append_missing=True,
            prefer_compact_append=True,
        )

        self.assertIn("Pleural effusion is present.", prediction)
        self.assertNotIn("right basilar opacity", prediction)

    def test_failed_constraint_gate_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "verifier.json"
            predictions.write_text(json.dumps(_toy_rows()), encoding="utf-8")

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--min-samples",
                    "1",
                    "--constraint-baseline-arm",
                    "real_prior",
                    "--max-fp-delta",
                    "-100",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))

        self.assertEqual(2, rc)
        self.assertFalse(report["tuning"]["best"]["constraint"]["satisfied"])

    def test_analysis_only_features_require_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "verifier.json"
            predictions.write_text(json.dumps(_toy_rows()), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Analysis-only arms are not allowed"):
                main(
                    [
                        "--predictions",
                        str(predictions),
                        "--output-json",
                        str(out_json),
                        "--min-samples",
                        "1",
                        "--feature-arms",
                        "current_only,real_prior,matched_shuffled_prior",
                    ]
                )

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--min-samples",
                    "1",
                    "--feature-arms",
                    "current_only,real_prior,matched_shuffled_prior",
                    "--allow-analysis-only-features",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertTrue(report["prediction_config"]["allow_analysis_only_features"])

    def test_token_length_constraint_reports_reason(self):
        result = constraint_status(
            {
                "finding_f1": 0.5,
                "finding_status_accuracy": 0.8,
                "finding_fp_count": 1.0,
                "finding_fn_count": 1.0,
                "prediction_tokens": 12.0,
                "token_f1": 0.4,
                "objective": 0.3,
            },
            baseline_metrics={
                "finding_f1": 0.5,
                "finding_status_accuracy": 0.8,
                "finding_fp_count": 1.0,
                "finding_fn_count": 1.0,
                "prediction_tokens": 10.0,
                "token_f1": 0.4,
                "objective": 0.3,
            },
            max_fp_delta=None,
            max_fn_delta=None,
            max_prediction_tokens_delta=0.0,
            min_finding_f1_delta=None,
            min_status_accuracy_delta=None,
            min_token_f1_delta=0.0,
            min_objective_delta=None,
        )

        self.assertFalse(result["satisfied"])
        self.assertIn("prediction_tokens_delta_above_max", result["reasons"])
        self.assertEqual(2.0, result["deltas"]["prediction_tokens"])


if __name__ == "__main__":
    unittest.main()

