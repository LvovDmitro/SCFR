import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_ledger_composer import compose_report, main


class SourceCausalLedgerComposerTests(unittest.TestCase):
    def test_compose_filters_non_ledger_positive_sentence_and_appends_missing(self):
        row = {
            "real_prior": "Pneumothorax is present. Pleural effusion is present. No edema.",
            "current_only": "Pleural effusion is present.",
            "prior_report_only": "",
        }

        result = compose_report(
            row,
            selected_findings=["pleural_effusion"],
            base_arm="real_prior",
            append_missing=True,
            keep_neutral_sentences=True,
            max_appended_findings=2,
        )

        self.assertNotIn("Pneumothorax is present", result["prediction"])
        self.assertIn("Pleural effusion is present", result["prediction"])
        self.assertIn("No edema", result["prediction"])

    def test_cli_runs_validate_smoke_without_test_pairs(self):
        rows = []
        for idx in range(4):
            split = "train" if idx < 2 else "validate"
            rows.append(
                {
                    "id": f"row_{idx}",
                    "split": split,
                    "reference": "Pleural effusion is present.",
                    "current_only": "Pleural effusion is present.",
                    "real_prior": "Pleural effusion is present. Pneumothorax is present.",
                    "prior_report_only": "Pneumothorax is present.",
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.json"
            output_json = root / "report.json"
            output_md = root / "report.md"
            predictions.write_text(json.dumps(rows), encoding="utf-8")

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--min-samples",
                    "1",
                    "--threshold-grid",
                    "0.5",
                    "--min-support-grid",
                    "1",
                    "--fallbacks",
                    "current",
                    "--base-arms",
                    "real_prior",
                    "--append-missing-options",
                    "true",
                    "--keep-neutral-options",
                    "true",
                    "--max-appended-findings-grid",
                    "1",
                ]
            )
            report = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual(["validate"], report["eval_splits"])
        self.assertIn("ledger_composer", report["metrics_by_split"]["validate"])


if __name__ == "__main__":
    unittest.main()

