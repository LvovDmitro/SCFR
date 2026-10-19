import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_per_finding_gate import main


def _toy_rows():
    rows = []
    for split in ("train", "validate", "test"):
        rows.append(
            {
                "id": f"{split}_valid_prior",
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
                "id": f"{split}_wrong_history",
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


class SourceCausalPerFindingGateTests(unittest.TestCase):
    def test_gate_trains_table_and_writes_predictions(self):
        rows = _toy_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "gate.json"
            out_md = root / "gate.md"
            out_predictions = root / "selected.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")
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
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected_rows = json.loads(out_predictions.read_text(encoding="utf-8"))
            markdown_exists = out_md.exists()

        self.assertEqual(0, rc)
        self.assertIn("per_finding_gate", report["metrics_by_split"]["test"])
        self.assertEqual(2, len([row for row in selected_rows if row["split"] == "test"]))
        self.assertTrue(markdown_exists)

    def test_constraint_tuning_prefers_satisfying_trial(self):
        rows = _toy_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "gate.json"
            out_md = root / "gate.md"
            predictions.write_text(json.dumps(rows), encoding="utf-8")
            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--min-samples",
                    "1",
                    "--feature-arms",
                    "current_only,real_prior,prior_report_only",
                    "--constraint-baseline-arm",
                    "real_prior",
                    "--max-fp-delta",
                    "0",
                    "--min-finding-f1-delta",
                    "0",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            markdown = out_md.read_text(encoding="utf-8")

        self.assertEqual(0, rc)
        self.assertEqual("real_prior", report["constraint_baseline_arm"])
        self.assertGreater(report["tuning"]["satisfying_trials"], 0)
        self.assertTrue(report["tuning"]["best"]["constraint"]["satisfied"])
        self.assertIn("Constraint baseline: `real_prior`", markdown)


if __name__ == "__main__":
    unittest.main()

