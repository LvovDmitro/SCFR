import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_counterfactual_arm_predictions import main


class MergeCounterfactualArmPredictionsTests(unittest.TestCase):
    def test_merge_outputs_one_row_per_sample_with_all_arm_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_a = root / "current_only.json"
            manifest_b = root / "real_prior.json"
            result_a = root / "result_a.json"
            result_b = root / "result_b.json"
            output = root / "merged.json"
            row = {"id": "s1", "reference": "ref", "prior_report": "prior"}
            manifest_a.write_text(json.dumps([row]), encoding="utf-8")
            manifest_b.write_text(json.dumps([dict(row, context="prior")]), encoding="utf-8")
            result_a.write_text(json.dumps([{"id": "s1", "prediction": "a"}]), encoding="utf-8")
            result_b.write_text(json.dumps([{"id": "s1", "prediction": "b"}]), encoding="utf-8")

            rc = main(
                [
                    "--arm-manifest",
                    f"current_only={manifest_a}",
                    "--arm-manifest",
                    f"real_prior={manifest_b}",
                    "--arm-results",
                    f"current_only={result_a}",
                    "--arm-results",
                    f"real_prior={result_b}",
                    "--output",
                    str(output),
                    "--require-complete",
                ]
            )
            rows = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual(1, len(rows))
        self.assertEqual("a", rows[0]["current_only"])
        self.assertEqual("b", rows[0]["real_prior"])
        self.assertEqual("prior", rows[0]["context"])

    def test_require_complete_rejects_manifest_sample_set_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_a = root / "current_only.json"
            manifest_b = root / "real_prior.json"
            result_a = root / "result_a.json"
            result_b = root / "result_b.json"
            output = root / "merged.json"
            manifest_a.write_text(
                json.dumps([{"id": "s1", "reference": "ref"}]),
                encoding="utf-8",
            )
            manifest_b.write_text(
                json.dumps([{"id": "s2", "reference": "ref"}]),
                encoding="utf-8",
            )
            result_a.write_text(json.dumps([{"id": "s1", "prediction": "a"}]), encoding="utf-8")
            result_b.write_text(json.dumps([{"id": "s2", "prediction": "b"}]), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "manifest sample-set mismatch"):
                main(
                    [
                        "--arm-manifest",
                        f"current_only={manifest_a}",
                        "--arm-manifest",
                        f"real_prior={manifest_b}",
                        "--arm-results",
                        f"current_only={result_a}",
                        "--arm-results",
                        f"real_prior={result_b}",
                        "--output",
                        str(output),
                        "--require-complete",
                    ]
                )

    def test_require_complete_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_a = root / "current_only.json"
            manifest_b = root / "real_prior.json"
            result_a = root / "result_a.json"
            result_b = root / "result_b.json"
            output = root / "merged.json"
            manifest_a.write_text(
                json.dumps([
                    {"id": "s1", "reference": "ref"},
                    {"id": "s1", "reference": "ref"},
                ]),
                encoding="utf-8",
            )
            manifest_b.write_text(
                json.dumps([{"id": "s1", "reference": "ref"}]),
                encoding="utf-8",
            )
            result_a.write_text(json.dumps([{"id": "s1", "prediction": "a"}]), encoding="utf-8")
            result_b.write_text(json.dumps([{"id": "s1", "prediction": "b"}]), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "duplicate sample ids"):
                main(
                    [
                        "--arm-manifest",
                        f"current_only={manifest_a}",
                        "--arm-manifest",
                        f"real_prior={manifest_b}",
                        "--arm-results",
                        f"current_only={result_a}",
                        "--arm-results",
                        f"real_prior={result_b}",
                        "--output",
                        str(output),
                        "--require-complete",
                    ]
                )

    def test_only_complete_results_filters_subset_without_requiring_full_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_a = root / "current_only.json"
            manifest_b = root / "real_prior.json"
            result_a = root / "result_a.json"
            result_b = root / "result_b.json"
            output = root / "merged.json"
            summary = root / "summary.json"
            rows = [
                {"id": "s1", "subject_id": "p1", "reference": "ref1"},
                {"id": "s2", "subject_id": "p2", "reference": "ref2"},
            ]
            manifest_a.write_text(json.dumps(rows), encoding="utf-8")
            manifest_b.write_text(json.dumps(rows), encoding="utf-8")
            result_a.write_text(json.dumps([{"id": "s1", "prediction": "a"}]), encoding="utf-8")
            result_b.write_text(json.dumps([{"id": "s1", "prediction": "b"}]), encoding="utf-8")

            rc = main(
                [
                    "--arm-manifest",
                    f"current_only={manifest_a}",
                    "--arm-manifest",
                    f"real_prior={manifest_b}",
                    "--arm-results",
                    f"current_only={result_a}",
                    "--arm-results",
                    f"real_prior={result_b}",
                    "--output",
                    str(output),
                    "--summary-output",
                    str(summary),
                    "--only-complete-results",
                ]
            )
            merged_rows = json.loads(output.read_text(encoding="utf-8"))
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual(["s1"], [row["sample_id"] for row in merged_rows])
        self.assertEqual("p1", merged_rows[0]["subject_id"])
        self.assertEqual(1, summary_payload["rows"])
        self.assertEqual(1, summary_payload["complete_prediction_rows"])
        self.assertEqual("complete_results", summary_payload["output_row_filter"])


if __name__ == "__main__":
    unittest.main()

