import json
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_counterfactual_arms import main


class MaterializeCounterfactualArmsTests(unittest.TestCase):
    def _write_source_manifest(self, path: Path) -> None:
        rows = [
            {
                "id": "s1",
                "subject_id": "1001",
                "study_id": "5001",
                "image_paths": ["cur1.jpg"],
                "reference": "No acute findings.",
                "prior_study_id": "4901",
                "prior_image_paths": ["prior1.jpg"],
                "prior_report": "Stable left basilar opacity.",
                "context": "Stable left basilar opacity.",
                "split": "train",
                "prior_study_datetime": "2020-01-01T00:00:00",
                "temporal_gap_days": 10,
            },
            {
                "id": "s2",
                "subject_id": "1002",
                "study_id": "5002",
                "image_paths": ["cur2.jpg"],
                "reference": "Mild edema improved.",
                "prior_study_id": "4902",
                "prior_image_paths": ["prior2.jpg"],
                "prior_report": "Pulmonary edema is worse.",
                "context": "Pulmonary edema is worse.",
                "split": "train",
                "prior_study_datetime": "2020-01-02T00:00:00",
                "temporal_gap_days": 10,
            },
            {
                "id": "s3",
                "subject_id": "1003",
                "study_id": "5003",
                "image_paths": ["cur3.jpg"],
                "reference": "Pleural effusion unchanged.",
                "prior_study_id": "4903",
                "prior_image_paths": ["prior3.jpg"],
                "prior_report": "Small right pleural effusion.",
                "context": "Small right pleural effusion.",
                "split": "validate",
                "prior_study_datetime": "2020-01-03T00:00:00",
                "temporal_gap_days": 10,
            },
            {
                "id": "s4",
                "subject_id": "1004",
                "study_id": "5004",
                "image_paths": ["cur4.jpg"],
                "reference": "No pleural effusion.",
                "prior_study_id": "4904",
                "prior_image_paths": ["prior4.jpg"],
                "prior_report": "No focal opacity.",
                "context": "No focal opacity.",
                "split": "validate",
                "prior_study_datetime": "2020-01-04T00:00:00",
                "temporal_gap_days": 10,
            },
        ]
        path.write_text(json.dumps(rows), encoding="utf-8")

    def test_materializes_all_counterfactual_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            out_dir = root / "out"
            self._write_source_manifest(source)

            rc = main(
                [
                    "--source-manifest",
                    str(source),
                    "--output-dir",
                    str(out_dir),
                    "--seed",
                    "7",
                    "--allow-reference-conditioned-matching",
                ]
            )

            self.assertEqual(0, rc)
            current_only = json.loads((out_dir / "source_current_only.json").read_text(encoding="utf-8"))
            real_prior = json.loads((out_dir / "source_real_prior.json").read_text(encoding="utf-8"))
            shuffled = json.loads((out_dir / "source_shuffled_prior.json").read_text(encoding="utf-8"))
            matched_shuffled = json.loads(
                (out_dir / "source_matched_shuffled_prior.json").read_text(encoding="utf-8")
            )
            prior_report_only = json.loads(
                (out_dir / "source_prior_report_only.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (out_dir / "source_counterfactual_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(4, len(current_only))
        self.assertEqual(4, len(real_prior))
        self.assertEqual(4, len(shuffled))
        self.assertEqual(4, len(matched_shuffled))
        self.assertEqual(4, len(prior_report_only))

        self.assertEqual("current_only", current_only[0]["counterfactual_arm"])
        self.assertEqual([], current_only[0]["prior_image_paths"])
        self.assertEqual("", current_only[0]["prior_report"])
        self.assertEqual("", current_only[0]["context"])
        self.assertIsNone(current_only[0]["temporal_gap_days"])

        self.assertEqual("prior_report_only", prior_report_only[0]["counterfactual_arm"])
        self.assertEqual([], prior_report_only[0]["prior_image_paths"])
        self.assertTrue(prior_report_only[0]["prior_report"])
        self.assertEqual(prior_report_only[0]["prior_report"], prior_report_only[0]["context"])

        for row in shuffled:
            info = row["counterfactual_source"]
            self.assertNotEqual(info["target_subject_id"], info["shuffled_from_subject_id"])
            self.assertEqual(info["target_split"], info["shuffled_from_split"])
            self.assertEqual("shuffled_prior", row["counterfactual_arm"])
            self.assertTrue(row["prior_report"])
            self.assertEqual(info["shuffled_from_prior_study_datetime"], row["prior_study_datetime"])
            self.assertIsNone(row["temporal_gap_days"])

        for row in matched_shuffled:
            info = row["counterfactual_source"]
            self.assertNotEqual(info["target_subject_id"], info["matched_from_subject_id"])
            self.assertEqual(info["target_split"], info["matched_from_split"])
            self.assertEqual("matched_shuffled_prior", row["counterfactual_arm"])
            self.assertEqual(
                "same_split_reference_finding_signature_prior_length",
                info["match_strategy"],
            )
            self.assertIn("reference_finding_symmetric_difference", info)
            self.assertTrue(row["prior_report"])
            self.assertIsNone(row["temporal_gap_days"])

        matched_stats = summary["arm_stats"]["matched_shuffled_prior"]
        self.assertEqual(4, matched_stats["subject_mismatch_rows"])
        self.assertEqual(4, matched_stats["split_match_rows"])
        self.assertIn("mean_reference_finding_symmetric_difference", matched_stats)
        self.assertTrue(summary["reference_conditioned_matching"])
        self.assertEqual(["matched_shuffled_prior"], summary["analysis_only_arms"])

    def test_matched_shuffled_requires_explicit_reference_conditioned_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            out_dir = root / "out"
            self._write_source_manifest(source)

            with self.assertRaisesRegex(SystemExit, "reference-conditioned"):
                main(
                    [
                        "--source-manifest",
                        str(source),
                        "--output-dir",
                        str(out_dir),
                        "--arms",
                        "matched_shuffled_prior",
                    ]
                )

    def test_shuffled_controls_require_subject_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            out_dir = root / "out"
            rows = [
                {
                    "id": "s1",
                    "study_id": "5001",
                    "image_paths": ["cur1.jpg"],
                    "reference": "No acute findings.",
                    "prior_study_id": "4901",
                    "prior_image_paths": ["prior1.jpg"],
                    "prior_report": "Stable left basilar opacity.",
                    "split": "train",
                },
                {
                    "id": "s2",
                    "study_id": "5002",
                    "image_paths": ["cur2.jpg"],
                    "reference": "Mild edema improved.",
                    "prior_study_id": "4902",
                    "prior_image_paths": ["prior2.jpg"],
                    "prior_report": "Pulmonary edema is worse.",
                    "split": "train",
                },
            ]
            source.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing subject_id"):
                main(
                    [
                        "--source-manifest",
                        str(source),
                        "--output-dir",
                        str(out_dir),
                        "--arms",
                        "shuffled_prior",
                    ]
                )


if __name__ == "__main__":
    unittest.main()

