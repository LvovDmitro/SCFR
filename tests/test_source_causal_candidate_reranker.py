import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_candidate_reranker import main, prepare_rows


class SourceCausalCandidateRerankerTests(unittest.TestCase):
    def test_reranker_learns_heldout_candidate_choice(self):
        rows = [
            {
                "id": "train_1",
                "split": "train",
                "reference": "Small pleural effusion is present.",
                "prior_report": "Prior report notes a small pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "Small pleural effusion is present.",
                "prior_report_only": "Small pleural effusion is present.",
            },
            {
                "id": "train_2",
                "split": "train",
                "reference": "Pneumothorax is present.",
                "prior_report": "Prior report mentions pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "Pneumothorax is present.",
                "prior_report_only": "Pneumothorax is present.",
            },
            {
                "id": "test_1",
                "split": "test",
                "reference": "Small pleural effusion is present.",
                "prior_report": "Prior report notes a small pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "Small pleural effusion is present.",
                "prior_report_only": "Small pleural effusion is present.",
            },
            {
                "id": "test_2",
                "split": "test",
                "reference": "Pneumothorax is present.",
                "prior_report": "Prior report mentions pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "Pneumothorax is present.",
                "prior_report_only": "Pneumothorax is present.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "reranker.json"
            out_md = root / "reranker.md"
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
                    "--eval-splits",
                    "test",
                    "--min-samples",
                    "1",
                    "--epochs",
                    "20",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected = json.loads(out_predictions.read_text(encoding="utf-8"))
            markdown_exists = out_md.exists()

        self.assertEqual(0, rc)
        self.assertEqual(2, report["metrics_by_split"]["test"]["candidate_reranker"]["rows"])
        self.assertGreaterEqual(
            report["metrics_by_split"]["test"]["candidate_reranker"]["metrics"]["objective"],
            report["metrics_by_split"]["test"]["current_only"]["metrics"]["objective"],
        )
        self.assertEqual(2, len(selected))
        self.assertTrue(markdown_exists)

    def test_reranker_rejects_leaky_eval_without_flag(self):
        rows = [
            {
                "id": "train_1",
                "split": "train",
                "reference": "No pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "No pneumothorax.",
                "prior_report_only": "No pneumothorax.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "reranker.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "leaky eval split requested"):
                main(
                    [
                        "--predictions",
                        str(predictions),
                        "--output-json",
                        str(out_json),
                        "--eval-splits",
                        "all",
                        "--min-samples",
                        "1",
                    ]
                )

    def test_reranker_rejects_underpowered_eval_without_flag(self):
        rows = [
            {
                "id": "train_1",
                "split": "train",
                "reference": "No pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "No pneumothorax.",
                "prior_report_only": "No pneumothorax.",
            },
            {
                "id": "train_2",
                "split": "train",
                "reference": "No pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "No pleural effusion.",
                "prior_report_only": "No pleural effusion.",
            },
            {
                "id": "test_1",
                "split": "test",
                "reference": "No pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "No pneumothorax.",
                "prior_report_only": "No pneumothorax.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "reranker.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "eval split rows below min-samples gate"):
                main(
                    [
                        "--predictions",
                        str(predictions),
                        "--output-json",
                        str(out_json),
                        "--tune-split",
                        "train",
                        "--eval-splits",
                        "test",
                        "--min-samples",
                        "2",
                    ]
                )

    def test_reranker_allows_combined_eval_split_gate(self):
        rows = [
            {
                "id": "train_1",
                "split": "train",
                "reference": "No pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "No pneumothorax.",
                "prior_report_only": "No pneumothorax.",
            },
            {
                "id": "train_2",
                "split": "train",
                "reference": "No pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "No pleural effusion.",
                "prior_report_only": "No pleural effusion.",
            },
            {
                "id": "val_1",
                "split": "validate",
                "reference": "No pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "No pneumothorax.",
                "prior_report_only": "No pneumothorax.",
            },
            {
                "id": "test_1",
                "split": "test",
                "reference": "No pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "No pleural effusion.",
                "prior_report_only": "No pleural effusion.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "reranker.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")

            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--tune-split",
                    "train",
                    "--eval-splits",
                    "validate+test",
                    "--min-samples",
                    "2",
                    "--epochs",
                    "2",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual(2, report["metrics_by_split"]["validate+test"]["candidate_reranker"]["rows"])

    def test_reranker_scores_normalized_reference_text(self):
        rows = [
            {
                "id": "row_1",
                "split": "train",
                "reference": "There is no\npneumothorax.",
                "current_only": "There is no pneumothorax.",
                "real_prior": "There is no pneumothorax.",
                "prior_report_only": "There is no pneumothorax.",
            }
        ]

        prepared = prepare_rows(
            rows,
            candidate_arms=["current_only", "real_prior", "prior_report_only"],
            split_field="split",
            fp_weight=0.1,
            fn_weight=0.0,
        )
        current = next(candidate for candidate in prepared[0].candidates if candidate.arm == "current_only")

        self.assertEqual(0.0, current.true_score["finding_fn_count"])
        self.assertEqual(1.0, current.true_score["finding_status_accuracy"])


if __name__ == "__main__":
    unittest.main()

