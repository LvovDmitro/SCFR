import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_meta_controller import main


class SourceCausalMetaControllerTests(unittest.TestCase):
    def test_meta_controller_materializes_candidates_and_writes_predictions(self):
        base_rows = [
            {
                "id": "train_1",
                "split": "train",
                "reference": "Pleural effusion is present.",
                "prior_report": "Prior report mentions pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "Pleural effusion is present.",
                "prior_report_only": "Pleural effusion is present.",
            },
            {
                "id": "train_2",
                "split": "train",
                "reference": "No pneumothorax.",
                "prior_report": "Prior report mentions pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "Pneumothorax is present.",
                "prior_report_only": "Pneumothorax is present.",
            },
            {
                "id": "test_1",
                "split": "test",
                "reference": "Pleural effusion is present.",
                "prior_report": "Prior report mentions pleural effusion.",
                "current_only": "No pleural effusion.",
                "real_prior": "Pleural effusion is present.",
                "prior_report_only": "Pleural effusion is present.",
            },
            {
                "id": "test_2",
                "split": "test",
                "reference": "No pneumothorax.",
                "prior_report": "Prior report mentions pneumothorax.",
                "current_only": "No pneumothorax.",
                "real_prior": "Pneumothorax is present.",
                "prior_report_only": "Pneumothorax is present.",
            },
        ]
        candidate_rows = [
            {**row, "prediction": row["real_prior"] if row["id"].endswith("_1") else row["current_only"]}
            for row in base_rows
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.json"
            cand_path = root / "cand.json"
            out_json = root / "meta.json"
            out_md = root / "meta.md"
            out_preds = root / "predictions.json"
            out_candidates = root / "candidates.json"
            base_path.write_text(json.dumps(base_rows), encoding="utf-8")
            cand_path.write_text(json.dumps(candidate_rows), encoding="utf-8")

            rc = main(
                [
                    "--base-predictions",
                    str(base_path),
                    "--candidate",
                    f"cand={cand_path}",
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--output-predictions",
                    str(out_preds),
                    "--output-candidates",
                    str(out_candidates),
                    "--eval-splits",
                    "test",
                    "--min-samples",
                    "1",
                    "--epochs",
                    "10",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected = json.loads(out_preds.read_text(encoding="utf-8"))
            materialized = json.loads(out_candidates.read_text(encoding="utf-8"))
            markdown_exists = out_md.exists()

        self.assertEqual(0, rc)
        self.assertEqual("source_causal_meta_controller", report["controller"])
        self.assertEqual(2, report["metrics_by_split"]["test"]["candidate_reranker"]["rows"])
        self.assertEqual(2, len(selected))
        self.assertEqual("Pleural effusion is present.", materialized[0]["cand"])
        self.assertTrue(markdown_exists)


if __name__ == "__main__":
    unittest.main()

