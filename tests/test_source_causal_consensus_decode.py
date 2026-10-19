import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_consensus_decode import main


class SourceCausalConsensusDecodeTests(unittest.TestCase):
    def test_consensus_decode_vetoes_wrong_history_addition(self):
        rows = [
            {
                "id": "row1",
                "split": "test",
                "reference": "Pleural effusion is present.",
                "current_only": "No pneumothorax. Pleural effusion is present.",
                "real_prior": "Pleural effusion is present. Pneumothorax is present.",
                "shuffled_prior": "Pneumothorax is present.",
                "matched_shuffled_prior": "Pneumothorax is present.",
                "prior_report_only": "Pneumothorax is present.",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "merged.json"
            out_json = root / "consensus.json"
            out_predictions = root / "decoded.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")
            rc = main(
                [
                    "--predictions",
                    str(predictions),
                    "--output-json",
                    str(out_json),
                    "--output-predictions",
                    str(out_predictions),
                    "--eval-splits",
                    "test",
                    "--min-samples",
                    "1",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            decoded = json.loads(out_predictions.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertIn("wrong_history_veto", report["metrics_by_split"]["test"])
        veto_rows = [row for row in decoded if row["consensus_policy"] == "wrong_history_veto"]
        self.assertEqual(["pleural_effusion"], veto_rows[0]["consensus_findings"])


if __name__ == "__main__":
    unittest.main()

