import json
import tempfile
import unittest
from pathlib import Path

from scripts.source_causal_pairwise_reranker import build_report, load_pairs, main


def _write_pairs(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _toy_rows():
    rows = []
    for split in ("train", "train", "validate", "validate"):
        rows.append(
            {
                "sample_id": f"{split}_{len(rows)}",
                "split": split,
                "context": "No edema. No pneumothorax.",
                "chosen": "No edema. No pneumothorax.",
                "rejected": "Edema and pneumothorax are present.",
                "chosen_source": "current_only",
                "rejected_source": "shuffled_prior",
                "reward_margin": 1.0,
            }
        )
    return rows


class SourceCausalPairwiseRerankerTests(unittest.TestCase):
    def test_no_source_model_learns_validation_pair_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs_path = root / "pairs.jsonl"
            candidates_path = root / "candidates.json"
            out_json = root / "report.json"
            out_md = root / "report.md"
            out_predictions = root / "selected.json"
            _write_pairs(pairs_path, _toy_rows())
            candidates_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "val_1",
                            "split": "validate",
                            "context": "No edema. No pneumothorax.",
                            "reference": "No edema. No pneumothorax.",
                            "current_only": "No edema. No pneumothorax.",
                            "shuffled_prior": "Edema and pneumothorax are present.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rc = main(
                [
                    "--pairs",
                    str(pairs_path),
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                    "--output-predictions",
                    str(out_predictions),
                    "--candidate-predictions",
                    str(candidates_path),
                    "--candidate-arms",
                    "current_only,shuffled_prior",
                    "--min-train-pairs",
                    "1",
                    "--min-eval-pairs",
                    "1",
                    "--min-candidate-rows",
                    "1",
                    "--min-validate-accuracy",
                    "0.5",
                    "--sample-weight-mode",
                    "none",
                ]
            )
            report = json.loads(out_json.read_text(encoding="utf-8"))
            selected = json.loads(out_predictions.read_text(encoding="utf-8"))
            markdown_exists = out_md.exists()

        self.assertEqual(0, rc)
        self.assertFalse(report["include_source_identity"])
        self.assertEqual("GO", report["gate"]["verdict"])
        self.assertGreaterEqual(report["metrics_by_split"]["validate"]["accuracy"], 0.5)
        self.assertIn("candidate_prediction_eval", report)
        self.assertEqual("current_only", selected[0]["selected_arm"])
        self.assertTrue(markdown_exists)

    def test_test_pairs_are_blocked_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = Path(tmp) / "pairs.jsonl"
            rows = _toy_rows()
            rows.append({**rows[0], "sample_id": "test_1", "split": "test"})
            _write_pairs(pairs_path, rows)
            pairs = load_pairs(pairs_path)

            with self.assertRaisesRegex(RuntimeError, "test pairs are forbidden"):
                build_report(
                    pairs,
                    train_split="train",
                    eval_splits=["validate"],
                    include_source_identity=False,
                    min_train_pairs=1,
                    min_eval_pairs=1,
                    min_validate_accuracy=0.5,
                    c_value=1.0,
                    max_iter=100,
                    sample_weight_mode="none",
                    max_sample_weight=3.0,
                    allow_test_pairs=False,
                )


if __name__ == "__main__":
    unittest.main()

