"""Release contracts use invented reports, never clinical records."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ReleaseTests(unittest.TestCase):
    def pipeline(self):
        self.assertTrue((ROOT / "scfr_release" / "pipeline.py").is_file(), "reference-free release pipeline is missing")
        from scfr_release import pipeline
        return pipeline

    def test_tied_reranker_returns_first_report_without_reference(self):
        p = self.pipeline()
        row = {"id": "invented-1", "current_only": "No pneumothorax.", "real_prior": "Pneumothorax.", "prior_report_only": "No effusion.", "prior_report": "No effusion."}
        model = {"candidate_arms": list(p.BASE_ARMS), "feature_names": p.ranker.feature_names(p.BASE_ARMS), "weights": [0.0] * 17}
        self.assertEqual("No pneumothorax.", p.rank_report(row, model)[1])

    def test_reranker_rejects_wrong_feature_contract(self):
        p = self.pipeline()
        with self.assertRaisesRegex(ValueError, "feature"):
            p.rank_report({"current_only": "No pneumothorax."}, {"candidate_arms": list(p.BASE_ARMS), "feature_names": ["bad"], "weights": [0.0]})

    def test_training_rejects_subject_overlap(self):
        p = self.pipeline()
        rows = p.synthetic_rows()
        rows[-1]["subject_id"] = rows[0]["subject_id"]
        with self.assertRaisesRegex(ValueError, "subject"):
            p.validate_training_rows(rows, min_samples=1)

    def test_inference_rejects_reference_and_wrong_history(self):
        p = self.pipeline()
        for field in ("reference", "shuffled_prior", "matched_shuffled_prior"):
            row = {"id": "invented", "current_only": "No effusion.", "real_prior": "No effusion.", "prior_report_only": "No effusion.", field: "forbidden"}
            with self.assertRaisesRegex(ValueError, "inference"):
                p.validate_inference_row(row)

    def test_inference_rejects_structured_context_and_sample_ids(self):
        p = self.pipeline()
        base = {"id": "invented", "current_only": "No effusion.", "real_prior": "No effusion.", "prior_report_only": "No effusion."}
        for field in ("prior_report", "context", "id"):
            with self.assertRaisesRegex(ValueError, "inference"):
                p.validate_inference_row({**base, field: {"reference": "forbidden nested report"}})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional verifier extra is not installed")
    def test_full_original_proposals_train_export_and_reference_free_inference(self):
        p = self.pipeline()
        rows = p.synthetic_rows()
        bundle = p.train_bundle(rows, min_samples=1, diagnostic=True)
        self.assertIn("verifiers", bundle, "each realization budget needs its independently selected verifier")
        self.assertEqual({"verifier_t0", "verifier_t2"}, set(bundle["verifiers"]))
        self.assertEqual(0, bundle["verifiers"]["verifier_t0"]["realization_budget"])
        self.assertEqual(2, bundle["verifiers"]["verifier_t2"]["realization_budget"])
        self.assertIsNot(bundle["verifiers"]["verifier_t0"], bundle["verifiers"]["verifier_t2"])
        restored = json.loads(json.dumps(bundle, allow_nan=False))
        heldout = [row for row in rows if row["split"] == "test"]
        inputs = [{k: v for k, v in row.items() if k in ("id", "prior_report", *p.BASE_ARMS)} for row in heldout]
        predictions = p.infer(inputs, restored)
        self.assertEqual(len(heldout), len(predictions))
        self.assertTrue(all(row["selected_arm"] in p.FINAL_ARMS for row in predictions))
        self.assertTrue(all("reference" not in row for row in predictions))
        self.assertEqual(predictions, p.infer(inputs, restored))
        altered = copy.deepcopy(rows)
        for row in altered:
            if row["split"] == "test":
                row["reference"] = "Pneumonia and edema."
        self.assertEqual(bundle, p.train_bundle(altered, min_samples=1, diagnostic=True))

    def test_cli_help_is_offline(self):
        result = subprocess.run([sys.executable, "-B", "-m", "scfr_release", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("infer", result.stdout)

    def test_cli_evaluation_requires_separate_unique_aligned_references(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "predictions.json").write_text(json.dumps([{"id": "invented", "prediction": "No pneumothorax."}]), encoding="utf-8")
            refs = root / "references.json"
            refs.write_text(json.dumps([{"id": "invented", "reference": "No pneumothorax."}]), encoding="utf-8")
            command = [sys.executable, "-B", "-m", "scfr_release", "evaluate", "--predictions", str(root / "predictions.json"), "--references", str(refs), "--output", str(root / "aggregate.json")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(1, aggregate["rows"])
            self.assertEqual(1.0, aggregate["metrics"]["finding_f1"])
            refs.write_text(json.dumps([{"id": "other", "reference": "No pneumothorax."}]), encoding="utf-8")
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional verifier extra is not installed")
    def test_original_empty_realization_uses_canonical_fallback_outside_append_budget(self):
        from scripts.source_causal_verifier import natural_corrected_report
        # The append budget is not an absolute total-length cap in the original code.
        result = natural_corrected_report({"row": {"current_only": "Edema."}}, [], base_arm="current_only", append_arms=(), append_missing=True, max_token_delta=0, prefer_compact_append=True)
        self.assertEqual("No source-valid positive findings.", result)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional verifier extra is not installed")
    def test_numeric_json_verifier_matches_original_sklearn_predictions(self):
        p = self.pipeline()
        from scripts import source_causal_verifier as v
        from scripts.source_causal_per_finding_gate import prepare_row
        from sklearn.linear_model import LogisticRegression
        training = [prepare_row(row, p.BASE_ARMS) for row in p.synthetic_rows() if row["split"] == "train"]
        x, y, _ = v.make_xy(training, feature_arms=p.BASE_ARMS, wrong_arms=(), use_finding_identity=True)
        model = LogisticRegression(C=.1, solver="liblinear", random_state=0).fit(x, y)
        frozen = {"feature_names": v.feature_names(p.BASE_ARMS, use_finding_identity=True), "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "threshold": .5, "thresholds_by_finding": {}}
        row = {k: val for k, val in p.synthetic_rows()[-1].items() if k in p.INFERENCE_FIELDS}
        prepared = prepare_row(row, p.BASE_ARMS)
        features = [v.row_features(prepared, finding, feature_arms=p.BASE_ARMS, wrong_arms=(), use_finding_identity=True) for finding in v.FINDING_PATTERNS]
        findings = [finding for finding, probability in zip(v.FINDING_PATTERNS, model.predict_proba(features)[:, 1]) if probability >= .5]
        expected = tuple(v.prediction_for_findings(prepared, findings, prediction_mode="natural_filter_compact_append", natural_base_arm="current_only", natural_append_arms=("real_prior", "current_only", "prior_report_only"), natural_max_token_delta=delta) for delta in (0, 2))
        independent_models = {f"verifier_t{delta}": {**frozen, "realization_budget": delta} for delta in (0, 2)}
        self.assertEqual(expected, p._verifier_reports(row, independent_models))


if __name__ == "__main__":
    unittest.main()
