"""Archived router inference needs candidate text, never target references."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scfr_release import pipeline


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "scfr_controller_vectors_full_precision.json"


class ArchivedRouterTests(unittest.TestCase):
    def setUp(self):
        self.archive = json.loads(WEIGHTS.read_text(encoding="utf-8"))
        self.row = {
            "id": "invented-1", "current_only": "No pleural effusion.",
            "real_prior": "Pleural effusion.", "prior_report_only": "Pleural effusion.",
            "prior_report": "Pleural effusion.", "cand_fp02": "No pleural effusion.",
            "cand_fp08": "No pleural effusion.",
            "verifier_t0": "No pleural effusion.", "verifier_t2": "No pleural effusion.",
        }

    def test_bundled_vectors_apply_with_no_reference(self):
        for panel in ("seed911", "seed1221", "locked1000"):
            selected = pipeline.route_candidates([self.row], self.archive, panel)
            self.assertEqual([{"id": "invented-1", "prediction": "No pleural effusion.",
                               "selected_arm": "current_only"}], selected)

    def test_routing_rejects_references_wrong_history_and_missing_proposals(self):
        for field in ("reference", "matched_shuffled_prior", "image_paths"):
            with self.assertRaisesRegex(ValueError, "routing"):
                pipeline.route_candidates([{**self.row, field: "forbidden"}], self.archive)
        row = dict(self.row)
        del row["verifier_t2"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            pipeline.route_candidates([row], self.archive)

    def test_routing_selects_a_noncurrent_proposal_and_preserves_first_maximum_ties(self):
        row = {**self.row, "cand_fp02": "Proposal one.", "cand_fp08": "Proposal two.",
               "verifier_t0": "Proposal three.", "verifier_t2": "Proposal four."}
        for weights, arm, report in (
            ([0.0] * 18 + [1.0], "verifier_t2", "Proposal four."),
            ([0.0] * 19, "current_only", "No pleural effusion."),
        ):
            archive = {"feature_names": self.archive["feature_names"],
                       "panels": {"locked1000": {"weights": weights}}}
            self.assertEqual([{"id": "invented-1", "prediction": report, "selected_arm": arm}],
                             pipeline.route_candidates([row], archive))

    def test_routing_rejects_duplicate_ids_and_bad_weight_contracts(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pipeline.route_candidates([self.row, self.row], self.archive)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            pipeline.route_candidates([self.row], self.archive, "missing")
        archive = {**self.archive, "feature_names": ["wrong"]}
        with self.assertRaisesRegex(ValueError, "feature"):
            pipeline.route_candidates([self.row], archive)

    def test_cli_uses_bundled_vectors_and_emits_only_selected_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.json"
            target = root / "selected.json"
            source.write_text(json.dumps([self.row]), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", "-m", "scfr_release", "route", "--input", str(source),
                 "--panel", "locked1000", "--output", str(target)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual([{"id": "invented-1", "prediction": "No pleural effusion.",
                               "selected_arm": "current_only"}],
                             json.loads(target.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
