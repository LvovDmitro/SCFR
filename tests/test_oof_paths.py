"""OOF orchestration uses explicit storage paths without internal directories."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OOFPathTests(unittest.TestCase):
    def test_explicit_weights_are_checked_before_protected_outputs_are_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "scripts.run_scfr_cross_generator_oof_rotation",
                 "--protected-root", str(root / "protected"),
                 "--weights-receipt", str(root / "missing-weights.json"),
                 "--output-dir", str(root / "aggregates")],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing archived controller weights", result.stderr)
            self.assertFalse((root / "protected").exists())
            self.assertFalse((root / "aggregates").exists())

    def test_bundled_weights_work_without_a_rebuttal_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run(
                [sys.executable, "-B", "-m", "scripts.run_scfr_cross_generator_oof_rotation",
                 "--protected-root", str(root / "protected"),
                 "--output-dir", str(root / "aggregates")],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing protected merged input for radinfer", result.stderr)
            self.assertTrue((root / "protected" / "oof5" / "status.json").is_file())
            self.assertFalse((root / "aggregates").exists())


if __name__ == "__main__":
    unittest.main()
