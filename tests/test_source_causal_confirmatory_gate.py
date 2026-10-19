import unittest

from scripts.source_causal_confirmatory_gate import holm, signed_direction_ok


class SourceCausalConfirmatoryGateTests(unittest.TestCase):
    def test_holm_respects_direction_and_adjusts_p_values(self):
        rows = [
            {"target": "a", "metric": "m", "p_value": 0.01, "direction_ok": True},
            {"target": "b", "metric": "m", "p_value": 0.02, "direction_ok": False},
            {"target": "c", "metric": "m", "p_value": 0.20, "direction_ok": True},
        ]

        adjusted = holm(rows, alpha=0.05)
        by_target = {row["target"]: row for row in adjusted}

        self.assertTrue(by_target["a"]["holm_reject"])
        self.assertFalse(by_target["b"]["holm_reject"])
        self.assertFalse(by_target["c"]["holm_reject"])
        self.assertGreaterEqual(by_target["b"]["holm_adjusted_p"], by_target["a"]["holm_adjusted_p"])

    def test_signed_direction_ok(self):
        self.assertTrue(signed_direction_ok(-0.1, "negative"))
        self.assertFalse(signed_direction_ok(0.1, "negative"))
        self.assertTrue(signed_direction_ok(0.1, "positive"))
        self.assertFalse(signed_direction_ok(-0.1, "positive"))

    def test_holm_does_not_restart_after_first_ordered_failure(self):
        rows = [
            {"target": "a", "metric": "m", "p_value": 0.03, "direction_ok": True},
            {"target": "b", "metric": "m", "p_value": 0.04, "direction_ok": True},
        ]

        adjusted = holm(rows, alpha=0.05)
        by_target = {row["target"]: row for row in adjusted}

        self.assertFalse(by_target["a"]["holm_reject"])
        self.assertFalse(by_target["b"]["holm_reject"])
        self.assertGreater(by_target["b"]["holm_adjusted_p"], 0.05)


if __name__ == "__main__":
    unittest.main()

