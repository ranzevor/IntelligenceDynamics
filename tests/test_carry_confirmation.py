import math
import unittest

from intelligence_dynamics.research.confirmation import (
    confirm_carry_effects,
    summarize_confirmation_pair,
)

try:
    import numpy as np
except ImportError:
    np = None


class ConfirmationInferenceTests(unittest.TestCase):
    def results(self, effects):
        return [
            {
                "index": index,
                "seed": index,
                "run_identity": "same-run",
                "R_i_reset_minus_continued_post_mean_reward": effect,
            }
            for index, effect in enumerate(effects, 25)
        ]

    def test_six_seven_and_eight_positive_effects(self):
        for positives, expected, p_value in (
            (6, False, 37 / 256),
            (7, True, 9 / 256),
            (8, True, 1 / 256),
        ):
            with self.subTest(positives=positives):
                result = confirm_carry_effects(
                    self.results([1.0] * positives + [-0.1] * (8 - positives)),
                    indices=range(25, 33),
                )
                self.assertEqual(result["criterion_met"], expected)
                self.assertEqual(result["one_sided_sign_test"]["p_value"], p_value)
                self.assertEqual(result["positive_count"], positives)

    def test_zero_is_nonpositive_and_mean_must_be_positive(self):
        result = confirm_carry_effects(self.results([0.0] * 8), indices=range(25, 33))
        self.assertEqual(result["positive_count"], 0)
        self.assertEqual(result["zero_count"], 8)
        self.assertFalse(result["criterion_met"])
        result = confirm_carry_effects(self.results([1.0] * 7 + [-8.0]), indices=range(25, 33))
        self.assertEqual(result["positive_count"], 7)
        self.assertFalse(result["criterion_met"])

    def test_exact_interval_endpoints_and_independent_reference(self):
        for effects, expected in (
            ([1.0] * 8, (0.025 ** (1 / 8), 1.0)),
            ([-1.0] * 8, (0.0, 1 - 0.025 ** (1 / 8))),
            ([1.0] * 7 + [-1.0], (0.47349032912479344, 0.9968402764687481)),
        ):
            result = confirm_carry_effects(self.results(effects), indices=range(25, 33))
            interval = result["positive_fraction_interval"]
            self.assertAlmostEqual(interval["lower"], expected[0], places=12)
            self.assertAlmostEqual(interval["upper"], expected[1], places=12)

    def test_input_order_preserves_configured_output_order(self):
        rows = self.results([1.0] * 8)
        result = confirm_carry_effects(list(reversed(rows)), indices=range(25, 33))
        self.assertEqual([row["index"] for row in result["samples"]], list(range(25, 33)))

    def test_missing_duplicate_unknown_and_mixed_identity(self):
        rows = self.results([1.0] * 8)
        bad = [rows[:-1], rows + [rows[0]], rows + [{**rows[0], "index": 100}]]
        bad.append([{**rows[0], "run_identity": "other"}, *rows[1:]])
        bad.append([{**rows[0], "seed": rows[1]["seed"]}, *rows[1:]])
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                confirm_carry_effects(values, indices=range(25, 33))
        for indices in ([], [25, 25], [True], [[25]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                confirm_carry_effects(rows, indices=indices)

    def test_invalid_effect_values_and_finite_extremes(self):
        for value in (math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                confirm_carry_effects(self.results([value] * 8), indices=range(25, 33))
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        result = confirm_carry_effects(self.results([largest] * 8), indices=range(25, 33))
        self.assertEqual(result["mean_paired_reward_effect"], largest)
        smallest = math.nextafter(0.0, 1.0)
        result = confirm_carry_effects(self.results([smallest] * 8), indices=range(25, 33))
        self.assertEqual(result["mean_paired_reward_effect"], smallest)
        self.assertTrue(result["criterion_met"])


@unittest.skipIf(np is None, "NumPy is required for carry-study traces")
class ConfirmationWindowTests(unittest.TestCase):
    def traces(self, length=4096):
        return {
            name: np.zeros(length, dtype=np.int32 if name == "biome_id" else np.float16)
            for name in ("rewards", "biome_id", "loss", "abs_td_error", "weight_change")
        }

    def test_early_and_later_windows_use_exact_separate_ranges(self):
        left, right = self.traces(), self.traces()
        right["rewards"][1024:2048] = 2
        right["rewards"][2048:] = -1
        result = summarize_confirmation_pair(left, right, boundary=1024, endpoint=4096)
        early, later = (result["fixed_response_windows"][name] for name in ("early", "later"))
        self.assertEqual((early["start"], early["stop"], early["samples"]), (1024, 2048, 1024))
        self.assertEqual((later["start"], later["stop"], later["samples"]), (2048, 4096, 2048))
        self.assertEqual(early["reset_minus_continued"], 2)
        self.assertEqual(later["reset_minus_continued"], -1)
        self.assertEqual(result["R_i_reset_minus_continued_post_mean_reward"], 0)

    def test_windows_require_a_later_period_and_identical_history(self):
        left, right = self.traces(), self.traces()
        right["rewards"][0] = 1
        with self.assertRaisesRegex(ValueError, "prehistory"):
            summarize_confirmation_pair(left, right, boundary=1024, endpoint=4096)
        left, right = self.traces(2048), self.traces(2048)
        with self.assertRaisesRegex(ValueError, "more than 1024"):
            summarize_confirmation_pair(left, right, boundary=1024, endpoint=2048)


if __name__ == "__main__":
    unittest.main()
