import json
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from intelligence_dynamics.research.carry import TRACE_FIELDS, summarize_carry_pair


@unittest.skipIf(np is None, "NumPy is required for carry-study traces")
class CarrySummaryTests(unittest.TestCase):
    @staticmethod
    def traces(length=4096, reward=1):
        return {
            name: np.full(
                length,
                0 if name == "biome_id" else reward,
                dtype=(np.int32 if name == "biome_id" else np.float16),
            )
            for name in TRACE_FIELDS
        }

    def test_pair_uses_only_post_boundary_rewards(self):
        continued = self.traces()
        reset = self.traces()
        reset["rewards"][1024:] = 3
        report = summarize_carry_pair(continued, reset, boundary=1024, endpoint=4096)
        self.assertEqual(report["R_i_reset_minus_continued_post_mean_reward"], 2)
        self.assertEqual(report["continued"]["post_boundary_samples"], 3072)
        self.assertTrue(report["matched_history_exact"])
        self.assertEqual(report["prehistory_sha256"]["rewards"]["shape"], [1024])
        json.dumps(report, allow_nan=False)

    def test_empty_events_keep_absence_separate_from_zero(self):
        traces = self.traces()
        report = summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)["continued"]
        self.assertEqual(report["post_boundary_eligible_biome_events"], 0)
        self.assertEqual(report["recovery_censored_count"], 0)
        self.assertIsNone(report["recovery_censored_fraction"])
        self.assertIsNone(report["recovery_median_steps"])
        self.assertIsNone(report["event_cost_pre_minus_post_mean"])

    def test_events_use_complete_nonoverlapping_windows(self):
        traces = self.traces()
        for step in (1087, 1088, 1100, 1216, 4032, 4033):
            traces["biome_id"][step:] += 1
        report = summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)["continued"]
        self.assertEqual([event["step"] for event in report["events"]], [1088, 1216, 4032])
        self.assertEqual(report["native_biome_change_count"], 6)
        self.assertEqual(report["recovery_recovered_count"], 3)
        self.assertEqual(report["recovery_median_steps"], 64)

    def test_recovery_uses_literal_threshold_and_block_end(self):
        for before, after, expected in ((10, 9, 64), (0, 0, 64), (-10, -9, 64), (-10, -10, None)):
            with self.subTest(before=before, after=after):
                traces = self.traces(reward=before)
                traces["rewards"][2048:] = after
                traces["biome_id"][2048:] = 1
                report = summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)[
                    "continued"
                ]
                event = report["events"][0]
                self.assertEqual(event["recovery_target"], 0.9 * before)
                self.assertEqual(event["recovery_steps"], expected)
                self.assertEqual(event["observation_steps"], 2048)

    def test_recovery_crosses_later_changes_and_counts_endpoint_censoring(self):
        traces = self.traces()
        traces["biome_id"][2048:2112] = 1
        traces["biome_id"][4032:] = 1
        traces["rewards"][2048:2112] = 0
        traces["rewards"][4032:] = 0
        report = summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)["continued"]
        self.assertEqual([event["step"] for event in report["events"]], [2048, 4032])
        self.assertEqual(report["events"][0]["recovery_steps"], 128)
        self.assertEqual(report["events"][1]["observation_steps"], 64)
        self.assertIsNone(report["events"][1]["recovery_steps"])
        self.assertEqual(report["recovery_censored_count"], 1)
        self.assertEqual(report["recovery_endpoint_censored_count"], 1)

    def test_recovery_includes_last_full_block_at_observation_limit(self):
        traces = self.traces()
        traces["biome_id"][2048:] = 1
        traces["rewards"][2048:4032] = 0
        report = summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)["continued"]
        self.assertEqual(report["events"][0]["recovery_steps"], 2048)
        self.assertEqual(report["recovery_censored_count"], 0)

    def test_windows_use_float64_and_report_partial_sample_counts(self):
        traces = self.traces(length=100003, reward=60000)
        report = summarize_carry_pair(traces, traces, boundary=3, endpoint=100003)["continued"]
        self.assertEqual(report["post_boundary_mean_reward"], 60000)
        self.assertEqual(
            report["reward_windows_25k"][-1],
            {"start": 100000, "stop": 100003, "samples": 3, "mean": 60000},
        )
        self.assertEqual(
            report["update_blocks_50k"]["weight_change"],
            [
                {"start": 3, "stop": 50003, "samples": 50000, "mean": 60000},
                {"start": 50003, "stop": 100003, "samples": 50000, "mean": 60000},
            ],
        )
        json.dumps(report, allow_nan=False)

    def test_pre_history_requires_identical_dtype_and_bytes(self):
        for field in TRACE_FIELDS:
            with self.subTest(field=field):
                left, right = self.traces(), self.traces()
                right[field][0] += 1
                with self.assertRaisesRegex(ValueError, "prehistory differs"):
                    summarize_carry_pair(left, right, boundary=1024, endpoint=4096)
        left, right = self.traces(), self.traces()
        right["rewards"] = right["rewards"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "prehistory differs"):
            summarize_carry_pair(left, right, boundary=1024, endpoint=4096)
        left, right = self.traces(reward=0), self.traces(reward=0)
        right["rewards"][0] = -0.0
        with self.assertRaisesRegex(ValueError, "prehistory differs"):
            summarize_carry_pair(left, right, boundary=1024, endpoint=4096)

    def test_invalid_keys_shapes_types_and_values(self):
        broken = []
        missing = self.traces()
        del missing["loss"]
        broken.append(missing)
        extra = self.traces()
        extra["other"] = extra["loss"]
        broken.append(extra)
        for value in (
            np.zeros((4096, 1)),
            np.zeros(4095),
            [0] * 4096,
            np.zeros(4096, dtype=complex),
            np.zeros(4096, dtype=bool),
            np.full(4096, np.nan),
            np.full(4096, np.inf),
        ):
            malformed = self.traces()
            malformed["loss"] = value
            broken.append(malformed)
        noninteger_biomes = self.traces()
        noninteger_biomes["biome_id"] = np.zeros(4096)
        broken.append(noninteger_biomes)
        for position, traces in enumerate(broken):
            with self.subTest(position=position), self.assertRaises(ValueError):
                summarize_carry_pair(traces, self.traces(), boundary=1024, endpoint=4096)

    def test_reduction_overflow_does_not_emit_nonfinite_json(self):
        traces = self.traces()
        traces["loss"] = np.full(4096, np.finfo(np.float64).max)
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            summarize_carry_pair(traces, traces, boundary=1024, endpoint=4096)

    def test_boundaries_require_nonempty_pre_and_post_periods(self):
        traces = self.traces()
        for boundary, endpoint in (
            (0, 4096),
            (True, 4096),
            (1024.0, 4096),
            (4096, 4096),
            (4097, 4096),
            (1024, True),
        ):
            with self.subTest(boundary=boundary, endpoint=endpoint), self.assertRaises(ValueError):
                summarize_carry_pair(traces, traces, boundary=boundary, endpoint=endpoint)


if __name__ == "__main__":
    unittest.main()
