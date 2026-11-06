import unittest
from copy import deepcopy
from dataclasses import dataclass

from intelligence_dynamics.research import (
    LearningRule,
    Series,
    TaskConfig,
    aggregate_comparisons,
    common_support,
    compare_blocks,
    compare_branches,
    select_events,
    state_changes,
    state_digest,
    trace_digest,
    verify_intervention,
    verify_paired_history,
)


@dataclass
class Memory:
    values: list
    carry: tuple


class StateTests(unittest.TestCase):
    def test_digest_preserves_type_shape_and_signed_zero(self):
        values = (1, True, 1.0, [1], (1,), {"0": 1}, 0.0, -0.0)
        self.assertEqual(len(set(state_digest(value) for value in values)), len(values))
        self.assertEqual(state_digest({"b": 2, "a": 1}), state_digest({"a": 1, "b": 2}))
        original = Memory([1, 2], (3, 4))
        self.assertEqual(state_digest(original), state_digest(deepcopy(original)))

    def test_cycle_and_nonfinite_state_are_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        for state in (cyclic, {1: "value"}, float("nan"), float("inf")):
            with self.subTest(state=type(state)), self.assertRaises(ValueError):
                state_digest(state)
        with self.assertRaises(TypeError):
            state_digest(object())

    def test_structured_paths_do_not_conflate_punctuation(self):
        before = {"a.b": [1], "a": {"b": [1]}}
        after = {"a.b": [2], "a": {"b": [1]}}
        self.assertEqual(state_changes(before, after), (("a.b", 0),))
        self.assertEqual(
            verify_intervention(before, after, allowed_paths=(("a.b",),)), (("a.b", 0),)
        )
        with self.assertRaises(ValueError):
            verify_intervention(before, after, allowed_paths=(("a", "b"),))

    def test_intervention_detects_unselected_state_and_noop(self):
        before = {"agent": Memory([1, 2], (3, 4)), "replay": [4, 5]}
        after = deepcopy(before)
        after["agent"].carry = ()
        changed = verify_intervention(before, after, allowed_paths=(("agent", "carry"),))
        self.assertEqual(changed, (("agent", "carry", 0), ("agent", "carry", 1)))
        after["replay"][0] = 9
        with self.assertRaises(ValueError):
            verify_intervention(before, after, allowed_paths=(("agent", "carry"),))
        with self.assertRaises(ValueError):
            verify_intervention(before, before, allowed_paths=(("agent", "carry"),))
        self.assertEqual(
            verify_intervention(before, before, allowed_paths=(("agent",),), require_change=False),
            (),
        )

    def test_branch_preserves_preboundary_and_environment_stream(self):
        task = TaskConfig(
            steps=32, cue_count=2, delay=2, switches=(16,), mappings=((0, 1), (1, 0)), seed=4
        )
        rule = LearningRule(priors=(0, 0), rate=1, exploration=0)
        first = compare_branches(task, rule, boundary=16, seed=3)
        second = compare_branches(task, rule, boundary=16, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first.common_prefix), 16)
        self.assertEqual(len(first.continued), 16)
        self.assertEqual(first.changed_paths, (("policy", "learning_enabled"),))
        self.assertEqual(first.intervened_updates, 0)
        self.assertGreater(first.continued_updates, 0)
        self.assertEqual(
            tuple(row.cue for row in first.continued), tuple(row.cue for row in first.intervened)
        )
        self.assertEqual(first.continued[0].step, 16)

    def test_memory_reset_reports_empty_intervention(self):
        task = TaskConfig(steps=8, switches=(4,))
        result = compare_branches(
            task, LearningRule(), boundary=1, mode="buffered", intervention="reset_memory"
        )
        self.assertFalse(result.intervention_changed)
        self.assertEqual(result.changed_paths, ())
        self.assertEqual(result.continued, result.intervened)

    def test_memory_reset_changes_only_gate_memory(self):
        task = TaskConfig(steps=10, delay=0, switches=(5,))
        result = compare_branches(
            task, LearningRule(), boundary=2, mode="buffered", intervention="reset_memory"
        )
        self.assertTrue(result.intervention_changed)
        self.assertTrue(all(path[:2] == ("policy", "memory") for path in result.changed_paths))

    def test_invalid_branch_bounds(self):
        for boundary in (0, True, 96, 2.5):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                compare_branches(TaskConfig(), LearningRule(), boundary=boundary)

    def test_trace_history_requires_exact_boundary_and_typed_identity(self):
        columns = {"reward": [1.0, 0.0], "context": [0, 1]}
        digest = trace_digest(columns, boundary=2)
        self.assertEqual(verify_paired_history(columns, deepcopy(columns), boundary=2), digest)
        with self.assertRaises(ValueError):
            trace_digest(columns, boundary=1)
        with self.assertRaises(ValueError):
            trace_digest({"reward": [1], "context": [0, 1]}, boundary=2)
        changed = deepcopy(columns)
        changed["context"][0] = False
        with self.assertRaises(ValueError):
            verify_paired_history(columns, changed, boundary=2)


class CommonSupportTests(unittest.TestCase):
    def test_configured_reward_blocks(self):
        control = Series((0, 0, 2, 2), (0,) * 4)
        treatment = Series((1, 1, 1, 1), (0,) * 4)
        result = compare_blocks(control, treatment, edges=(0, 2, 4))
        self.assertEqual(tuple(row.delta for row in result), (1, -1))
        self.assertEqual(tuple((row.start, row.stop) for row in result), ((0, 2), (2, 4)))
        for edges in ((0,), (0, 0), (0, 5), (0, True), (2, 1)):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                compare_blocks(control, treatment, edges=edges)

    def test_hand_calculated_shared_event(self):
        states = (0, 0, 0, 0, 1, 1, 1, 1)
        control = Series((0,) * 8, states)
        treatment = Series((0, 0, 0, 0, 1, 1, 1, 1), states)
        result = common_support(control, treatment, window=2)
        self.assertEqual(result.selected_events, (4,))
        self.assertEqual(result.control_response, 0)
        self.assertEqual(result.treatment_response, 1)
        self.assertEqual(result.delta, 1)
        self.assertEqual(result.event_deltas, (1,))
        self.assertEqual(result.jaccard, 1)

    def test_intersection_precedes_spacing(self):
        left_states = (0, 0, 1, 1, 2, 2, 2, 2)
        right_states = (0, 0, 0, 0, 1, 1, 1, 1)
        result = common_support(
            Series((0,) * 8, left_states), Series((0,) * 8, right_states), window=2
        )
        self.assertEqual(result.control_events, (2, 4))
        self.assertEqual(result.treatment_events, (4,))
        self.assertEqual(result.selected_events, (4,))
        self.assertEqual(result.jaccard, 0.5)

    def test_window_endpoints_are_inclusive(self):
        states = (0, 0, 1, 1, 1, 1, 2, 2)
        result = common_support(Series((0,) * 8, states), Series((0,) * 8, states), window=2)
        self.assertEqual(result.selected_events, (2, 6))
        truncated = common_support(
            Series((0,) * 8, states), Series((0,) * 8, states), start=1, stop=7, window=2
        )
        self.assertEqual(truncated.selected_events, ())

    def test_empty_support_and_partial_aggregate_not_zero(self):
        left = Series((0,) * 6, (0, 0, 0, 1, 1, 1))
        right = Series((0,) * 6, (0,) * 6)
        report = common_support(left, right, window=1)
        self.assertIsNone(report.delta)
        self.assertIsNone(report.control_response)
        self.assertEqual(report.jaccard, 0)
        none = common_support(right, right, window=1)
        self.assertIsNone(none.jaccard)
        aggregate = aggregate_comparisons((report.delta, 1))
        self.assertIsNone(aggregate.observations.mean)
        self.assertIsNone(aggregate.median)
        self.assertEqual(aggregate.observations.missing, 1)
        self.assertEqual(aggregate.positive, 1)
        complete = aggregate_comparisons((1, 3))
        self.assertEqual(complete.observations.mean, 2)
        self.assertEqual(complete.median, 2)
        large = aggregate_comparisons((1e308, 1e308))
        self.assertEqual(large.observations.mean, 1e308)
        self.assertEqual(large.median, 1e308)

    def test_boundaries_and_numeric_validation(self):
        for rewards, states in (
            ((), ()),
            ((1,), ()),
            ((float("nan"),), (0,)),
            ((1,), (True,)),
            ((True,), (0,)),
            ((1,), (0.0,)),
        ):
            with self.subTest(rewards=rewards, states=states), self.assertRaises(ValueError):
                Series(rewards, states)
        series = Series((0,) * 8, (0,) * 8)
        for change in (
            dict(window=0),
            dict(window=True),
            dict(start=-1),
            dict(stop=9),
            dict(start=4, stop=4),
            dict(spacing=1),
            dict(spacing=0),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                common_support(series, series, **change)
        for events in ((2, 2), (3, 1), (True,), (1.5,)):
            with self.subTest(events=events), self.assertRaises(ValueError):
                select_events(events, spacing=2)
        self.assertEqual(select_events((1, 2, 3, 4, 7), spacing=3), (1, 4, 7))


if __name__ == "__main__":
    unittest.main()
