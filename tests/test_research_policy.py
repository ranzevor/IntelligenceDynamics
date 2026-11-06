import dataclasses
import unittest

from intelligence_dynamics.research import Feedback, LearningRule, Observation, TabularPolicy


class PolicyTests(unittest.TestCase):
    def deliver(self, policy, values):
        for credit, value in enumerate(values):
            policy.act(Observation(credit % len(policy.rule.priors)))
            policy.observe(Feedback(credit, value))

    def test_credit_id_targets_the_original_action(self):
        policy = TabularPolicy(LearningRule(priors=(0.5, -0.5), rate=1, exploration=0))
        self.assertEqual(policy.act(Observation(0)), 1)
        self.assertEqual(policy.act(Observation(1)), 0)
        policy.observe(Feedback(1, 0))
        self.assertEqual(policy.values, ((0.25, 0.75), (0, 0.25)))
        policy.observe(Feedback(0, 1))
        self.assertEqual(policy.values, ((0.25, 1), (0, 0.25)))
        before = policy.snapshot()
        with self.assertRaises(ValueError):
            policy.observe(Feedback(0, 1))
        self.assertEqual(before, policy.snapshot())

    def test_conditional_boundary_is_inclusive(self):
        policy = TabularPolicy(LearningRule(priors=(0,), threshold=0.5), "conditional")
        self.deliver(policy, [1])
        self.assertEqual(len(policy.updates), 1)
        self.assertEqual(policy.updates[0].error, 0.5)

    def test_persistence_does_not_replay_earlier_credits(self):
        policy = TabularPolicy(LearningRule(), "persistence")
        self.deliver(policy, [1, 1])
        self.assertEqual(policy.last_committed, (1,))
        self.assertEqual(len(policy.updates), 1)

    def test_buffered_gate_commits_both_window_edges(self):
        policy = TabularPolicy(LearningRule(), "buffered")
        self.deliver(policy, [1, 0.5, 1])
        self.assertEqual(policy.last_committed, (0, 2))
        self.assertEqual(tuple(row.feedback_index for row in policy.updates), (0, 2))
        self.assertEqual(tuple(row.committed_at for row in policy.updates), (2, 2))
        self.assertEqual(policy.feedback_count, 3)

    def test_buffered_gate_expires_old_credit(self):
        policy = TabularPolicy(LearningRule(priors=(0,) * 5), "buffered")
        self.deliver(policy, [1, 0.5, 0.5, 1])
        self.assertEqual(policy.updates, ())
        self.assertEqual(
            tuple(row.credit_id for row in policy.snapshot()["memory"]["eligible"]), (3,)
        )

    def test_buffered_updates_do_not_commit_twice(self):
        policy = TabularPolicy(LearningRule(priors=(0,) * 8), "buffered")
        self.deliver(policy, [1] * 8)
        self.assertEqual(tuple(row.credit_id for row in policy.updates), tuple(range(8)))

    def test_accumulator_leaks_and_resets(self):
        rule = LearningRule(priors=(0,) * 4, threshold=0.25, decay=0.5, accumulation=0.375)
        policy = TabularPolicy(rule, "accumulator")
        self.deliver(policy, [1, 1, 1, 1])
        self.assertEqual(tuple(row.credit_id for row in policy.updates), (1, 3))
        self.assertEqual(policy.snapshot()["memory"]["accumulated"], 0)

    def test_random_gate_extremes(self):
        for probability, expected in ((0, 0), (1, 4)):
            policy = TabularPolicy(LearningRule(random_probability=probability), "random")
            self.deliver(policy, [1] * 4)
            self.assertEqual(len(policy.updates), expected)

    def test_freezing_and_memory_reset_preserve_values_and_credits(self):
        policy = TabularPolicy(LearningRule(), "buffered")
        self.deliver(policy, [1])
        policy.act(Observation(1))
        values, pending = policy.values, policy.pending_count
        policy.clear_memory()
        self.assertEqual(policy.values, values)
        self.assertEqual(policy.pending_count, pending)
        policy.learning_enabled = False
        policy.observe(Feedback(1, 1))
        self.assertEqual(policy.updates, ())
        self.assertEqual(policy.feedback_count, 2)

    def test_invalid_inputs_leave_state_untouched(self):
        policy = TabularPolicy(LearningRule())
        policy.act(Observation(0))
        before = policy.snapshot()
        with self.assertRaises(ValueError):
            policy.act(Observation(10), Feedback(0, 1))
        self.assertEqual(before, policy.snapshot())
        with self.assertRaises(ValueError):
            policy.observe(Feedback(19, 1))
        self.assertEqual(before, policy.snapshot())

    def test_all_parameter_boundaries(self):
        for name in (
            "rate",
            "exploration",
            "threshold",
            "decay",
            "accumulation",
            "random_probability",
        ):
            for value in (float("nan"), float("inf"), True, "0.1", -0.01):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LearningRule(**{name: value})
        for changes in (
            dict(priors=()),
            dict(priors=(float("nan"),)),
            dict(priors=(True,)),
            dict(rate=0),
            dict(accumulation=0),
            dict(window=0),
            dict(required=0),
            dict(required=4),
            dict(window=2.5),
            dict(required=True),
            dict(exploration=1.1),
            dict(decay=1.1),
            dict(threshold=1.1),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                LearningRule(**changes)
        with self.assertRaises(ValueError):
            TabularPolicy(LearningRule(), "unknown")
        for arguments in ((True, 1), (0, float("nan")), (-1, 1), (0, True), (0, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Feedback(*arguments)

    def test_rule_contains_no_acquired_state(self):
        rule = LearningRule()
        before = dataclasses.asdict(rule)
        policy = TabularPolicy(rule)
        self.deliver(policy, [1, 0, 1, 0])
        self.assertEqual(dataclasses.asdict(rule), before)
        policy.reset(seed=1)
        self.assertEqual(policy.values, ((0.5, 0.5),) * 4)


if __name__ == "__main__":
    unittest.main()
