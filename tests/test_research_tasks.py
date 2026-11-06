import dataclasses
import unittest

from intelligence_dynamics.research import (
    CueTask,
    Feedback,
    FixedPolicy,
    LearningRule,
    TabularPolicy,
    TaskConfig,
    adaptation_latency,
    corruption_plan,
    evaluate,
    make_bank,
    run_episode,
    summarize,
)


class TaskTests(unittest.TestCase):
    def test_hidden_observation_and_action_indexed_delay(self):
        task = CueTask(
            TaskConfig(steps=8, cue_count=1, delay=2, switches=(4,), mappings=((0,), (1,)), seed=4)
        )
        with self.assertRaises(RuntimeError):
            task.advance(0)
        observation = task.reset()
        self.assertEqual([field.name for field in dataclasses.fields(observation)], ["cue"])
        first, second, third = (task.advance(0) for _ in range(3))
        self.assertIsNone(first.feedback)
        self.assertIsNone(second.feedback)
        self.assertEqual(third.feedback, Feedback(0, 1))
        fourth, fifth = task.advance(0), task.advance(0)
        self.assertEqual(fourth.record.regime, 0)
        self.assertEqual(fifth.record.regime, 1)
        self.assertEqual(fifth.record.reward, 0)
        for _ in range(3):
            result = task.advance(1)
        self.assertTrue(result.done)
        self.assertIsNone(result.observation)
        with self.assertRaises(RuntimeError):
            task.advance(0)

    def test_reset_replays_cues(self):
        task = CueTask(TaskConfig(steps=9, switches=(4,)))
        runs = []
        for _ in range(2):
            task.reset()
            runs.append(tuple(task.advance(0) for _ in range(9)))
        self.assertEqual(*runs)

    def test_action_invalid_does_not_advance(self):
        task = CueTask(TaskConfig())
        task.reset()
        before = task.snapshot()
        for action in (True, 0.0, -1, 2, "0", None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                task.advance(action)
            self.assertEqual(before, task.snapshot())

    def test_task_boundaries(self):
        invalid = [
            dict(steps=True),
            dict(steps=0),
            dict(cue_count=0),
            dict(delay=-1),
            dict(delay=1.5),
            dict(seed=-1),
            dict(switches=(2, 2)),
            dict(switches=(96,)),
            dict(switches=(0,)),
            dict(switches=(True,)),
            dict(switches=(8, 4)),
            dict(mappings=((0, 1), (1, 0))),
            dict(mappings=()),
            dict(mappings=((False, 1, 0, 1), (1, 0, 1, 0))),
            dict(switches="bad"),
            dict(mappings=((0, 1, 0, 1),)),
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                TaskConfig(**change)

    def test_input_lists_are_immutable_snapshots(self):
        switches = [3]
        mappings = [[0], [1]]
        config = TaskConfig(steps=6, cue_count=1, switches=switches, mappings=mappings)
        switches.append(4)
        mappings[0][0] = 1
        self.assertEqual(config.switches, (3,))
        self.assertEqual(config.mappings, ((0,), (1,)))

    def test_mirrored_bank_cancels_every_fixed_policy(self):
        bank = make_bank(partition="train", seed=11, pairs=2, steps=16, cue_count=2)
        self.assertEqual(len(bank), 8)
        for mapping in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for index in range(0, len(bank), 2):
                a = run_episode(bank[index].task, FixedPolicy(mapping))
                b = run_episode(bank[index + 1].task, FixedPolicy(mapping))
                self.assertAlmostEqual(a.accuracy + b.accuracy, 1)
                self.assertEqual(tuple(x.cue for x in a.records), tuple(x.cue for x in b.records))
        self.assertEqual(sum(item.kind == "stationary" for item in bank), 4)

    def test_episode_resets_acquired_values(self):
        task = TaskConfig(steps=16, cue_count=1, delay=0, switches=(), mappings=((1,),))
        rule = LearningRule(priors=(0,), rate=1, exploration=0)
        learner = TabularPolicy(rule)
        first = run_episode(task, learner, seed=8)
        self.assertNotEqual(learner.values, ((0.5, 0.5),))
        second = run_episode(task, learner, seed=8)
        self.assertEqual(first, second)
        self.assertEqual(rule.priors, (0,))
        learner.reset(seed=8)
        self.assertEqual(learner.values, ((0.5, 0.5),))
        self.assertEqual(learner.pending_count, 0)
        self.assertEqual(learner.updates, ())

    def test_final_feedback_is_not_counted_as_an_online_decision(self):
        task = TaskConfig(steps=7, cue_count=1, delay=2, switches=(), mappings=((0,),))
        report = run_episode(task, TabularPolicy(LearningRule(priors=(0,))))
        self.assertEqual(len(report.deliveries), 4)
        self.assertEqual(report.undelivered, 3)
        self.assertEqual(tuple(row.credit_id for row in report.deliveries), (0, 1, 2, 3))
        self.assertEqual(tuple(row.step for row in report.deliveries), (3, 4, 5, 6))

    def test_latency_covers_last_complete_window(self):
        self.assertEqual(
            adaptation_latency((False, False, True, True), 0, 4, window=2, threshold=1), 2
        )
        self.assertIsNone(adaptation_latency((True,), 0, 1, window=2))
        with self.assertRaises(ValueError):
            adaptation_latency((1, 1), 0, 2)
        with self.assertRaises(ValueError):
            adaptation_latency((True,), 0, 2)

    def test_corruption_modes_are_deterministic_and_validate(self):
        self.assertEqual(corruption_plan(6), (False,) * 6)
        self.assertEqual(corruption_plan(6, kind="isolated", rate=1), (True, False) * 3)
        self.assertEqual(corruption_plan(8, kind="burst", rate=1), (True, True, True, False) * 2)
        self.assertEqual(
            corruption_plan(100, kind="burst", rate=0.2, seed=5),
            corruption_plan(100, kind="burst", rate=0.2, seed=5),
        )
        for arguments in (
            dict(length=-1),
            dict(length=True),
            dict(length=1, kind="wrong"),
            dict(length=1, rate=float("nan")),
            dict(length=1, rate=0.1),
            dict(length=1, kind="burst", burst_length=0),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                corruption_plan(**arguments)

    def test_missing_observations_remain_missing(self):
        summary = summarize([1, None])
        self.assertEqual((summary.count, summary.observed, summary.missing), (2, 1, 1))
        self.assertIsNone(summary.mean)
        self.assertIsNone(summarize([]).mean)
        self.assertEqual(summarize([1, 3]).mean, 2)
        self.assertEqual(summarize([1e308, 1e308]).mean, 1e308)

    def test_online_learning_tracks_a_reversed_target(self):
        config = TaskConfig(
            steps=96, cue_count=1, delay=1, switches=(32, 64), mappings=((0,), (1,)), seed=1
        )
        fixed = run_episode(config, FixedPolicy((0,)))
        adaptive = run_episode(
            config, TabularPolicy(LearningRule(priors=(0,), rate=1, exploration=0))
        )
        self.assertGreater(adaptive.accuracy, fixed.accuracy + 0.1)
        self.assertTrue(all(value is not None for value in adaptive.switch_latency))

    def test_zero_feedback_and_missing_categories_are_explicit(self):
        bank = make_bank(partition="heldout", seed=4, steps=8, delay=20)
        report = evaluate(LearningRule(), bank[:2])
        self.assertEqual(report.feedback_count, 0)
        self.assertIsNone(report.update_fraction)
        self.assertIsNone(report.clean_source_update_rate)
        self.assertIsNone(report.switching_accuracy)
        self.assertEqual(report.switch_latency.count, 0)

    def test_source_attribution_balances(self):
        bank = make_bank(partition="heldout", seed=7, steps=24)
        for mode in (
            "fixed",
            "plastic",
            "conditional",
            "accumulator",
            "persistence",
            "buffered",
            "random",
        ):
            with self.subTest(mode=mode):
                report = evaluate(
                    LearningRule(), bank, mode=mode, corruption="isolated", corruption_rate=0.4
                )
                self.assertEqual(
                    report.clean_source_updates + report.corrupt_source_updates, report.update_count
                )
                self.assertEqual(report.updated_errors.count, report.update_count)
                self.assertEqual(
                    report.updated_errors.count + report.skipped_errors.count, report.feedback_count
                )
                self.assertEqual(
                    report.clean_feedback_count + report.corrupt_feedback_count,
                    report.feedback_count,
                )
                self.assertLessEqual(report.update_count, report.feedback_count)
                self.assertGreaterEqual(report.corrupt_source_update_rate, 0)
                self.assertLessEqual(report.corrupt_source_update_rate, 1)
        self.assertEqual(evaluate(LearningRule(), bank, mode="fixed").update_count, 0)
        self.assertEqual(evaluate(LearningRule(), bank, mode="plastic").update_fraction, 1)
        self.assertIsNone(evaluate(LearningRule(), bank, mode="plastic").gate_error_separation)


if __name__ == "__main__":
    unittest.main()
