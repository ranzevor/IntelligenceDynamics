import dataclasses
import json
import unittest
from unittest.mock import patch

from intelligence_dynamics.research import (
    Scenario,
    SearchConfig,
    evaluate,
    evolve,
    make_bank,
    run_study,
)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.train = make_bank(partition="train", seed=5, steps=12, cue_count=2)
        self.heldout = make_bank(partition="heldout", seed=6, steps=12, cue_count=2)
        self.config = SearchConfig(population=3, generations=2, elites=1)

    def test_search_is_deterministic_and_elitist(self):
        first = evolve(self.train, config=self.config, seed=13)
        second = evolve(self.train, config=self.config, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(first.candidate_evaluations, 6)
        self.assertEqual(len(first.history_best), 2)
        self.assertGreaterEqual(first.history_best[1], first.history_best[0])
        self.assertEqual(
            first.training_fingerprints, tuple(row.task.fingerprint for row in self.train)
        )

    def test_heldout_never_reaches_search_evaluator(self):
        with patch("intelligence_dynamics.research.evolution.evaluate", wraps=evaluate) as spy:
            result = evolve(self.train, config=self.config, seed=8)
            self.assertEqual(spy.call_count, 12)
            for call in spy.call_args_list:
                self.assertTrue(all(row.partition == "train" for row in call.args[1]))
                self.assertEqual(call.kwargs["seed"], 8)
        self.assertEqual(len(result.rule.priors), 2)

    def test_changing_heldout_does_not_change_selection_or_calibration(self):
        first = run_study(self.train, self.heldout, config=self.config, seed=2)
        other = make_bank(partition="heldout", seed=9, steps=16, cue_count=2)
        second = run_study(self.train, other, config=self.config, seed=2)
        self.assertEqual(first.search, second.search)
        self.assertEqual(first.random_probability, second.random_probability)
        calibration = evaluate(first.search.rule, self.train, mode="buffered", seed=2)
        self.assertEqual(first.random_probability, calibration.update_fraction)
        self.assertEqual(first.frozen.update_count, 0)
        self.assertEqual(first.always.update_fraction, 1)
        json.dumps(dataclasses.asdict(first), allow_nan=False)

    def test_forged_partition_names_do_not_hide_reused_tasks(self):
        renamed = tuple(Scenario("held:" + row.name, "heldout", row.task) for row in self.train)
        with patch("intelligence_dynamics.research.evolution.evolve") as selection:
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_study(self.train, renamed, config=self.config)
            selection.assert_not_called()
        changed = tuple(
            Scenario("held:" + row.name, "heldout", dataclasses.replace(row.task, delay=0))
            for row in self.train
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            run_study(self.train, changed, config=self.config)

    def test_wrong_partition_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evolve(self.heldout)
        with self.assertRaises(ValueError):
            evolve(self.train + (self.train[0],))
        duplicate = Scenario("another-name", "train", self.train[0].task)
        with self.assertRaises(ValueError):
            evolve(self.train + (duplicate,))
        with self.assertRaises(ValueError):
            run_study(self.train, self.train)
        with self.assertRaises(ValueError):
            evolve(())

    def test_seed_changes_are_explicit(self):
        same_seed_held = make_bank(partition="heldout", seed=5, steps=12, cue_count=2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            run_study(self.train, same_seed_held, config=self.config)

    def test_search_rejects_no_feedback_objective(self):
        train = make_bank(partition="train", seed=1, steps=8, delay=8)
        with self.assertRaisesRegex(ValueError, "feedback"):
            evolve(train)

    def test_search_config_boundaries(self):
        changes = [
            dict(population=1),
            dict(population=True),
            dict(generations=0),
            dict(generations=1.5),
            dict(elites=0),
            dict(elites=4),
            dict(mutation=-1),
            dict(update_cost=float("nan")),
            dict(noise_weight=1.1),
            dict(corruption_rate=-0.1),
            dict(mutation=float("inf")),
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                SearchConfig(**change)

    def test_every_mode_uses_same_candidate_budget(self):
        for mode in (
            "fixed",
            "plastic",
            "conditional",
            "persistence",
            "accumulator",
            "buffered",
            "random",
        ):
            with self.subTest(mode=mode):
                report = evolve(
                    self.train, mode=mode, config=SearchConfig(population=2, generations=1)
                )
                self.assertEqual(report.candidate_evaluations, 2)
                self.assertEqual(report.mode, mode)
                self.assertGreaterEqual(report.train_update_fraction, 0)
                self.assertLessEqual(report.train_update_fraction, 1)


if __name__ == "__main__":
    unittest.main()
