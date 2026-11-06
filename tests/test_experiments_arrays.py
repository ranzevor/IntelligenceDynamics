import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

from intelligence_dynamics.adapters import forager_npz_records, npz_records
from intelligence_dynamics.experiments import (
    IndividualObservation,
    ScaleProfile,
    population_snapshot,
    qualify_environment,
)


class ExperimentTests(unittest.TestCase):
    def test_scale_retains_learning(self):
        learning = {"rate": 0.2, "optimizer": {"steps": [2, 5]}}
        profile = ScaleProfile()
        result = profile.configuration(learning)
        result["learning"]["optimizer"]["steps"][0] = 7
        self.assertEqual(learning["optimizer"]["steps"], [2, 5])
        self.assertEqual(profile.interactions_per_individual, 1600)
        for values in (
            {"elites": 5},
            {"parent_pool": 9},
            {"validation_trials": 10},
            {"seed": True},
        ):
            with self.assertRaises(ValueError):
                ScaleProfile(**values)

    def test_population_snapshot(self):
        population = [IndividualObservation(i, 1.0, 2, 3, 1, 0.1, 0.2, 0.3, 0.4) for i in (2, 1, 3)]
        before = copy.deepcopy(population)
        result = population_snapshot(population, generation=3, parent_pool=2, elites=1)
        self.assertEqual(population, before)
        self.assertEqual([i["individual_id"] for i in result["individuals"]], [1, 2, 3])
        self.assertEqual([i["selected_parent"] for i in result["individuals"]], [True, True, False])
        with self.assertRaises(ValueError):
            population_snapshot([population[0]] * 2, generation=0, parent_pool=1, elites=1)

    def test_environment(self):
        class Environment:
            def reset(self, seed):
                self.state = seed
                return [self.state]

            def step(self, action):
                self.state += action
                return [self.state], float(action), False, {"state_scalars": 1}

        result = qualify_environment(Environment, [1, 2], seed=7)
        self.assertEqual(result["trajectory"]["transitions"][-1]["observation"], [10])

        class Broken(Environment):
            def step(self, action):
                return [0], float("nan"), False, {}

        with self.assertRaises(ValueError):
            qualify_environment(Broken, [1])
        with self.assertRaises(ValueError):
            qualify_environment(Environment, [])


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "arrays dependency is not installed")
class ArrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace.npz"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, **values):
        np.savez(self.path, **values)

    def test_invalid_containers_report_npz_format(self):
        for native in (b"unrecognized-native-input", b"PK\x03\x04broken-archive", None):
            if native is None:
                with self.path.open("wb") as stream:
                    np.save(stream, np.array([1.0, 2.0]))
            else:
                self.path.write_bytes(native)
            for reader in (npz_records, forager_npz_records):
                with self.subTest(native=native, reader=reader.__name__):
                    with self.assertRaisesRegex(ValueError, "NPZ .*valid.*archive"):
                        list(reader(self.path, run_id="synthetic"))

    def test_nonarray_zip_members_are_rejected(self):
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("rewards", b"not an array")
            archive.writestr("total_loss", b"not an array")
        for reader in (npz_records, forager_npz_records):
            with self.subTest(reader=reader.__name__):
                with self.assertRaisesRegex(ValueError, "NPZ members.*NumPy arrays"):
                    list(reader(self.path, run_id="synthetic"))

    def test_input_is_closed_before_measurements_are_yielded(self):
        self.write(rewards=[1.0, 2.0], total_loss=[0.5, 0.75])
        expected = self.path.read_bytes()
        for reader in (npz_records, forager_npz_records):
            with self.subTest(reader=reader.__name__):
                rows = reader(self.path, run_id="synthetic")
                self.assertEqual(next(rows).capability, 1.0)
                self.path.unlink()
                self.assertEqual(next(rows).capability, 2.0)
                rows.close()
                self.path.write_bytes(expected)

    def test_valid_archive_preserves_member_errors(self):
        for reader in (npz_records, forager_npz_records):
            for rewards, expected in (
                ([float("nan"), 2.0], "finite"),
                (np.array([{}], dtype=object), "Object arrays"),
            ):
                with self.subTest(reader=reader.__name__, expected=expected):
                    self.write(rewards=rewards, total_loss=[0.5])
                    with self.assertRaisesRegex(ValueError, expected):
                        list(reader(self.path, run_id="synthetic"))

    def test_multirate_context_and_static(self):
        self.write(
            rewards=[1.0, 3.0, 2.0, 6.0],
            total_loss=[0.5, 0.25],
            gradient=[4.0, 5.0],
            rank=[7.0],
            pos=[[0, 1], [1, 2], [2, 3], [3, 4]],
            biome_id=[1, 1, 2, 2],
            object_collected_id=[-1, 2, -1, -1],
            biome_regret=[4.0, 2.0, 1.0, 3.0],
        )
        rows = list(forager_npz_records(self.path, run_id="synthetic", expected_rollout_steps=2))
        self.assertEqual([row.step for row in rows], [2, 4])
        self.assertEqual([row.capability for row in rows], [2.0, 4.0])
        self.assertEqual(rows[0].extras["native"]["pos"], [[0, 1], [1, 2]])
        self.assertEqual(rows[0].extras["native"]["object_collected_count"], 1)
        self.assertEqual(rows[1].extras["native"]["rank"], 7.0)
        self.assertIsNone(rows[0].plasticity_probe_score)

    def test_bad_shapes_values_interval(self):
        cases = [
            dict(rewards=[1, 2, 3], total_loss=[1, 2]),
            dict(rewards=[1, 2], total_loss=[]),
            dict(rewards=[1, 2], total_loss=[1], pos=[1]),
            dict(rewards=[1, 2], total_loss=[1], bad=[[1]]),
            dict(rewards=[1, float("nan")], total_loss=[1]),
            dict(rewards=[1, 2], total_loss=[1], bad=np.array([{}], dtype=object)),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.write(**case)
                with self.assertRaises(ValueError):
                    list(forager_npz_records(self.path, run_id="r"))
        self.write(rewards=[1, 2], total_loss=[1])
        with self.assertRaises(ValueError):
            list(forager_npz_records(self.path, run_id="r", expected_rollout_steps=3))

    def test_large_finite_mean(self):
        self.write(rewards=[1e308, 1e308], total_loss=[1])
        self.assertEqual(next(forager_npz_records(self.path, run_id="r")).capability, 1e308)

    def test_equal_rate(self):
        self.write(rewards=[1, 2], metric=[3.0, 4.0])
        rows = list(npz_records(self.path, run_id="r", interval=5))
        self.assertEqual([row.step for row in rows], [5, 10])
