import copy
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from intelligence_dynamics.experiments import IndividualObservation, population_snapshot
from intelligence_dynamics.native import ForagaxAdapter, dal_population, dal_remote_population


class Environment:
    def __init__(self):
        self.default_params = {"nested": {"gain": 1}}
        self.mutate = None

    def action_space(self, params):
        if self.mutate == "action_space":
            params["nested"]["gain"] = 99

        def sample(key):
            if self.mutate == "sample":
                params["nested"]["gain"] = 99
            return key % 2

        return SimpleNamespace(n=2, sample=sample)

    def reset(self, key, params):
        if self.mutate == "reset":
            params["nested"]["gain"] = 99
        return [0], {"x": 0}

    def step(self, key, state, action, params):
        if self.mutate == "step":
            params["nested"]["gain"] = 99
        x = state["x"] + params["nested"]["gain"]
        return [x], {"x": x}, float(x), False, {}


def adapter(environment, params=None):
    return ForagaxAdapter(
        environment,
        params=params,
        key=lambda seed: seed,
        split=lambda key, count: (key + 1, key + 2),
    )


class ConfigurationTests(unittest.TestCase):
    def test_default_parameters_are_owned_and_exact_resume_is_preserved(self):
        native = Environment()
        env = adapter(native)
        env.reset(1)
        saved = env.snapshot()
        first = env.step(0)
        native.default_params["nested"]["gain"] = 7
        env.restore(saved)
        self.assertEqual(env.step(0), first)
        self.assertEqual(env.snapshot().identity, saved.identity)

    def test_explicit_parameters_and_public_copy_do_not_alias(self):
        supplied = {"nested": {"gain": 3}}
        env = adapter(Environment(), supplied)
        supplied["nested"]["gain"] = 7
        exposed = env.params
        exposed["nested"]["gain"] = 9
        env.reset(1)
        self.assertEqual(env.step(0)[1], 3.0)
        self.assertEqual(env.params, {"nested": {"gain": 3}})

    def test_configuration_properties_cannot_be_reassigned(self):
        env = adapter(Environment())
        for name, value in (("params", {}), ("identity", "other"), ("transition", "step_env")):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(env, name, value)

    def test_different_parameters_reject_restore_without_state_change(self):
        env = adapter(Environment())
        env.reset(1)
        other = adapter(Environment(), {"nested": {"gain": 7}})
        other.reset(1)
        before = other.snapshot()
        with self.assertRaises(ValueError):
            other.restore(env.snapshot())
        self.assertEqual(other.snapshot(), before)

    def test_native_mutation_is_rejected_without_snapshot_advance(self):
        for operation in ("reset", "action_space", "sample", "step"):
            with self.subTest(operation=operation):
                native = Environment()
                env = adapter(native)
                env.reset(1)
                before = env.snapshot()
                native.mutate = operation
                with self.assertRaisesRegex(ValueError, "changed environment parameters"):
                    if operation == "reset":
                        env.reset(2)
                    elif operation == "sample":
                        env.sample_action()
                    else:
                        env.step(0)
                self.assertEqual(env.snapshot(), before)
                self.assertEqual(env.params, {"nested": {"gain": 1}})
                native.mutate = None
                self.assertEqual(env.step(0)[1], 1.0)

    def test_direct_internal_corruption_is_detected_before_use(self):
        env = adapter(Environment())
        env.reset(1)
        saved = env.snapshot()
        env._params["nested"]["gain"] = 7
        for operation in (
            env.snapshot,
            lambda: env.restore(saved),
            lambda: env.reset(1),
            lambda: env.step(0),
            env.sample_action,
            lambda: env.params,
        ):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()

    def test_dataclass_parameters_preserve_type_and_nested_copy(self):
        @dataclass
        class Params:
            gains: list

        supplied = Params([1])
        native = Environment()
        native.default_params = supplied
        native.reset = lambda key, params: ([params.gains[0]], {})
        env = adapter(native)
        supplied.gains[0] = 7
        self.assertIs(type(env.params), Params)
        self.assertEqual(env.reset(1), [1])

    def test_array_parameter_copy_and_identity(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        supplied = {"gain": np.array([1.0, 2.0])}
        native = Environment()
        native.reset = lambda key, params: (params["gain"], {})
        env = adapter(native, supplied)
        supplied["gain"][0] = 7.0
        exposed = env.params
        exposed["gain"][1] = 9.0
        self.assertEqual(env.reset(1), [1.0, 2.0])
        different = adapter(native, supplied)
        self.assertNotEqual(env.identity, different.identity)


def individual(identifier, fitness):
    return SimpleNamespace(
        individual_id=identifier,
        normalised_fitness=fitness,
        connectivity_grid_a=[[0, 1]],
        connectivity_grid_m=[[0]],
        io_list_a=[(0, 1, 0.3)],
        io_list_m=[],
        rl_learning_rate=0.2,
        total_dw_list_nm=[0.1],
        total_dw_list_rl=[0.2],
        mean_active_rl_learning_rate=0.2,
        mutations_applied=[],
        is_unmutated_copy=False,
    )


class StableDALRankingTests(unittest.TestCase):
    def test_tied_elite_follows_native_population_order(self):
        for ids in ((9, 2), (2, 9)):
            population = [individual(identifier, 1.0) for identifier in ids]
            result = dal_population(population, generation=1, parent_pool=1, elites=1)
            self.assertEqual([row["individual_id"] for row in result["individuals"]], list(ids))
            self.assertEqual([row["elite"] for row in result["individuals"]], [True, False])
            self.assertEqual(
                [row["selected_parent"] for row in result["individuals"]], [True, False]
            )

    def test_mixed_fitness_and_ties_match_native_stable_sort(self):
        population = [
            individual(9, 1.0),
            individual(5, 2.0),
            individual(2, 1.0),
            individual(8, 2.0),
        ]
        before = copy.deepcopy([vars(item) for item in population])
        result = dal_population(population, generation=1, parent_pool=3, elites=2)
        expected = sorted(population, key=lambda item: item.normalised_fitness, reverse=True)
        self.assertEqual(
            [row["individual_id"] for row in result["individuals"]],
            [item.individual_id for item in expected],
        )
        self.assertEqual(
            [row["elite"] for row in result["individuals"]], [True, True, False, False]
        )
        self.assertEqual(
            [row["selected_parent"] for row in result["individuals"]], [True, True, True, False]
        )
        self.assertEqual([vars(item) for item in population], before)

    def test_remote_worker_order_is_preserved_at_tie(self):
        class Worker:
            def __init__(self, identifier):
                self.individual_id = identifier
                self.is_unmutated_copy = False
                self.pipe = SimpleNamespace(poll=lambda timeout: True)
                self.requests = []

            def send(self, message):
                self.requests.append(message)

            def recv(self):
                return individual(self.individual_id, 1.0)

        workers = [Worker(9), Worker(2)]
        result = dal_remote_population(workers, generation=1, parent_pool=1, elites=1)
        self.assertEqual([row["individual_id"] for row in result["individuals"]], [9, 2])
        self.assertEqual([w.requests for w in workers], [["get_over_here"], ["get_over_here"]])

    def test_generic_tie_policy_is_explicit_and_preserves_default(self):
        population = [IndividualObservation(i, 1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0) for i in (9, 2)]
        kwargs = {"generation": 1, "parent_pool": 1, "elites": 1}
        default = population_snapshot(population, **kwargs)
        stable = population_snapshot(population, **kwargs, tie_break="input_order")
        self.assertEqual([row["individual_id"] for row in default["individuals"]], [2, 9])
        self.assertEqual([row["individual_id"] for row in stable["individuals"]], [9, 2])
        with self.assertRaises(ValueError):
            population_snapshot(population, **kwargs, tie_break="unknown")
