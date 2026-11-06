import copy
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace

from intelligence_dynamics.native import (
    DRQNPolicy,
    ForagaxAdapter,
    dal_individual,
    dal_population,
)


class DALTests(unittest.TestCase):
    def individual(self, identifier=3):
        return SimpleNamespace(
            individual_id=identifier,
            normalised_fitness=2.0,
            connectivity_grid_a=[[0, 1], [-1, 0]],
            connectivity_grid_m=[[0, 0], [2, 0]],
            io_list_a=[(0, 1, 0.2), (1, 0, 0.4)],
            io_list_m=[(1, 0)],
            rl_learning_rate=0.03,
            is_unmutated_copy=False,
            stats={
                "total_weight_change_nm": 0.5,
                "total_weight_change_rl": 0.7,
                "mean_active_rl_learning_rate": 0.02,
                "mutations_applied": ["add"],
            },
        )

    def test_extracts_live_and_genotype_diagnostics_without_mutation(self):
        individual = self.individual()
        before = copy.deepcopy(vars(individual))
        result = dal_individual(individual)
        self.assertEqual(result.activation_projections, 2)
        self.assertEqual(result.modulatory_projections, 1)
        self.assertEqual(result.live_modulatory_projections, 1)
        self.assertAlmostEqual(result.mean_susceptibility, 0.3)
        self.assertEqual(result.weight_change_nm, 0.5)
        self.assertEqual(result.weight_change_rl, 0.7)
        self.assertEqual(vars(individual), before)
        individual.stats["mutations_applied"].append("later")
        self.assertEqual(result.mutations, ("add",))

    def test_population_reads_all_individuals_and_retains_empty_susceptibility(self):
        first, second = self.individual(1), self.individual(2)
        second.normalised_fitness, second.io_list_a = 3.0, []
        result = dal_population([first, second], generation=0, parent_pool=1, elites=1)
        self.assertEqual([x["individual_id"] for x in result["individuals"]], [2, 1])
        self.assertIsNone(result["individuals"][0]["mean_susceptibility"])
        self.assertTrue(result["individuals"][0]["selected_parent"])
        self.assertFalse(result["individuals"][1]["elite"])

    def test_invalid_individual_contracts(self):
        for field, value in (
            ("stats", {}),
            ("individual_id", True),
            ("connectivity_grid_a", [float("nan")]),
            ("io_list_a", [(1, 2)]),
            ("is_unmutated_copy", 1),
        ):
            individual = self.individual()
            setattr(individual, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                dal_individual(individual)
        with self.assertRaises(ValueError):
            dal_population(
                [self.individual(), self.individual()], generation=0, parent_pool=1, elites=0
            )


class NativeSpace:
    n = 2

    def sample(self, key):
        return key % self.n


class NativeEnvironment:
    default_params = {"offset": 1}

    def __init__(self):
        self.calls = []

    def action_space(self, params):
        self.calls.append(("action_space", params))
        return NativeSpace()

    def reset(self, key, params):
        self.calls.append(("reset", key, params))
        return [0], {"position": 0, "reset_key": key}

    def step(self, key, state, action, params):
        self.calls.append(("step", key, action, params))
        updated = {**state, "position": state["position"] + action + params["offset"]}
        return [updated["position"]], updated, float(action), False, {"key": key}


def adapter(environment=None, identity="synthetic"):
    return ForagaxAdapter(
        environment or NativeEnvironment(),
        key=lambda seed: seed * 2,
        split=lambda key, count: (key + 1, key + 10),
        identity=identity,
    )


class ForagaxTests(unittest.TestCase):
    def test_native_calls_key_advance_and_exact_resume(self):
        native = NativeEnvironment()
        environment = adapter(native)
        self.assertEqual(environment.reset(3), [0])
        self.assertEqual(native.calls[0], ("reset", 16, {"offset": 1}))
        initial = environment.snapshot()
        expected = environment.step(1)
        self.assertEqual(expected[:3], ([2], 1.0, False))
        environment.restore(initial)
        self.assertEqual(environment.step(1), expected)
        self.assertEqual(environment.snapshot().steps, 1)
        action = environment.sample_action()
        self.assertIn(action, (0, 1))
        self.assertEqual(environment.snapshot().key, 9)

    def test_registry_factory_receives_explicit_configuration(self):
        calls = []

        def factory(name, **options):
            calls.append((name, options))
            return NativeEnvironment()

        env = ForagaxAdapter.from_factory(
            factory,
            "Synthetic-v1",
            options={"aperture_size": 3},
            key=lambda seed: seed,
            split=lambda key, count: (key + 1, key + 2),
        )
        env.reset(0)
        self.assertEqual(calls, [("Synthetic-v1", {"aperture_size": 3})])

    def test_reset_required_actions_and_identity(self):
        env = adapter()
        with self.assertRaises(ValueError):
            env.step(0)
        env.reset(0)
        for action in (-1, 2, True, 0.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        with self.assertRaises(ValueError):
            adapter(identity="other").restore(env.snapshot())
        env.restore(replace(env.snapshot(), terminated=True))
        with self.assertRaises(ValueError):
            env.step(0)

    def test_rejects_nonfinite_reward_and_invalid_native_tuple(self):
        native, env = NativeEnvironment(), None
        env = adapter(native)
        env.reset(0)
        native.step = lambda *args: ([0], {}, float("nan"), False, {})
        with self.assertRaises(ValueError):
            env.step(0)
        native.step = lambda *args: ([0], {}, 1.0, 0, {})
        with self.assertRaises(ValueError):
            env.step(0)


@dataclass
class SmallState:
    params: dict
    carry: object
    key: int = 1


class SmallAgent:
    def __init__(self):
        self.calls = []

    def phi(self, params, image, *, scalars, carry):
        self.calls.append(("phi", copy.deepcopy(params), image, scalars, carry))
        return None, None, [[params["initial"]]]

    def _carry_for_buffer(self, carry):
        self.calls.append(("carry", copy.deepcopy(carry)))
        return carry[0]

    def act(self, state, observation, scalars):
        self.calls.append(("act", state.carry))
        carry = state.params["initial"] if state.carry is None else state.carry[0][0]
        return replace(state, carry=[[carry + 1]], key=state.key + 1), 1


class DRQNPolicyTests(unittest.TestCase):
    def test_current_learned_initial_carry_is_used_for_future_replay(self):
        agent = SmallAgent()
        policy = DRQNPolicy(agent, expand_batch=lambda x: [x])
        state = SmallState({"initial": 7}, None)
        result = policy.act(state, [2], [3])
        self.assertEqual(result.replay_carry, [7])
        self.assertEqual(result.state.carry, [[8]])
        self.assertTrue(result.initialized_carry)
        self.assertIsNone(state.carry)
        self.assertEqual([call[0] for call in agent.calls], ["phi", "carry", "act"])
        self.assertIsNone(agent.calls[-1][1])

    def test_existing_carry_skips_phi(self):
        agent = SmallAgent()
        result = DRQNPolicy(agent, expand_batch=lambda x: [x]).act(
            SmallState({"initial": 7}, [[4]]), [0], []
        )
        self.assertEqual(result.replay_carry, [4])
        self.assertEqual([call[0] for call in agent.calls], ["carry", "act"])

    def test_rejects_mutating_phi_and_missing_live_carry(self):
        agent = SmallAgent()

        def bad_phi(params, *args, **kwargs):
            params["initial"] = 99
            return None, None, [[99]]

        agent.phi = bad_phi
        state = SmallState({"initial": 7}, None)
        with self.assertRaises(ValueError):
            DRQNPolicy(agent, expand_batch=lambda x: [x]).act(state, [], [])
        self.assertEqual(state.params["initial"], 7)
        agent = SmallAgent()
        agent.act = lambda state, *args: (replace(state, carry=None), 0)
        with self.assertRaises(ValueError):
            DRQNPolicy(agent, expand_batch=lambda x: [x]).act(state, [], [])


class NativeDALObjectTests(unittest.TestCase):
    def raw(self):
        value = DALTests().individual()
        del value.stats
        del value.is_unmutated_copy
        value.total_dw_list_nm = [[[1.0, -2.0]], [[0.5]]]
        value.total_dw_list_rl = [[-4.0, 2.0]]
        value.mean_active_rl_learning_rate = 0.04
        value.mutations_applied = ["projection_insertion_A[add]"]
        return value

    def test_raw_nmnn_object_does_not_require_instrumented_stats(self):
        individual = self.raw()
        before = copy.deepcopy(vars(individual))
        result = dal_individual(individual, unmutated=True)
        self.assertEqual(result.weight_change_nm, 3.5)
        self.assertEqual(result.weight_change_rl, 6.0)
        self.assertTrue(result.unmutated)
        self.assertEqual(vars(individual), before)
        with self.assertRaises(ValueError):
            dal_individual(individual)

    def test_spnn_send_poll_recv_calls_and_identity_validation(self):
        from intelligence_dynamics.native import dal_remote_population

        calls = []
        value = self.raw()
        worker = SimpleNamespace(
            individual_id=3,
            is_unmutated_copy=False,
            send=lambda command: calls.append(("send", command)),
            recv=lambda: (calls.append(("recv",)), value)[1],
            pipe=SimpleNamespace(poll=lambda timeout: (calls.append(("poll", timeout)), True)[1]),
        )
        result = dal_remote_population([worker], generation=1, parent_pool=1, elites=0, timeout=0.5)
        self.assertEqual(result["individuals"][0]["weight_change_nm"], 3.5)
        self.assertEqual(calls, [("send", "get_over_here"), ("poll", 0.5), ("recv",)])
        worker.individual_id = 4
        with self.assertRaises(ValueError):
            dal_remote_population([worker], generation=1, parent_pool=1, elites=0)
        worker.pipe.poll = lambda timeout: False
        with self.assertRaises(TimeoutError):
            dal_remote_population([worker], generation=1, parent_pool=1, elites=0)


class DirectForagaxTransitionTests(unittest.TestCase):
    def test_explicit_step_env_does_not_patch_native_termination_methods(self):
        native = NativeEnvironment()
        native.step_env = native.step
        native.step = lambda *args: (_ for _ in ()).throw(RuntimeError("wrapper unavailable"))
        environment = ForagaxAdapter(
            native,
            key=lambda seed: seed,
            split=lambda key, count: (key + 1, key + 2),
            transition="step_env",
        )
        environment.reset(0)
        self.assertEqual(environment.step(0)[0], [1])
        self.assertFalse(hasattr(native, "is_terminated"))
        with self.assertRaises(ValueError):
            ForagaxAdapter(
                native,
                key=lambda seed: seed,
                split=lambda key, count: (key + 1, key + 2),
                transition="unknown",
            )
