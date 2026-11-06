import copy
import json
import pickle
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path

from intelligence_dynamics.native import (
    ArrayOperations,
    CheckpointCodec,
    CheckpointStore,
    DRQNExecutor,
    NativeCheckpoint,
    NativeLifecycle,
    execution_state,
    verify_common_history,
)
from intelligence_dynamics.native._state import state_snapshot


@dataclass
class Hypers:
    gamma: float = 0.9
    epsilon: float = 0.3
    update_freq: int = 2
    freeze_steps: float = float("inf")


@dataclass
class Metrics:
    loss: float = 0.0
    weight_change: float = 0.0
    abs_td_error: float = 0.0


@dataclass
class AgentState:
    params: dict = field(default_factory=lambda: {"weight": 2.0, "initial": 5.0})
    target_params: dict = field(default_factory=lambda: {"weight": 2.0, "initial": 5.0})
    optim: tuple = ("adam", 0)
    buffer_state: tuple = ()
    key: int = 7
    carry: object = None
    last_timestep: dict = field(default_factory=dict)
    steps: int = 0
    updates: int = 0
    hypers: Hypers = field(default_factory=Hypers)
    metrics: Metrics = field(default_factory=Metrics)
    reward_trace: float = 0.0
    hint_trace: tuple = ()


@dataclass
class EnvState:
    time: int = 0
    key: int = 10


class Buffer:
    def __init__(self, calls):
        self.calls = calls

    def add(self, state, batch):
        self.calls.append("buffer.add")
        if set(batch) != {"batch"} or len(batch["batch"]) != 1 or len(batch["batch"][0]) != 1:
            raise ValueError("missing replay batch/time axes")
        return (*state, copy.deepcopy(batch["batch"][0][0]))

    def can_sample(self, state):
        self.calls.append("buffer.can_sample")
        return len(state) >= 2

    def sample(self, state, key):
        self.calls.append("buffer.sample")
        return state[-1]


class Agent:
    reward_clip = 1.5

    def __init__(self):
        self.calls = []
        self.state = AgentState()
        self.buffer = Buffer(self.calls)

    def phi(self, params, observation, *, scalars, carry):
        self.calls.append("phi")
        return None, None, [[params["initial"]]]

    def _carry_for_buffer(self, carry):
        self.calls.append("carry_for_buffer")
        return copy.deepcopy(carry[0])

    def encode_scalar_features(self, action, reward, trace, hint=None):
        self.calls.append("encode_scalar_features")
        return [action, reward, trace]

    def act(self, state, observation, scalars):
        self.calls.append("act:none" if state.carry is None else "act:carry")
        previous = state.params["initial"] if state.carry is None else state.carry[0][0]
        return replace(state, carry=[[previous + 1]], key=state.key + 1), state.key % 2

    def _start(self, state, observation):
        self.calls.append("start")
        scalars = self.encode_scalar_features(-1, 0.0, 0.0)
        state, action = self.act(state, observation, scalars)
        state = replace(
            state,
            last_timestep={
                "x": observation,
                "a": action,
                "scalars": scalars,
                "carry": self._carry_for_buffer(state.carry),
                "reset": True,
            },
        )
        return self._advance_update_clock(state), action

    def _compute_reward_trace(self, state, reward):
        self.calls.append("reward_trace")
        value = state.reward_trace * 0.9 + reward * 0.1
        return replace(state, reward_trace=value), value

    def _update_state_with_metrics(self, state):
        self.calls.append("update")
        sample = self.buffer.sample(state.buffer_state, state.key + 1)
        self.calls.append("optimizer.update")
        params = {
            **state.params,
            "weight": state.params["weight"] + sample["r"],
            "initial": state.params["initial"] + 0.25,
        }
        updates = state.updates + 1
        target = params if updates % 2 == 0 else state.target_params
        self.calls.append("target.refresh" if updates % 2 == 0 else "target.retain")
        return replace(
            state,
            params=params,
            target_params=copy.deepcopy(target),
            optim=("adam", state.optim[1] + 1),
            updates=updates,
            key=state.key + 1,
            metrics=Metrics(0.5, 1.0, 0.25),
        )

    def _advance_update_clock(self, state):
        self.calls.append("clock")
        return replace(
            state,
            steps=state.steps + 1,
            hypers=replace(state.hypers, epsilon=max(0.0, state.hypers.epsilon - 0.01)),
        )


class Environment:
    def __init__(self, terminal_after=None):
        self.state = EnvState()
        self.calls = []
        self.terminal_after = terminal_after

    def _start(self, state):
        self.calls.append("start")
        return replace(state, key=state.key + 1), [0]

    def _step(self, state, action):
        self.calls.append(("step", action))
        updated = replace(state, time=state.time + 1, key=state.key + 2)
        terminal = self.terminal_after is not None and updated.time >= self.terminal_after
        return updated, (
            [updated.time],
            float(action + 1),
            terminal,
            False,
            {"gamma": 1.0, "biome_id": updated.time // 2},
        )


def make_lifecycle():
    agent, environment = Agent(), Environment()
    operations = ArrayOperations(
        lambda value: [value],
        lambda value: {"batch": [[copy.deepcopy(value)]]},
        float,
        bool,
        lambda value, low, high: max(low, min(high, value)),
    )
    executor = DRQNExecutor(agent, environment, operations=operations)
    return NativeLifecycle(executor), agent, environment


def codec():
    return CheckpointCodec("synthetic-local-pickle", pickle.dumps, pickle.loads)


class NativeExecutionTests(unittest.TestCase):
    def test_default_execution_calls_replay_policy_optimizer_and_prng(self):
        lifecycle, agent, environment = make_lifecycle()
        initial = lifecycle.start()
        self.assertEqual(initial.state.agent_state.steps, 1)
        original = initial.sha256
        agent.calls.clear()
        after = lifecycle.run(initial, interactions=4)
        state = after.state.agent_state
        self.assertEqual(state.steps, 5)
        self.assertEqual(state.updates, 2)
        self.assertEqual(state.optim, ("adam", 2))
        self.assertEqual(state.target_params, state.params)
        self.assertEqual(state.key, initial.state.agent_state.key + 6)
        self.assertEqual(len(state.buffer_state), 4)
        self.assertEqual(agent.calls.count("buffer.add"), 4)
        self.assertEqual(agent.calls.count("buffer.sample"), 2)
        self.assertEqual(agent.calls.count("optimizer.update"), 2)
        self.assertEqual(agent.calls.count("target.refresh"), 1)
        self.assertEqual([row.audit.learned for row in after.history], [False, True, False, True])
        self.assertEqual(initial.sha256, original)
        self.assertEqual(environment.calls[0], "start")
        self.assertEqual(after.state.env_state.time, 4)
        self.assertEqual(state.buffer_state[0]["r"], 1.5)
        self.assertAlmostEqual(state.buffer_state[0]["gamma"], 0.9)

    def test_checkpoint_reset_first_step_and_continuation(self):
        lifecycle, agent, _ = make_lifecycle()
        common = lifecycle.run(lifecycle.start(), interactions=3)
        self.assertEqual(common.state.agent_state.steps % 2, 0)
        original_buffer = copy.deepcopy(common.state.agent_state.buffer_state)
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, codec())
            store.save("common", common)
            continued = store.clone("common", "continued")
            before_reset = store.clone("common", "reset-copy")
            self.assertEqual(
                (Path(root) / "common/payload.bin").read_bytes(),
                (Path(root) / "reset-copy/payload.bin").read_bytes(),
            )
            reset = lifecycle.intervene(before_reset, {("agent_state", "carry"): None})
            self.assertEqual(reset.state.agent_state.buffer_state, original_buffer)
            self.assertEqual(reset.state.agent_state.params, common.state.agent_state.params)
            self.assertEqual(reset.state.env_state, common.state.env_state)
            self.assertEqual(reset.state.agent_state.key, common.state.agent_state.key)
            self.assertEqual(reset.history, common.history)
            store.save("reset", reset)
            reloaded = store.load("reset")
            self.assertEqual(reloaded.sha256, reset.sha256)
            agent.calls.clear()
            first = lifecycle.run(reloaded, interactions=1)
            audit = first.history[-1].audit
            self.assertEqual(audit.agent_steps_after - audit.agent_steps_before, 1)
            self.assertEqual(audit.updates_after - audit.updates_before, 1)
            self.assertTrue(audit.initialized_carry)
            self.assertIn("phi", agent.calls)
            self.assertIn("act:none", agent.calls)
            self.assertEqual(
                first.state.agent_state.last_timestep["carry"],
                [common.state.agent_state.params["initial"]],
            )
            self.assertEqual(first.state.agent_state.buffer_state[:3], original_buffer)
            store.save("first", first)
            resumed = lifecycle.run(store.load("first"), interactions=2)
            control = lifecycle.run(continued, interactions=3)
            self.assertEqual(resumed.state.total_steps, 6)
            self.assertEqual(verify_common_history(resumed, control, boundary=3)["boundary"], 3)
            self.assertEqual(store.load("common").sha256, common.sha256)

    def test_disabled_learning_preserves_parameters_optimizer_but_collects(self):
        lifecycle, _, _ = make_lifecycle()
        common = lifecycle.run(lifecycle.start(), interactions=2)
        result = lifecycle.run(common, interactions=3, learning=False)
        for name in ("params", "target_params", "optim", "updates"):
            self.assertEqual(
                getattr(result.state.agent_state, name), getattr(common.state.agent_state, name)
            )
        self.assertEqual(len(result.state.agent_state.buffer_state), 5)
        self.assertEqual(result.state.agent_state.steps, common.state.agent_state.steps + 3)
        self.assertGreater(result.state.agent_state.key, common.state.agent_state.key)
        self.assertNotEqual(result.state.agent_state.carry, common.state.agent_state.carry)

    def test_selected_freeze_restores_only_selected_fields(self):
        lifecycle, _, _ = make_lifecycle()
        initial = lifecycle.start()
        plan = lifecycle.freeze(
            initial,
            [("agent_state", "params"), ("agent_state", "target_params"), ("agent_state", "optim")],
        )
        after = lifecycle.run(initial, interactions=4, freeze=plan)
        self.assertEqual(after.state.agent_state.params, initial.state.agent_state.params)
        self.assertEqual(after.state.agent_state.optim, initial.state.agent_state.optim)
        self.assertEqual(after.state.agent_state.updates, 2)
        self.assertEqual(len(after.state.agent_state.buffer_state), 4)
        self.assertEqual(after.state.total_steps, 4)

    def test_native_freeze_boundary_preserves_clock_and_epsilon(self):
        lifecycle, agent, _ = make_lifecycle()
        agent.state.hypers.freeze_steps = 3
        initial = lifecycle.start()
        after = lifecycle.run(initial, interactions=5)
        self.assertEqual(after.state.agent_state.steps, 3)
        self.assertEqual(after.state.agent_state.updates, 1)
        self.assertEqual(
            [x.audit.native_frozen for x in after.history], [False, False, True, True, True]
        )
        self.assertEqual(len(after.state.agent_state.buffer_state), 5)
        self.assertAlmostEqual(after.state.agent_state.hypers.epsilon, 0.27)

    def test_terminal_collection_stops_and_uses_zero_discount(self):
        lifecycle, _, environment = make_lifecycle()
        environment.terminal_after = 2
        result = lifecycle.run(lifecycle.start(), interactions=5)
        self.assertEqual(result.state.total_steps, 2)
        self.assertTrue(result.state.terminated)
        self.assertEqual(result.state.agent_state.buffer_state[-1]["gamma"], 0.0)
        self.assertEqual(result.state.num_episodes, 1)
        with self.assertRaises(ValueError):
            lifecycle.executor.step(result.state)

    def test_rejects_wrong_clock_or_unselected_training_change(self):
        lifecycle, agent, _ = make_lifecycle()
        initial = lifecycle.start()
        agent._advance_update_clock = lambda state: replace(state, steps=state.steps + 2)
        with self.assertRaisesRegex(ValueError, "schedule"):
            lifecycle.run(initial, interactions=1)
        lifecycle, agent, _ = make_lifecycle()
        initial = lifecycle.start()
        original = agent.act

        def corrupt(state, *args):
            state, action = original(state, *args)
            state.params["weight"] = -999
            return state, action

        agent.act = corrupt
        with self.assertRaisesRegex(ValueError, "no-learning"):
            lifecycle.run(initial, interactions=1, learning=False)
        self.assertEqual(initial.state.agent_state.params["weight"], 2.0)

    def test_intervention_rejects_counters_overlap_missing_and_nochange(self):
        lifecycle, _, _ = make_lifecycle()
        initial = lifecycle.start()
        for values in (
            {("agent_state", "steps"): 0},
            {("unknown",): 0},
            {("agent_state", "carry"): initial.state.agent_state.carry},
            {("agent_state", "params"): {}, ("agent_state", "params", "weight"): 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                lifecycle.intervene(initial, values)
        with self.assertRaises(ValueError):
            lifecycle.freeze(initial, [("total_steps",)])

    def test_common_history_checks_missing_and_changed_data(self):
        lifecycle, _, _ = make_lifecycle()
        common = lifecycle.run(lifecycle.start(), interactions=2)
        changed = NativeCheckpoint(
            common.state, (replace(common.history[0], reward=99.0), common.history[1])
        )
        with self.assertRaisesRegex(ValueError, "histories differ"):
            verify_common_history(common, changed, boundary=2)
        with self.assertRaises(ValueError):
            verify_common_history(common, changed, boundary=3)
        with self.assertRaises(ValueError):
            NativeCheckpoint(common.state)

    def test_native_glue_copy_has_all_fields_and_no_aliasing(self):
        lifecycle, _, _ = make_lifecycle()
        state = lifecycle.start().state
        result = execution_state(state)
        self.assertEqual(state_snapshot(result)["sha256"], state_snapshot(state)["sha256"])
        result.agent_state.params["weight"] = 99
        self.assertEqual(state.agent_state.params["weight"], 2.0)


class NativeCheckpointTests(unittest.TestCase):
    def checkpoint(self):
        lifecycle, _, _ = make_lifecycle()
        return lifecycle.run(lifecycle.start(), interactions=2)

    def test_tampering_rejected_before_decode(self):
        value = self.checkpoint()
        called = []
        trusted = CheckpointCodec(
            "test", pickle.dumps, lambda raw: (called.append(True), pickle.loads(raw))[1]
        )
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, trusted)
            store.save("sample", value)
            called.clear()
            path = Path(root) / "sample/payload.bin"
            raw = path.read_bytes()
            path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.assertRaisesRegex(ValueError, "digest"):
                store.load("sample")
            self.assertEqual(called, [])

    def test_lossy_codec_rejected_before_save(self):
        value = self.checkpoint()

        def decode(raw):
            checkpoint = pickle.loads(raw)
            checkpoint.state.agent_state.buffer_state = ()
            return checkpoint

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, CheckpointCodec("lossy", pickle.dumps, decode))
            with self.assertRaisesRegex(ValueError, "changed state"):
                store.save("sample", value)
            self.assertFalse((Path(root) / "sample").exists())

    def test_size_names_overwrite_and_codec_identity(self):
        value = self.checkpoint()
        with tempfile.TemporaryDirectory() as root:
            small = CheckpointStore(root, codec(), max_bytes=1)
            with self.assertRaises(ValueError):
                small.save("sample", value)
            store = CheckpointStore(root, codec())
            for name in ("../sample", "x/y", "", "."):
                with self.assertRaises(ValueError):
                    store.save(name, value)
            store.save("sample", value)
            with self.assertRaises(FileExistsError):
                store.save("sample", value)
            manifest = Path(root) / "sample/manifest.json"
            data = json.loads(manifest.read_text())
            data["codec"] = "other"
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                store.load("sample")

    def test_symlink_is_not_a_checkpoint_container(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "target").mkdir()
            from tests.support import create_symlink

            create_symlink(self, directory / "link", directory / "target", directory=True)
            with self.assertRaises(ValueError):
                CheckpointStore(directory / "link", codec())


class StrictNativeManifestTests(unittest.TestCase):
    def test_duplicate_manifest_keys_and_nonfinite_json_are_rejected(self):
        lifecycle, _, _ = make_lifecycle()
        for corruption in ("duplicate", "nan"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as root:
                store = CheckpointStore(root, codec())
                store.save("sample", lifecycle.start())
                path = Path(root) / "sample/manifest.json"
                original = path.read_text()
                if corruption == "duplicate":
                    path.write_text('{"codec":' + json.dumps(codec().name) + "," + original[1:])
                else:
                    path.write_text(original.replace('"bytes":', '"bytes":NaN,"unused":'))
                with self.assertRaises(ValueError):
                    store.load("sample")


class NativeAdapterIntegrationTests(unittest.TestCase):
    def test_foragax_adapter_runs_through_default_drqn_executor(self):
        from types import SimpleNamespace

        from intelligence_dynamics.native import ForagaxAdapter

        calls = []

        class NativeEnvironment:
            default_params = {}

            def action_space(self, params):
                return SimpleNamespace(n=2)

            def reset(self, key, params):
                calls.append(("reset", key))
                return [0], {"time": 0}

            def step(self, key, state, action, params):
                calls.append(("step", key, action))
                state = {"time": state["time"] + 1}
                return [state["time"]], state, 1.0, False, {"gamma": 1.0}

        environment = ForagaxAdapter(
            NativeEnvironment(), key=lambda seed: seed, split=lambda key, count: (key + 1, key + 2)
        )
        source, agent, _ = make_lifecycle()
        lifecycle = NativeLifecycle(
            DRQNExecutor(agent, environment, operations=source.executor.operations)
        )
        initial = lifecycle.start(seed=2)
        first = lifecycle.run(initial, interactions=2)
        repeated = lifecycle.run(initial, interactions=2)
        self.assertEqual(first.sha256, repeated.sha256)
        self.assertEqual(first.state.agent_state.updates, 1)
        self.assertEqual(calls[0], ("reset", 4))
        self.assertEqual(first.state.env_state.steps, 2)

    def test_zero_agent_clock_is_rejected_before_native_environment_call(self):
        lifecycle, _, environment = make_lifecycle()
        checkpoint = lifecycle.start()
        checkpoint.state.agent_state.steps = 0
        before = len(environment.calls)
        with self.assertRaisesRegex(ValueError, "positive initialized"):
            lifecycle.run(checkpoint, interactions=1)
        self.assertEqual(len(environment.calls), before)


class NamedOptimizerInterventionTests(unittest.TestCase):
    def test_namedtuple_paths_and_index_aliases_share_one_contract(self):
        from collections import namedtuple

        Optimizer = namedtuple("Optimizer", "count momentum")
        lifecycle, agent, _ = make_lifecycle()
        agent.state.optim = Optimizer(0, {"weight": 0.2})
        initial = lifecycle.start()
        changed = lifecycle.intervene(
            initial, {("agent_state", "optim", "momentum", "weight"): 0.4}
        )
        self.assertIs(type(changed.state.agent_state.optim), Optimizer)
        self.assertEqual(changed.state.agent_state.optim.momentum["weight"], 0.4)
        plan = lifecycle.freeze(initial, [("agent_state", "optim", "momentum", "weight")])
        after = lifecycle.run(initial, interactions=1, learning=False, freeze=plan)
        self.assertIs(type(after.state.agent_state.optim), Optimizer)
        self.assertEqual(after.state.agent_state.optim.momentum["weight"], 0.2)
        with self.assertRaises(ValueError):
            lifecycle.intervene(
                initial, {("agent_state", "optim", "count"): 1, ("agent_state", "optim", 0): 2}
            )
