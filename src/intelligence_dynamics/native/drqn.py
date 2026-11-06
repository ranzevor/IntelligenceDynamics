"""DRQN action and replay-carry execution."""

from copy import deepcopy
from dataclasses import dataclass

from ._state import state_snapshot
from ._values import read_field


@dataclass(frozen=True)
class DRQNDecision:
    state: object
    action: object
    replay_carry: object
    initialized_carry: bool


class DRQNPolicy:
    """Call native phi, act and carry conversion without modifying the agent."""

    def __init__(self, agent, *, expand_batch):
        for name in ("phi", "act", "_carry_for_buffer"):
            if not callable(getattr(agent, name, None)):
                raise ValueError(f"DRQN agent requires callable {name}")
        if not callable(expand_batch):
            raise ValueError("expand_batch must add a leading batch dimension")
        self.agent, self.expand_batch = agent, expand_batch

    def act(self, state, observation, scalars):
        """Return the next state, action and pre-action carry for new replay."""
        working = deepcopy(state)
        original = state_snapshot(working)["sha256"]
        carry = read_field(working, ("carry",))
        initialized = carry is None
        if initialized:
            output = self.agent.phi(
                read_field(working, ("params",)),
                self.expand_batch(observation),
                scalars=self.expand_batch(scalars),
                carry=None,
            )
            if not isinstance(output, tuple) or len(output) < 3 or output[2] is None:
                raise ValueError("phi must return a non-None initial carry as its third result")
            carry = output[2]
            if state_snapshot(working)["sha256"] != original:
                raise ValueError("phi changed the supplied agent state")
        replay_carry = deepcopy(self.agent._carry_for_buffer(carry))
        if replay_carry is None:
            raise ValueError("replay carry cannot be None")
        state_snapshot(replay_carry)
        result = self.agent.act(working, observation, scalars)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("native act must return state and action")
        next_state, action = result
        if read_field(next_state, ("carry",)) is None:
            raise ValueError("native act did not initialize live recurrent carry")
        state_snapshot(next_state)
        state_snapshot(action)
        return DRQNDecision(deepcopy(next_state), deepcopy(action), replay_carry, initialized)


@dataclass(frozen=True)
class ArrayOperations:
    expand_batch: object
    replay_batch: object
    float_scalar: object
    boolean: object
    clip: object

    def __post_init__(self):
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("array operations must be callable")

    @classmethod
    def jax(cls):
        """Use JAX arrays and its pytree mapper for replay batch axes."""
        import jax
        import jax.numpy as jnp

        return cls(
            expand_batch=lambda value: jnp.expand_dims(value, 0),
            replay_batch=lambda value: jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (1, 1, *leaf.shape)), value
            ),
            float_scalar=jnp.float32,
            boolean=jnp.bool_,
            clip=jnp.clip,
        )


@dataclass(frozen=True)
class ExecutionState:
    agent_state: object
    env_state: object
    last_action: object
    num_episodes: int = 0
    num_steps: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    terminated: bool = False


@dataclass(frozen=True)
class StepAudit:
    agent_steps_before: int
    agent_steps_after: int
    updates_before: int
    updates_after: int
    update_due: bool
    replay_ready: bool
    learned: bool
    native_frozen: bool
    initialized_carry: bool


@dataclass(frozen=True)
class NativeTransition:
    step: int
    reward: float
    action: object
    terminated: bool
    truncated: bool
    extra: object
    metrics: object
    audit: StepAudit


class DRQNExecutor:
    """Execute one native environment/replay/action/optimizer transition."""

    def __init__(self, agent, environment, *, operations=None):
        from .foragax import ForagaxAdapter

        self.operations = operations if operations is not None else ArrayOperations.jax()
        if not isinstance(self.operations, ArrayOperations):
            raise ValueError("operations must be ArrayOperations")
        required = (
            "_start",
            "_compute_reward_trace",
            "encode_scalar_features",
            "_update_state_with_metrics",
            "_advance_update_clock",
        )
        for name in required:
            if not callable(getattr(agent, name, None)):
                raise ValueError(f"native agent requires callable {name}")
        for name in ("add", "can_sample"):
            if not callable(getattr(getattr(agent, "buffer", None), name, None)):
                raise ValueError(f"native replay buffer requires callable {name}")
        if not isinstance(environment, ForagaxAdapter) and any(
            not callable(getattr(environment, name, None)) for name in ("_start", "_step")
        ):
            raise ValueError("environment requires native _start and _step or ForagaxAdapter")
        self.agent, self.environment = agent, environment
        self.policy = DRQNPolicy(agent, expand_batch=self.operations.expand_batch)

    def initialize(self, *, seed=0):
        """Start a fresh native environment and agent using their supplied states."""
        from ._values import count
        from .foragax import ForagaxAdapter

        count(seed, "seed")
        if isinstance(self.environment, ForagaxAdapter):
            self.environment.reset(seed)
            env_state = self.environment.snapshot()
            observation = env_state.observation
        else:
            env_state, observation = self.environment._start(deepcopy(self.environment.state))
        agent_state, action = self.agent._start(deepcopy(self.agent.state), observation)
        state = ExecutionState(deepcopy(agent_state), deepcopy(env_state), deepcopy(action))
        self._counters(state.agent_state)
        state_snapshot(state)
        return state

    @staticmethod
    def _counters(state):
        from ._values import count

        return (
            count(read_field(state, ("steps",)), "agent steps"),
            count(read_field(state, ("updates",)), "agent updates"),
            count(read_field(state, ("hypers", "update_freq")), "update frequency", 1),
        )

    def step(self, state, *, learning=True):
        """Advance one interaction; schedule learning from the pre-step clock."""
        import math
        from collections.abc import Mapping
        from dataclasses import replace

        from ._values import count, finite, scalar
        from .foragax import ForagaxAdapter

        if not isinstance(state, ExecutionState) or type(learning) is not bool:
            raise ValueError("step requires ExecutionState and boolean learning")
        if state.terminated or state.last_action is None:
            raise ValueError("state must be started and nonterminal")
        working = deepcopy(state)
        old_steps, old_updates, frequency = self._counters(working.agent_state)
        if old_steps == 0:
            raise ValueError("native continuation requires a positive initialized agent clock")
        training_before = {
            name: state_snapshot(read_field(working.agent_state, (name,)))["sha256"]
            for name in ("params", "target_params", "optim")
        }
        threshold = scalar(read_field(working.agent_state, ("hypers", "freeze_steps")))
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or math.isnan(threshold)
            or threshold < 0
        ):
            raise ValueError("freeze_steps must be nonnegative or positive infinity")
        native_frozen = old_steps >= threshold
        ops = self.operations
        if isinstance(self.environment, ForagaxAdapter):
            self.environment.restore(working.env_state)
            _, reward, term, extra = self.environment.step(working.last_action)
            env_state = self.environment.snapshot()
            observation, trunc = env_state.observation, False
        else:
            env_state, result = self.environment._step(working.env_state, working.last_action)
            if not isinstance(result, tuple) or len(result) != 5:
                raise ValueError("native environment _step must return five interaction fields")
            observation, reward, term, trunc, extra = result
        reward = finite(reward, "reward")
        term, trunc = scalar(term), scalar(trunc)
        if type(term) is not bool or type(trunc) is not bool or not isinstance(extra, Mapping):
            raise ValueError("native termination flags and extra have invalid types")
        agent_state = working.agent_state
        collected = dict(read_field(agent_state, ("last_timestep",)))
        clipped = (
            ops.clip(reward, -self.agent.reward_clip, self.agent.reward_clip)
            if self.agent.reward_clip > 0
            else ops.float_scalar(reward)
        )
        discount = (
            0.0
            if term
            else finite(extra.get("gamma", 1.0), "environment discount")
            * finite(read_field(agent_state, ("hypers", "gamma")), "agent discount")
        )
        if discount < 0 or discount > 1:
            raise ValueError("combined discount must lie in [0, 1]")
        collected.update(r=clipped, gamma=ops.float_scalar(discount))
        buffer_state = self.agent.buffer.add(agent_state.buffer_state, ops.replay_batch(collected))
        agent_state = replace(agent_state, buffer_state=buffer_state, last_timestep=collected)
        agent_state, reward_trace = self.agent._compute_reward_trace(agent_state, clipped)
        initialized = False
        next_action = working.last_action
        if not term:
            image = observation["image"] if isinstance(observation, Mapping) else observation
            hint = observation.get("hint") if isinstance(observation, Mapping) else None
            features = self.agent.encode_scalar_features(
                collected["a"], clipped, reward_trace, hint
            )
            decision = self.policy.act(agent_state, image, features)
            agent_state, next_action, initialized = (
                decision.state,
                decision.action,
                decision.initialized_carry,
            )
            collected.update(
                x=image,
                a=next_action,
                scalars=features,
                carry=decision.replay_carry,
                reset=ops.boolean(False),
            )
            agent_state = replace(agent_state, last_timestep=collected)
        ready = scalar(self.agent.buffer.can_sample(agent_state.buffer_state))
        if type(ready) is not bool:
            raise ValueError("buffer.can_sample must return a scalar boolean")
        due = old_steps % frequency == 0
        learned = learning and not native_frozen and due and ready
        if learned:
            agent_state = self.agent._update_state_with_metrics(agent_state)
        if not native_frozen:
            agent_state = self.agent._advance_update_clock(agent_state)
        new_steps, new_updates, new_frequency = self._counters(agent_state)
        if (
            new_steps != old_steps + int(not native_frozen)
            or new_updates != old_updates + int(learned)
            or new_frequency != frequency
        ):
            raise ValueError("native step/update counters violate the selected schedule")
        if not learned and any(
            state_snapshot(read_field(agent_state, (name,)))["sha256"] != value
            for name, value in training_before.items()
        ):
            raise ValueError("parameters or optimizer changed on a no-learning transition")
        metrics = deepcopy(read_field(agent_state, ("metrics",)))
        state_snapshot(metrics)
        for name in ("loss", "weight_change", "abs_td_error"):
            finite(read_field(metrics, (name,)), name)
        total_steps = count(working.total_steps, "total steps") + 1
        total_reward = finite(working.total_reward + reward, "total reward")
        result = ExecutionState(
            deepcopy(agent_state),
            deepcopy(env_state),
            deepcopy(next_action),
            working.num_episodes + int(term),
            working.num_steps + 1,
            total_reward,
            total_steps,
            term or trunc,
        )
        state_snapshot(result)
        audit = StepAudit(
            old_steps,
            new_steps,
            old_updates,
            new_updates,
            due,
            ready,
            learned,
            native_frozen,
            initialized,
        )
        transition = NativeTransition(
            total_steps,
            reward,
            deepcopy(working.last_action),
            term,
            trunc,
            deepcopy(extra),
            metrics,
            audit,
        )
        return result, transition
