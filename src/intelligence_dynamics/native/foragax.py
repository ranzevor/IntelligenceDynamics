"""Stateful adapter for the native Foragax reset and step APIs."""

from copy import deepcopy
from dataclasses import dataclass

from ..validation import integer, text
from ._state import state_snapshot
from ._values import count, finite, plain, scalar


@dataclass(frozen=True)
class ForagaxSnapshot:
    identity: str
    key: object
    environment_state: object
    observation: object
    steps: int
    terminated: bool


class ForagaxAdapter:
    """Execute a supplied Foragax instance with explicit PRNG operations."""

    def __init__(self, environment, *, key, split, params=None, identity=None, transition="step"):
        if not callable(key) or not callable(split):
            raise ValueError("key and split must be callable PRNG operations")
        if transition not in ("step", "step_env"):
            raise ValueError("transition must be step or step_env")
        self._transition = transition
        for method in ("reset", transition, "action_space"):
            if not callable(getattr(environment, method, None)):
                raise ValueError(f"environment requires callable {method}")
        self.environment = environment
        supplied_params = environment.default_params if params is None else params
        self._params_sha256 = state_snapshot(supplied_params)["sha256"]
        self._params = deepcopy(supplied_params)
        self._check_configuration()
        self.key_factory, self.split = key, split
        self._identity = text(
            identity or f"{type(environment).__module__}.{type(environment).__qualname__}",
            "identity",
        )
        self._identity += ":" + transition + ":" + self._params_sha256
        self._snapshot = None

    @property
    def transition(self):
        return self._transition

    @property
    def identity(self):
        return self._identity

    @property
    def params(self):
        """Return a detached copy of the fixed environment parameters."""
        self._check_configuration()
        params = deepcopy(self._params)
        if state_snapshot(params)["sha256"] != self._params_sha256:
            raise ValueError("parameter copying changed the environment configuration")
        return params

    def _check_configuration(self):
        if state_snapshot(self._params)["sha256"] != self._params_sha256:
            raise ValueError("environment parameters changed")

    def _with_params(self, operation):
        params = self.params
        result = operation(params)
        if state_snapshot(params)["sha256"] != self._params_sha256:
            raise ValueError("native operation changed environment parameters")
        return result

    @classmethod
    def from_factory(cls, factory, env_id, *, options, key, split, transition="step"):
        """Call a native registry factory with an explicit ID and options."""
        text(env_id, "env_id")
        if not callable(factory) or not isinstance(options, dict):
            raise ValueError("factory must be callable and options must be a mapping")
        checked = plain(options)
        environment = factory(env_id, **checked)
        identity = env_id + ":" + state_snapshot(checked)["sha256"]
        return cls(environment, key=key, split=split, identity=identity, transition=transition)

    def _keys(self, key):
        values = self.split(key, 2)
        if len(values) != 2:
            raise ValueError("split(key, 2) must return two keys")
        return values[0], values[1]

    def reset(self, seed):
        integer(seed, "seed")
        next_key, reset_key = self._keys(self.key_factory(seed))
        result = self._with_params(lambda params: self.environment.reset(reset_key, params))
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("native reset must return observation and state")
        observation, state = result
        converted = plain(observation)
        state_snapshot(state)
        state_snapshot(next_key)
        self._snapshot = ForagaxSnapshot(
            self.identity, deepcopy(next_key), deepcopy(state), deepcopy(observation), 0, False
        )
        return converted

    def snapshot(self):
        """Copy environment state, current observation and the next PRNG key."""
        self._check_configuration()
        if self._snapshot is None:
            raise ValueError("environment has not been reset")
        return deepcopy(self._snapshot)

    def restore(self, snapshot):
        self._check_configuration()
        if not isinstance(snapshot, ForagaxSnapshot) or snapshot.identity != self.identity:
            raise ValueError("snapshot belongs to a different environment configuration")
        integer(snapshot.steps, "steps")
        if type(snapshot.terminated) is not bool:
            raise ValueError("terminated must be boolean")
        plain(snapshot.observation)
        state_snapshot(snapshot.environment_state)
        state_snapshot(snapshot.key)
        self._snapshot = deepcopy(snapshot)

    def sample_action(self):
        """Sample the native action space and advance the stored PRNG key."""
        current = self.snapshot()
        next_key, action_key = self._keys(current.key)
        action = self._with_params(
            lambda params: self.environment.action_space(params).sample(action_key)
        )
        self._snapshot = ForagaxSnapshot(
            current.identity,
            deepcopy(next_key),
            current.environment_state,
            current.observation,
            current.steps,
            current.terminated,
        )
        return action

    def step(self, action):
        current = self.snapshot()
        if current.terminated:
            raise ValueError("reset is required after a terminal transition")
        action = count(action, "action")
        size = count(
            self._with_params(lambda params: self.environment.action_space(params).n),
            "action count",
            1,
        )
        if action >= size:
            raise ValueError("action is outside the native action space")
        next_key, step_key = self._keys(current.key)
        result = self._with_params(
            lambda params: getattr(self.environment, self.transition)(
                step_key, current.environment_state, action, params
            )
        )
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError("native step must return observation, state, reward, done, info")
        observation, state, reward, done, info = result
        reward, done = finite(reward, "reward"), scalar(done)
        if type(done) is not bool or not isinstance(info, dict):
            raise ValueError("native done must be boolean and info must be a mapping")
        converted, details = plain(observation), plain(info)
        state_snapshot(state)
        state_snapshot(next_key)
        self._snapshot = ForagaxSnapshot(
            self.identity,
            deepcopy(next_key),
            deepcopy(state),
            deepcopy(observation),
            current.steps + 1,
            done,
        )
        return converted, reward, done, details
