"""Finite cue tasks with hidden mappings and action-indexed delayed feedback."""

import json
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from random import Random

from ._validation import choice, integer, real, sequence


@dataclass(frozen=True)
class TaskConfig:
    """Action t produces feedback before action t + delay + 1."""

    steps: int = 96
    cue_count: int = 4
    delay: int = 1
    switches: tuple[int, ...] = (32, 64)
    mappings: tuple[tuple[int, ...], ...] = ((0, 1, 0, 1), (1, 0, 1, 0))
    seed: int = 0

    def __post_init__(self):
        integer(self.steps, "steps", 1)
        integer(self.cue_count, "cue_count", 1)
        integer(self.delay, "delay")
        integer(self.seed, "seed")
        switches = sequence(self.switches, "switches")
        for step in switches:
            integer(step, "switch", 1)
            if step >= self.steps:
                raise ValueError("switches must precede the final action")
        if switches != tuple(sorted(set(switches))):
            raise ValueError("switches must be strictly increasing")
        mappings = tuple(sequence(row, "mapping") for row in sequence(self.mappings, "mappings"))
        if not mappings or (switches and len(mappings) < 2):
            raise ValueError("switching tasks require at least two mappings")
        for row in mappings:
            if len(row) != self.cue_count:
                raise ValueError("mapping width must equal cue_count")
            if any(type(action) is not int or action not in (0, 1) for action in row):
                raise ValueError("mappings must contain binary integer actions")
        object.__setattr__(self, "switches", switches)
        object.__setattr__(self, "mappings", mappings)

    @property
    def fingerprint(self):
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    cue: int

    def __post_init__(self):
        integer(self.cue, "cue")


@dataclass(frozen=True)
class Feedback:
    credit_id: int
    value: float

    def __post_init__(self):
        integer(self.credit_id, "credit_id")
        object.__setattr__(self, "value", real(self.value, "feedback", 0, 1))


@dataclass(frozen=True)
class StepRecord:
    step: int
    cue: int
    action: int
    target: int
    regime: int
    reward: float


@dataclass(frozen=True)
class Transition:
    observation: Observation | None
    feedback: Feedback | None
    done: bool
    record: StepRecord


class CueTask:
    def __init__(self, config: TaskConfig):
        if not isinstance(config, TaskConfig):
            raise TypeError("config must be TaskConfig")
        self.config = config
        self._started = False

    def reset(self):
        self._random = Random(self.config.seed)
        self._time = 0
        self._regime = 0
        self._cue = self._random.randrange(self.config.cue_count)
        self._due = deque()
        self._started = True
        return Observation(self._cue)

    @property
    def time(self):
        if not self._started:
            raise RuntimeError("task has not been reset")
        return self._time

    def advance(self, action):
        if not self._started or self._time >= self.config.steps:
            raise RuntimeError("task must be active")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("action must be the integer 0 or 1")
        t = self._time
        if t in self.config.switches:
            self._regime = (self._regime + 1) % len(self.config.mappings)
        target = self.config.mappings[self._regime][self._cue]
        reward = float(action == target)
        record = StepRecord(t, self._cue, action, target, self._regime, reward)
        self._due.append((t + self.config.delay + 1, Feedback(t, reward)))
        self._time += 1
        delivered = None
        if self._due and self._due[0][0] == self._time:
            delivered = self._due.popleft()[1]
        done = self._time == self.config.steps
        observation = None
        if not done:
            self._cue = self._random.randrange(self.config.cue_count)
            observation = Observation(self._cue)
        return Transition(observation, delivered, done, record)

    def snapshot(self):
        if not self._started:
            raise RuntimeError("task has not been reset")
        return {
            "config": asdict(self.config),
            "time": self._time,
            "regime": self._regime,
            "cue": self._cue,
            "due": tuple(self._due),
            "random": self._random.getstate(),
        }


@dataclass(frozen=True)
class Scenario:
    name: str
    partition: str
    task: TaskConfig

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be nonempty")
        choice(self.partition, "partition", ("train", "heldout"))
        if not isinstance(self.task, TaskConfig):
            raise TypeError("scenario task must be TaskConfig")

    @property
    def kind(self):
        return "switching" if self.task.switches else "stationary"


def make_bank(*, partition, seed, pairs=1, steps=64, cue_count=4, delay=1):
    """Build stationary and switching pairs with opposite mapping targets."""
    choice(partition, "partition", ("train", "heldout"))
    integer(seed, "seed")
    integer(pairs, "pairs", 1)
    integer(steps, "steps", 8)
    integer(cue_count, "cue_count", 1)
    integer(delay, "delay")
    generator = Random(seed)
    bank = []
    for kind in ("stationary", "switching"):
        for pair in range(pairs):
            base = tuple(generator.randrange(2) for _ in range(cue_count))
            opposite = tuple(1 - action for action in base)
            switches = ()
            if kind == "switching":
                switches = tuple(
                    generator.randrange(k * steps // 4, (k + 1) * steps // 4) for k in (1, 2, 3)
                )
            stream_seed = int.from_bytes(
                sha256(f"{seed}:{kind}:{pair}".encode()).digest()[:16], "big"
            )
            for mirror, mappings in enumerate(((base, opposite), (opposite, base))):
                config = TaskConfig(steps, cue_count, delay, switches, mappings, stream_seed)
                name = f"{partition}:{seed}:{kind}:{pair}:{mirror}"
                bank.append(Scenario(name, partition, config))
    return tuple(bank)


def validate_bank(scenarios, partition=None):
    bank = sequence(scenarios, "scenarios")
    if not bank or any(not isinstance(item, Scenario) for item in bank):
        raise ValueError("scenarios must contain Scenario instances")
    if partition is not None:
        choice(partition, "partition", ("train", "heldout"))
        if any(item.partition != partition for item in bank):
            raise ValueError(f"all scenarios must belong to {partition}")
    if len({item.name for item in bank}) != len(bank):
        raise ValueError("scenario names must be unique")
    if len({item.task.fingerprint for item in bank}) != len(bank):
        raise ValueError("duplicate task configurations")
    if len({item.task.cue_count for item in bank}) != 1:
        raise ValueError("all scenarios must have the same cue_count")
    return bank
