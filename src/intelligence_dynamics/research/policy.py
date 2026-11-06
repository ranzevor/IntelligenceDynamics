"""Tabular learning rules with delayed credit and feedback-driven update gates."""

from collections import deque
from dataclasses import asdict, dataclass
from random import Random

from ._validation import choice, integer, real, sequence
from .task import Feedback, Observation

MODES = ("fixed", "plastic", "conditional", "accumulator", "persistence", "buffered", "random")


@dataclass(frozen=True)
class LearningRule:
    priors: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    rate: float = 0.5
    exploration: float = 0.05
    threshold: float = 0.4
    decay: float = 0.7
    accumulation: float = 0.18
    window: int = 3
    required: int = 2
    random_probability: float = 0.5

    def __post_init__(self):
        priors = sequence(self.priors, "priors")
        if not priors:
            raise ValueError("priors must contain at least one cue")
        object.__setattr__(self, "priors", tuple(real(x, "prior", -1, 1) for x in priors))
        for name in ("rate", "exploration", "threshold", "decay", "random_probability"):
            object.__setattr__(self, name, real(getattr(self, name), name, 0, 1))
        if self.rate == 0:
            raise ValueError("rate must be positive")
        object.__setattr__(self, "accumulation", real(self.accumulation, "accumulation", 0))
        if self.accumulation == 0:
            raise ValueError("accumulation must be positive")
        integer(self.window, "window", 1)
        integer(self.required, "required", 1)
        if self.required > self.window:
            raise ValueError("required cannot exceed window")


@dataclass(frozen=True)
class AppliedUpdate:
    credit_id: int
    feedback_index: int
    committed_at: int
    old_value: float
    new_value: float
    error: float


@dataclass(frozen=True)
class _Eligible:
    credit_id: int
    feedback_index: int
    cue: int
    action: int
    target: float
    error: float


class TabularPolicy:
    def __init__(self, rule: LearningRule, mode="plastic"):
        if not isinstance(rule, LearningRule):
            raise TypeError("rule must be LearningRule")
        choice(mode, "mode", MODES)
        self.rule = rule
        self.mode = mode
        self.reset(seed=0)

    def reset(self, *, seed=0):
        integer(seed, "seed")
        self._action_random = Random(seed)
        self._gate_random = Random(f"gate:{seed}")
        self._values = [[(1 - prior) / 2, (1 + prior) / 2] for prior in self.rule.priors]
        self._pending = {}
        self._next_credit = 0
        self._recent = deque()
        self._eligible = deque()
        self._accumulated = 0.0
        self._updates = []
        self.feedback_count = 0
        self.last_committed = ()
        self.last_error = None
        self.learning_enabled = True

    @property
    def values(self):
        return tuple(tuple(row) for row in self._values)

    @property
    def updates(self):
        return tuple(self._updates)

    @property
    def pending_count(self):
        return len(self._pending)

    def clear_memory(self):
        """Clear gate memory without altering values, action credits, or RNGs."""
        self._recent.clear()
        self._eligible.clear()
        self._accumulated = 0.0

    def _commit(self, event, at):
        before = self._values[event.cue][event.action]
        after = before + self.rule.rate * (event.target - before)
        self._values[event.cue][event.action] = after
        self._updates.append(
            AppliedUpdate(event.credit_id, event.feedback_index, at, before, after, event.error)
        )

    def observe(self, feedback):
        """Apply a delivered action credit once and return committed credit IDs."""
        if not isinstance(feedback, Feedback):
            raise TypeError("feedback must be Feedback")
        if feedback.credit_id not in self._pending:
            raise ValueError("feedback has no outstanding action credit")
        cue, action = self._pending.pop(feedback.credit_id)
        index = self.feedback_count
        self.feedback_count += 1
        error = abs(feedback.value - self._values[cue][action])
        self.last_error = error
        event = _Eligible(feedback.credit_id, index, cue, action, feedback.value, error)
        commits = ()
        if self.learning_enabled and self.mode != "fixed":
            if self.mode == "plastic":
                commits = (event,)
            elif self.mode == "random":
                if self._gate_random.random() < self.rule.random_probability:
                    commits = (event,)
            elif self.mode == "conditional":
                if error >= self.rule.threshold:
                    commits = (event,)
            elif self.mode == "accumulator":
                self._accumulated = self.rule.decay * self._accumulated + max(
                    0, error - self.rule.threshold
                )
                if self._accumulated >= self.rule.accumulation:
                    commits = (event,)
                    self._accumulated = 0.0
            else:
                high = error >= self.rule.threshold
                self._recent.append((index, high))
                if high and self.mode == "buffered":
                    self._eligible.append(event)
                earliest = index - self.rule.window + 1
                while self._recent and self._recent[0][0] < earliest:
                    self._recent.popleft()
                while self._eligible and self._eligible[0].feedback_index < earliest:
                    self._eligible.popleft()
                if sum(flag for _, flag in self._recent) >= self.rule.required:
                    commits = tuple(self._eligible) if self.mode == "buffered" else (event,)
                    self._recent.clear()
                    self._eligible.clear()
        for selected in commits:
            self._commit(selected, index)
        self.last_committed = tuple(selected.credit_id for selected in commits)
        return self.last_committed

    def act(self, observation, feedback=None):
        if not isinstance(observation, Observation):
            raise TypeError("observation must be Observation")
        if observation.cue >= len(self._values):
            raise ValueError("observation cue exceeds policy width")
        if feedback is not None:
            self.observe(feedback)
        else:
            self.last_committed = ()
            self.last_error = None
        # Consume the same number of draws for every decision in paired branches.
        explore = self._action_random.random()
        alternate = self._action_random.randrange(2)
        left, right = self._values[observation.cue]
        action = (
            alternate if explore < self.rule.exploration or left == right else int(right > left)
        )
        self._pending[self._next_credit] = (observation.cue, action)
        self._next_credit += 1
        return action

    def snapshot(self):
        return {
            "rule": asdict(self.rule),
            "mode": self.mode,
            "values": self.values,
            "pending": tuple(sorted(self._pending.items())),
            "next_credit": self._next_credit,
            "memory": {
                "recent": tuple(self._recent),
                "eligible": tuple(self._eligible),
                "accumulated": self._accumulated,
            },
            "updates": self.updates,
            "feedback_count": self.feedback_count,
            "last_committed": self.last_committed,
            "last_error": self.last_error,
            "learning_enabled": self.learning_enabled,
            "action_random": self._action_random.getstate(),
            "gate_random": self._gate_random.getstate(),
        }


class FixedPolicy:
    def __init__(self, mapping):
        self.mapping = sequence(mapping, "mapping")
        if not self.mapping or any(type(x) is not int or x not in (0, 1) for x in self.mapping):
            raise ValueError("mapping must contain binary integer actions")
        self.reset()

    def reset(self, *, seed=0):
        integer(seed, "seed")
        self.feedback_count = 0
        self.last_committed = ()
        self._pending = set()
        self._next_credit = 0

    @property
    def updates(self):
        return ()

    @property
    def pending_count(self):
        return len(self._pending)

    def act(self, observation, feedback=None):
        if not isinstance(observation, Observation) or observation.cue >= len(self.mapping):
            raise ValueError("observation cue exceeds mapping width")
        if feedback is not None:
            if not isinstance(feedback, Feedback) or feedback.credit_id not in self._pending:
                raise ValueError("feedback has no outstanding action credit")
            self._pending.remove(feedback.credit_id)
            self.feedback_count += 1
        self._pending.add(self._next_credit)
        self._next_credit += 1
        return self.mapping[observation.cue]
