"""Matched-state interventions and branching continuations."""

import json
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from statistics import fmean

from ._validation import choice, integer, real, sequence
from .policy import LearningRule, TabularPolicy
from .task import CueTask, StepRecord, TaskConfig


def _typed(value, active):
    kind = type(value)
    if value is None:
        return ["null"]
    if kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        real(value, "state number")
        return ["float", value.hex()]
    token = id(value)
    if token in active:
        raise ValueError("state must be acyclic")
    active.add(token)
    try:
        if kind is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("state dictionary keys must be strings")
            return ["dict", [[key, _typed(value[key], active)] for key in sorted(value)]]
        if kind in (list, tuple):
            return [kind.__name__, [_typed(item, active) for item in value]]
        if is_dataclass(value) and not isinstance(value, type):
            return [
                "dataclass",
                f"{kind.__module__}.{kind.__qualname__}",
                [
                    [field.name, _typed(getattr(value, field.name), active)]
                    for field in fields(value)
                ],
            ]
        raise TypeError(f"unsupported state type: {kind.__name__}")
    finally:
        active.remove(token)


def state_digest(state):
    """Hash typed scalar, sequence, dictionary, and dataclass state recursively."""
    encoded = json.dumps(_typed(state, set()), separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def trace_digest(columns, *, boundary):
    """Hash named trace columns after requiring exactly boundary rows per column."""
    integer(boundary, "boundary")
    if type(columns) is not dict or not columns:
        raise ValueError("trace columns must be a nonempty dictionary")
    normalized = {}
    for name, values in columns.items():
        if type(name) is not str or not name:
            raise ValueError("trace column names must be nonempty strings")
        rows = sequence(values, name)
        if len(rows) != boundary:
            raise ValueError(f"trace column {name} must contain exactly {boundary} rows")
        normalized[name] = rows
    return state_digest(normalized)


def verify_paired_history(control, treatment, *, boundary):
    """Require identical typed trace histories and return their shared digest."""
    left = trace_digest(control, boundary=boundary)
    right = trace_digest(treatment, boundary=boundary)
    if left != right:
        raise ValueError("paired trace histories differ")
    return left


def state_changes(before, after):
    """Return changed paths as tuples of dictionary keys or sequence indices."""
    state_digest(before)
    state_digest(after)
    found = []

    def visit(left, right, path):
        if type(left) is not type(right):
            found.append(path)
        elif isinstance(left, dict):
            for key in sorted(left.keys() | right.keys()):
                if key not in left or key not in right:
                    found.append((*path, key))
                else:
                    visit(left[key], right[key], (*path, key))
        elif isinstance(left, (list, tuple)):
            for index in range(max(len(left), len(right))):
                if index >= len(left) or index >= len(right):
                    found.append((*path, index))
                else:
                    visit(left[index], right[index], (*path, index))
        elif is_dataclass(left) and not isinstance(left, type):
            for field in fields(left):
                visit(getattr(left, field.name), getattr(right, field.name), (*path, field.name))
        elif _typed(left, set()) != _typed(right, set()):
            found.append(path)

    visit(before, after, ())
    return tuple(found)


def verify_intervention(before, after, *, allowed_paths, require_change=True):
    paths = tuple(
        sequence(path, "allowed path") for path in sequence(allowed_paths, "allowed_paths")
    )
    if not paths or any(not path for path in paths):
        raise ValueError("allowed_paths must contain nonempty paths")
    if any(
        type(part) not in (str, int) or (type(part) is int and part < 0)
        for path in paths
        for part in path
    ):
        raise ValueError("path elements must be string keys or nonnegative integer indices")
    if type(require_change) is not bool:
        raise ValueError("require_change must be boolean")
    changed = state_changes(before, after)
    if require_change and not changed:
        raise ValueError("intervention changed no state")
    unexpected = [
        path for path in changed if not any(path[: len(prefix)] == prefix for prefix in paths)
    ]
    if unexpected:
        raise ValueError(f"intervention changed unselected state: {unexpected}")
    return changed


@dataclass(frozen=True)
class BranchComparison:
    boundary: int
    intervention: str
    intervention_changed: bool
    changed_paths: tuple[tuple[str | int, ...], ...]
    shared_state_digest: str
    common_prefix: tuple[StepRecord, ...]
    continued: tuple[StepRecord, ...]
    intervened: tuple[StepRecord, ...]
    continued_accuracy: float
    intervened_accuracy: float
    delta: float
    continued_updates: int
    intervened_updates: int


def compare_branches(config, rule, *, boundary, mode="plastic", seed=0, intervention="freeze"):
    """Branch before action boundary; delta is continued minus intervened accuracy."""
    if not isinstance(config, TaskConfig) or not isinstance(rule, LearningRule):
        raise TypeError("config and rule must be TaskConfig and LearningRule")
    if len(rule.priors) != config.cue_count:
        raise ValueError("rule width must equal cue_count")
    integer(boundary, "boundary", 1)
    integer(seed, "seed")
    if boundary >= config.steps:
        raise ValueError("boundary must leave at least one continuation action")
    choice(intervention, "intervention", ("freeze", "reset_memory"))
    task, policy = CueTask(config), TabularPolicy(rule, mode)
    observation = task.reset()
    policy.reset(seed=seed)
    feedback = None
    prefix = []
    for _ in range(boundary):
        transition = task.advance(policy.act(observation, feedback))
        prefix.append(transition.record)
        observation, feedback = transition.observation, transition.feedback
    branched_task, branched_policy = deepcopy(task), deepcopy(policy)
    before = {
        "task": task.snapshot(),
        "policy": policy.snapshot(),
        "observation": observation,
        "feedback": feedback,
    }
    clone = {
        "task": branched_task.snapshot(),
        "policy": branched_policy.snapshot(),
        "observation": observation,
        "feedback": feedback,
    }
    if state_digest(before) != state_digest(clone):
        raise RuntimeError("branch copies do not match")
    if intervention == "freeze":
        branched_policy.learning_enabled = False
        allowed = (("policy", "learning_enabled"),)
    else:
        branched_policy.clear_memory()
        allowed = (("policy", "memory"),)
    after = {
        "task": branched_task.snapshot(),
        "policy": branched_policy.snapshot(),
        "observation": observation,
        "feedback": feedback,
    }
    changes = verify_intervention(before, after, allowed_paths=allowed, require_change=False)

    def continue_from(environment, learner):
        current, delivery = observation, feedback
        start_updates = len(learner.updates)
        rows = []
        for _ in range(boundary, config.steps):
            tr = environment.advance(learner.act(current, delivery))
            rows.append(tr.record)
            current, delivery = tr.observation, tr.feedback
        return tuple(rows), len(learner.updates) - start_updates

    continued, continued_updates = continue_from(task, policy)
    intervened, intervened_updates = continue_from(branched_task, branched_policy)
    if tuple(row.cue for row in continued) != tuple(row.cue for row in intervened):
        raise RuntimeError("paired environment streams diverged")
    control_mean = fmean(row.reward for row in continued)
    treatment_mean = fmean(row.reward for row in intervened)
    return BranchComparison(
        boundary,
        intervention,
        bool(changes),
        changes,
        state_digest(before),
        tuple(prefix),
        continued,
        intervened,
        control_mean,
        treatment_mean,
        control_mean - treatment_mean,
        continued_updates,
        intervened_updates,
    )
