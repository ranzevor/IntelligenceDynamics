"""Native continuation, selective interventions and common-history checks."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from ..validation import integer
from ._state import state_differences, state_snapshot
from ._values import canonical_path, field_path, read_field, replace_field
from .checkpoint import NativeCheckpoint
from .drqn import DRQNExecutor, ExecutionState


def execution_state(glue, *, terminated=False):
    """Copy the seven native RLGlue fields into a resumable execution state."""
    if type(terminated) is not bool:
        raise ValueError("terminated must be boolean")
    names = (
        "agent_state",
        "env_state",
        "last_action",
        "num_episodes",
        "num_steps",
        "total_reward",
        "total_steps",
    )
    values = {name: deepcopy(read_field(glue, (name,))) for name in names}
    from ._values import count, finite

    for name in ("num_episodes", "num_steps", "total_steps"):
        values[name] = count(values[name], name)
    values["total_reward"] = finite(values["total_reward"], "total reward")
    result = ExecutionState(**values, terminated=terminated)
    state_snapshot(result)
    return result


def verify_common_history(left, right, *, boundary):
    """Require exact typed observations through a specified interaction boundary."""
    integer(boundary, "boundary")
    if not isinstance(left, NativeCheckpoint) or not isinstance(right, NativeCheckpoint):
        raise ValueError("common-history comparison requires checkpoints")
    if boundary > len(left.history) or boundary > len(right.history):
        raise ValueError("requested history boundary is not present in both checkpoints")
    first = state_snapshot(left.history[:boundary])["sha256"]
    second = state_snapshot(right.history[:boundary])["sha256"]
    if first != second:
        raise ValueError("branch histories differ before the boundary")
    return {"boundary": boundary, "sha256": first}


def _paths(paths):
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("paths must be a nonempty sequence")
    result = tuple(field_path(path) for path in paths)
    if len(set(result)) != len(result):
        raise ValueError("field paths must be unique")
    if any(a != b and a[: len(b)] == b for a in result for b in result):
        raise ValueError("field paths cannot overlap")
    protected = (
        ("agent_state", "steps"),
        ("agent_state", "updates"),
        ("agent_state", "hypers", "update_freq"),
        ("total_steps",),
        ("num_steps",),
        ("num_episodes",),
        ("terminated",),
    )
    if any(a[: len(b)] == b or b[: len(a)] == a for a in result for b in protected):
        raise ValueError("interventions cannot change scheduling or lifecycle counters")
    return result


@dataclass(frozen=True)
class FreezePlan:
    fields: tuple

    def apply(self, state):
        paths = _paths(tuple(canonical_path(state, path) for path, _ in self.fields))
        result = deepcopy(state)
        for path, value in self.fields:
            result = replace_field(result, path, value)
        changed = state_differences(state, result)
        if any(not any(path[: len(prefix)] == prefix for prefix in paths) for path in changed):
            raise ValueError("freeze changed an unselected field")
        return result


class NativeLifecycle:
    """Run the concrete DRQN executor and retain complete paired histories."""

    def __init__(self, executor):
        if not isinstance(executor, DRQNExecutor):
            raise ValueError("executor must be DRQNExecutor")
        self.executor = executor

    def start(self, *, seed=0):
        return NativeCheckpoint(self.executor.initialize(seed=seed))

    def intervene(self, checkpoint, replacements):
        """Copy a checkpoint and replace only explicitly selected state fields."""
        if not isinstance(checkpoint, NativeCheckpoint) or not isinstance(replacements, Mapping):
            raise ValueError("intervention requires a checkpoint and path-to-value mapping")
        selected = _paths(tuple(replacements))
        paths = _paths(tuple(canonical_path(checkpoint.state, path) for path in selected))
        original = checkpoint.sha256
        state = deepcopy(checkpoint.state)
        for path, selected_path in zip(paths, selected):
            state = replace_field(state, path, replacements[selected_path])
        changed = state_differences(checkpoint.state, state)
        if not changed:
            raise ValueError("intervention changed no state")
        if any(not any(path[: len(prefix)] == prefix for prefix in paths) for path in changed):
            raise ValueError("intervention changed an unselected field")
        result = NativeCheckpoint(state, deepcopy(checkpoint.history))
        if checkpoint.sha256 != original:
            raise ValueError("intervention mutated its source checkpoint")
        verify_common_history(checkpoint, result, boundary=len(checkpoint.history))
        return result

    def freeze(self, checkpoint, paths):
        """Capture selected values to restore after each executed transition."""
        if not isinstance(checkpoint, NativeCheckpoint):
            raise ValueError("freeze requires a checkpoint")
        selected = _paths(tuple(canonical_path(checkpoint.state, path) for path in _paths(paths)))
        return FreezePlan(
            tuple((path, deepcopy(read_field(checkpoint.state, path))) for path in selected)
        )

    def run(self, checkpoint, *, interactions, learning=True, freeze=None):
        """Execute additional interactions, stopping at a terminal transition."""
        if not isinstance(checkpoint, NativeCheckpoint):
            raise ValueError("run requires a checkpoint")
        integer(interactions, "interactions")
        if type(learning) is not bool or (
            freeze is not None and not isinstance(freeze, FreezePlan)
        ):
            raise ValueError("learning or freeze has an invalid type")
        source_identity = checkpoint.sha256
        state, history = deepcopy(checkpoint.state), list(deepcopy(checkpoint.history))
        for _ in range(interactions):
            if state.terminated:
                break
            state, transition = self.executor.step(state, learning=learning)
            if freeze is not None:
                state = freeze.apply(state)
            history.append(transition)
        result = NativeCheckpoint(state, tuple(history))
        if checkpoint.sha256 != source_identity:
            raise ValueError("native execution mutated its source checkpoint")
        verify_common_history(checkpoint, result, boundary=len(checkpoint.history))
        return result
