"""Executable adapters and lifecycle controls for explicit native objects."""

from ._state import state_snapshot as _state_snapshot
from .checkpoint import CheckpointCodec, CheckpointStore, NativeCheckpoint
from .dal import dal_individual, dal_population, dal_remote_population
from .drqn import (
    ArrayOperations,
    DRQNDecision,
    DRQNExecutor,
    DRQNPolicy,
    ExecutionState,
    NativeTransition,
    StepAudit,
)
from .foragax import ForagaxAdapter, ForagaxSnapshot
from .lifecycle import FreezePlan, NativeLifecycle, execution_state, verify_common_history

__all__ = [
    "ArrayOperations",
    "CheckpointCodec",
    "CheckpointStore",
    "DRQNDecision",
    "DRQNExecutor",
    "DRQNPolicy",
    "ExecutionState",
    "ForagaxAdapter",
    "ForagaxSnapshot",
    "FreezePlan",
    "NativeCheckpoint",
    "NativeLifecycle",
    "NativeTransition",
    "StepAudit",
    "dal_individual",
    "dal_population",
    "dal_remote_population",
    "execution_state",
    "verify_common_history",
    "native_state_snapshot",
]


def native_state_snapshot(value):
    """Return typed state identity and numeric tensor element/byte counts."""
    return _state_snapshot(value)
