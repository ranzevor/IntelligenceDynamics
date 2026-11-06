"""Detached, typed observations of model state and tensor storage."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from .storage import canonical_bytes, digest
from .validation import number


def state_snapshot(value):
    """Observe scalar/container state and numeric arrays without changing it."""
    active, tensors = set(), []

    def encode(item, path):
        kind = type(item)
        if item is None:
            return ["null"]
        if kind in (bool, int, str):
            return [kind.__name__, item]
        if kind is float:
            return ["float", number(item, "state float").hex()]
        identity = id(item)
        if identity in active:
            raise ValueError("state must be acyclic")
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                if any(type(key) not in (str, int) for key in item):
                    raise ValueError("state mapping keys must be strings or integers")
                keys = sorted(item, key=lambda key: canonical_bytes([type(key).__name__, key]))
                return [
                    "mapping",
                    f"{kind.__module__}.{kind.__qualname__}",
                    [[encode(key, path), encode(item[key], (*path, key))] for key in keys],
                ]
            if kind in (tuple, list):
                return [
                    kind.__name__,
                    [encode(child, (*path, index)) for index, child in enumerate(item)],
                ]
            if is_dataclass(item) and not isinstance(item, type):
                return [
                    "dataclass",
                    f"{kind.__module__}.{kind.__qualname__}",
                    [
                        [field.name, encode(getattr(item, field.name), (*path, field.name))]
                        for field in fields(item)
                    ],
                ]
            if hasattr(item, "__array__"):
                try:
                    import numpy as np
                except ImportError as exc:
                    raise ValueError("tensor state requires the arrays dependency group") from exc
                array = np.asarray(item)
                if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
                    raise ValueError("state arrays must have finite numeric values")
                content = array.tobytes(order="C")
                observation = {
                    "path": list(path),
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "elements": int(array.size),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                tensors.append(observation)
                return ["array", observation["dtype"], observation["shape"], observation["sha256"]]
            raise ValueError(f"unsupported state type: {kind.__name__}")
        finally:
            active.remove(identity)

    encoded = encode(value, ())
    return {
        "sha256": digest(encoded),
        "state": encoded,
        "tensors": tensors,
        "tensor_elements": sum(item["elements"] for item in tensors),
        "tensor_bytes": sum(item["bytes"] for item in tensors),
    }


def state_differences(before, after):
    """Return semantic field paths whose typed values differ."""
    state_snapshot(before)
    state_snapshot(after)
    changed = []

    def visit(left, right, path):
        if type(left) is not type(right):
            changed.append(path)
        elif isinstance(left, Mapping):
            keys = sorted(
                left.keys() | right.keys(),
                key=lambda key: canonical_bytes([type(key).__name__, key]),
            )
            for key in keys:
                if key not in left or key not in right:
                    changed.append((*path, key))
                else:
                    visit(left[key], right[key], (*path, key))
        elif type(left) in (list, tuple):
            for index in range(max(len(left), len(right))):
                if index >= len(left) or index >= len(right):
                    changed.append((*path, index))
                else:
                    visit(left[index], right[index], (*path, index))
        elif is_dataclass(left) and not isinstance(left, type):
            for field in fields(left):
                visit(getattr(left, field.name), getattr(right, field.name), (*path, field.name))
        elif state_snapshot(left)["sha256"] != state_snapshot(right)["sha256"]:
            changed.append(path)

    visit(before, after, ())
    return tuple(changed)


def verify_state_intervention(before, after, *, allowed_paths, require_change=True):
    if not isinstance(allowed_paths, (list, tuple)) or not allowed_paths:
        raise ValueError("allowed_paths must contain nonempty field paths")
    paths = []
    for path in allowed_paths:
        if (
            not isinstance(path, (tuple, list))
            or not path
            or any(type(part) not in (str, int) for part in path)
        ):
            raise ValueError("each field path requires string keys or integer indices")
        paths.append(tuple(path))
    if type(require_change) is not bool:
        raise ValueError("require_change must be boolean")
    changed = state_differences(before, after)
    if require_change and not changed:
        raise ValueError("intervention changed no state")
    if any(not any(path[: len(prefix)] == prefix for prefix in paths) for path in changed):
        raise ValueError("intervention changed unselected state")
    return changed
