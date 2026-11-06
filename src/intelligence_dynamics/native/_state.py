"""Typed native-state identity including optimizer tuples and JAX keys."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from ..storage import canonical_bytes


def _numeric_dtype(array):
    if array.dtype.kind in "biufc":
        return array.dtype.str
    if array.dtype.name == "bfloat16":
        try:
            import ml_dtypes
            import numpy as np
        except ImportError:
            pass
        else:
            if array.dtype == np.dtype(ml_dtypes.bfloat16):
                return "bfloat16"
    raise ValueError("native arrays must be finite and numeric")


def _snapshot(value, *, root_path=(), array_types=False):
    active = set()
    tensors = []

    def encode(item, path):
        kind = type(item)
        identity = f"{kind.__module__}.{kind.__qualname__}"
        if item is None or kind in (bool, str, int):
            return [identity, item]
        if kind is float:
            allowed_infinity = (
                len(path) >= 2 and path[-2:] == ("hypers", "freeze_steps") and item == math.inf
            )
            if not math.isfinite(item) and not allowed_infinity:
                raise ValueError("native scalar state must be finite outside freeze_steps")
            return [identity, item.hex()]
        if id(item) in active:
            raise ValueError("native state must be acyclic")
        active.add(id(item))
        try:
            if is_dataclass(item) and not isinstance(item, type):
                return [
                    identity,
                    [
                        [f.name, encode(getattr(item, f.name), (*path, f.name))]
                        for f in fields(item)
                    ],
                ]
            if isinstance(item, Mapping):
                if any(type(key) not in (str, int) for key in item):
                    raise ValueError("native mapping keys must be strings or integers")
                keys = sorted(item, key=lambda key: canonical_bytes([type(key).__name__, key]))
                return [
                    identity,
                    [[encode(key, ()), encode(item[key], (*path, key))] for key in keys],
                ]
            if isinstance(item, (tuple, list)):
                return [identity, [encode(child, (*path, i)) for i, child in enumerate(item)]]
            if hasattr(item, "__array__"):
                import numpy as np

                key_type = str(getattr(item, "dtype", ""))
                if key_type.startswith("key<"):
                    from jax.random import key_data

                    array = np.asarray(key_data(item))
                else:
                    array = np.asarray(item)
                dtype = _numeric_dtype(array)
                allowed_infinity = (
                    len(path) >= 2
                    and path[-2:] == ("hypers", "freeze_steps")
                    and array.size == 1
                    and bool(np.isposinf(array).all())
                )
                if not np.isfinite(array).all() and not allowed_infinity:
                    raise ValueError("native arrays must be finite and numeric")
                raw = array.tobytes(order="C")
                record = {
                    "path": list(path),
                    "shape": list(array.shape),
                    "dtype": dtype,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "elements": int(array.size),
                    "bytes": len(raw),
                }
                tensors.append(record)
                encoded = [
                    "array",
                    key_type if key_type.startswith("key<") else dtype,
                    list(array.shape),
                    record["sha256"],
                ]
                if array_types:
                    encoded.extend((identity, getattr(item, "weak_type", None)))
                return encoded
            raise ValueError(f"unsupported native state type: {identity}")
        finally:
            active.remove(id(item))

    encoded = encode(value, root_path)
    return {
        "sha256": hashlib.sha256(canonical_bytes(encoded)).hexdigest(),
        "tensors": tensors,
        "tensor_elements": sum(item["elements"] for item in tensors),
        "tensor_bytes": sum(item["bytes"] for item in tensors),
    }


def state_snapshot(value, *, root_path=()):
    return _snapshot(value, root_path=root_path)


def codec_state_identity(value):
    """Compare codec results without changing established checkpoint digests."""
    return _snapshot(value, array_types=True)["sha256"]


def state_differences(before, after, path=()):
    if (
        state_snapshot(before, root_path=path)["sha256"]
        == state_snapshot(after, root_path=path)["sha256"]
    ):
        return ()
    if type(before) is not type(after):
        return (path,)
    if is_dataclass(before) and not isinstance(before, type):
        return tuple(
            child
            for f in fields(before)
            for child in state_differences(
                getattr(before, f.name), getattr(after, f.name), (*path, f.name)
            )
        )
    if isinstance(before, Mapping):
        result = []
        for key in sorted(
            before.keys() | after.keys(), key=lambda key: canonical_bytes([type(key).__name__, key])
        ):
            result.extend(
                ((*path, key),)
                if key not in before or key not in after
                else state_differences(before[key], after[key], (*path, key))
            )
        return tuple(result)
    if isinstance(before, (tuple, list)):
        result = []
        for i in range(max(len(before), len(after))):
            result.extend(
                ((*path, i),)
                if i >= len(before) or i >= len(after)
                else state_differences(before[i], after[i], (*path, i))
            )
        return tuple(result)
    return (path,)
