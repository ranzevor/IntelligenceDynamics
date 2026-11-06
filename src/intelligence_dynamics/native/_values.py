"""Native scalar conversion and immutable field replacement."""

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

from ..validation import integer, json_value, number


def scalar(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError) as exc:
            raise ValueError("expected a scalar native value") from exc
    return value


def count(value, name, minimum=0):
    return integer(scalar(value), name, minimum=minimum)


def finite(value, name):
    return number(scalar(value), name)


def plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return json_value({key: plain(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return json_value(scalar(value))


def field_path(path):
    if not isinstance(path, (tuple, list)) or not path:
        raise ValueError("field paths must be nonempty tuples or lists")
    if any(type(part) not in (str, int) for part in path):
        raise ValueError("field path components must be strings or integers")
    return tuple(path)


def read_field(value, path):
    for part in field_path(path):
        if is_dataclass(value) and not isinstance(value, type):
            if part not in {field.name for field in fields(value)}:
                raise ValueError(f"missing dataclass field: {part}")
            value = getattr(value, part)
        elif isinstance(value, Mapping):
            if part not in value:
                raise ValueError(f"missing mapping field: {part}")
            value = value[part]
        elif isinstance(value, tuple) and hasattr(value, "_fields") and isinstance(part, str):
            if part not in value._fields:
                raise ValueError(f"missing namedtuple field: {part}")
            value = getattr(value, part)
        elif isinstance(value, (tuple, list)) and type(part) is int:
            if part < 0 or part >= len(value):
                raise ValueError("field index is out of range")
            value = value[part]
        else:
            raise ValueError("field path does not address structured state")
    return value


def replace_field(value, path, replacement):
    path = field_path(path)
    read_field(value, path)
    part, rest = path[0], path[1:]
    child = read_field(value, (part,))
    result = replace_field(child, rest, replacement) if rest else deepcopy(replacement)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{part: result})
    if isinstance(value, Mapping):
        if not isinstance(value, MutableMapping):
            raise ValueError("nested replacement requires a mutable mapping")
        updated = deepcopy(value)
        if updated is value:
            raise ValueError("mapping deepcopy must return a detached object")
        updated[part] = result
        return updated
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        name = value._fields[part] if type(part) is int else part
        return value._replace(**{name: result})
    updated = list(value)
    updated[part] = result
    if type(value) is tuple:
        return tuple(updated)
    if type(value) is list:
        return updated
    raise ValueError("nested replacement requires a built-in sequence or namedtuple")


def canonical_path(value, path):
    """Resolve namedtuple field names to their equivalent positional indices."""
    result = []
    for part in field_path(path):
        if isinstance(value, tuple) and hasattr(value, "_fields") and isinstance(part, str):
            if part not in value._fields:
                raise ValueError(f"missing namedtuple field: {part}")
            part = value._fields.index(part)
        result.append(part)
        value = read_field(value, (part,))
    return tuple(result)
