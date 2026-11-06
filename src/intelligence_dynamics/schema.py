"""JSON Schema representation of protocol 0.1 records."""

from dataclasses import MISSING, fields
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

from .records import (
    RECORD_TYPES,
    ComputationCounters,
    MemoryAccounting,
    RoutingSummary,
)


def record_schema():
    classes = (*RECORD_TYPES.values(), ComputationCounters, MemoryAccounting, RoutingSummary)
    definitions = {}
    nonnegative = {
        "step",
        "lifetime",
        "start_step",
        "end_step",
        "recovery_steps",
        "state_norm",
        "prediction_error",
        "salience",
        "uncertainty",
        "routing_entropy",
        "wall_time_seconds",
    }

    def value_schema(annotation, name):
        origin = get_origin(annotation)
        if origin is UnionType:
            return {"anyOf": [value_schema(part, name) for part in get_args(annotation)]}
        if annotation is type(None):
            return {"type": "null"}
        if annotation is Any:
            return {}
        if annotation in classes:
            return {"$ref": "#/$defs/" + annotation.__name__}
        if origin is dict:
            return {
                "type": "object",
                "additionalProperties": value_schema(get_args(annotation)[1], ""),
            }
        primitive = {str: "string", int: "integer", float: "number", bool: "boolean"}
        result = {"type": primitive[annotation]}
        if annotation is str:
            result["minLength"] = 1
        if annotation in (int, float) and (
            name in nonnegative
            or name.endswith(
                (
                    "_bytes",
                    "_updates",
                    "_events",
                    "_ops",
                    "_count",
                    "_iterations",
                    "_transitions",
                    "_decisions",
                    "_steps",
                    "_modules",
                    "_units",
                    "_edges",
                )
            )
        ):
            result["minimum"] = 0
        if name == "protocol_version":
            result["const"] = "0.1"
        if name == "track":
            result["enum"] = ["A", "B", "C", "AB", "cross"]
        if name == "target_recovery_fraction":
            result.update(minimum=0, maximum=1)
        return result

    for cls in classes:
        hints = get_type_hints(cls)
        properties = {
            field.name: value_schema(hints[field.name], field.name) for field in fields(cls)
        }
        required = [
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        ]
        if cls.__name__ in RECORD_TYPES:
            properties["record_type"] = {"const": cls.__name__}
            required.extend(["record_type", "protocol_version"])
        definitions[cls.__name__] = {
            "type": "object",
            "properties": properties,
            "required": sorted(set(required)),
            "additionalProperties": False,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "IntelligenceDynamics measurement protocol 0.1",
        "oneOf": [{"$ref": "#/$defs/" + name} for name in RECORD_TYPES],
        "$defs": definitions,
    }
