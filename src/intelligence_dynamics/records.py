"""Versioned records for run identity, measurements, changes and interventions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .validation import integer, json_value, number, text

PROTOCOL_VERSION = "0.1"


def _optional(value: Any, name: str, validator, **bounds) -> None:
    if value is not None:
        validator(value, name, **bounds)


def _protocol(value: str) -> None:
    if value != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return json_value(value, name)


@dataclass(frozen=True)
class ComputationCounters:
    active_unit_updates: int | None = None
    potential_unit_updates: int | None = None
    active_edge_events: int | None = None
    potential_edge_events: int | None = None
    matrix_multiply_ops: int | None = None
    spike_or_event_count: int | None = None
    recurrent_iterations: int | None = None
    state_transitions: int | None = None
    routing_decisions: int | None = None
    learning_updates: int | None = None
    wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "wall_time_seconds":
                _optional(value, item.name, number, minimum=0)
            else:
                _optional(value, item.name, integer)
        for active, potential in (
            (self.active_unit_updates, self.potential_unit_updates),
            (self.active_edge_events, self.potential_edge_events),
        ):
            if active is not None and potential is not None and active > potential:
                raise ValueError("active computation exceeds potential computation")

    @property
    def unit_activation_ratio(self) -> float | None:
        return _ratio(self.active_unit_updates, self.potential_unit_updates)

    @property
    def edge_activation_ratio(self) -> float | None:
        return _ratio(self.active_edge_events, self.potential_edge_events)


def _ratio(active: int | None, potential: int | None) -> float | None:
    return None if active is None or potential in (None, 0) else active / potential


@dataclass(frozen=True)
class MemoryAccounting:
    persistent_state_bytes: int | None = None
    parameter_bytes: int | None = None
    replay_or_episodic_bytes: int | None = None
    external_memory_bytes: int | None = None
    read_bytes: int | None = None
    write_bytes: int | None = None
    retained_history_steps: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            _optional(getattr(self, item.name), item.name, integer)


@dataclass(frozen=True)
class RoutingSummary:
    active_modules: int | None = None
    available_modules: int | None = None
    active_units: int | None = None
    available_units: int | None = None
    active_edges: int | None = None
    available_edges: int | None = None
    reused_units: int | None = None
    newly_allocated_units: int | None = None
    routing_entropy: float | None = None
    route_id: str | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            validator = (
                text
                if item.name == "route_id"
                else number
                if item.name == "routing_entropy"
                else integer
            )
            bounds = {"minimum": 0} if item.name == "routing_entropy" else {}
            _optional(getattr(self, item.name), item.name, validator, **bounds)
        for group in ("modules", "units", "edges"):
            active, available = (
                getattr(self, "active_" + group),
                getattr(self, "available_" + group),
            )
            if active is not None and available is not None and active > available:
                raise ValueError(f"active_{group} exceeds available_{group}")


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    track: str
    system_name: str
    environment_name: str
    seed: int
    source_revision: str
    config_digest: str
    protocol_version: str = PROTOCOL_VERSION
    upstream_revision: str | None = None
    environment_revision: str | None = None
    exploratory: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _protocol(self.protocol_version)
        for name in (
            "run_id",
            "system_name",
            "environment_name",
            "source_revision",
            "config_digest",
        ):
            text(getattr(self, name), name)
        if not isinstance(self.track, str) or self.track not in {"A", "B", "C", "AB", "cross"}:
            raise ValueError("track must be A, B, C, AB or cross")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.exploratory) is not bool:
            raise ValueError("exploratory must be a boolean")
        for name in ("upstream_revision", "environment_revision"):
            _optional(getattr(self, name), name, text)
        object.__setattr__(self, "notes", _mapping(self.notes, "notes"))


@dataclass(frozen=True)
class StepRecord:
    run_id: str
    step: int
    lifetime: float
    capability: float | None = None
    reward_or_return: float | None = None
    loss: float | None = None
    computation: ComputationCounters | None = None
    memory: MemoryAccounting | None = None
    routing: RoutingSummary | None = None
    state_norm: float | None = None
    prediction_error: float | None = None
    salience: float | None = None
    uncertainty: float | None = None
    plasticity_probe_score: float | None = None
    context_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol(self.protocol_version)
        text(self.run_id, "run_id")
        integer(self.step, "step")
        number(self.lifetime, "lifetime", minimum=0)
        for name in ("capability", "reward_or_return", "loss", "plasticity_probe_score"):
            _optional(getattr(self, name), name, number)
        for name in ("state_norm", "prediction_error", "salience", "uncertainty"):
            _optional(getattr(self, name), name, number, minimum=0)
        _optional(self.context_id, "context_id", text)
        for name, cls in (
            ("computation", ComputationCounters),
            ("memory", MemoryAccounting),
            ("routing", RoutingSummary),
        ):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be {cls.__name__}")
        object.__setattr__(self, "extras", _mapping(self.extras, "extras"))


@dataclass(frozen=True)
class AdaptationEvent:
    event_id: str
    step: int
    event_type: str
    hidden_from_agent: bool
    description: str
    target_recovery_fraction: float = 0.9
    tags: dict[str, str] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol(self.protocol_version)
        for name in ("event_id", "event_type", "description"):
            text(getattr(self, name), name)
        integer(self.step, "step")
        if type(self.hidden_from_agent) is not bool:
            raise ValueError("hidden_from_agent must be a boolean")
        number(self.target_recovery_fraction, "target_recovery_fraction", minimum=0, maximum=1)
        tags = _mapping(self.tags, "tags")
        for key, value in tags.items():
            text(value, "tags." + key)
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True)
class InterventionRecord:
    run_id: str
    intervention_id: str
    mechanism: str
    intervention_type: str
    start_step: int
    end_step: int
    delta_capability: float | None = None
    delta_adaptation_metric: float | None = None
    delta_compute_ratio: float | None = None
    delta_plasticity: float | None = None
    recovery_steps: int | None = None
    matched_compute_budget: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _protocol(self.protocol_version)
        for name in ("run_id", "intervention_id", "mechanism", "intervention_type"):
            text(getattr(self, name), name)
        integer(self.start_step, "start_step")
        integer(self.end_step, "end_step", minimum=self.start_step)
        for name in (
            "delta_capability",
            "delta_adaptation_metric",
            "delta_compute_ratio",
            "delta_plasticity",
        ):
            _optional(getattr(self, name), name, number)
        _optional(self.recovery_steps, "recovery_steps", integer)
        if type(self.matched_compute_budget) is not bool:
            raise ValueError("matched_compute_budget must be a boolean")
        object.__setattr__(self, "details", _mapping(self.details, "details"))


RECORD_TYPES = {
    cls.__name__: cls for cls in (RunMetadata, StepRecord, AdaptationEvent, InterventionRecord)
}


def to_object(record: Any) -> dict[str, Any]:
    if type(record).__name__ not in RECORD_TYPES or RECORD_TYPES[type(record).__name__] is not type(
        record
    ):
        raise ValueError("unsupported measurement record")

    def encode(value):
        if type(value) in (
            *RECORD_TYPES.values(),
            ComputationCounters,
            MemoryAccounting,
            RoutingSummary,
        ):
            return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
        return json_value(value)

    result = encode(record)
    result["record_type"] = type(record).__name__
    from_object(result)
    return result


def from_object(value: Any):
    if not isinstance(value, dict):
        raise ValueError("record must be a JSON object")
    payload = json_value(value)
    kind = payload.pop("record_type", None)
    if "protocol_version" not in payload:
        raise ValueError("protocol_version is required")
    if not isinstance(kind, str) or kind not in RECORD_TYPES:
        raise ValueError("record_type is unknown")
    cls = RECORD_TYPES[kind]
    allowed = {f.name for f in fields(cls)}
    if set(payload) - allowed:
        raise ValueError("unknown record fields: " + ", ".join(sorted(set(payload) - allowed)))
    if cls is StepRecord:
        for name, nested in (
            ("computation", ComputationCounters),
            ("memory", MemoryAccounting),
            ("routing", RoutingSummary),
        ):
            if payload.get(name) is not None:
                if not isinstance(payload[name], dict):
                    raise ValueError(f"{name} must be an object or null")
                try:
                    payload[name] = nested(**payload[name])
                except TypeError as exc:
                    raise ValueError(f"invalid {name} fields") from exc
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError("missing or invalid record fields") from exc
