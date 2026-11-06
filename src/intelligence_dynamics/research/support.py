"""Paired event responses measured on shared event timestamps."""

from dataclasses import dataclass
from statistics import mean

from ._validation import integer, real, sequence
from .evaluation import Observations, summarize


@dataclass(frozen=True)
class Series:
    rewards: tuple[float, ...]
    states: tuple[int | str, ...]

    def __post_init__(self):
        rewards = tuple(real(value, "reward") for value in sequence(self.rewards, "rewards"))
        states = sequence(self.states, "states")
        if not rewards or len(states) != len(rewards):
            raise ValueError("rewards and states must have equal positive lengths")
        if any(
            type(value) not in (int, str) or (type(value) is str and not value) for value in states
        ):
            raise ValueError("states must contain integer or nonempty string identifiers")
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "states", states)


def select_events(events, *, spacing):
    integer(spacing, "spacing", 1)
    events = sequence(events, "events")
    for event in events:
        integer(event, "event")
    if any(right <= left for left, right in zip(events, events[1:])):
        raise ValueError("events must be strictly increasing")
    selected = []
    for event in events:
        if not selected or event - selected[-1] >= spacing:
            selected.append(event)
    return tuple(selected)


@dataclass(frozen=True)
class SupportComparison:
    control_events: tuple[int, ...]
    treatment_events: tuple[int, ...]
    shared_events: tuple[int, ...]
    selected_events: tuple[int, ...]
    jaccard: float | None
    control_response: float | None
    treatment_response: float | None
    delta: float | None
    event_deltas: tuple[float, ...]


def common_support(control, treatment, *, start=0, stop=None, window=3, spacing=None):
    """Compare post-minus-pre reward at shared events; delta is treatment minus control."""
    if not isinstance(control, Series) or not isinstance(treatment, Series):
        raise TypeError("control and treatment must be Series")
    if len(control.rewards) != len(treatment.rewards):
        raise ValueError("paired series must share a time horizon")
    integer(start, "start")
    end = len(control.rewards) if stop is None else integer(stop, "stop")
    integer(window, "window", 1)
    separation = 2 * window if spacing is None else integer(spacing, "spacing", 1)
    if not start < end <= len(control.rewards):
        raise ValueError("invalid comparison interval")
    if separation < 2 * window:
        raise ValueError("spacing must be at least twice window")

    def events(series):
        return tuple(
            t
            for t in range(max(1, start + window), end - window + 1)
            if series.states[t] != series.states[t - 1]
        )

    control_events, treatment_events = events(control), events(treatment)
    shared = tuple(sorted(set(control_events) & set(treatment_events)))
    selected = select_events(shared, spacing=separation)
    union = set(control_events) | set(treatment_events)

    def responses(series):
        return tuple(
            real(
                mean(series.rewards[t : t + window]) - mean(series.rewards[t - window : t]),
                "event response",
            )
            for t in selected
        )

    left, right = responses(control), responses(treatment)
    differences = tuple(real(b - a, "event delta") for a, b in zip(left, right))
    return SupportComparison(
        control_events,
        treatment_events,
        shared,
        selected,
        len(shared) / len(union) if union else None,
        float(mean(left)) if left else None,
        float(mean(right)) if right else None,
        float(mean(differences)) if differences else None,
        differences,
    )


@dataclass(frozen=True)
class ComparisonAggregate:
    observations: Observations
    deltas: tuple[float | None, ...]
    median: float | None
    minimum: float | None
    maximum: float | None
    positive: int
    negative: int


@dataclass(frozen=True)
class BlockComparison:
    start: int
    stop: int
    control_mean: float
    treatment_mean: float
    delta: float


def compare_blocks(control, treatment, *, edges):
    """Compare reward means in consecutive half-open intervals defined by edges."""
    if not isinstance(control, Series) or not isinstance(treatment, Series):
        raise TypeError("control and treatment must be Series")
    if len(control.rewards) != len(treatment.rewards):
        raise ValueError("paired series must share a time horizon")
    boundaries = sequence(edges, "edges")
    if len(boundaries) < 2:
        raise ValueError("edges must contain at least two boundaries")
    for boundary in boundaries:
        integer(boundary, "edge")
        if boundary > len(control.rewards):
            raise ValueError("edge exceeds the time horizon")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("edges must be strictly increasing")
    blocks = []
    for start, stop in zip(boundaries, boundaries[1:]):
        left = float(mean(control.rewards[start:stop]))
        right = float(mean(treatment.rewards[start:stop]))
        blocks.append(BlockComparison(start, stop, left, right, real(right - left, "block delta")))
    return tuple(blocks)


def aggregate_comparisons(deltas):
    values = tuple(None if value is None else real(value, "delta") for value in deltas)
    summary = summarize(values)
    known = tuple(value for value in values if value is not None)
    complete = bool(known) and not summary.missing
    middle = None
    if complete:
        ordered = sorted(known)
        index = len(ordered) // 2
        middle = (
            float(mean(ordered[index - 1 : index + 1])) if len(ordered) % 2 == 0 else ordered[index]
        )
    return ComparisonAggregate(
        summary,
        values,
        middle,
        min(known) if complete else None,
        max(known) if complete else None,
        sum(value > 0 for value in known),
        sum(value < 0 for value in known),
    )
