"""Recovery, lifetime plasticity and resource-frontier measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .validation import integer, number


@dataclass(frozen=True)
class Recovery:
    baseline: float
    floor: float
    target: float
    first_recovery_step: int | None
    normalized_auc: float
    final_value: float


def recovery(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    baseline: float,
    floor: float = 0.0,
    fraction: float = 0.9,
) -> Recovery:
    """Integrate normalized higher-is-better capability with trapezoidal weights."""
    if not steps or len(steps) != len(values):
        raise ValueError("steps and values require equal nonzero lengths")
    for step in steps:
        integer(step, "step")
    if any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must increase strictly")
    observed = [number(v, "capability") for v in values]
    base = number(baseline, "baseline")
    low = number(floor, "floor")
    f = number(fraction, "fraction", minimum=0, maximum=1)
    if base <= low:
        raise ValueError("baseline must exceed floor")
    scale = max(abs(base), abs(low), 1.0)
    denominator = base / scale - low / scale
    if denominator <= 0:
        raise ValueError("baseline and floor are numerically indistinguishable")
    normalized = [(v / scale - low / scale) / denominator for v in observed]
    if not all(math.isfinite(v) for v in normalized):
        raise ValueError("normalized capability exceeds finite range")
    target = base * f + low * (1 - f)
    horizon = steps[-1] - steps[0]
    area = (
        normalized[0]
        if horizon == 0
        else math.fsum(
            (b - a) / horizon * (left / 2 + right / 2)
            for a, b, left, right in zip(steps, steps[1:], normalized, normalized[1:])
        )
    )
    number(area, "normalized_auc")
    number(target, "target")
    first = next((s for s, v in zip(steps, observed) if v >= target), None)
    return Recovery(base, low, target, first, area, observed[-1])


def plasticity(
    points: Iterable[tuple[float, float]], *, reference: int = 0
) -> list[tuple[float, float]]:
    """Divide each probe score by one selected nonzero reference score."""
    values = [(number(t, "lifetime", minimum=0), number(s, "probe score")) for t, s in points]
    integer(reference, "reference")
    if not values or reference >= len(values):
        raise ValueError("reference must select an observed point")
    if any(a[0] >= b[0] for a, b in zip(values, values[1:])):
        raise ValueError("lifetimes must increase strictly")
    denominator = values[reference][1]
    if denominator == 0:
        raise ValueError("reference score must be nonzero")
    result = [(t, s / denominator) for t, s in values]
    for _, s in result:
        number(s, "normalized plasticity")
    return result


def observations(values: Iterable[float | None]) -> dict:
    items = list(values)
    known = [number(v, "observation") for v in items if v is not None]
    missing = len(items) - len(known)
    mean = math.fsum(v / len(known) for v in known) if known and missing == 0 else None
    if mean is not None:
        number(mean, "mean")
    return {"count": len(items), "observed": len(known), "missing": missing, "mean": mean}


def pareto_front(
    points: Iterable[dict], *, maximize: Sequence[str], minimize: Sequence[str]
) -> list[int]:
    """Return input indices that no other complete point strictly dominates."""
    if not maximize and not minimize:
        raise ValueError("at least one objective is required")
    names = list(maximize) + list(minimize)
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("objective names must be nonempty and distinct")
    rows = list(points)
    vectors = []
    for row in rows:
        if not isinstance(row, dict) or any(k not in row or row[k] is None for k in names):
            raise ValueError("each point requires every objective")
        vectors.append(tuple(number(row[k], k) * (1 if k in maximize else -1) for k in names))
    return [
        i
        for i, v in enumerate(vectors)
        if not any(
            j != i and all(a >= b for a, b in zip(w, v)) and any(a > b for a, b in zip(w, v))
            for j, w in enumerate(vectors)
        )
    ]
