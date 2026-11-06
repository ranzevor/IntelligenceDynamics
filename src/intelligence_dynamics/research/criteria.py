"""Configured numeric thresholds with three-valued joint decisions."""

import operator
from collections.abc import Mapping
from dataclasses import dataclass

from ._validation import choice, real, sequence

_OPERATORS = {
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
    "eq": operator.eq,
}


@dataclass(frozen=True)
class Criterion:
    name: str
    metric: str
    direction: str
    threshold: float

    def __post_init__(self):
        for name in ("name", "metric"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"criterion {name} must be a nonempty string")
        choice(self.direction, "direction", tuple(_OPERATORS))
        object.__setattr__(self, "threshold", real(self.threshold, "threshold"))


@dataclass(frozen=True)
class CriterionResult:
    name: str
    metric: str
    direction: str
    threshold: float
    value: float | None
    passed: bool | None


@dataclass(frozen=True)
class Decision:
    results: tuple[CriterionResult, ...]
    combine: str
    passed: bool | None
    evaluated: int
    missing: int


def assess(values, criteria, *, combine="all"):
    """For all, any false decides false; for any, any true decides true.

    Otherwise missing metrics produce None. Absent keys and explicit None are
    missing. A fully observed decision uses the selected logical combination.
    """
    choice(combine, "combine", ("all", "any"))
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    checked = {}
    for name, value in values.items():
        if type(name) is not str or not name.strip():
            raise ValueError("metric names must be nonempty strings")
        checked[name] = None if value is None else real(value, f"metric {name}")
    rules = sequence(criteria, "criteria")
    if not rules or any(not isinstance(rule, Criterion) for rule in rules):
        raise ValueError("criteria must contain Criterion instances")
    if len({rule.name for rule in rules}) != len(rules):
        raise ValueError("criterion names must be unique")
    results = []
    for rule in rules:
        value = checked.get(rule.metric)
        passed = None if value is None else _OPERATORS[rule.direction](value, rule.threshold)
        results.append(
            CriterionResult(rule.name, rule.metric, rule.direction, rule.threshold, value, passed)
        )
    outcomes = tuple(row.passed for row in results)
    missing = sum(value is None for value in outcomes)
    decisive = False if combine == "all" else True
    if any(value is decisive for value in outcomes):
        passed = decisive
    elif missing:
        passed = None
    else:
        passed = not decisive
    return Decision(tuple(results), combine, passed, len(outcomes) - missing, missing)
