"""Data adapters for tabular measurements and array traces."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import BadZipFile

from .records import StepRecord
from .validation import integer, json_value, number, text


@dataclass(frozen=True)
class DalScale:
    task_instances: int
    trials: int
    trial_steps: int

    def __post_init__(self):
        for name in ("task_instances", "trials", "trial_steps"):
            integer(getattr(self, name), name, minimum=1)

    @property
    def interactions(self) -> int:
        return self.task_instances * self.trials * self.trial_steps


_DAL_FIELDS = (
    "generation",
    "focal_fitness",
    "unmutated_fitness",
    "mean_fitness",
    "weight_change_nm",
    "weight_change_rl",
    "rl_learning_rate",
    "action_sigma_bias",
    "nm_update_fraction",
    "nm_only_fitness",
    "rl_only_fitness",
)


def dal_rows(lines: Iterable[str], *, run_id: str, scale: DalScale) -> Iterable[StepRecord]:
    """Convert eleven leading DAL CSV columns and trailing trial scores."""
    text(run_id, "run_id")
    if not isinstance(scale, DalScale):
        raise ValueError("scale must be DalScale")
    previous = -1
    for line_number, parts in enumerate(csv.reader(lines, strict=True), 1):
        if len(parts) < 11 or any(not p.strip() for p in parts):
            raise ValueError(f"DAL row {line_number}: expected eleven nonempty leading columns")
        parts = [p.strip() for p in parts]
        try:
            generation = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"DAL row {line_number}: invalid generation") from exc
        integer(generation, "generation")
        if parts[0] != str(generation):
            raise ValueError("generation requires canonical decimal integer")
        if generation <= previous:
            raise ValueError("DAL generations must increase strictly")
        previous = generation
        native = {"generation": generation}
        for i, name in enumerate(_DAL_FIELDS[1:], 1):
            if i in (9, 10) and parts[i] == "None":
                native[name] = None
            else:
                try:
                    value = float(parts[i])
                except ValueError as exc:
                    raise ValueError(f"DAL row {line_number}: invalid {name}") from exc
                native[name] = number(value, name)
        number(native["nm_update_fraction"], "nm_update_fraction", minimum=0, maximum=1)
        progress = []
        for v in parts[11:]:
            try:
                progress.append(number(float(v), "learning_progress"))
            except ValueError as exc:
                raise ValueError(f"DAL row {line_number}: invalid trial score") from exc
        native["learning_progress"] = progress
        capability = native.pop("focal_fitness")
        step = generation * scale.interactions
        yield StepRecord(
            run_id,
            step,
            number(step, "lifetime"),
            capability=capability,
            context_id=f"generation:{generation}",
            extras={"format": "dal-generation-csv", "native": native},
        )


def forager_rows(
    rows: Iterable[dict[str, Any]], *, run_id: str, capability_field: str, step_field: str = "steps"
) -> Iterable[StepRecord]:
    """Map explicit fields and preserve remaining native diagnostics."""
    text(run_id, "run_id")
    text(capability_field, "capability_field")
    text(step_field, "step_field")
    if capability_field == step_field:
        raise ValueError("step and capability fields must differ")
    previous = -1
    for row in rows:
        value = json_value(row, "row")
        if not isinstance(value, dict):
            raise ValueError("native row must be an object")
        if step_field not in value or capability_field not in value:
            raise ValueError("native row is missing required fields")
        step = integer(value[step_field], step_field)
        if step <= previous:
            raise ValueError("measurement steps must increase strictly")
        previous = step
        capability = number(value[capability_field], capability_field)
        native = {k: v for k, v in value.items() if k not in (step_field, capability_field)}
        reward = value.get("mean_reward", value.get("return"))
        loss = value.get("total_loss")
        yield StepRecord(
            run_id,
            step,
            number(step, "lifetime"),
            capability=capability,
            reward_or_return=None if reward is None else number(reward, "reward"),
            loss=None if loss is None else number(loss, "loss"),
            extras={
                "format": "forager-row",
                "capability_field": capability_field,
                "native": native,
            },
        )


def _npz_arrays(path):
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("NPZ input requires the arrays dependency group") from exc
    with Path(path).open("rb") as source:
        try:
            archive = np.load(source, allow_pickle=False)
        except (ValueError, BadZipFile) as exc:
            raise ValueError("NPZ input must be a valid NumPy ZIP archive") from exc
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("NPZ input must be a valid NumPy ZIP archive")
        with archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("NPZ array names must be unique")
            arrays = {name: archive[name] for name in archive.files}
            if any(not isinstance(array, np.ndarray) for array in arrays.values()):
                raise ValueError("NPZ members must contain NumPy arrays")
    return np, arrays


def npz_records(
    path: str | Path, *, run_id: str, capability_field: str = "rewards", interval: int = 1
) -> Iterable[StepRecord]:
    """Convert equally sampled one-dimensional numeric NPZ arrays."""
    integer(interval, "interval", minimum=1)
    text(capability_field, "capability_field")
    np, arrays = _npz_arrays(path)
    if capability_field not in arrays:
        raise ValueError("NPZ is missing the capability array")
    length = len(arrays[capability_field]) if arrays[capability_field].ndim == 1 else 0
    if not length:
        raise ValueError("NPZ arrays require at least one sample")
    for name, array in arrays.items():
        text(name, "array name")
        if (
            array.ndim != 1
            or len(array) != length
            or array.dtype.kind not in "iuf"
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"NPZ array {name} must contain {length} finite numeric samples")

    def rows():
        for index in range(length):
            row = {name: array[index].item() for name, array in arrays.items()}
            row["measurement_step"] = (index + 1) * interval
            yield row

    if "measurement_step" in arrays:
        raise ValueError("NPZ name measurement_step is reserved")
    yield from forager_rows(
        rows(), run_id=run_id, capability_field=capability_field, step_field="measurement_step"
    )


def forager_npz_records(
    path: str | Path, *, run_id: str, expected_rollout_steps: int | None = None
) -> Iterable[StepRecord]:
    """Aggregate step rewards onto optimizer updates and retain native context."""
    if expected_rollout_steps is not None:
        integer(expected_rollout_steps, "expected_rollout_steps", minimum=1)
    np, arrays = _npz_arrays(path)
    for name, array in arrays.items():
        text(name, "array name")
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(f"array {name} must be finite and numeric")
    if not {"rewards", "total_loss"} <= arrays.keys():
        raise ValueError("NPZ requires rewards and total_loss")
    rewards, losses = arrays["rewards"], arrays["total_loss"]
    if rewards.ndim != 1 or losses.ndim != 1 or not len(rewards) or not len(losses):
        raise ValueError("rewards and total_loss require nonempty one-dimensional arrays")
    updates, samples = len(losses), len(rewards)
    if samples % updates:
        raise ValueError("reward samples must divide evenly into optimizer updates")
    interval = samples // updates
    if expected_rollout_steps is not None and interval != expected_rollout_steps:
        raise ValueError("derived rollout interval differs from expected_rollout_steps")
    per_step = {"object_collected_id", "biome_regret", "biome_rank", "pos", "biome_id"}
    for name, array in arrays.items():
        if name == "rewards":
            continue
        if name in per_step:
            if array.ndim < 1 or len(array) != samples:
                raise ValueError(f"array {name} must have {samples} step samples")
            if name not in {"pos", "biome_id"} and array.ndim != 1:
                raise ValueError(f"array {name} must be one-dimensional")
        elif array.shape not in {(updates,), (1,)}:
            raise ValueError(f"array {name} must contain update samples or one static scalar")
    from fractions import Fraction

    def mean(values):
        total = sum((Fraction(float(value)) for value in values), Fraction())
        return float(total / len(values))

    for index in range(updates):
        start, end = index * interval, (index + 1) * interval
        native = {
            name: array[index if len(array) == updates else 0].item()
            for name, array in arrays.items()
            if name not in per_step | {"rewards"}
        }
        native["rollout_steps"] = interval
        if "object_collected_id" in arrays:
            native["object_collected_count"] = int(
                (arrays["object_collected_id"][start:end] >= 0).sum()
            )
        for name in ("biome_regret", "biome_rank"):
            if name in arrays:
                native[name + "_mean"] = mean(arrays[name][start:end])
        for name in ("pos", "biome_id"):
            if name in arrays:
                native[name] = arrays[name][start:end].tolist()
        reward = mean(rewards[start:end])
        yield StepRecord(
            run_id,
            end,
            float(end),
            capability=reward,
            reward_or_return=reward,
            loss=float(losses[index]),
            extras={"format": "forager-update-npz", "native": native},
        )
