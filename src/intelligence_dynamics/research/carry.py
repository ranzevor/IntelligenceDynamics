"""Paired reward and adaptation summaries for a single live-carry reset."""

import hashlib
from collections.abc import Mapping

from ..validation import json_value
from ._validation import integer

TRACE_FIELDS = ("rewards", "biome_id", "loss", "abs_td_error", "weight_change")
DIAGNOSTIC_FIELDS = ("loss", "abs_td_error", "weight_change")


def _validate_traces(traces, endpoint):
    import numpy as np

    if not isinstance(traces, Mapping) or set(traces) != set(TRACE_FIELDS):
        raise ValueError("traces must contain exactly the five carry-study fields")
    for name in TRACE_FIELDS:
        values = traces[name]
        if not isinstance(values, np.ndarray) or values.shape != (endpoint,):
            raise ValueError(f"{name} must be a one-dimensional array of length endpoint")
        allowed = "iu" if name == "biome_id" else "fiu"
        if values.dtype.kind not in allowed or not bool(np.isfinite(values).all()):
            raise ValueError(f"{name} must contain finite numeric samples")


def _windows(values, start, stop, width):
    import numpy as np

    return [
        {
            "start": position,
            "stop": min(position + width, stop),
            "samples": min(width, stop - position),
            "mean": float(
                np.mean(values[position : min(position + width, stop)], dtype=np.float64)
            ),
        }
        for position in range(start, stop, width)
    ]


def _events(rewards, changes, boundary, endpoint):
    import numpy as np

    events = []
    last_stop = boundary
    for position in changes:
        step = int(position)
        if step - 64 < last_stop or step + 64 > endpoint:
            continue
        last_stop = step + 64
        before = float(np.mean(rewards[step - 64 : step], dtype=np.float64))
        after = float(np.mean(rewards[step : step + 64], dtype=np.float64))
        target = 0.9 * before
        observed = min(2048, ((endpoint - step) // 64) * 64)
        recovery = None
        for offset in range(0, observed, 64):
            block = rewards[step + offset : step + offset + 64]
            if float(np.mean(block, dtype=np.float64)) >= target:
                recovery = offset + 64
                break
        events.append(
            {
                "step": step,
                "pre_mean": before,
                "post_mean": after,
                "cost_pre_minus_post": before - after,
                "recovery_target": target,
                "observation_steps": observed,
                "recovery_steps": recovery,
            }
        )
    return events


def _diagnostics(traces, boundary, endpoint):
    import numpy as np

    rewards = traces["rewards"]
    biomes = traces["biome_id"]
    changes = np.flatnonzero(biomes[1:] != biomes[:-1]) + 1
    events = _events(rewards, changes, boundary, endpoint)
    recovered = [event["recovery_steps"] for event in events if event["recovery_steps"] is not None]
    costs = [event["cost_pre_minus_post"] for event in events]
    censored = len(events) - len(recovered)
    return {
        "post_boundary_mean_reward": float(np.mean(rewards[boundary:endpoint], dtype=np.float64)),
        "post_boundary_samples": endpoint - boundary,
        "reward_windows_25k": _windows(rewards, 0, endpoint, 25000),
        "update_blocks_50k": {
            name: _windows(traces[name], boundary, endpoint, 50000) for name in DIAGNOSTIC_FIELDS
        },
        "native_biome_change_count": int(len(changes)),
        "post_boundary_biome_change_count": int(np.count_nonzero(changes >= boundary)),
        "biome_occupancy": {
            str(int(biome)): int(np.count_nonzero(biomes == biome)) for biome in np.unique(biomes)
        },
        "post_boundary_eligible_biome_events": len(events),
        "events": events,
        "event_cost_pre_minus_post_mean": float(np.mean(costs, dtype=np.float64))
        if costs
        else None,
        "recovery_median_steps": float(np.median(recovered)) if recovered else None,
        "recovery_recovered_count": len(recovered),
        "recovery_censored_count": censored,
        "recovery_censored_fraction": censored / len(events) if events else None,
        "recovery_endpoint_censored_count": sum(
            event["recovery_steps"] is None and event["observation_steps"] < 2048
            for event in events
        ),
    }


def summarize_carry_pair(continued, reset, *, boundary=150000, endpoint=300000):
    """Compare complete paired traces, requiring identical typed prehistory.

    Event windows [step-64, step+64) are selected chronologically without
    overlap. Recovery is the first 64-step block mean at least 0.9 times the
    pre-event mean, including zero and negative baselines. Its observation
    extends across later biome changes, up to 2048 steps or the endpoint.
    """
    integer(boundary, "boundary", 1)
    integer(endpoint, "endpoint", boundary + 1)
    _validate_traces(continued, endpoint)
    _validate_traces(reset, endpoint)
    prehistory = {}
    for name in TRACE_FIELDS:
        left, right = continued[name][:boundary], reset[name][:boundary]
        if left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise ValueError(f"{name} prehistory differs in dtype or bytes")
        prehistory[name] = {
            "dtype": left.dtype.str,
            "shape": [boundary],
            "sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        }
    control = _diagnostics(continued, boundary, endpoint)
    intervention = _diagnostics(reset, boundary, endpoint)
    result = {
        "boundary": boundary,
        "endpoint": endpoint,
        "matched_history_exact": True,
        "prehistory_sha256": prehistory,
        "continued": control,
        "reset": intervention,
        "R_i_reset_minus_continued_post_mean_reward": (
            intervention["post_boundary_mean_reward"] - control["post_boundary_mean_reward"]
        ),
    }
    return json_value(result)
