"""Fixed-sample inference and time windows for paired carry interventions."""

import math
import statistics
from collections.abc import Mapping

from ..validation import json_value
from ._validation import integer, real, sequence
from .carry import summarize_carry_pair


def _binomial_tail(successes, count, probability):
    if probability == 0.5:
        return sum(math.comb(count, value) for value in range(successes, count + 1)) / 2**count
    if probability == 0:
        return float(successes == 0)
    if probability == 1:
        return 1.0
    log_p, log_q = math.log(probability), math.log1p(-probability)
    total = math.fsum(
        math.exp(
            math.lgamma(count + 1)
            - math.lgamma(value + 1)
            - math.lgamma(count - value + 1)
            + value * log_p
            + (count - value) * log_q
        )
        for value in range(successes, count + 1)
    )
    return min(1.0, total)


def _tail_probability(successes, count, target):
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _binomial_tail(successes, count, middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def confirm_carry_effects(results, *, indices):
    """Apply the mean-positive and one-sided exact sign-test criterion."""
    indices = sequence(indices, "indices")
    for index in indices:
        integer(index, "index")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("indices must be nonempty and unique")
    rows = {}
    seeds = set()
    identities = set()
    for result in sequence(results, "results"):
        if not isinstance(result, Mapping):
            raise ValueError("each result must be a mapping")
        index = integer(result.get("index"), "index")
        seed = integer(result.get("seed"), "seed")
        if index not in indices or index in rows or seed in seeds:
            raise ValueError("results contain an unexpected or duplicate index or seed")
        identity = result.get("run_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("each result requires a run identity")
        identities.add(identity)
        seeds.add(seed)
        effect = real(result.get("R_i_reset_minus_continued_post_mean_reward"), "effect")
        rows[index] = {"index": index, "seed": seed, "effect": effect}
    if set(rows) != set(indices) or len(identities) != 1:
        raise ValueError("results require every expected index and one run identity")
    effects = [rows[index]["effect"] for index in indices]
    count = len(effects)
    positives = sum(value > 0 for value in effects)
    mean = statistics.mean(effects)
    p_value = _binomial_tail(positives, count, 0.5)
    lower = _tail_probability(positives, count, 0.025) if positives else 0.0
    upper = _tail_probability(positives + 1, count, 0.975) if positives < count else 1.0
    return json_value(
        {
            "run_identity": next(iter(identities)),
            "samples": [rows[index] for index in indices],
            "sample_count": count,
            "positive_count": positives,
            "zero_count": sum(value == 0 for value in effects),
            "mean_paired_reward_effect": mean,
            "one_sided_sign_test": {"null_positive_probability": 0.5, "p_value": p_value},
            "positive_fraction_interval": {
                "method": "Clopper-Pearson",
                "confidence_level": 0.95,
                "sidedness": "two-sided",
                "lower": lower,
                "upper": upper,
            },
            "criterion": {"alpha": 0.05, "mean_effect_threshold": 0.0},
            "criterion_met": mean > 0 and p_value <= 0.05,
        }
    )


def summarize_confirmation_pair(continued, reset, *, boundary=150000, endpoint=300000):
    """Add the first 1,024 post-reset interactions and the remaining interval."""
    import numpy as np

    result = summarize_carry_pair(continued, reset, boundary=boundary, endpoint=endpoint)
    split = boundary + 1024
    if split >= endpoint:
        raise ValueError("confirmation requires more than 1024 post-boundary interactions")
    windows = {}
    for name, start, stop in (
        ("early", boundary, split),
        ("later", split, endpoint),
    ):
        control = float(np.mean(continued["rewards"][start:stop], dtype=np.float64))
        intervention = float(np.mean(reset["rewards"][start:stop], dtype=np.float64))
        windows[name] = {
            "start": start,
            "stop": stop,
            "samples": stop - start,
            "continued_mean_reward": control,
            "reset_mean_reward": intervention,
            "reset_minus_continued": intervention - control,
        }
    result["fixed_response_windows"] = windows
    return json_value(result)
