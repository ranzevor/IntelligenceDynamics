"""Episode evaluation, feedback corruption, and observation-aware summaries."""

from dataclasses import dataclass
from random import Random
from statistics import fmean, mean

from ._validation import choice, integer, real, sequence
from .policy import MODES, AppliedUpdate, FixedPolicy, LearningRule, TabularPolicy
from .task import CueTask, Feedback, StepRecord, TaskConfig, validate_bank


@dataclass(frozen=True)
class Observations:
    count: int
    observed: int
    missing: int
    mean: float | None


def summarize(values):
    data = tuple(values)
    known = [real(value, "observation") for value in data if value is not None]
    missing = len(data) - len(known)
    return Observations(
        len(data), len(known), missing, float(mean(known)) if known and not missing else None
    )


def corruption_plan(length, *, kind="clean", rate=0.0, seed=0, burst_length=3):
    """Rate is a start probability; isolated flips require a clean following event."""
    integer(length, "length")
    integer(seed, "seed")
    integer(burst_length, "burst_length", 1)
    choice(kind, "corruption", ("clean", "isolated", "burst"))
    rate = real(rate, "rate", 0, 1)
    if kind == "clean" and rate != 0:
        raise ValueError("clean corruption requires rate=0")
    flags = [False] * length
    if kind == "clean":
        return tuple(flags)
    generator = Random(seed)
    index = 0
    while index < length:
        if generator.random() < rate:
            width = 1 if kind == "isolated" else burst_length
            for position in range(index, min(index + width, length)):
                flags[position] = True
            index += width + 1
        else:
            index += 1
    return tuple(flags)


def adaptation_latency(correct, start, stop, *, window=6, threshold=0.75):
    """Return the first qualifying window offset within the half-open interval."""
    data = sequence(correct, "correct")
    if any(type(value) is not bool for value in data):
        raise ValueError("correct must contain booleans")
    integer(start, "start")
    integer(stop, "stop")
    integer(window, "window", 1)
    threshold = real(threshold, "threshold", 0, 1)
    if not start <= stop <= len(data):
        raise ValueError("invalid adaptation interval")
    for position in range(start, stop - window + 1):
        if sum(data[position : position + window]) / window >= threshold:
            return position - start
    return None


@dataclass(frozen=True)
class Delivery:
    credit_id: int
    step: int
    corrupted: bool
    value: float
    committed_ids: tuple[int, ...]
    error: float | None


@dataclass(frozen=True)
class EpisodeResult:
    accuracy: float
    segment_accuracy: tuple[float, ...]
    switch_latency: tuple[int | None, ...]
    corruption_recovery: tuple[int | None, ...]
    records: tuple[StepRecord, ...]
    deliveries: tuple[Delivery, ...]
    updates: tuple[AppliedUpdate, ...]
    undelivered: int


def run_episode(
    config,
    policy,
    *,
    seed=0,
    corruption="clean",
    corruption_rate=0.0,
    corruption_seed=None,
    latency_window=6,
    latency_threshold=0.75,
):
    """Reset policy state and run actions; feedback after the final action is not learned."""
    if not isinstance(config, TaskConfig):
        raise TypeError("config must be TaskConfig")
    if not isinstance(policy, (TabularPolicy, FixedPolicy)):
        raise TypeError("policy must be TabularPolicy or FixedPolicy")
    width = len(policy.rule.priors) if isinstance(policy, TabularPolicy) else len(policy.mapping)
    if width != config.cue_count:
        raise ValueError("policy width must equal cue_count")
    integer(seed, "seed")
    integer(latency_window, "latency_window", 1)
    real(latency_threshold, "latency_threshold", 0, 1)
    noise_seed = seed if corruption_seed is None else integer(corruption_seed, "corruption_seed")
    plan = corruption_plan(config.steps, kind=corruption, rate=corruption_rate, seed=noise_seed)
    task = CueTask(config)
    observation = task.reset()
    policy.reset(seed=seed)
    feedback = None
    records, deliveries = [], []
    for step in range(config.steps):
        corrupted = False
        if feedback is not None:
            corrupted = plan[len(deliveries)]
            if corrupted:
                feedback = Feedback(feedback.credit_id, 1.0 - feedback.value)
        action = policy.act(observation, feedback)
        if feedback is not None:
            deliveries.append(
                Delivery(
                    feedback.credit_id,
                    step,
                    corrupted,
                    feedback.value,
                    policy.last_committed,
                    policy.last_error if isinstance(policy, TabularPolicy) else None,
                )
            )
        transition = task.advance(action)
        records.append(transition.record)
        if not transition.done and transition.observation is None:
            raise RuntimeError("active task returned no observation")
        observation, feedback = transition.observation, transition.feedback
    correct = tuple(record.reward == 1 for record in records)
    bounds = (0, *config.switches, config.steps)
    segments = tuple(fmean(correct[a:b]) for a, b in zip(bounds, bounds[1:]))
    latencies = tuple(
        adaptation_latency(correct, a, b, window=latency_window, threshold=latency_threshold)
        for a, b in zip(bounds[1:-1], bounds[2:])
    )
    corrupt_steps = [delivery.step for delivery in deliveries if delivery.corrupted]
    ends = [
        step
        for index, step in enumerate(corrupt_steps)
        if index + 1 == len(corrupt_steps) or corrupt_steps[index + 1] != step + 1
    ]
    recovery = []
    for end in ends:
        later = [step for step in (*corrupt_steps, *config.switches) if step > end]
        stop = min(later, default=config.steps)
        recovery.append(
            adaptation_latency(
                correct, end + 1, stop, window=latency_window, threshold=latency_threshold
            )
        )
    return EpisodeResult(
        fmean(correct),
        segments,
        latencies,
        tuple(recovery),
        tuple(records),
        tuple(deliveries),
        policy.updates,
        config.steps - len(deliveries),
    )


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    stationary_accuracy: float | None
    switching_accuracy: float | None
    episode_scores: tuple[float, ...]
    feedback_count: int
    update_count: int
    update_fraction: float | None
    corrupt_feedback_count: int
    clean_feedback_count: int
    corrupt_source_updates: int
    clean_source_updates: int
    corrupt_source_update_rate: float | None
    clean_source_update_rate: float | None
    switch_feedback_count: int
    switch_source_update_rate: float | None
    switch_latency: Observations
    corruption_recovery: Observations
    updated_errors: Observations
    skipped_errors: Observations
    gate_error_separation: float | None
    corrupted_feedback_fraction: float | None


def evaluate(rule, scenarios, *, mode="plastic", seed=0, corruption="clean", corruption_rate=0.0):
    """Average episode accuracy and count updates by their originating feedback."""
    if not isinstance(rule, LearningRule):
        raise TypeError("rule must be LearningRule")
    choice(mode, "mode", MODES)
    integer(seed, "seed")
    bank = validate_bank(scenarios)
    if len(rule.priors) != bank[0].task.cue_count:
        raise ValueError("rule width must equal cue_count")
    episodes = []
    categories = {"stationary": [], "switching": []}
    feedbacks = updates = corrupt = corrupt_updates = switch_feedbacks = switch_updates = 0
    latencies, recovery, updated_errors, skipped_errors = [], [], [], []
    for scenario in bank:
        stream = seed ^ scenario.task.seed
        report = run_episode(
            scenario.task,
            TabularPolicy(rule, mode),
            seed=stream,
            corruption=corruption,
            corruption_rate=corruption_rate,
            corruption_seed=stream ^ 0x3972,
        )
        episodes.append(report.accuracy)
        categories[scenario.kind].append(report.accuracy)
        feedbacks += len(report.deliveries)
        updates += len(report.updates)
        corrupt += sum(row.corrupted for row in report.deliveries)
        by_credit = {row.credit_id: row for row in report.deliveries}
        corrupt_updates += sum(by_credit[row.credit_id].corrupted for row in report.updates)
        committed = {row.credit_id for row in report.updates}
        for delivery in report.deliveries:
            (updated_errors if delivery.credit_id in committed else skipped_errors).append(
                delivery.error
            )
        near_switch = {
            row.credit_id
            for row in report.deliveries
            if any(s <= row.credit_id < s + 12 for s in scenario.task.switches)
        }
        switch_feedbacks += len(near_switch)
        switch_updates += sum(row.credit_id in near_switch for row in report.updates)
        latencies.extend(report.switch_latency)
        recovery.extend(report.corruption_recovery)
    clean = feedbacks - corrupt
    clean_updates = updates - corrupt_updates

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    error_updated, error_skipped = summarize(updated_errors), summarize(skipped_errors)
    separation = None
    if error_updated.mean is not None and error_skipped.mean is not None:
        separation = error_updated.mean - error_skipped.mean
    return Evaluation(
        fmean(episodes),
        fmean(categories["stationary"]) if categories["stationary"] else None,
        fmean(categories["switching"]) if categories["switching"] else None,
        tuple(episodes),
        feedbacks,
        updates,
        ratio(updates, feedbacks),
        corrupt,
        clean,
        corrupt_updates,
        clean_updates,
        ratio(corrupt_updates, corrupt),
        ratio(clean_updates, clean),
        switch_feedbacks,
        ratio(switch_updates, switch_feedbacks),
        summarize(latencies),
        summarize(recovery),
        error_updated,
        error_skipped,
        separation,
        ratio(corrupt, feedbacks),
    )
