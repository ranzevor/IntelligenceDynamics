"""Elitist rule search using training tasks and fixed common random numbers."""

from dataclasses import dataclass, replace
from random import Random
from statistics import fmean

from ._validation import choice, integer, real
from .evaluation import Evaluation, evaluate
from .policy import MODES, LearningRule
from .task import validate_bank


@dataclass(frozen=True)
class SearchConfig:
    population: int = 4
    generations: int = 2
    elites: int = 1
    mutation: float = 0.15
    update_cost: float = 0.05
    corruption_rate: float = 0.1
    noise_weight: float = 0.5

    def __post_init__(self):
        integer(self.population, "population", 2)
        integer(self.generations, "generations", 1)
        integer(self.elites, "elites", 1)
        if self.elites >= self.population:
            raise ValueError("elites must be less than population")
        for name in ("mutation", "update_cost"):
            object.__setattr__(self, name, real(getattr(self, name), name, 0, 1))
        for name in ("corruption_rate", "noise_weight"):
            object.__setattr__(self, name, real(getattr(self, name), name, 0, 1))


@dataclass(frozen=True)
class SearchResult:
    rule: LearningRule
    mode: str
    fitness: float
    train_accuracy: float
    train_update_fraction: float
    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    candidate_evaluations: int
    training_fingerprints: tuple[str, ...]


def _sample(generator, width):
    return LearningRule(
        priors=tuple(generator.uniform(-1, 1) for _ in range(width)),
        rate=generator.uniform(0.05, 1),
        exploration=generator.uniform(0, 0.3),
        threshold=generator.random(),
        decay=generator.random(),
        accumulation=generator.uniform(0.05, 1.5),
    )


def _offspring(rule, generator, scale):
    def move(value, lower, upper):
        return min(upper, max(lower, value + generator.gauss(0, scale)))

    return replace(
        rule,
        priors=tuple(move(value, -1, 1) for value in rule.priors),
        rate=move(rule.rate, 0.01, 1),
        exploration=move(rule.exploration, 0, 1),
        threshold=move(rule.threshold, 0, 1),
        decay=move(rule.decay, 0, 1),
        accumulation=move(rule.accumulation, 0.01, 3),
    )


def evolve(train, *, mode="buffered", seed=0, config=SearchConfig()):
    """Select a rule from training data; this function accepts no heldout input."""
    if not isinstance(config, SearchConfig):
        raise TypeError("config must be SearchConfig")
    choice(mode, "mode", MODES)
    integer(seed, "seed")
    bank = validate_bank(train, "train")
    if any(item.task.delay + 1 >= item.task.steps for item in bank):
        raise ValueError("search tasks must deliver feedback before the last action")
    generator = Random(seed)
    population = [_sample(generator, bank[0].task.cue_count) for _ in range(config.population)]
    best, history_best, history_mean = None, [], []
    for generation in range(config.generations):
        ranked = []
        for rule in population:
            clean = evaluate(rule, bank, mode=mode, seed=seed)
            noisy = evaluate(
                rule,
                bank,
                mode=mode,
                seed=seed,
                corruption="isolated",
                corruption_rate=config.corruption_rate,
            )
            if clean.update_fraction is None or noisy.update_fraction is None:
                raise ValueError("fitness requires delivered feedback")
            weight = config.noise_weight
            accuracy = (1 - weight) * clean.accuracy + weight * noisy.accuracy
            update_fraction = (1 - weight) * clean.update_fraction + weight * noisy.update_fraction
            fitness = accuracy - config.update_cost * update_fraction
            ranked.append((fitness, rule, accuracy, update_fraction))
        ranked.sort(key=lambda row: row[0], reverse=True)
        history_best.append(ranked[0][0])
        history_mean.append(fmean(row[0] for row in ranked))
        if best is None or ranked[0][0] > best[0]:
            best = ranked[0]
        if generation + 1 < config.generations:
            parents = [row[1] for row in ranked[: config.elites]]
            population = parents.copy()
            while len(population) < config.population:
                population.append(_offspring(generator.choice(parents), generator, config.mutation))
    if best is None:
        raise RuntimeError("search produced no candidates")
    return SearchResult(
        best[1],
        mode,
        best[0],
        best[2],
        best[3],
        tuple(history_best),
        tuple(history_mean),
        config.population * config.generations,
        tuple(item.task.fingerprint for item in bank),
    )


@dataclass(frozen=True)
class StudyResult:
    search: SearchResult
    heldout: Evaluation
    frozen: Evaluation
    always: Evaluation
    matched_random: Evaluation
    isolated: Evaluation
    burst: Evaluation
    random_probability: float
    heldout_fingerprints: tuple[str, ...]


def run_study(train, heldout, *, mode="buffered", seed=0, config=SearchConfig()):
    """Evaluate one selected rule and controls after checking split disjointness."""
    training = validate_bank(train, "train")
    testing = validate_bank(heldout, "heldout")
    if training[0].task.cue_count != testing[0].task.cue_count:
        raise ValueError("training and heldout cue counts differ")
    for values in (
        lambda item: item.name,
        lambda item: item.task.fingerprint,
        lambda item: item.task.seed,
    ):
        if {values(item) for item in training} & {values(item) for item in testing}:
            raise ValueError("training and heldout tasks overlap")
    selected = evolve(training, mode=mode, seed=seed, config=config)
    calibration = evaluate(selected.rule, training, mode=mode, seed=seed)
    if calibration.update_fraction is None:
        raise ValueError("random control calibration requires feedback")
    probability = calibration.update_fraction
    random_rule = replace(selected.rule, random_probability=probability)
    options = {"seed": seed}
    clean = evaluate(selected.rule, testing, mode=mode, **options)
    frozen = evaluate(selected.rule, testing, mode="fixed", **options)
    always = evaluate(selected.rule, testing, mode="plastic", **options)
    random_control = evaluate(random_rule, testing, mode="random", **options)
    isolated = evaluate(
        selected.rule,
        testing,
        mode=mode,
        corruption="isolated",
        corruption_rate=config.corruption_rate,
        **options,
    )
    burst = evaluate(
        selected.rule,
        testing,
        mode=mode,
        corruption="burst",
        corruption_rate=config.corruption_rate,
        **options,
    )
    return StudyResult(
        selected,
        clean,
        frozen,
        always,
        random_control,
        isolated,
        burst,
        probability,
        tuple(item.task.fingerprint for item in testing),
    )
