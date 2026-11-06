"""Experiment scale, population observations and environment contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from .storage import digest
from .validation import integer, json_value, number, text


@dataclass(frozen=True)
class ScaleProfile:
    seed: int = 2
    generations: int = 10
    population: int = 8
    parent_pool: int = 4
    elites: int = 2
    task_instances: int = 16
    trials: int = 10
    trial_steps: int = 10
    validation_trials: int = 4
    save_interval: int = 5

    def __post_init__(self):
        integer(self.seed, "seed")
        for name in self.__dataclass_fields__:
            if name != "seed":
                integer(getattr(self, name), name, minimum=1)
        if not self.elites <= self.parent_pool <= self.population:
            raise ValueError("elites <= parent_pool <= population is required")
        if self.validation_trials >= self.trials:
            raise ValueError("validation_trials must be smaller than trials")

    @property
    def interactions_per_individual(self):
        return self.task_instances * self.trials * self.trial_steps

    def configuration(self, learning: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(learning, dict):
            raise ValueError("learning must be an object")
        return {"scale": asdict(self), "learning": json_value(learning)}


@dataclass(frozen=True)
class IndividualObservation:
    individual_id: int
    fitness: float
    activation_projections: int
    modulatory_projections: int
    live_modulatory_projections: int
    weight_change_nm: float
    weight_change_rl: float
    active_learning_rate: float
    learning_rate_gene: float
    mean_susceptibility: float | None = None
    unmutated: bool = False
    mutations: tuple[str, ...] = ()

    def __post_init__(self):
        for name in (
            "individual_id",
            "activation_projections",
            "modulatory_projections",
            "live_modulatory_projections",
        ):
            integer(getattr(self, name), name)
        for name in (
            "fitness",
            "weight_change_nm",
            "weight_change_rl",
            "active_learning_rate",
            "learning_rate_gene",
        ):
            number(getattr(self, name), name)
        if self.mean_susceptibility is not None:
            number(self.mean_susceptibility, "mean_susceptibility")
        if type(self.unmutated) is not bool:
            raise ValueError("unmutated must be boolean")
        if not isinstance(self.mutations, (tuple, list)):
            raise ValueError("mutations must be a sequence")
        for item in self.mutations:
            text(item, "mutation")
        object.__setattr__(self, "mutations", tuple(self.mutations))


def population_snapshot(
    individuals, *, generation: int, parent_pool: int, elites: int, tie_break="individual_id"
):
    """Rank by fitness with explicit ID or stable input-order tie breaking."""
    integer(generation, "generation")
    population = list(individuals)
    if not population or any(not isinstance(item, IndividualObservation) for item in population):
        raise ValueError("population requires individual observations")
    integer(parent_pool, "parent_pool", minimum=1)
    integer(elites, "elites")
    if not elites <= parent_pool <= len(population):
        raise ValueError("elites <= parent_pool <= population is required")
    if len({item.individual_id for item in population}) != len(population):
        raise ValueError("individual IDs must be unique")
    if tie_break == "input_order":
        ranked = sorted(population, key=lambda item: item.fitness, reverse=True)
    elif tie_break == "individual_id":
        ranked = sorted(population, key=lambda item: (-item.fitness, item.individual_id))
    else:
        raise ValueError("tie_break must be individual_id or input_order")
    return {
        "generation": generation,
        "individuals": [
            {
                **asdict(item),
                "rank": rank,
                "selected_parent": rank < parent_pool,
                "elite": rank < elites,
            }
            for rank, item in enumerate(ranked)
        ],
    }


class Environment(Protocol):
    def reset(self, seed: int) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]: ...


def qualify_environment(
    factory: Callable[[], Environment],
    actions,
    *,
    seed: int = 0,
    observation_check: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Check deterministic reset and finite transitions through a declared adapter."""
    integer(seed, "seed")
    actions = tuple(json_value(action) for action in actions)
    if not actions:
        raise ValueError("at least one action is required")

    def trajectory():
        environment = factory()
        observation = json_value(environment.reset(seed), "observation")
        if observation_check is not None:
            observation_check(observation)
        initial = observation
        transitions = []
        for action in actions:
            result = environment.step(action)
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError("step must return observation, reward, terminated, info")
            observation, reward, terminated, info = result
            observation = json_value(observation, "observation")
            if observation_check is not None:
                observation_check(observation)
            number(reward, "reward")
            if type(terminated) is not bool or not isinstance(info, dict):
                raise ValueError("step terminated/info fields have invalid types")
            transitions.append(
                {
                    "observation": observation,
                    "reward": reward,
                    "terminated": terminated,
                    "info": json_value(info),
                }
            )
            if terminated:
                break
        return {"initial": initial, "transitions": transitions}

    first, second = trajectory(), trajectory()
    if digest(first) != digest(second):
        raise ValueError("same-seed environment trajectories differ")
    return {
        "seed": seed,
        "steps": len(first["transitions"]),
        "trajectory_sha256": digest(first),
        "trajectory": first,
    }
