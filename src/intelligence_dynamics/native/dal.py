"""Extract diagnostics from evaluated DAL individuals."""

from collections.abc import Mapping
from fractions import Fraction

from ..experiments import IndividualObservation, population_snapshot
from ._values import count, finite, plain, scalar


def _required(individual, name):
    try:
        return getattr(individual, name)
    except AttributeError as exc:
        raise ValueError(f"DAL individual is missing {name}") from exc


def _nonzero(value, name):
    value = plain(value)
    if isinstance(value, list):
        return sum(_nonzero(item, name) for item in value)
    if type(value) is bool:
        return int(value)
    return int(finite(value, name) != 0)


def dal_individual(individual, *, unmutated=None):
    """Read evaluated phenotype/genotype fields into a detached observation."""
    stats = getattr(individual, "stats", None)
    if stats is None:

        def absolute_sum(value):
            value = plain(value)
            if isinstance(value, list):
                return sum((absolute_sum(child) for child in value), Fraction())
            return Fraction(abs(finite(value, "weight change")))

        stats = {
            "total_weight_change_nm": float(
                absolute_sum(_required(individual, "total_dw_list_nm"))
            ),
            "total_weight_change_rl": float(
                absolute_sum(_required(individual, "total_dw_list_rl"))
            ),
            "mean_active_rl_learning_rate": _required(individual, "mean_active_rl_learning_rate"),
            "mutations_applied": _required(individual, "mutations_applied"),
        }
    if not isinstance(stats, Mapping):
        raise ValueError("DAL stats must be a mapping")
    required = (
        "total_weight_change_nm",
        "total_weight_change_rl",
        "mean_active_rl_learning_rate",
        "mutations_applied",
    )
    if any(name not in stats for name in required):
        raise ValueError("DAL stats are missing evaluated diagnostics")
    activation = tuple(_required(individual, "io_list_a"))
    modulation = tuple(_required(individual, "io_list_m"))
    susceptibilities = []
    for projection in activation:
        if not isinstance(projection, (list, tuple)) or len(projection) != 3:
            raise ValueError("activation projections require source, target, susceptibility")
        susceptibilities.append(finite(projection[2], "susceptibility"))
    mean = None
    if susceptibilities:
        mean = float(
            sum((Fraction(v) for v in susceptibilities), Fraction()) / len(susceptibilities)
        )
    mutations = stats["mutations_applied"]
    if not isinstance(mutations, (list, tuple)):
        raise ValueError("mutations_applied must be a sequence")
    unmutated = scalar(
        _required(individual, "is_unmutated_copy") if unmutated is None else unmutated
    )
    if type(unmutated) is not bool:
        raise ValueError("is_unmutated_copy must be boolean")
    return IndividualObservation(
        individual_id=count(_required(individual, "individual_id"), "individual_id"),
        fitness=finite(_required(individual, "normalised_fitness"), "normalised_fitness"),
        activation_projections=_nonzero(
            _required(individual, "connectivity_grid_a"), "connectivity_grid_a"
        ),
        modulatory_projections=_nonzero(
            _required(individual, "connectivity_grid_m"), "connectivity_grid_m"
        ),
        live_modulatory_projections=len(modulation),
        weight_change_nm=finite(stats["total_weight_change_nm"], "total_weight_change_nm"),
        weight_change_rl=finite(stats["total_weight_change_rl"], "total_weight_change_rl"),
        active_learning_rate=finite(
            stats["mean_active_rl_learning_rate"], "mean_active_rl_learning_rate"
        ),
        learning_rate_gene=finite(_required(individual, "rl_learning_rate"), "rl_learning_rate"),
        mean_susceptibility=mean,
        unmutated=unmutated,
        mutations=tuple(mutations),
    )


def dal_population(individuals, *, generation, parent_pool, elites, unmutated_flags=None):
    """Rank evaluated diagnostics using DAL's stable descending-fitness order."""
    individuals = tuple(individuals)
    flags = (None,) * len(individuals) if unmutated_flags is None else tuple(unmutated_flags)
    if len(flags) != len(individuals):
        raise ValueError("unmutated_flags must align with every individual")
    return population_snapshot(
        [
            dal_individual(individual, unmutated=flag)
            for individual, flag in zip(individuals, flags)
        ],
        generation=generation,
        parent_pool=parent_pool,
        elites=elites,
        tie_break="input_order",
    )


def dal_remote_population(workers, *, generation, parent_pool, elites, timeout=10.0):
    """Read evaluated DAL SPNN objects through their bounded existing pipe API."""
    from ..validation import number

    number(timeout, "timeout", minimum=0)
    if timeout == 0:
        raise ValueError("timeout must be positive")
    population, flags = [], []
    for worker in workers:
        for name in ("send", "recv"):
            if not callable(getattr(worker, name, None)):
                raise ValueError(f"DAL worker requires callable {name}")
        poll = getattr(getattr(worker, "pipe", None), "poll", None)
        if not callable(poll):
            raise ValueError("DAL worker pipe requires poll(timeout)")
        worker.send("get_over_here")
        if not poll(timeout):
            raise TimeoutError("DAL diagnostic response timed out")
        individual = worker.recv()
        if count(_required(individual, "individual_id"), "individual_id") != count(
            _required(worker, "individual_id"), "worker individual_id"
        ):
            raise ValueError("DAL response belongs to another individual")
        population.append(individual)
        flags.append(_required(worker, "is_unmutated_copy"))
    return dal_population(
        population,
        generation=generation,
        parent_pool=parent_pool,
        elites=elites,
        unmutated_flags=flags,
    )
