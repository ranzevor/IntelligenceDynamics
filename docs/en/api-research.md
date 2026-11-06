# Research API

[中文](../api-research.md) · [Research contract](research.md)

The public entry point is `intelligence_dynamics.research`. Python 3.10 or later is required; the subpackage uses only the standard library. Every Python block below runs independently in an environment where the package is importable. For a source checkout, add the project's `src` directory to `PYTHONPATH`.

Results are dataclasses. Convert them with `dataclasses.asdict`, then serialize with `json.dumps(..., allow_nan=False)`. JSON represents tuples as arrays and `None` as `null`.

## 1. Types and validation

Counts, times, cues, action identifiers, and seeds use Python integers; booleans are not accepted as integers. Seeds are nonnegative. Numeric parameters must be finite, and probabilities lie in the closed interval from 0 to 1. Invalid values generally raise `ValueError`, incorrect object types generally raise `TypeError`, and invalid lifecycle operations raise `RuntimeError`.

`TaskConfig`, `LearningRule`, `Scenario`, `SearchConfig`, and result dataclasses are immutable. Configurable sequences are converted to tuples at construction, so later changes to the caller's input lists do not change the configuration.

## 2. Tasks and scenarios

### `TaskConfig`

| Field | Default | Contract |
|---|---|---|
| `steps` | `96` | Positive integer action count |
| `cue_count` | `4` | Positive integer number of cues |
| `delay` | `1` | Nonnegative integer; feedback for action t arrives before action t + delay + 1 |
| `switches` | `(32, 64)` | Strictly increasing integer positions inside the lifetime |
| `mappings` | `((0,1,0,1),(1,0,1,0))` | Each row has cue_count entries, each the integer 0 or 1 |
| `seed` | `0` | Environment random seed |

Switching requires at least two mapping rows; a stationary task may use one. Each switch advances to the next row, wrapping after the final row. `fingerprint` is a SHA-256 identifier for the complete task configuration. When changing `steps` or `cue_count`, provide compatible switch positions and mappings.

`Scenario(name, partition, task)` has partition `train` or `heldout`, a nonempty name, and a `kind` derived from the presence of switches.

`make_bank(*, partition, seed, pairs=1, steps=64, cue_count=4, delay=1)` returns `4 * pairs` scenarios: `pairs` mirrored pairs for each stationary/switching condition. `steps` must be at least 8. Mirrored partners share cue randomness and switch times but have opposite targets. Changing only the partition label while retaining the same master seed does not create independent data.

```python
from intelligence_dynamics.research import (
    FixedPolicy, LearningRule, TabularPolicy, TaskConfig, run_episode,
)

config = TaskConfig(
    steps=24, cue_count=2, delay=1, switches=(8, 16),
    mappings=((0, 1), (1, 0)), seed=7,
)
fixed = run_episode(config, FixedPolicy((0, 1)), seed=3)
learner = TabularPolicy(LearningRule(priors=(0.0, 0.0)), mode="plastic")
adaptive = run_episode(config, learner, seed=3)
print(len(adaptive.records), adaptive.undelivered)
print(fixed.accuracy, adaptive.accuracy)
```

### Step interface

- `CueTask(config).reset()` returns `Observation(cue)`.
- `CueTask.advance(action)` accepts the integer 0 or 1 and returns `Transition`.
- `Transition` contains `observation`, `feedback`, `done`, and an evaluator-only `record`.
- `Feedback(credit_id, value)` has a value from 0 to 1 and identifies the original action.
- `StepRecord` contains `step`, `cue`, `action`, `target`, `regime`, and `reward`.
- A terminal transition has `observation=None`. Calling `advance` before reset or after termination raises an exception.
- `CueTask.time` is the next action index; `snapshot()` returns an environment-state record supported by state_digest.

`record` and `snapshot()` are for evaluation or branch checks, not the policy observation interface.

## 3. Policies and learning rules

### `LearningRule`

| Field | Default | Range and meaning |
|---|---|---|
| `priors` | `(0.0, 0.0, 0.0, 0.0)` | Nonempty; each entry from -1 to 1 |
| `rate` | `0.5` | Greater than 0 and at most 1 |
| `exploration` | `0.05` | Exploration probability |
| `threshold` | `0.4` | High-error threshold, from 0 to 1 |
| `decay` | `0.7` | Accumulator retention coefficient, from 0 to 1 |
| `accumulation` | `0.18` | Strictly positive accumulation trigger |
| `window` | `3` | Positive integer measured in delivered feedback events |
| `required` | `2` | Positive integer not exceeding window |
| `random_probability` | `0.5` | Random update probability |

A cue preference `b` initializes `Q(cue,0)=(1-b)/2` and `Q(cue,1)=(1+b)/2`. A committed update is `Q ← Q + rate * (feedback - Q)`.

`TabularPolicy(rule, mode="plastic")` supports the seven entries in `MODES`:

| Mode | Commitment condition |
|---|---|
| `fixed` | No value updates; retain initial preferences, exploration, and random tie-breaking |
| `plastic` | Update on every delivered feedback event |
| `conditional` | Current absolute error is at least threshold |
| `accumulator` | Update accumulation as `decay * previous + max(0, error-threshold)`; at accumulation or above, commit the current credit and reset the accumulator |
| `persistence` | At least required high errors in the most recent window events; commit the current credit and clear the window |
| `buffered` | Under the same persistence condition, commit all unexpired high-error credits in order, then clear the window |
| `random` | Update with random_probability using a gate RNG separate from the action RNG |

At feedback index `i`, the buffer window includes indices `i-window+1` through `i`. Each credit is committed at most once. Buffered commitment uses the current value at commitment time and the original feedback target.

Policy methods and properties:

- `reset(*, seed=0)` reconstructs initial values, random streams, and lifetime state.
- `act(observation, feedback=None)` processes optional feedback before selecting and registering an action.
- `observe(feedback)` processes feedback separately and returns committed credit IDs; unknown or already processed IDs raise an error.
- `clear_memory()` clears gate memory but preserves values, undelivered credits, and random streams.
- `values`, `updates`, and `pending_count` expose current values, update records, and the number of undelivered credits.
- `last_committed` contains original action IDs committed by the latest processing step; `feedback_count` counts processed feedback.
- `AppliedUpdate` records the original credit ID, original feedback index, commitment index, old and new values, and original feedback error.

`FixedPolicy(mapping)` is a deterministic cue mapping, unlike the exploratory `mode="fixed"` policy.

```python
from intelligence_dynamics.research import Feedback, LearningRule, Observation, TabularPolicy

policy = TabularPolicy(
    LearningRule(priors=(0.5,), rate=1.0, exploration=0.0),
    mode="plastic",
)
action = policy.act(Observation(0))
committed = policy.observe(Feedback(credit_id=0, value=1.0))
print(action, committed, policy.values)
```

## 4. Running episodes and measuring behavior

`run_episode(config, policy, *, seed=0, corruption="clean", corruption_rate=0.0, corruption_seed=None, latency_window=6, latency_threshold=0.75)` resets the policy before running the complete action sequence. Feedback arriving after the final action is not learned. `undelivered` counts action feedback never processed by the policy, including feedback left unprocessed at the endpoint.

`EpisodeResult` contains overall and segment accuracy, switch latencies, recovery latencies after corruption runs, action records, delivery records, updates, and the unprocessed count. `Delivery` retains the source credit, delivery time, corruption flag, delivered value, committed IDs, and error when processed.

`evaluate(rule, scenarios, *, mode="plastic", seed=0, corruption="clean", corruption_rate=0.0)` returns `Evaluation`:

- `accuracy` is an equal-weight mean of episode accuracies, not a pooled accuracy weighted by episode length.
- `stationary_accuracy`, `switching_accuracy`, and `episode_scores` retain condition-specific and per-scenario performance.
- `feedback_count`, `update_count`, and `update_fraction` count delivered feedback and actual commitments.
- `corrupt_source_updates`, `clean_source_updates`, and their denominators and rates use original-feedback attribution.
- `switch_feedback_count` and `switch_source_update_rate` use the union of the first 12 original action positions beginning at each switch, preventing double counting of overlapping windows.
- `updated_errors` and `skipped_errors` summarize source-feedback errors; `gate_error_separation` is the difference between their means.
- `switch_latency` and `corruption_recovery` retain total, observed, and missing counts and a mean.
- `corrupted_feedback_fraction` is the realized flipped fraction of delivered feedback.

Rates with no feedback are `None`; accuracy for an absent scenario category is `None`. Summary means are `None` if any observation is missing.

```python
from intelligence_dynamics.research import LearningRule, MODES, evaluate, make_bank

bank = make_bank(partition="heldout", seed=12, pairs=1, steps=16, cue_count=2)
rule = LearningRule(priors=(0.0, 0.0))
for mode in MODES:
    result = evaluate(
        rule, bank, mode=mode, seed=4,
        corruption="isolated", corruption_rate=0.2,
    )
    print(mode, result.accuracy, result.update_fraction)
```

### Measurement helpers

- `corruption_plan(length, *, kind="clean", rate=0.0, seed=0, burst_length=3)` returns a boolean tuple. `clean` requires rate zero. `isolated` leaves one clean event after a flip. `burst` flips up to burst_length consecutive events, then leaves one clean event.
- `adaptation_latency(correct, start, stop, *, window=6, threshold=0.75)` takes boolean correctness values and returns the start offset of the first qualifying complete window in the half-open interval. The final complete window is included. No qualifying window returns `None`.
- `summarize(values)` returns `Observations(count, observed, missing, mean)`. Empty input or any missing value makes mean `None`.

Built-in corruption recovery starts at the action after a contiguous corruption run and ends at the earliest next corruption, environment switch, or lifetime endpoint.

```python
from intelligence_dynamics.research import adaptation_latency, corruption_plan, summarize

print(corruption_plan(6, kind="isolated", rate=1.0))
print(adaptation_latency((False, False, True, True), 0, 4, window=2, threshold=1.0))
print(summarize((1.0, None, 3.0)))
```

## 5. Evolutionary search and independent evaluation

`SearchConfig(population=4, generations=2, elites=1, mutation=0.15, update_cost=0.05, corruption_rate=0.1, noise_weight=0.5)` requires population at least 2, positive integer generations and elites, and elites smaller than population. Other values lie from 0 to 1.

Scenario collections must be nonempty, with unique scenario names and task fingerprints and a consistent cue width.

`evolve(train, *, mode="buffered", seed=0, config=SearchConfig())` accepts training scenarios only, and every training task must deliver feedback before the final action. Every candidate and generation uses the same evaluation randomness. Initial preferences, learning rate, exploration, error threshold, decay, and accumulation threshold mutate. Window size, required count, and random probability retain their fixed search defaults.

Fitness is:

`fitness = (1-w)*clean_accuracy + w*isolated_accuracy - update_cost*((1-w)*clean_updates + w*isolated_updates)`

Here `w=noise_weight`, and updates denotes update fractions. Selection retains elites and fills remaining positions with mutated elite descendants. `SearchResult` contains the rule, mode, fitness, weighted training accuracy and update fraction, generation-best and generation-mean histories, candidate-evaluation count, and training fingerprints. Candidate evaluations equal population × generations; each evaluates both clean and isolated-corruption conditions.

`run_study(train, heldout, *, mode="buffered", seed=0, config=SearchConfig())` checks disjoint names, task fingerprints, and environment seeds before selection. `StudyResult` contains:

- `search`: the training selection result.
- `heldout`: clean evaluation of the selected mode.
- `frozen`, `always`, and `matched_random`: controls using the same selected rule, not separately re-evolved models.
- `isolated` and `burst`: corruption evaluations of that selected rule.
- `random_probability`: random-gating probability calibrated on clean training data.
- `heldout_fingerprints`: final task-configuration fingerprints.

Call `evolve` or `run_study` separately for different mechanism families and match total candidate, interaction, and resource budgets. Random gating matches an expected update probability; realized update counts must still be compared.

```python
from dataclasses import asdict
import json
from intelligence_dynamics.research import SearchConfig, evolve, make_bank, run_study

train = make_bank(partition="train", seed=21, steps=16, cue_count=2)
heldout = make_bank(partition="heldout", seed=22, steps=16, cue_count=2)
config = SearchConfig(population=3, generations=2, elites=1)
selected = evolve(train, mode="buffered", seed=8, config=config)
study = run_study(train, heldout, mode="buffered", seed=8, config=config)
print(selected.candidate_evaluations, study.random_probability)
print(json.dumps(asdict(study.heldout), allow_nan=False))
```

## 6. Branching and state integrity

`compare_branches(config, rule, *, boundary, mode="plastic", seed=0, intervention="freeze")` branches before action boundary. Boundary is at least 1 and less than steps. Interventions are `freeze` or `reset_memory`.

`BranchComparison` contains the common prefix, both continuations, shared-state digest, changed paths, whether state actually changed, both accuracies, and post-boundary update counts. Here `delta = continued_accuracy - intervened_accuracy`.

`state_digest` supports finite native scalar values, lists, tuples, string-keyed dictionaries, and dataclass instances. It preserves scalar and container types and rejects cycles. `state_changes` returns tuple paths, so dotted keys are not confused with nested paths.

`verify_intervention(before, after, *, allowed_paths, require_change=True)` permits changes only at specified paths or their descendants and rejects no-ops by default. `trace_digest(columns, *, boundary)` requires exactly boundary rows in every named column. `verify_paired_history` additionally requires identical column names, row order within each column, and value types.

```python
from intelligence_dynamics.research import (
    LearningRule, TaskConfig, compare_branches, verify_intervention, verify_paired_history,
)

config = TaskConfig(
    steps=12, cue_count=2, delay=1, switches=(6,),
    mappings=((0, 1), (1, 0)), seed=4,
)
result = compare_branches(
    config, LearningRule(priors=(0.0, 0.0)), boundary=6, seed=3,
)
print(len(result.common_prefix), result.intervened_updates)
before = {"memory": [1.0], "value": 0.5}
after = {"memory": [], "value": 0.5}
print(verify_intervention(before, after, allowed_paths=(("memory",),)))
history = {"reward": (1.0, 0.0), "cue": (0, 1)}
print(verify_paired_history(history, history, boundary=2))
```

## 7. Shared event support and fixed-time blocks

`Series(rewards, states)` requires equal-length nonempty sequences. Rewards are finite; state identifiers are integers or nonempty strings.

`common_support(control, treatment, *, start=0, stop=None, window=3, spacing=None)` requires equal horizons. Complete pre/post windows must lie within `[start, stop)`. Spacing defaults to `2*window` and cannot be smaller when specified.

`SupportComparison` retains eligible events in both arms, shared events, selected events, Jaccard overlap, both event responses, event-level differences, and the mean difference. Here `delta = treatment_response - control_response`, unlike the direction in the branch API. With no selected shared events, responses and differences are `None`. With no events in either arm, Jaccard is `None`.

`select_events(events, *, spacing)` requires strictly increasing timestamps. `compare_blocks(control, treatment, *, edges)` constructs consecutive half-open intervals and returns `BlockComparison` objects; their deltas also subtract control from treatment.

`aggregate_comparisons(deltas)` retains individual values, observation counts, and positive/negative counts. Any missing value makes the mean, median, and range `None`, while observed sign counts remain available.

```python
from intelligence_dynamics.research import (
    Series, aggregate_comparisons, common_support, compare_blocks,
)

states = (0, 0, 0, 0, 1, 1, 1, 1)
control = Series((0.0,) * 8, states)
treatment = Series((0.0,) * 4 + (1.0,) * 4, states)
shared = common_support(control, treatment, window=2)
blocks = compare_blocks(control, treatment, edges=(0, 4, 8))
print(shared.selected_events, shared.delta)
print(tuple(block.delta for block in blocks))
print(aggregate_comparisons((shared.delta, None)).observations)
```

## 8. Configured criteria

`Criterion(name, metric, direction, threshold)` accepts finite thresholds. Directions `ge`, `gt`, `le`, `lt`, and `eq` mean greater-or-equal, greater, less-or-equal, less, and exact equality. Criterion names must be unique; several criteria may refer to one metric.

`assess(values, criteria, *, combine="all")` returns `Decision(results, combine, passed, evaluated, missing)`. Each `CriterionResult` retains its name, metric, direction, threshold, actual value, and outcome. Missing keys and `None` produce an undetermined criterion. Nonfinite values and boolean metric inputs raise errors.

| Combination | Decisive outcome | No decisive outcome, with missing values | Complete, with no decisive outcome |
|---|---|---|---|
| `all` | Any False → False | None | True |
| `any` | Any True → True | None | False |

Empty criterion sets raise an error. The criterion engine does not estimate effects or conduct significance tests.

```python
from intelligence_dynamics.research import Criterion, assess

criteria = (
    Criterion("capability", "accuracy", "ge", 0.75),
    Criterion("update_budget", "update_fraction", "le", 0.5),
)
result = assess({"accuracy": 0.8, "update_fraction": None}, criteria)
print(tuple(item.passed for item in result.results), result.passed)
```
