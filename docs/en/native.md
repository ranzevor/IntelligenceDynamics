# Native models and environments

[中文](../native.md) · [Home](../../README.en.md)

`intelligence_dynamics.native` executes explicitly supplied DAL, Foragax and DRQN objects.

## Dependencies and API contract

The core installation and synthetic-object examples need only the Python standard library. Array state requires NumPy. The default DRQN numerical backend uses JAX; a real model additionally needs its upstream Haiku, Optax, Flashbax, Chex, Foragax and other dependencies.

The DRQN interface follows the public object shapes in [continual-foragax-agents](https://github.com/steventango/continual-foragax-agents), and DAL fields follow the public [DAL](https://github.com/trsjp/dal) implementation. The in-project JAX state-codec tests use Python 3.12 and the exact CPU dependency versions in [`requirements-native.txt`](../../requirements-native.txt). Connecting a Foragax model still requires the additional dependencies of an upstream version selected and reviewed by the caller. Software licensing is recorded in [Third-party notices](../../THIRD_PARTY_NOTICES.md).

| Object | Required interface |
| --- | --- |
| Evaluated DAL individual | `individual_id`, `normalised_fitness`, both `connectivity_grid_*` fields, `io_list_a`, `io_list_m`, `rl_learning_rate`; evaluated statistics described below |
| DAL SPNN wrapper | `send`, `recv`, `pipe.poll`, `individual_id`, `is_unmutated_copy` |
| Foragax environment | `default_params`, `reset(key, params)`, `step(key, state, action, params)`, `action_space(params)` |
| DRQN agent | `state`, `buffer.add/can_sample`, `phi`, `act`, `_carry_for_buffer`, `_start`, `_compute_reward_trace`, `encode_scalar_features`, `_update_state_with_metrics`, `_advance_update_clock`, `reward_clip` |
| Native RLGlue environment wrapper | `state`, `_start(state)`, `_step(state, action)` |

DRQN execution operates on one agent. Separate batched training state according to its upstream layout and execute each agent individually. State includes parameters, target parameters, optimizer, replay, PRNG, live carry, previous transition, reward traces, hyperparameters and counters. It must support detached copying and explicit codec round trips. JAX typed-key identity uses `key_data` and retains the PRNG implementation identifier. Optax namedtuple types are retained. Parameters and arrays must be finite; positive infinity is accepted for `hypers.freeze_steps` to indicate no freeze boundary.

`native_state_snapshot(value)` returns a type-sensitive digest, each array's shape/dtype/content digest, and tensor element and byte counts. Ordinary scalar values in lists are not counted as tensor storage.

Arrays support NumPy numeric dtypes and `ml_dtypes.bfloat16`. The latter retains its semantic `bfloat16` dtype. Both undergo nonfinite-value checks.

## DAL individual diagnostics

`dal_individual` reads an evaluated individual directly. For an unmodified `nmnn.nn`, the adapter sums elementwise absolute values in `total_dw_list_nm` and `total_dw_list_rl`, and reads `mean_active_rl_learning_rate` and `mutations_applied`. An existing complete `stats` mapping is also accepted. Raw individuals do not contain the parent wrapper's `is_unmutated_copy`; provide `unmutated` explicitly.

This synthetic-object example runs directly.

```python
from types import SimpleNamespace
from intelligence_dynamics.native import dal_individual

individual = SimpleNamespace(
    individual_id=0, normalised_fitness=1.25,
    connectivity_grid_a=[[0, 1], [0, 0]],
    connectivity_grid_m=[[[0, 1]]],
    io_list_a=[(0, 1, 0.5)], io_list_m=[(0, 0, 1)],
    rl_learning_rate=0.02,
    total_dw_list_nm=[[0.1, -0.2]], total_dw_list_rl=[[0.3]],
    mean_active_rl_learning_rate=0.01,
    mutations_applied=["synthetic_change"],
)
observed = dal_individual(individual, unmutated=False)
print(observed.activation_projections, observed.mean_susceptibility)
```

`dal_population` consumes every individual and returns detached rankings, parent-selection flags and elite flags. Ranking uses descending fitness and preserves input population order for ties; parent and elite boundaries use that same order. Use `unmutated_flags` for aligned metadata on raw individuals. Mean susceptibility is `None` when there are no activation projections. Missing fields, duplicate IDs, nonfinite diagnostics and invalid ranking parameters fail rather than becoming zero.

`dal_remote_population(workers, generation=..., parent_pool=..., elites=..., timeout=...)` calls `send("get_over_here")`, `pipe.poll(timeout)` and `recv()` on existing SPNN wrappers. It validates the response ID before extracting diagnostics. It does not create, evolve or terminate workers. A pipe timeout raises `TimeoutError`; the pipe owner must handle any late response.

## Foragax environment adapter

The adapter explicitly manages PRNG keys for reset, transitions and action sampling. `snapshot()` and `restore()` include environment state, current observation, next key, interaction count and termination flag. Different configuration identities cannot restore each other's snapshots. Parameters are copied at construction and remain fixed for the adapter's lifetime. The `params` property returns a detached copy. Native calls receive separate copies; a call that changes its parameters is rejected without advancing the saved state. To use other parameters, construct another adapter.

```python
from types import SimpleNamespace
from intelligence_dynamics.native import ForagaxAdapter

class SyntheticEnvironment:
    default_params = {"increment": 1}
    def action_space(self, params):
        return SimpleNamespace(n=2, sample=lambda key: key % 2)
    def reset(self, key, params):
        return [0], {"position": 0}
    def step(self, key, state, action, params):
        state = {"position": state["position"] + action + params["increment"]}
        return [state["position"]], state, float(action), False, {"key": key}

adapter = ForagaxAdapter(
    SyntheticEnvironment(), key=lambda seed: seed,
    split=lambda key, count: (key + 1, key + 2),
)
adapter.reset(3)
saved = adapter.snapshot()
first = adapter.step(1)
adapter.restore(saved)
if adapter.step(1) != first:
    raise RuntimeError("same-state replay differs")
print(first[0], first[1])
```

`from_factory(factory, env_id, options=..., key=..., split=...)` calls the actual registry factory. The default `transition="step"` uses the environment's public wrapper. Explicit `transition="step_env"` uses the underlying transition, bypassing wrapper autoreset and termination helpers. Use `step_env` for continuing streams and the `step` wrapper for episodic termination and autoreset. A transition after termination requires reset.

## Connecting an installed DRQN

The following defines an integration function. Its caller supplies the installed native stack, environment ID and options, agent parameters and upstream collector.

```python
from copy import deepcopy
from intelligence_dynamics.native import DRQNExecutor, ForagaxAdapter, NativeLifecycle

def connect_foragax_drqn(*, env_id, environment_options, agent_parameters,
                        collector, seed, transition="step"):
    import jax
    from foragax.registry import make
    from algorithms.nn.DRQN import DRQN

    environment = ForagaxAdapter.from_factory(
        make, env_id, options=environment_options,
        key=jax.random.key, split=jax.random.split, transition=transition,
    )
    raw = environment.environment
    space = raw.observation_space(environment.params)
    observations = (space.shape if hasattr(space, "shape")
                    else {name: item.shape for name, item in space.spaces.items()})
    actions = int(raw.action_space(environment.params).n)
    agent = DRQN(observations, actions, deepcopy(agent_parameters), collector, seed)
    return NativeLifecycle(DRQNExecutor(agent, environment))
```

`agent_parameters` must include at least `total_steps`, `epsilon`, `gamma`, `target_refresh`, `buffer_size`, `buffer_min_size`, `batch`, `update_freq`, `sequence_length`, `optimizer` and `representation`. This interface's Adam configuration uses `alpha`, `beta1`, `beta2` and `eps`. The native network builder validates representation type and dimensions. Do not pass an entire experiment file as agent parameters. `DRQNExecutor` also accepts an existing agent and an environment wrapper with `_start/_step`.

The default execution chain advances the environment, adds the previous transition to native replay, computes reward traces and scalar features, executes the native policy, decides whether to update the optimizer, and advances the agent clock. The agent's `_update_state_with_metrics` performs actual sampling, gradients, optimizer updates and target-parameter refresh.

When live carry is `None`, `DRQNPolicy` evaluates `phi(..., carry=None)` using the current learned parameters and retains its initial carry for the next replay entry. The actual `act` still receives `carry=None`. The reset itself does not rewrite existing replay, environment state or RNG. The first transition executes directly and initializes carry.

The update decision uses pre-transition `steps % update_freq == 0`, plus replay readiness, enabled learning and the native `freeze_steps` boundary. Updates must increase by exactly zero or one; an unfrozen agent clock must advance exactly once. The native freeze boundary stops the agent clock and epsilon schedule while environment interaction, policy action and replay collection continue. The initialized agent clock must be positive.

## Checkpoints, interventions and continuation

`NativeCheckpoint.history` must cover every environment interaction and have length `state.total_steps`. Start a fresh task with `lifecycle.start(seed=...)`. Importing an already-running RLGlue state requires its complete history; do not omit or truncate it:

```python
from intelligence_dynamics.native import NativeCheckpoint, execution_state

def checkpoint_from_running_glue(glue_state, full_history, *, terminated=False):
    return NativeCheckpoint(
        execution_state(glue_state, terminated=terminated),
        tuple(full_history),
    )
```

A caller explicitly supplies the `CheckpointStore` codec, which must preserve complete state, types and history. Save verifies a codec round trip. Load checks container size and digest before decoding, then checks complete typed state identity. Clone retains the original payload bytes. Names cannot overwrite existing checkpoints or traverse directories or symbolic links.

Save separately compares actual array types and JAX weak-type attributes across the round trip; clone compares these attributes between its two decoded results. Existing ordinary-array digests and the `native-checkpoint-v1` format remain unchanged, so old checkpoints need no migration. V1 persists array numeric dtype, shape, and content in its digest; the codec reconstructs the array backend type during loading. Load with the trusted codec used to create the checkpoint.

Saving and loading both execute the caller-specified codec, including its decoder. Pickle decoding can execute code. Before loading, verify the source through controlled storage or an authenticated channel, then use the specified trusted codec. Container size, digests, and JSON validation check content integrity.

The branch example receives a running lifecycle, a common checkpoint, and a new trusted store. `additional` includes the first-step probe. The procedure loads and copies branches, resets carry, saves and reloads, executes continuation transitions, and checks common history.

```python
from intelligence_dynamics.native import verify_common_history

def continue_carry_branches(lifecycle, common, store, *, additional=3):
    if type(additional) is not int or additional < 1:
        raise ValueError("additional must be a positive integer")
    if common.state.terminated:
        raise ValueError("common state must be nonterminal")
    store.save("common", common)
    control = store.clone("common", "control")
    reset_copy = store.clone("common", "reset-copy")
    reset = lifecycle.intervene(reset_copy, {("agent_state", "carry"): None})
    store.save("reset", reset)
    first = lifecycle.run(store.load("reset"), interactions=1)
    audit = first.history[-1].audit
    if audit.native_frozen or audit.agent_steps_after - audit.agent_steps_before != 1:
        raise ValueError("first-action clock requires an unfrozen native state")
    if not audit.initialized_carry:
        raise ValueError("first post-reset transition did not execute recurrent policy")
    expected = int(audit.update_due and audit.replay_ready)
    if audit.updates_after - audit.updates_before != expected:
        raise ValueError("first-action update schedule differs")
    store.save("first", first)
    reset_final = lifecycle.run(store.load("first"), interactions=additional - 1)
    control_final = lifecycle.run(control, interactions=additional)
    matched = verify_common_history(
        control_final, reset_final, boundary=common.state.total_steps,
    )
    return control_final, reset_final, matched
```

`learning=False` suppresses optimizer updates and retains parameters, target parameters, optimizer state and update count, while continuing replay collection and the unfrozen clock. `freeze(checkpoint, paths)` is a different intervention: after each transition it restores selected fields to their captured values. Updates still execute, and update counters and metrics record that computation before the selected state is clamped.

Field paths support dataclass fields, mutable mapping keys, sequence indices and namedtuple field names. Overlapping paths, counter changes, no-op resets, missing fields and changes outside selected fields are rejected. Nested immutable Mapping replacement is unsupported; namedtuple replacement retains its actual type. Common-history checks include field types, values and array identity. A missing boundary is an error.

`run` stops adding interactions after termination or truncation. Exceptions from native methods propagate to the caller; partially written checkpoint files are not successful results.
