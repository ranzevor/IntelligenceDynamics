# Recurrent-state reset confirmation study

[中文](../carry-confirmation.md) · [Research design](research.md) · [Two-trajectory pilot](carry-study.md)

## Research question

Under the fixed DRQN, Foragax environment and training configuration, does the positive reward effect of a single live recurrent-state reset recur across new independent training trajectories?

The pilot reset-minus-retention mean reward differences were `+0.1903333333` and `+0.0617200000` for indices 23 and 24. Confirmation uses indices and seeds 25–32, with eight independent training trajectories as the analysis units.

## Fixed design

The study uses `ForagaxSquareWaveTwoBiome-v11`, 9×9 color observations and native DRQN. The upstream source commit is `9710f60fa30da5badc451ad7ce3ff296d5070830` and dependency-lock object is `18b5340e7ce408d03ee518ac40419f4269c0a1c0`. Model, optimizer, exploration and update settings follow the pilot configuration. [Study configuration](../../configs/carry-confirmation.json)

Each seed first completes 150,000 common interactions. Two branches then start from the same complete state. The retention branch keeps the live recurrent state. The reset branch sets `agent_state.carry` to `None` before the next policy computation. Each branch continues for 150,000 interactions.

Parameters, optimizer, replay, random state and environment state share the branch starting point. Each seed contributes 450,000 effective interactions, totaling 3,600,000 across eight seeds. Completed indices 23 and 24 remain a separate exploratory group.

## Primary endpoint and criterion

For each seed, use float64 reductions over environment interactions `[150000, 300000)` to compute

```text
R_i = mean reward(single reset) - mean reward(retained state)
```

Confirmation requires both a positive arithmetic mean of the eight `R_i` values and `p ≤ 0.05` in a one-sided exact binomial test of the positive-effect fraction. The null hypothesis is a positive-effect probability at most `0.5`. A positive effect means `R_i > 0`, and zero counts as nonpositive.

With eight trajectories, the one-sided probabilities for seven, eight and six positive effects are `9/256`, `1/256` and `37/256`, respectively. Confirmation therefore requires at least seven positive effects and a positive paired mean.

Report all eight differences, their paired mean, positive and zero counts, the one-sided test probability, and a two-sided 95% Clopper–Pearson interval for the positive-effect fraction. Evaluate the criterion after all eight fixed trajectories finish.

## Secondary measurements

- Mean reward difference during the first 1,024 post-reset interactions, `[150000, 151024)`.
- Mean reward difference over `[151024, 300000)`.
- Reward curves in 25,000-interaction windows and post-branch loss, absolute TD error and weight-change means in 50,000-interaction windows.
- The pilot's complete event windows, recovery threshold and observation horizon, with event, recovered and unrecovered counts for each branch.
- Live-state bytes before reset, after reset and after the first subsequent policy computation, plus stage execution time and peak memory.

These secondary measurements describe the response trajectory. The primary confirmation conclusion uses the fixed joint criterion above.

## Data and analysis

Each seed retains five typed traces, the common checkpoint and branch checkpoints. Paired analysis checks common prehistory, native-state provenance and actual intervention paths. A no-op is retained as an observation for that seed.

Analysis recomputes the primary and secondary endpoints from final NPZ files, records source, configuration and dependency identities, and retains each trajectory's results. Interactions repeated during recovery are counted separately from effective interactions. See the [contribution guide](../../CONTRIBUTING.md#recurrent-state-confirmation-study) for execution, continuation and resource settings.
