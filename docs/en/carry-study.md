# Native recurrent-state reset pilot

[中文](../carry-study.md) · [Research design](research.md) · [Native interfaces](native.md)

## Scientific question

How does a single reset of live recurrent state change subsequent behavior, recovery and learning after the same native DRQN history? Two branches start from the same parameters, optimizer, replay, random state and environment state.

The pilot follows the A2-P1 package comparison and examines the contribution of live recurrent state. See [research results](research-results.md).

## Design

Use native DRQN on `ForagaxSquareWaveTwoBiome-v11` with 9×9 color observations. Resolve indices 23 and 24 through the upstream experiment configuration. Each index defines an independent training trajectory.

Run 150,000 environment interactions per index and save the common state and complete history. Create two branches.

| Branch | Boundary operation | Additional interactions |
| --- | --- | ---: |
| Retained carry | Continue from the common state | 150,000 |
| Single carry reset | Set live `carry` to `None` before the next policy computation | 150,000 |

The budget is 450,000 interactions per index and 900,000 in total. Both branches reuse the executed prehistory. The boundary uses environment interactions, while the agent retains its native clock and update schedule.

The intervention changes live carry alone. Existing replay state, parameters, target parameters, optimizer, random state, exploration parameters, reward traces, previous transition and environment state retain their common starting values. New behavior, replay and learning after branching form the intervention response.

## Primary measurement

For each index, compute mean reward over environment interactions `[150000, 300000)` and report

```text
R_i = mean_reward(single carry reset) - mean_reward(retained carry)
```

Report `R_23`, `R_24` and their arithmetic mean. The index is the analysis unit and the branches remain paired.

## Secondary measurements

- Mean reward in 25,000-interaction blocks.
- Mean loss, absolute TD error and weight change in post-branch 50,000-interaction blocks, using float64 reductions.
- Reward differences across complete 64-interaction pre/post change windows within `[150000, 300000)`, selected chronologically without overlap. Adjacent windows may meet at their endpoints.
- Recovery to 90% of the pre-change reward mean, checked in 64-step blocks for up to 2,048 interactions. Multiply the pre-change mean directly by 0.9, including zero and negative values. Recovery observation continues across later biome changes.
- Eligible event counts, recovered and unrecovered counts, biome occupancy and resource use for each branch.
- Live recurrent-state size before reset, after reset and after the first subsequent policy computation.

Loss, absolute TD error and weight change are sampled once per environment interaction. Native learning steps refresh these diagnostics. Other interactions retain their current values.

Present fixed-time reward contrasts and event-conditioned measurements separately. Preserve counts and distinct missing-value meanings for empty event sets and recovery censored at the observation horizon.

## State and data

Retain the rewards, biome_id, loss, abs_td_error and weight_change traces with their dtypes, shapes and content digests. Restore both branches from the same serialized state. Compare common state fields before intervention and record the actual changed paths afterward.

Outputs include the native configuration, environment versions, five traces, common checkpoint, final branch checkpoints, first-step scheduling record, paired results and resource records. Execution and debugging logs belong in the experiment output directory.

## Subsequent research

The pilot supplies two native paired results and an execution basis for state interventions. A subsequent confirmatory study defines independent trajectories, primary effect criteria and resource matching before comparing the roles of live state, learning state and experience memory.

See the [contributing guide](../../CONTRIBUTING.md#native-recurrent-state-pilot) for execution and resume steps.

The experiment configuration is [carry-study.json](../../configs/carry-study.json). The original design is recorded in the [archived implementation protocol](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/track-a-engineered/a2-p2-state-mechanism-decomposition-implementation-protocol.md).
