# Research results

[中文](../research-results.md) · [Research design](research.md) · [Carry-reset pilot](carry-study.md)

This page summarizes experiments on continued updates, recurrent state and conditional updates. A2-P2 follows the current pilot design. The original A1, A2-P1 and E1 reports are recorded in the archive. [Intelligence-Dynamics archive](https://github.com/ranzevor/Intelligence-Dynamics/tree/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79)

## Continued updates over a lifetime

A1-P4 branched trajectories 7 and 8 at 75k, 150k and 225k environment interactions and observed each native frozen continuation for another 75k interactions. Both trajectories had slightly negative paired effects at 150k and strongly positive effects at 225k, showing heterogeneous lifetime effects. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p4-lifetime-plasticity-pilot-decision.md)

A1-P5 used fresh trajectories 9–12 at the 225k boundary. All four continued-minus-frozen reward contrasts were positive, with mean `+1.915283203125`. A1-P6 independently used trajectories 13–16. All four contrasts were positive, with mean `+2.33642578125`. The results support behavioral value from the late-life update path under this native DQN/Foragax condition. [A1-P5](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p5-late-life-confirmatory-decision.md) · [A1-P6](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p6-update-path-necessity-decision.md)

### Sum of recorded A1-P6 weight_change values

Summing the archived NPZ `weight_change` field over `[225000, 300000)` with a float64 accumulator gives the values below. Each sum contains 75,000 original float16 samples taken once per environment interaction. Native learning steps refresh this diagnostic. Other interactions retain its current value. The low-precision accumulation in the original summary overflowed. [Original run](https://github.com/ranzevor/Intelligence-Dynamics/actions/runs/35204051815)

| Trajectory | Continued updates | Native freeze |
| --- | ---: | ---: |
| 13 | 751884.3984375 | 1088671.875 |
| 14 | 729580.26171875 | 1170703.125 |
| 15 | 705263.759765625 | 1224609.375 |
| 16 | 798530.9765625 | 1116210.9375 |

## Native DQN and DRQN

A2-P1 ran native DQN and DRQN for trajectories 19–22, with 300k environment interactions per run. The primary comparison was mean reward over `[150k, 300k)`, expressed as `DRQN − DQN`.

| Trajectory | Paired contrast |
| --- | ---: |
| 19 | -0.2154533333 |
| 20 | +0.1387466667 |
| 21 | -0.1109866667 |
| 22 | +0.2342800000 |

Two of four contrasts were positive, with mean `+0.0116466667`. The joint criterion of at least three positive pairs and a positive mean was not met. The package comparison produced a negative result, leading to state-intervention research within the same native DRQN. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-18-track-a-a2-p1-state-construction-confirmatory-decision.md)

## Single recurrent-state reset

A2-P2 compares native DRQN branches for indices 23 and 24. Each pair shares 150,000 environment interactions of prehistory, then either retains live recurrent state or resets it once to `None`. The table reports mean reward per interaction over `[150000, 300000)`. [Pilot design](carry-study.md)

| Index | Retained state | Single state reset | Reset − retained |
| --- | ---: | ---: | ---: |
| 23 | 0.8749733333 | 1.0653066667 | +0.1903333333 |
| 24 | 0.5910800000 | 0.6528000000 | +0.0617200000 |

The arithmetic mean of the paired differences is **+0.1260266667**. The reset branch has higher mean reward for both trajectories in the specified observation interval. The next stage uses independent trajectories to examine this difference, measuring short-term response, sustained reward and state resource use separately.

## Conditional updates and corrupted feedback

E1.2 completed a conditional-update comparison across five independent evolutionary seeds on fixed training and heldout task banks. Conditional gating reached accuracy `0.91870`, compared with `0.91650` for always updating, at update fraction `0.07024`. It met the experiment's joint criteria. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-2-confirmatory-gate.md)

E1.3b compared instantaneous, persistence and buffered-persistence mechanisms across three independent evolutionary seeds. Buffering improved mean clean switching latency from `15.75` to `12.52` steps. Under this configuration, the fraction of corrupted burst-feedback records eventually committed was `0.87542`. Subsequent research examines independent discovery processes and consistency information that can distinguish environmental change from corrupted feedback. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-3b-buffered-persistence-decision.md)
