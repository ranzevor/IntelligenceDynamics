# Research results

[中文](../research-results.md) · [Research design](research.md) · [Carry-reset pilot](carry-study.md) · [Confirmation study](carry-confirmation.md)

This page summarizes experiments on continued updates, recurrent state and conditional updates. A2-P2 and the eight-trajectory confirmation study follow the current design. The original A1, A2-P1 and E1 reports are recorded in the archive. [Intelligence-Dynamics archive](https://github.com/ranzevor/Intelligence-Dynamics/tree/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79)

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

### Exploratory pilot (A2-P2, indices 23–24)

A2-P2 compares native DRQN branches for indices 23 and 24. Each pair shares 150,000 environment interactions of prehistory, then either retains live recurrent state or resets it once to `None`. The table reports mean reward per interaction over `[150000, 300000)`. [Pilot design](carry-study.md)

| Index | Retained state | Single state reset | Reset − retained |
| --- | ---: | ---: | ---: |
| 23 | 0.8749733333 | 1.0653066667 | +0.1903333333 |
| 24 | 0.5910800000 | 0.6528000000 | +0.0617200000 |

The arithmetic mean of the exploratory pilot differences is **+0.1260266667** (`2/2` positive; `900,000` effective interactions, plus `100,000` interactions replayed during interruption recovery).

### Eight-trajectory confirmation study (indices 25–32)

The fixed confirmation study tested eight independent seeds 25–32 under the same 150,000-interaction common prehistory, configuration, and upstream commit, evaluating `R_i = mean reward(single reset) - mean reward(retained state)` over `[150000, 300000)` (`3,600,000` effective interactions, with `35,000` logged repeated recovery interactions reported separately without adding independent samples). [Confirmation study](carry-confirmation.md)

| Index | Retained `[150k, 300k)` | Reset `[150k, 300k)` | Reset − retained | Early `[150k, 151024)` diff | Later `[151024, 300k)` diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 | 1.0970666667 | 1.0955733333 | -0.0014933333 | -0.1152343750 | -0.0007115240 |
| 26 | 0.9166133333 | 0.6622133333 | -0.2544000000 | +0.0390625000 | -0.2564171410 |
| 27 | 0.8318533333 | 0.9907333333 | +0.1588800000 | +2.2363281250 | +0.1446004726 |
| 28 | 0.8502800000 | 1.0355066667 | +0.1852266667 | +0.0703125000 | +0.1860165396 |
| 29 | 0.8567066667 | 1.1368400000 | +0.2801333333 | +0.0253906250 | +0.2818843304 |
| 30 | 0.8676133333 | 1.0234666667 | +0.1558533333 | +0.0351562500 | +0.1566829556 |
| 31 | 0.9450000000 | 0.7088933333 | -0.2361066667 | -0.1855468750 | -0.2364541940 |
| 32 | 1.1091866667 | 0.9969733333 | -0.1122133333 | +0.1738281250 | -0.1141794651 |

Across the eight confirmation trajectories, the arithmetic mean of `R_i` was **+0.0219850000**, with `4/8` positive, `0` zeros, and `4/8` negative. The one-sided exact binomial test gave `p = 0.63671875`, with a two-sided 95% Clopper–Pearson interval of `[0.157013, 0.842987]`. **The prespecified joint confirmation criterion was not met.** All eight seeds verified intervention confinement to `agent_state.carry` (`577,536` bytes → `0` bytes → `577,536` bytes after the first step).

While reset branches tended to outperform retained branches over the first `1,024` steps (mean `+0.284912`, `6/8` positive) and first `25,000` steps (mean `+0.236880`, `7/8` positive), both arms continued online replay insertion and optimizer updates, causing rapid divergence in visited biome-switch sequences (Jaccard overlap `0.0473`–`0.0605`) and bidirectional outcomes over the full `150,000`-step continuation. A `2×2` factorial design crossing `retain/reset carry` with `continue/freeze updates` from the `150,000`-step checkpoint is needed to isolate live recurrent-state control from induced learning divergence.

## Conditional updates and corrupted feedback

E1.2 completed a conditional-update comparison across five independent evolutionary seeds on fixed training and heldout task banks. Conditional gating reached accuracy `0.91870`, compared with `0.91650` for always updating, at update fraction `0.07024`. It met the experiment's joint criteria. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-2-confirmatory-gate.md)

E1.3b compared instantaneous, persistence and buffered-persistence mechanisms across three independent evolutionary seeds. Buffering improved mean clean switching latency from `15.75` to `12.52` steps. Under this configuration, the fraction of corrupted burst-feedback records eventually committed was `0.87542`. Subsequent research examines independent discovery processes and consistency information that can distinguish environmental change from corrupted feedback. [Original report](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-3b-buffered-persistence-decision.md)
