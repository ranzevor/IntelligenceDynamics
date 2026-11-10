# 研究结果

[English](en/research-results.md) · [研究设计](research.md) · [循环状态试点](carry-study.md)

本页汇总持续更新、循环状态与条件更新实验。A2-P2 按当前试点设计执行，A1、A2-P1 和 E1 的原始报告来自归档主线。[Intelligence-Dynamics 归档主线](https://github.com/ranzevor/Intelligence-Dynamics/tree/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79)

## 持续更新与生命周期

A1-P4 在轨迹 7、8 的 75k、150k、225k 环境交互处建立原生冻结分支，并各观察后续 75k 交互。两个轨迹在 150k 的配对效应略负，在 225k 的配对效应均为强正，呈现随生命周期变化的异质作用。[原始报告](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p4-lifetime-plasticity-pilot-decision.md)

A1-P5 使用新轨迹 9–12，在 225k 边界比较持续更新与原生冻结，四组奖励配对效应全部为正，均值为 `+1.915283203125`。A1-P6 再使用新轨迹 13–16，四组效应全部为正，均值为 `+2.33642578125`。结果支持该原生 DQN／Foragax 条件下晚期更新路径的行为价值。[A1-P5](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p5-late-life-confirmatory-decision.md) · [A1-P6](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-17-track-a-a1-p6-update-path-necessity-decision.md)

### A1-P6 的 weight_change 记录值之和

对归档 NPZ 的 `weight_change` 字段，在 `[225000, 300000)` 区间以 float64 累加，得到以下结果。每个值包含按环境交互采样的 75,000 个原始 float16 记录。原生学习步骤刷新该诊断值，其他交互保留其当前值。原汇总文件在该计算中使用的低精度累加产生了溢出。[原始运行](https://github.com/ranzevor/Intelligence-Dynamics/actions/runs/35204051815)

| 轨迹 | 持续更新 | 原生冻结 |
| --- | ---: | ---: |
| 13 | 751884.3984375 | 1088671.875 |
| 14 | 729580.26171875 | 1170703.125 |
| 15 | 705263.759765625 | 1224609.375 |
| 16 | 798530.9765625 | 1116210.9375 |

## 原生 DQN 与 DRQN 比较

A2-P1 对轨迹 19–22 分别执行原生 DQN、DRQN，各运行 300k 次环境交互。主要比较为 `[150k, 300k)` 的平均奖励差 `DRQN − DQN`。

| 轨迹 | 配对差 |
| --- | ---: |
| 19 | -0.2154533333 |
| 20 | +0.1387466667 |
| 21 | -0.1109866667 |
| 22 | +0.2342800000 |

四组中两组为正，均值为 `+0.0116466667`，预定的至少三组为正且均值为正的联合判据未满足。该包级比较得到阴性结果，后续研究转向同一原生 DRQN 内的状态干预。[原始报告](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-18-track-a-a2-p1-state-construction-confirmatory-decision.md)

## 单次循环状态重置

A2-P2 比较索引 23、24 的原生 DRQN 分支。各索引共享 150,000 次环境交互的前史，随后分别保留实时循环状态或将其单次重置为 `None`。下表给出 `[150000, 300000)` 的每步平均奖励。[试点设计](carry-study.md)

| 索引 | 保留状态 | 单次重置状态 | 重置 − 保留 |
| --- | ---: | ---: | ---: |
| 23 | 0.8749733333 | 1.0653066667 | +0.1903333333 |
| 24 | 0.5910800000 | 0.6528000000 | +0.0617200000 |

两组配对差的算术平均值为 **+0.1260266667**。在这两条轨迹的指定观察区间，单次重置组的平均奖励均高于保留组。下一阶段使用独立轨迹复核该差异，并分别测量短期响应、持续奖励与状态资源占用。

## 条件更新与反馈污染

E1.2 的五个独立演化种子在固定训练／留出任务银行上完成条件更新比较。条件门控准确率为 `0.91870`，始终更新为 `0.91650`，条件更新比例为 `0.07024`，满足该实验的预定联合判据。[原始报告](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-2-confirmatory-gate.md)

E1.3b 的三个独立演化种子比较瞬时、持续性和缓冲持续性机制。缓冲将平均清洁切换延迟从持续性机制的 `15.75` 步降至 `12.52` 步，改善了响应速度。该配置中，突发污染反馈记录最终进入更新的比例为 `0.87542`。后续研究据此考察独立发现过程及能区分环境变化与污染反馈的一致性信息。[原始报告](https://github.com/ranzevor/Intelligence-Dynamics/blob/da9b4bbd35e3d0df3fc116bb0f50d24356bfed79/research/results/2026-09-16-e1-3b-buffered-persistence-decision.md)
