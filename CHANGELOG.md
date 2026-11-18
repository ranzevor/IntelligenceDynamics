# 版本 / Versions

## 0.1.1

- **生命周期状态 / Lifecycle status:** `maintained`
- **维护范围 / Maintenance scope:** 0.1.1 的源码包、CLI、Python API、JSONL 0.1 兼容性与随附文档；软件版本 `0.1.1` 标识包与文档修订，不改变 `JSONL 0.1` 与 `native-checkpoint-v1` 的数据格式身份。Source package, CLI, Python API, `JSONL 0.1` compatibility, and accompanying documentation for 0.1.1; software version `0.1.1` identifies the package and documentation revision without changing `JSONL 0.1` or `native-checkpoint-v1` format identities.
- **入口与数据 / Entries and data:** 安装、升级与命令见[软件指南](docs/software.md)，数据契约见[数据契约](docs/data.md)，原生检查点见[原生模型接口](docs/native.md)，确认实验见[确认实验](docs/carry-confirmation.md)，实验汇总见[研究结果](docs/research-results.md)；See the [software guide](docs/en/software.md) for installation, upgrade, and commands, the [data contract](docs/en/data.md) for data formats, [native model interfaces](docs/en/native.md) for native checkpoints, the [confirmation study](docs/en/carry-confirmation.md), and [research results](docs/en/research-results.md).
- **安全报告 / Security reporting:** 使用[私密报告入口或邮箱](SECURITY.md)；use the [private reporting entry or email](SECURITY.md).

- 新增固定的原生 DRQN／Foragax 循环状态重置试点配置、运行器、检查点续跑、标量弱类型校验、配对轨迹诊断接口（`intelligence_dynamics.research.carry`、`tools/run_carry_study.py`），以及归档 `A1-P6` 的 float64 `weight_change` 重算结果。
- 新增八轨迹固定样本循环状态确认配置、单侧精确二项符号检验与 Clopper–Pearson 区间接口（`intelligence_dynamics.research.confirmation`）、按检查点退出的顺序调度工具（`tools/run_carry_confirmation.py`），以及已完成的八轨迹确认实验结果与机制分析（[确认实验](docs/carry-confirmation.md)、[研究结果](docs/research-results.md) 及其英文对应文档）。
- 补充从 `0.1.0` 就地升级与数据兼容说明，保持现有 `JSONL 0.1` 流、队列数据库与 `native-checkpoint-v1` 检查点无需迁移即可读取；已中断的原生试点或确认输出目录续跑（`--resume`）仍绑定其 `run.json` 记录的源码提交。

- Added pinned native DRQN/Foragax recurrent-state reset pilot configuration, runner, checkpoint continuation, scalar weak-type checks, paired trace diagnostics (`intelligence_dynamics.research.carry`, `tools/run_carry_study.py`), and archived `A1-P6` float64 `weight_change` recomputation.
- Added fixed-sample eight-trajectory confirmation configuration, one-sided exact binomial sign test and Clopper–Pearson interval API (`intelligence_dynamics.research.confirmation`), per-checkpoint worker scheduler (`tools/run_carry_confirmation.py`), and completed confirmation results and mechanism analysis ([Confirmation study](docs/en/carry-confirmation.md), [Research results](docs/en/research-results.md), and Chinese counterparts).
- Documented in-place upgrades from `0.1.0` and compatibility boundaries, keeping existing `JSONL 0.1` streams, queue databases, and `native-checkpoint-v1` checkpoints readable without migration; resuming an interrupted native pilot or confirmation run with `--resume` remains bound to the source revision recorded in `run.json`.

## 0.1.0

- **生命周期状态 / Lifecycle status:** `maintained`
- **维护范围 / Maintenance scope:** 0.1.0 的源码包、CLI、Python API、JSONL 0.1 与随附文档，自首次公开提交起生效；source package, CLI, Python API, JSONL 0.1, and accompanying documentation for 0.1.0, effective from the initial public commit.
- **入口与数据 / Entries and data:** 安装与命令见[软件指南](docs/software.md)，数据读取、校验与转换见[数据契约](docs/data.md)，checkpoint 保存与续跑见[原生模型接口](docs/native.md)；See the [software guide](docs/en/software.md) for installation and commands, the [data contract](docs/en/data.md) for reading, validation, and conversion, and [native model interfaces](docs/en/native.md) for checkpoint storage and continuation.
- **安全报告 / Security reporting:** 使用[私密报告入口或邮箱](SECURITY.md)；use the [private reporting entry or email](SECURITY.md).

- 项目自有材料采用 Apache-2.0；保留第三方 MIT 通知，提供 DCO 贡献与软件引用信息。
- 测量协议、数据转换与恢复指标。
- 合成任务、学习规则搜索、配对干预与比较。
- DAL 与 Foragax/DRQN 原生接口、checkpoint 续跑和选择性状态干预。
- 本地运行、事务任务队列和 Batch 接口。
- 中文与英文使用文档、Issue/PR 协作入口、CI 与依赖更新配置。
- wheel、完整源码包及绑定源码身份的发行清单与校验表。

- Project-owned materials use Apache-2.0; third-party MIT notices, DCO contribution terms, and software citation metadata are provided.
- Measurement protocol, data conversion, and recovery metrics.
- Synthetic tasks, learning-rule search, paired interventions, and comparisons.
- Native DAL and Foragax/DRQN interfaces, checkpoint continuation, and selective state interventions.
- Local execution, transactional queue, and Batch interfaces.
- Chinese and English documentation, Issue/PR contribution entries, CI, and dependency update configuration.
- Wheel, complete source archive, and source-bound release manifests and checksums.
