# 测量与数据契约

[English](en/data.md) · [首页](../README.md)

## 测量流

每行是一个 UTF-8 JSON 对象，必须包含 `record_type` 与 `protocol_version: "0.1"`。单条记录最多 1 MiB。重复键、未知字段、空行、非有限数、无效 UTF-8 和布尔型计数均为错误。

|类型|用途|主要字段|
|---|---|---|
|`RunMetadata`|标识一次运行|run_id、track、system_name、environment_name、seed、source_revision、config_digest|
|`StepRecord`|一个测量时点|run_id、step、lifetime、capability、computation、memory、routing|
|`AdaptationEvent`|环境或目标变化|event_id、step、event_type、hidden_from_agent、target_recovery_fraction|
|`InterventionRecord`|机制干预及差值|run_id、mechanism、intervention_type、start_step、end_step、delta_capability|

`track` 取 A、B、C、AB 或 cross。运行流先写唯一的 `RunMetadata`，同一运行的 step 严格递增，lifetime 不递减。事件记录以其 event_id 标识，不隐式归属于最近一条运行记录。

`read_records` 检查单条数据合同；CLI `validate` 额外检查流中的运行身份和时序。`to_object` 与 `from_object` 负责类型化对象和 JSON 对象之间的转换。

## 单位与缺失

- `step` 是非负整数测量索引；`lifetime` 的单位由运行协议给出。转换器使用环境交互数。
- 能力、奖励与损失的方向和单位由任务定义，不能跨任务直接相加。
- `ComputationCounters` 保留单位更新、边事件、矩阵运算、脉冲、循环、路由和学习更新等不同计数。
- `MemoryAccounting` 分开记录持久状态、参数、回放、外部记忆及读写字节。
- `RoutingSummary` 记录活跃/可用模块、单元和边，以及复用、新分配与路由熵。
- 未观测字段用 `null`，不以 0 代替。潜在计数为 0 时，激活比例为 `None`。
- `plasticity_probe_score` 是指定探针的实测结果。权重变化量、奖励或更新次数不会自动成为可塑性分数。

## 指标

`metrics.recovery(steps, values, baseline=..., floor=..., fraction=...)` 对高值更优的能力序列计算恢复阈值和归一化面积。

```text
target = floor + fraction × (baseline − floor)
normalized = (capability − floor) / (baseline − floor)
```

baseline 必须高于 floor。面积用实际步间距做梯形积分，再除以观测跨度；单点序列返回该点的归一化值。超出基线的观测不截断。`first_recovery_step` 是首个达到阈值的绝对 step，未达到时为 `None`。

`metrics.plasticity` 将每个探针分数除以所选非零参考分数。`metrics.observations` 报告观测数和缺失数；只在全部观测存在时返回均值。`metrics.pareto_front` 显式指定最大化和最小化目标，拒绝缺项，返回不被严格支配的原始行索引。

## 文件写入

`write_records` 与 `write_json` 在同目录临时文件中完成序列化与 fsync 后发布。默认不覆盖现有目标，显式 `overwrite=True` 才原子替换。生成器失败时不发布不完整文件。

`append_records` 读取并验证现有流，再原子替换为扩展流。该接口用于单写入者，内存占用与完整流大小成正比。符号链接路径与非普通目标文件被拒绝。

## Python 示例

```python
from intelligence_dynamics.records import ComputationCounters, StepRecord
from intelligence_dynamics.storage import write_records

record = StepRecord(
    run_id="example",
    step=1,
    lifetime=1.0,
    capability=0.75,
    computation=ComputationCounters(learning_updates=2),
)
write_records("one-step.jsonl", [record])
```

这是单条记录文件；需要完整可验证运行流时，先加入对应 `RunMetadata`。

## 外部模型状态

`state.state_snapshot` 接受标量、列表、元组、Mapping、dataclass 和 NumPy 可转换的数值数组。摘要保留类型、映射键、数组 dtype、shape 和按 C 顺序排列的字节身份。状态只读观察，不反序列化 checkpoint，也不更新模型。

`tensor_elements` 和 `tensor_bytes` 按状态树中的张量叶子累计逻辑元素与存储大小；共享底层存储的多个视图分别计数，不表示去重后的物理内存。实际进程 RSS 由运行管理接口测量。

`state_differences` 返回字段路径。`verify_state_intervention` 将变化限制在指定路径前缀内，可核对外部模型的 carry 重置而参数和优化器状态保持一致。数据加载和框架状态恢复由模型适配器提供。

`configs/measurement-record.schema.json` 提供记录结构的 JSON Schema。计数之间的大小关系、流顺序和数值有限性仍由运行时合同检查。
