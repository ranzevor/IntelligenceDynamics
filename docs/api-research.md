# 研究算法 API

[English](en/api-research.md) · [研究设计](research.md)

公开入口为 `intelligence_dynamics.research`。运行环境为 Python 3.10 或以上。合成实验和确认统计使用标准库，原生轨迹窗口汇总使用 NumPy。下面每个 Python 代码块都可在包可导入的环境中独立运行。直接使用源码时，将项目的 `src` 目录加入 `PYTHONPATH`。

合成实验的结果对象使用 dataclass，可由 `dataclasses.asdict` 转为普通嵌套数据，再通过 `json.dumps(..., allow_nan=False)` 输出。JSON 会把元组表示为数组，把 `None` 表示为 `null`。

## 1. 类型与验证

计数、时间、cue、动作编号与种子使用 Python 整数，不接受布尔值冒充整数。种子非负。数值参数须有限；概率在闭区间 0 到 1 内。取值、范围和序列格式校验失败抛出 `ValueError`。入口要求的配置、策略、反馈或 `Series` 对象类型错误时抛出 `TypeError`；未重置或已终止的任务继续推进时抛出 `RuntimeError`。

`TaskConfig`、`LearningRule`、`Scenario`、`SearchConfig` 和结果 dataclass 为不可变对象。可配置序列在构造时转成元组，调用方后来修改输入列表不会改变配置。

## 2. 任务与场景

### `TaskConfig`

| 字段 | 默认值 | 契约 |
|---|---|---|
| `steps` | `96` | 正整数，执行动作数 |
| `cue_count` | `4` | 正整数，cue 数量 |
| `delay` | `1` | 非负整数；动作 t 的反馈在动作 t + delay + 1 之前到达 |
| `switches` | `(32, 64)` | 严格递增整数，均位于生命周期内部 |
| `mappings` | `((0,1,0,1),(1,0,1,0))` | 每行宽度等于 cue_count，元素为整数 0 或 1 |
| `seed` | `0` | 环境随机种子 |

有切换时至少需要两行映射；无切换时允许一行。每次切换进入下一行，超过末行后回到首行。`fingerprint` 是完整任务配置的 SHA-256 标识。改变 `steps` 或 `cue_count` 时，应同时提供与其相容的切换位置和映射。

`Scenario(name, partition, task)` 的分区为 `train` 或 `heldout`，名称非空，`kind` 由是否存在切换决定。

`make_bank(*, partition, seed, pairs=1, steps=64, cue_count=4, delay=1)` 返回 `4 * pairs` 个场景：每种静态/切换条件各有 `pairs` 对镜像。`steps` 至少为 8。镜像对共享 cue 随机流和切换时刻，目标相反。相同主种子仅改变分区名称不会产生独立数据。

```python
from intelligence_dynamics.research import (
    FixedPolicy,
    LearningRule,
    TabularPolicy,
    TaskConfig,
    run_episode,
)

config = TaskConfig(
    steps=24,
    cue_count=2,
    delay=1,
    switches=(8, 16),
    mappings=((0, 1), (1, 0)),
    seed=7,
)
fixed = run_episode(config, FixedPolicy((0, 1)), seed=3)
learner = TabularPolicy(LearningRule(priors=(0.0, 0.0)), mode="plastic")
adaptive = run_episode(config, learner, seed=3)
print(len(adaptive.records), adaptive.undelivered)
print(fixed.accuracy, adaptive.accuracy)
```

### 逐步接口

- `CueTask(config).reset()` 返回 `Observation(cue)`。
- `CueTask.advance(action)` 接受整数 0 或 1，返回 `Transition`。
- `Transition` 包含 `observation`、`feedback`、`done` 与评估专用 `record`。
- `Feedback(credit_id, value)` 的值在 0 到 1 内，编号对应原始动作。
- `StepRecord` 包含 `step`、`cue`、`action`、`target`、`regime`、`reward`。
- 终止转移的 `observation` 为 `None`。重置前或终止后调用 `advance` 抛出异常。
- `CueTask.time` 返回下一动作编号；`snapshot()` 返回可交由 `state_digest` 处理的环境状态记录。

评估和分支检查读取 `record` 与 `snapshot()`；策略通过 `Observation(cue)` 接收观测。

## 3. 策略与学习规则

### `LearningRule`

| 字段 | 默认值 | 范围与含义 |
|---|---|---|
| `priors` | `(0.0, 0.0, 0.0, 0.0)` | 非空，每项在 -1 到 1 内 |
| `rate` | `0.5` | 大于 0，不超过 1 |
| `exploration` | `0.05` | 探索概率 |
| `threshold` | `0.4` | 高误差阈值，范围 0 到 1 |
| `decay` | `0.7` | 累积量保留系数，范围 0 到 1 |
| `accumulation` | `0.18` | 严格正的累积触发阈值 |
| `window` | `3` | 正整数，按已送达反馈计数 |
| `required` | `2` | 正整数，不超过 window |
| `random_probability` | `0.5` | 随机更新概率 |

cue 偏好 `b` 初始化 `Q(cue,0)=(1-b)/2`、`Q(cue,1)=(1+b)/2`。一次提交更新为 `Q ← Q + rate * (feedback - Q)`。

`TabularPolicy(rule, mode="plastic")` 支持 `MODES` 中的七种模式：

| 模式 | 提交条件 |
|---|---|
| `fixed` | 不更新价值，保留初始偏好、探索和并列值随机决策 |
| `plastic` | 每次送达反馈都更新 |
| `conditional` | 当前绝对误差大于或等于 threshold |
| `accumulator` | 累积量按 `decay * previous + max(0, error-threshold)` 更新；达到 accumulation 时提交当前信用并清零 |
| `persistence` | 最近 window 次反馈中至少 required 次高误差，提交当前信用并清空窗口 |
| `buffered` | 同一持续性条件成立时，依次提交窗口内所有未过期高误差信用，再清空窗口 |
| `random` | 以 random_probability 更新，门控随机流与行动随机流分开 |

缓冲窗口在第 `i` 次反馈时包含索引 `i-window+1` 到 `i`。每个信用至多提交一次。缓冲提交使用提交时的当前价值和原始反馈目标。

策略方法：

- `reset(*, seed=0)` 重建初始价值、随机流与生命周期状态。
- `act(observation, feedback=None)` 先处理可选反馈，再选择并登记一个动作。
- `observe(feedback)` 单独处理反馈，返回本次提交的信用编号元组；不存在或已处理的信用编号报错。
- `clear_memory()` 清空门控记忆，保留价值、未送达信用与随机流。
- `values`、`updates`、`pending_count` 返回当前价值、更新记录和未送达信用数量。
- `last_committed` 为本次处理提交的原动作编号；`feedback_count` 为已处理反馈数。
- `AppliedUpdate` 记录原信用编号、原反馈序号、提交序号、更新前后价值与原反馈误差。

`FixedPolicy(mapping)` 是确定性 cue 映射，区别于保留探索的 `mode="fixed"`。

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

## 4. 运行与测量

`run_episode(config, policy, *, seed=0, corruption="clean", corruption_rate=0.0, corruption_seed=None, latency_window=6, latency_threshold=0.75)` 每次先重置策略，再运行完整动作序列。最后一个动作之后才到达的反馈不会用于学习。`undelivered` 统计整个生命周期未被策略处理的动作反馈，包括终点未处理的反馈。

`EpisodeResult` 包含整体与分段正确率、每次切换延迟、每段扰动后的恢复延迟、动作记录、反馈送达记录、更新记录及未处理数量。`Delivery` 保留原信用、送达时刻、扰动标记、实际送达值、提交编号和处理时误差。

`evaluate(rule, scenarios, *, mode="plastic", seed=0, corruption="clean", corruption_rate=0.0)` 返回 `Evaluation`：

- `accuracy` 是各生命周期正确率的等权均值，不是按生命周期长度加权的总体正确率。
- `stationary_accuracy`、`switching_accuracy` 与 `episode_scores` 保留分类和逐场景表现。
- `feedback_count`、`update_count` 与 `update_fraction` 统计已送达反馈及实际提交。
- `corrupt_source_updates`、`clean_source_updates` 及对应分母和率按原反馈归因。
- `switch_feedback_count` 与 `switch_source_update_rate` 使用切换起始后 12 个原动作位置的并集，避免重叠窗口重复计数。
- `updated_errors`、`skipped_errors` 是原反馈误差的观测摘要；`gate_error_separation` 为两组均值之差。
- `switch_latency`、`corruption_recovery` 保留总数、已观测数、缺失数和均值。
- `corrupted_feedback_fraction` 是实际送达反馈中被翻转的比例。

没有反馈时相关率为 `None`；没有某类场景时该类正确率为 `None`。有缺失观测的摘要均值为 `None`。

```python
from intelligence_dynamics.research import LearningRule, MODES, evaluate, make_bank

bank = make_bank(partition="heldout", seed=12, pairs=1, steps=16, cue_count=2)
rule = LearningRule(priors=(0.0, 0.0))
for mode in MODES:
    result = evaluate(
        rule,
        bank,
        mode=mode,
        seed=4,
        corruption="isolated",
        corruption_rate=0.2,
    )
    print(mode, result.accuracy, result.update_fraction)
```

### 测量辅助函数

- `corruption_plan(length, *, kind="clean", rate=0.0, seed=0, burst_length=3)` 返回布尔元组。`clean` 要求 rate 为零；`isolated` 翻转后留一个干净事件；`burst` 连续翻转最多 burst_length 次，然后留一个干净事件。
- `adaptation_latency(correct, start, stop, *, window=6, threshold=0.75)` 输入布尔正确性序列，返回半开区间内首个合格完整窗口的起点相对 start 的偏移。最后一个完整窗口也参与查找；无合格窗口返回 `None`。
- `summarize(values)` 返回 `Observations(count, observed, missing, mean)`。空输入或任何缺失值都会使 mean 为 `None`。

内置扰动恢复查找从一段连续扰动结束后的动作开始，到下一次扰动、下一次环境切换或生命周期终点为止，以最早者为准。

```python
from intelligence_dynamics.research import adaptation_latency, corruption_plan, summarize

print(corruption_plan(6, kind="isolated", rate=1.0))
print(adaptation_latency((False, False, True, True), 0, 4, window=2, threshold=1.0))
print(summarize((1.0, None, 3.0)))
```

## 5. 进化搜索与独立评估

`SearchConfig(population=4, generations=2, elites=1, mutation=0.15, update_cost=0.05, corruption_rate=0.1, noise_weight=0.5)` 要求 population 至少为 2，generations 和 elites 为正整数，elites 小于 population，其余值在 0 到 1 内。

场景集合须非空，场景名和任务指纹不重复，cue 宽度一致。

`evolve(train, *, mode="buffered", seed=0, config=SearchConfig())` 只接受训练分区，且每个训练任务须在最后一次动作之前至少送达一次反馈。各候选和各代使用相同评估随机条件。初始偏好、学习率、探索率、误差阈值、衰减与累积阈值可变异；窗口大小、所需次数和随机概率保持该搜索的固定默认值。

适应度为：

`fitness = (1-w)*clean_accuracy + w*isolated_accuracy - update_cost*((1-w)*clean_updates + w*isolated_updates)`

其中 `w=noise_weight`，updates 指更新比例。选择保留精英，并由精英变异产生其余候选。`SearchResult` 返回规则、模式、适应度、加权训练正确率和更新比例、逐代最好与平均适应度、候选评估数及训练配置指纹。候选评估数为 population × generations；每个候选分别评估干净和孤立扰动条件。

`run_study(train, heldout, *, mode="buffered", seed=0, config=SearchConfig())` 先检验名称、任务指纹和环境随机种子不重叠，再选择规则。结果 `StudyResult` 包含：

- `search`：训练选择结果。
- `heldout`：所选模式的干净评估。
- `frozen`、`always`、`matched_random`：同一所选规则的控制条件；不是分别重新演化的模型。
- `isolated`、`burst`：同一所选规则的扰动评估。
- `random_probability`：由干净训练数据校准的随机门控概率。
- `heldout_fingerprints`：最终任务配置指纹。

不同机制家族需要分别调用 `evolve` 或 `run_study`，并匹配总候选、交互和资源预算。随机门控只匹配预期更新概率，实际更新数量仍应比较。

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

## 6. 分支与状态完整性

`compare_branches(config, rule, *, boundary, mode="plastic", seed=0, intervention="freeze")` 在动作 boundary 之前分叉，boundary 至少为 1 且小于 steps。干预为 `freeze` 或 `reset_memory`。

`BranchComparison` 包含共同前缀、两条后续轨迹、共同状态摘要、变化路径、干预是否确实改变状态、两臂正确率和边界后的更新数量。这里 `delta = continued_accuracy - intervened_accuracy`。

`state_digest` 支持有限标量、列表、元组、字符串键字典及 dataclass 实例，保留标量类型和容器类型，拒绝循环结构。`state_changes` 返回结构化路径元组，键中含点号也不会与嵌套路径混淆。

`verify_intervention(before, after, *, allowed_paths, require_change=True)` 只允许指定路径或其子路径发生变化。默认拒绝无变化。`trace_digest(columns, *, boundary)` 要求每个命名列恰好有 boundary 行。`verify_paired_history` 还要求两份历史的列名、每列行序与值类型一致。

```python
from intelligence_dynamics.research import (
    LearningRule,
    TaskConfig,
    compare_branches,
    verify_intervention,
    verify_paired_history,
)

config = TaskConfig(
    steps=12,
    cue_count=2,
    delay=1,
    switches=(6,),
    mappings=((0, 1), (1, 0)),
    seed=4,
)
result = compare_branches(
    config,
    LearningRule(priors=(0.0, 0.0)),
    boundary=6,
    seed=3,
)
print(len(result.common_prefix), result.intervened_updates)
before = {"memory": [1.0], "value": 0.5}
after = {"memory": [], "value": 0.5}
print(verify_intervention(before, after, allowed_paths=(("memory",),)))
history = {"reward": (1.0, 0.0), "cue": (0, 1)}
print(verify_paired_history(history, history, boundary=2))
```

## 7. 共同事件支持与固定时间块

`Series(rewards, states)` 要求两个等长非空序列。奖励须有限，state 标识须为整数或非空字符串。

`common_support(control, treatment, *, start=0, stop=None, window=3, spacing=None)` 要求两条序列具有相同长度。所有前后窗口必须完整落入 `[start, stop)`。间隔默认 `2*window`，指定时也不得小于该值。

`SupportComparison` 保留双方合格事件、共享事件、最终选取事件、Jaccard 重叠、两臂事件响应、逐事件差及均值差。这里 `delta = treatment_response - control_response`，与分支 API 的差值方向不同。无共享选取事件时响应与差值均为 `None`；双方无事件时 Jaccard 为 `None`。

`select_events(events, *, spacing)` 要求时间戳严格递增。`compare_blocks(control, treatment, *, edges)` 按相邻边界形成固定半开区间，返回 `BlockComparison` 元组，差值同样为 treatment 减 control。

`aggregate_comparisons(deltas)` 保留逐对值、观测计数与正负计数。有缺失时均值、中位数和范围均为 `None`，已观测正负计数仍保留。

```python
from intelligence_dynamics.research import (
    Series,
    aggregate_comparisons,
    common_support,
    compare_blocks,
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

## 8. 配置判据

`Criterion(name, metric, direction, threshold)` 接受有限阈值。方向为 `ge`、`gt`、`le`、`lt` 或 `eq`，分别对应大于等于、大于、小于等于、小于和精确等于。判据名称必须唯一；多个判据可引用同一指标。

`assess(values, criteria, *, combine="all")` 返回 `Decision(results, combine, passed, evaluated, missing)`。每项 `CriterionResult` 保留名称、指标、方向、阈值、实际值及通过状态。指标缺失或为 `None` 时该项未判定；非有限数和布尔指标输入报错。

| 组合 | 决定性结果 | 无决定性结果且存在缺失 | 完整且无决定性结果 |
|---|---|---|---|
| `all` | 任一 False → False | None | True |
| `any` | 任一 True → True | None | False |

空判据集报错。判据引擎不估计效应量、不执行显著性检验。

```python
from intelligence_dynamics.research import Criterion, assess

criteria = (
    Criterion("capability", "accuracy", "ge", 0.75),
    Criterion("update_budget", "update_fraction", "le", 0.5),
)
result = assess({"accuracy": 0.8, "update_fraction": None}, criteria)
print(tuple(item.passed for item in result.results), result.passed)
```

## 9. 循环状态确认统计

`intelligence_dynamics.research.confirmation.confirm_carry_effects(results, *, indices)` 返回 JSON 兼容字典。每条输入包含 `index`、`seed`、`run_identity` 和 `R_i_reset_minus_continued_post_mean_reward`，索引与种子须唯一，全部结果须对应同一运行身份。输出按 `indices` 的顺序排列，包含配对均值、正值与零值数量、单侧精确二项检验、正效应比例的双侧 95% Clopper–Pearson 区间和联合判据结果。

```python
from intelligence_dynamics.research.confirmation import confirm_carry_effects

indices = tuple(range(25, 33))
rows = [
    {
        "index": index,
        "seed": index,
        "run_identity": "example",
        "R_i_reset_minus_continued_post_mean_reward": 0.1,
    }
    for index in indices
]
result = confirm_carry_effects(rows, indices=indices)
print(result["positive_count"], result["criterion_met"])
```

`summarize_confirmation_pair(continued, reset, *, boundary=150000, endpoint=300000)` 接收与循环状态试点相同的五条 NumPy 轨迹，在配对汇总中增加边界后前 1,024 次交互和其余区间的平均奖励差。边界后必须包含超过 1,024 次交互。科学判据见[确认实验设计](carry-confirmation.md)。
