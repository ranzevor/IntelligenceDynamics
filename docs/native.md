# 原生模型与环境

[English](en/native.md) · [首页](../README.md)

`intelligence_dynamics.native` 执行显式传入的 DAL、Foragax 和 DRQN 对象。

## 依赖与接口合同

核心安装和合成对象示例只需 Python 标准库。数组状态需要 NumPy。默认 DRQN 数值后端使用 JAX；实际模型还需要其上游声明的 Haiku、Optax、Flashbax、Chex、Foragax 和其他依赖。

DRQN 接口对照 [continual-foragax-agents](https://github.com/steventango/continual-foragax-agents) 公开的对象形状；DAL 字段合同对照 [DAL](https://github.com/trsjp/dal) 公开实现。项目内 JAX 状态编解码测试使用 Python 3.12 与 [`requirements-native.txt`](../requirements-native.txt) 的精确 CPU 依赖版本；连接 Foragax 模型仍需按调用方选择并审查的上游版本配置其额外依赖。软件许可见 [第三方通知](../THIRD_PARTY_NOTICES.md)。

| 对象 | 必需接口 |
| --- | --- |
| DAL 受评个体 | `individual_id`、`normalised_fitness`、两类 `connectivity_grid_*`、`io_list_a`、`io_list_m`、`rl_learning_rate`；受评统计见下文 |
| DAL SPNN 包装器 | `send`、`recv`、`pipe.poll`、`individual_id`、`is_unmutated_copy` |
| Foragax 环境 | `default_params`、`reset(key, params)`、`step(key, state, action, params)`、`action_space(params)` |
| DRQN agent | `state`、`buffer.add/can_sample`、`phi`、`act`、`_carry_for_buffer`、`_start`、`_compute_reward_trace`、`encode_scalar_features`、`_update_state_with_metrics`、`_advance_update_clock`、`reward_clip` |
| 原生 RLGlue 环境包装器 | `state`、`_start(state)`、`_step(state, action)` |

DRQN 以单个 agent 为执行单位。批量训练状态按上游布局拆分后逐个执行。状态包含参数、目标参数、优化器、回放、PRNG、实时 carry、上一 transition、奖励轨迹、超参数和计数器。它们必须支持独立复制和显式 codec 回读。JAX typed key 使用 `key_data` 参与摘要，保留 PRNG 实现标识；Optax 命名元组保留类型。参数及数组要求有限，`hypers.freeze_steps` 可使用正无穷表示无冻结边界。

`native_state_snapshot(value)` 返回类型敏感的摘要、每个数组的 shape/dtype/内容摘要，以及张量元素数和字节数。列表中的普通标量不计作张量存储。

数组支持 NumPy 数值类型及 `ml_dtypes.bfloat16`，并保留 `bfloat16` 语义类型。两者均执行非有限值检查。

## DAL 个体诊断

`dal_individual` 直接读取受评个体。对于原始 `nmnn.nn`，适配器对 `total_dw_list_nm` 和 `total_dw_list_rl` 求逐元素绝对值之和，并读取 `mean_active_rl_learning_rate`、`mutations_applied`。若对象已有完整 `stats`，也可直接消费。原始个体不携带父代包装器的 `is_unmutated_copy`，此时显式提供 `unmutated`。

下面是可直接运行的合成对象例子。

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

`dal_population` 消费全部个体，返回分离的排序、父代和精英标记。按 fitness 降序排列，并列个体保持传入种群中的顺序；父代和精英边界使用同一排序。原始个体可用 `unmutated_flags` 提供逐个对应的元数据。空激活投影的平均 susceptibility 是 `None`。缺字段、重复 ID、非有限诊断或非法排名参数会失败，不补零。

`dal_remote_population(workers, generation=..., parent_pool=..., elites=..., timeout=...)` 对已有 SPNN 包装器调用 `send("get_over_here")`，随后 `pipe.poll(timeout)` 和 `recv()`。它核对响应 ID，再抽取个体诊断；不会创建、演化或终止 worker。管道超时抛出 `TimeoutError`，晚到响应的处理由管道所有者负责。

## Foragax 环境适配

适配器显式管理 reset、step 和动作采样的随机 key。`snapshot()` 与 `restore()` 包含环境状态、当前观察、下一 key、交互计数和终止标记；身份不匹配的配置不能互相恢复。参数在构造时复制，并在适配器生命周期内保持固定；`params` 属性返回独立副本。原生调用使用单独的参数副本，若调用修改参数则拒绝该结果，不推进保存的状态。使用另一套参数时应构造另一个适配器。

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

`from_factory(factory, env_id, options=..., key=..., split=...)` 调用真实注册工厂。默认 `transition="step"` 使用环境公开包装器；显式 `transition="step_env"` 使用其底层转移，跳过包装器的自动重置和终止辅助逻辑。持续流环境使用 `step_env`；回合任务通过 `step` 包装器处理终止与自动重置。任何终止后的后续 step 都要求 reset。

## 连接已安装的 DRQN

下面定义一个接入函数。调用者提供已安装原生栈、环境 ID、环境选项、agent 参数和上游 collector。

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

`agent_parameters` 至少包含 `total_steps`、`epsilon`、`gamma`、`target_refresh`、`buffer_size`、`buffer_min_size`、`batch`、`update_freq`、`sequence_length`、`optimizer` 和 `representation`。该接口的 Adam 配置使用 `alpha`、`beta1`、`beta2`、`eps`；representation 的类型与形状由原生 network builder 验证。不要把整个实验文件直接当作 agent 参数。`DRQNExecutor` 也可直接接收已有 agent 和具有 `_start/_step` 的环境包装器。

默认执行链依次推进环境、将上一 transition 加入原生 replay、计算奖励轨迹和标量特征、调用原生策略、按时钟决定 optimizer 更新，再推进 agent 时钟。实际采样、梯度、optimizer 和目标参数刷新由 agent 的 `_update_state_with_metrics` 执行。

当实时 carry 为 `None`，`DRQNPolicy` 用当前已学习参数执行 `phi(..., carry=None)`，取其初始 carry 供下一条 replay 使用。实际 `act` 仍接收 `carry=None`。已有 replay、环境状态和 RNG 不因重置操作被改写。首次转移直接执行，完成 carry 从 `None` 到张量的初始化。

更新判据使用转移前 `steps % update_freq == 0`，并同时要求 replay 可采样、允许学习且未达到原生 `freeze_steps`。更新数必须恰好增加零或一；未冻结 agent 的时钟必须恰好增加一。原生冻结边界会停止 agent 时钟与 epsilon 调度，但环境、动作和 replay 仍继续。起始 agent 时钟必须为正。

## 检查点、干预与续跑

`NativeCheckpoint` 的 history 必须覆盖全部环境交互，长度等于 `state.total_steps`。新任务从 `lifecycle.start(seed=...)` 开始。接入已运行 RLGlue 时必须同时提供完整历史，不能省略或截短：

```python
from intelligence_dynamics.native import NativeCheckpoint, execution_state

def checkpoint_from_running_glue(glue_state, full_history, *, terminated=False):
    return NativeCheckpoint(
        execution_state(glue_state, terminated=terminated),
        tuple(full_history),
    )
```

`CheckpointStore` 的 codec 由调用者显式指定，必须保留完整状态、类型和历史。保存时先做 codec 往返验证，加载时先核容器长度与摘要，再解码并核全部状态身份。复制保留原 payload 字节。名称不能覆盖已有检查点，也不能穿越目录或符号链接。

保存的往返验证另行比较实际数组类型及 JAX 弱类型属性，复制时也比较两次解码结果的这些属性。普通数组的既有摘要与 `native-checkpoint-v1` 格式保持不变，旧检查点无需迁移即可读取。v1 的持久化摘要记录数组数值类型、形状与内容，读取时的数组后端类型由 codec 重建。使用生成检查点时的可信 codec 加载。

保存和加载均会执行调用方指定的 codec，包括 decoder。Pickle 解码可执行代码；加载前通过受控存储或认证渠道核对来源，再使用已指定的可信 codec。容器长度、摘要和 JSON 格式用于内容完整性检查。

分支示例接收已运行的 lifecycle、共同 checkpoint 和新建的可信 store。`additional` 包含首步探针。流程依次加载并复制分支、重置 carry、保存并重载、执行后续转移，再核对共同历史。

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

`learning=False` 禁止 optimizer 更新，保持参数、目标参数、优化器和更新数不变，仍收集 replay 并推进未冻结时钟。`freeze(checkpoint, paths)` 是另一种干预：每次转移后将所选字段恢复到捕获值。此时更新确实执行，更新数和指标记录该次计算，所选状态随后被固定。

字段路径可使用 dataclass 字段、可变映射键、序列索引和命名元组字段名。重叠路径、计数器修改、无变化的 reset、缺失字段或非选中字段变化均被拒绝。嵌套不可变 Mapping 不支持替换；命名元组替换保留实际类型。共同历史比较包括字段类型、数值和数组身份，缺失边界直接报错。

终止或截断后 `run` 停止追加交互。原生方法的异常向调用者传播；检查点目录中的已写文件不会作为成功结果使用。
