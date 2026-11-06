# 软件指南

[English](en/software.md) · [首页](../README.md)

## 安装

核心包使用 Python 标准库，支持 Python 3.10 及以上。`requirements-build.lock` 固定官方 PyPI 索引上的精确构建工具版本；如 lock 包含 `--require-hashes`，pip 同时核对其中的完整 SHA-256；预期摘要应从可信发布来源取得。NumPy 2.2.6 是 NPZ 与原生数组状态接口的可选依赖，其发行物支持 Python 3.10–3.13。

```bash
python -m pip install -r requirements-build.lock
python -m pip install --no-build-isolation --no-deps .
```

启用 NPZ 接口：

```bash
python -m pip install -r requirements-arrays.lock
```

可用 `python -m intelligence_dynamics` 替代 `intelligence-dynamics`。所有示例均从项目根目录执行。输出文件默认拒绝覆盖，父目录需要预先存在。

## 命令

|命令|输入|输出|
|---|---|---|
|`check-source ROOT [--workspace]`|项目目录与检查模式|源码边界检查结果|
|`demo OUTPUT`|可选 `--seed`|合成任务的 JSONL 测量流|
|`validate INPUT`|测量 JSONL|记录类型计数、运行身份与顺序检查|
|`summarize INPUT`|测量 JSONL|完整观测的能力均值及缺失数量|
|`study CONFIG OUTPUT`|合成实验 JSON 配置|搜索、留出评估及对照组 JSON|
|`convert FORMAT INPUT CONFIG OUTPUT`|原生数据与元数据配置|标准测量 JSONL|
|`run CONFIG OUTPUT_DIR`|命令、时限及产物声明|日志、退出状态及产物摘要|
|`batch-spec CONFIG OUTPUT`|云目标、容器摘要及资源|Google Batch JSON 请求体|

常规成功返回 0，命令或数据错误返回 2。本地运行中的程序失败、产物不合格或超时返回 1。状态数据输出到标准输出，命令错误输出到标准错误。成功消息只显示输出文件名，不显示父目录；错误消息隐藏已知输入路径和主目录。实际文件仍写入指定位置。

## 原生数据转换

DAL 示例完全由合成数值构成：

```bash
intelligence-dynamics convert dal-csv examples/dal.csv configs/dal-conversion.json dal.jsonl
intelligence-dynamics validate dal.jsonl
```

前 11 列描述代数、适应度和更新统计，后续列为试次分数。代数乘以任务实例数、试次数、每试次步数，得到累计交互量。`None` 仅允许出现在两个消融适应度列。

创建并转换多采样率 NPZ：

```bash
python examples/make_array_fixture.py forager.npz
intelligence-dynamics convert forager-npz forager.npz configs/forager-conversion.json forager.jsonl
intelligence-dynamics validate forager.jsonl
```

`rewards` 是逐步序列，`total_loss` 是逐更新序列。逐步样本数必须能被更新数整除。每个更新区间计算奖励均值，并保留梯度、静态标量、位置及场景信息。`expected_rollout_steps` 用于确认采样间隔。

等间隔的一维数组使用 `array-npz` 格式；配置可指定 `capability_field` 和 `interval`。转换时逐块复制输入到系统临时目录，从该副本计算 SHA256 并解析测量值；结束时清理副本。临时目录需有容纳一份输入副本的空间。转换后的元数据包含该输入副本的 SHA256 与配置摘要。对象数组、非有限数、重复名称和不支持的形状均拒绝加载。

## 配置

- `configs/cue-study.json`：训练与留出任务种子、任务形状、搜索参数。
- `configs/dal-conversion.json`：运行身份和代数到交互量的尺度。
- `configs/forager-conversion.json`：运行身份和预期 rollout 间隔。
- `configs/local-command.json`：生成一个 JSON 产物的本地命令。
- `configs/batch-job.json`：Batch 请求结构模板。`image` 明确要求调用方在私有副本中提供运行时取得的不可变镜像引用；模板本身不嵌入具体摘要，不能直接提交。

配置使用严格 UTF-8 JSON，不接受重复键、NaN 或 Infinity。配置不会作为 Python 代码执行。机器私有配置放在 `local-config/`，运行产物放在 `runs/`；这两个目录与文档中的默认生成文件已排除在版本控制之外。自定义输出路径需自行加入忽略规则。

## 构建

先按 [README](../README.md#开发与测试)安装开发依赖和 editable 项目。下列命令可用于 Bash 和 PowerShell，从项目根目录执行；源码副本和构建目录必须尚不存在，且位于项目外。

```bash
python -B -m intelligence_dynamics check-source --workspace .
python -B tools/prepare_source.py ../IntelligenceDynamics-source
python -B -m intelligence_dynamics check-source ../IntelligenceDynamics-source
python -B -c "import shutil; shutil.copytree('../IntelligenceDynamics-source', '../IntelligenceDynamics-build')"
python -m pip wheel --no-build-isolation --no-deps ../IntelligenceDynamics-build --wheel-dir ../IntelligenceDynamics-wheels
```

`prepare_source.py` 先检查工作区，仅复制 `validation.PROJECT_FILES` 登记的普通源码文件，再核对内容并严格检查副本；目标位于项目内、已经存在、父目录不存在或路径经过符号链接或 Windows 重解析点（包括目录联接点）时拒绝执行。复制或核验失败返回非零，失败目标需检查后另选新路径重试。wheel 在独立构建副本中生成，源码副本与构建产物分开保存。

默认的 `check-source ROOT` 只接受登记源码与合法根 `.git` 管理目录或管理文件；根 `.git` 存在时，使用系统 Git 校验其工作树身份；无 `.git` 的源码包无需安装 Git。检查会拒绝缺失文件、意外文件或目录、嵌套 `.git` 和符号链接或 Windows 重解析点。`--workspace` 使用固定允许清单，额外容纳根目录中的 `.venv/`、`.pytest_cache/`、`.ruff_cache/`、`build/`、`dist/`、`runs/`、`local-config/`、`local-run/`，项目对应的根或 `src/` 下 egg-info，登记源码目录中的 `__pycache__/`，根 `.env`、`.env.*`、SQLite 数据库与旁文件，以及本文和 README 的默认输出文件。忽略规则的新增条目不扩大此清单；自定义产物应放入已有允许的产物目录。两种模式都检查登记源码的完整性与路径类型。

wheel 包含 `intelligence_dynamics` 模块、类型标识、Apache-2.0 正式文本、许可说明、第三方通知与命令入口。源码包另含测试、配置、示例和双语文档。安装验证应使用新的虚拟环境，先安装 wheel，再从项目外运行 `demo`、`validate`、`summarize`、DAL 转换和 `study`。

## 发行安装与恢复

从[发行页](https://github.com/ranzevor/IntelligenceDynamics/releases)选择具体版本，下载 wheel、完整源码包、`release-manifest.json` 和 `SHA256SUMS`。清单记录源码提交、文件长度及摘要；核对下载文件的 SHA-256 与该版本校验表。完整源码包提供运行示例所需的 `configs/` 和 `examples/`。

在新的虚拟环境中，进入 wheel 所在目录安装核心工具：

```bash
python -m pip install --no-index --no-deps intelligence_dynamics-0.1.0-py3-none-any.whl
intelligence-dynamics demo demo.jsonl
intelligence-dynamics validate demo.jsonl
intelligence-dynamics summarize demo.jsonl
```

核心 wheel 无额外运行依赖。NPZ 和原生接口的额外依赖仍按本文与原生接口指南安装。Python API 可从虚拟环境内用 `import intelligence_dynamics` 调用，模块命令与控制台命令等价。

卸载使用 `python -m pip uninstall intelligence-dynamics`。安装和卸载不迁移或删除已有 JSONL、队列数据库、运行日志或 checkpoint；保留这些数据及已使用的 wheel。恢复安装时在独立虚拟环境重新安装保存的 wheel，并先校验已有输入。数据与 checkpoint 兼容条件分别见[数据契约](data.md)和[原生接口](native.md)。

## 发行构建

从准备发行的准确 Git 提交构建。先按 README 安装锁定的构建工具；输出目录必须尚不存在，且位于项目外：

```bash
python -B tools/build_release.py ../IntelligenceDynamics-release
python -B tools/smoke_wheel.py ../IntelligenceDynamics-release/artifacts ../IntelligenceDynamics-consumer
```

构建工具核对源码字节与提交，从该提交的 Git 对象导出登记源码，再在独立副本构建 wheel；`artifacts/` 包含 wheel、完整源码 ZIP、源码与 wheel 内文件清单、构建版本、资产摘要及 `SHA256SUMS`。日志保存在输出目录。

消费者验证先从源码 ZIP 的实际文件字节重算 Git 树，核对 `release-manifest.json` 的 `source.tree`，再在临时虚拟环境中安装实际 wheel，从项目外调用 Python API、演示、校验、汇总、DAL 转换和合成实验，检查重复输出、未知命令与非法输入，再验证卸载和重装。原始输出及结果保存在独立消费者目录。

发行使用指向已核验提交的附注标签 `v<版本>`。上传前确认版本尚未占用；上传后回读标签提交、发行正文与资产身份，并从正式下载入口比较文件字节。已发布标签及资产不覆写；修复通过新版本交付。发行说明列运行条件、实际验证范围、安装与恢复方法、许可与兼容变化。
