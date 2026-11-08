# 开发与维护契约 / Development and maintenance contracts

| 开发对象 / Development area | 接口与维护要求 / Interface and maintenance requirements | 实现与测试入口 / Implementation and test references |
|---|---|---|
| 测量数据 / Measurement data | JSONL 0.1 记录类型、缺失值、运行身份、严格 JSON 解析与原子写入 / JSONL 0.1 record types, missing values, run identity, strict JSON parsing, and atomic writes | [`data.md`](data.md) / [`en/data.md`](en/data.md), [`measurement-record.schema.json`](../configs/measurement-record.schema.json), `records.py`, `schema.py`, `storage.py`, `test_records_storage.py` |
| 研究算法接口 / Research algorithm interfaces | 合成线索任务、延迟反馈、学习规则、训练/留出隔离、配对干预、共同时间支持与资源前沿 / synthetic cue tasks, delayed feedback, learning rules, train/heldout separation, paired interventions, common temporal support, and resource frontiers | [`api-research.md`](api-research.md) / [`en/api-research.md`](en/api-research.md), `research/`, `metrics.py`, `test_research_*.py`, `test_metrics_adapters.py` |
| 原生模型接口 / Native model interfaces | DAL 诊断、Foragax 环境、DRQN 交互与更新调度、完整状态身份、checkpoint 续跑与选择性干预 / DAL diagnostics, Foragax environments, DRQN interaction and update scheduling, complete state identity, checkpoint continuation, and selective interventions | [`native.md`](native.md) / [`en/native.md`](en/native.md), `native/`, `state.py`, `test_native_*.py`, [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) |
| 运行管理 / Execution management | 本地命令、声明产物、事务任务队列、Batch 请求构造与资源清理接口 / local commands, declared artifacts, transactional job queues, Batch request construction, and resource-cleanup interfaces | [`runtime.md`](runtime.md) / [`en/runtime.md`](en/runtime.md), `runtime/`, `test_runtime*.py`, `test_queue_recovery.py` |
| 安全要求 / Security requirements | 控制服务认证、命令与路径约束、错误及日志的敏感信息处理 / control-service authentication, command and path constraints, and sensitive-information handling in errors and logs | 技术接口 / technical interfaces: [`runtime.md`](runtime.md) / [`en/runtime.md`](en/runtime.md), `runtime/service.py`, `runtime/inspection.py`, `test_runtime_privacy.py`; 安全报告 / security reports: [`SECURITY.md`](../SECURITY.md) |
| 软件包与依赖 / Package and dependencies | Python 3.10+ 标准库核心、可选 NumPy 数组接口、固定构建与开发依赖、wheel 包与命令入口 / a Python 3.10+ standard-library core, optional NumPy array interfaces, pinned build and development dependencies, a wheel package, and a command entry point | [`software.md`](software.md) / [`en/software.md`](en/software.md), [`pyproject.toml`](../pyproject.toml), `requirements-*.txt`, `python -m unittest`, `ruff`, `pip wheel` |
| 弃用与退役 / Deprecation and retirement | 版本状态、旧入口响应、运行与队列收束、日志/checkpoint 处置及托管、索引与凭据撤销 / version status, old-entry responses, run and queue quiescence, log/checkpoint disposition, and hosting, index, and credential withdrawal | 本文件下节 / next section in this document, [`CHANGELOG.md`](../CHANGELOG.md), [`runtime.md`](runtime.md), [`SECURITY.md`](../SECURITY.md) |
| 视觉资产 / Visual assets | 品牌图、概念封面及 Python、数据协议、许可徽章的用途、替代文字与链接 / purposes, alternative text, and links for the mark, conceptual cover, and Python, protocol, and license badges | 资产说明 / asset descriptions: [`assets/README.md`](../assets/README.md); 图像 / images: [`brand.png`](../assets/brand.png), [`cover.png`](../assets/cover.png) |
| 权属与许可 / Rights and licensing | 项目材料的 Apache-2.0 许可范围、商标使用、第三方通知及研究材料来源 / Apache-2.0 scope for project materials, trademark use, third-party notices, and research-material origins | 许可与商标 / license and trademarks: [`LICENSE`](../LICENSE), [`LICENSE-NOTICE.md`](../LICENSE-NOTICE.md); 第三方许可 / third-party licenses: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md); 材料来源 / material origins: [`materials.md`](materials.md) |

## 弃用、停止维护与退役

IntelligenceDynamics 的每个进入发行或维护流程的版本必须在版本记录中使用 `maintained`、`deprecated`、`end-of-maintenance` 或 `retired` 之一。状态变化由项目负责人决定，并同时记录适用版本、决定生效边界、可用替代入口以及数据读取或导出要求。

- **弃用**：入口继续存在时必须产生明确弃用诊断，保持既有结果/错误语义，不静默改用不兼容行为。替代入口及已知数据转换边界写入版本记录和软件指南。
- **停止维护**：版本记录明确最后一个维护版本和停止接受的变更类别。已有运行产物、日志、队列状态或 checkpoint 不因状态变化自动删除或改写。
- **退役版本与旧入口**：CLI、模块入口或控制端点在退役后不得接受新工作并返回成功。仍随离线包提供的旧入口返回非零或非 2xx 的明确退役响应；不兼容数据或 checkpoint 由读取器显式拒绝，不猜测升级。
- **运行与队列**：退役开始后停止创建新运行和领取新任务。已接受任务按实际提供方状态完成、取消或进入可恢复的阻塞状态；幂等键、租约和终态保留，未知远端结果不得标为成功。
- **日志与 checkpoint**：处置前由数据负责人选择导出、保留或删除范围，核对运行身份、格式、访问权限和已完成的导出。导出按源数据的访问权限保存，checkpoint 由对应 codec 读取。
- **托管、索引与凭据**：如存在托管端点或包索引制品，获授权的运营方停止入口、阻止新安装或任务提交，并记录制品状态。如存在服务身份、访问令牌或密钥，按其提供方撤销并核对失效；凭据不写入源码、日志或退役记录。
- **安全报告渠道**：私密报告入口见 [SECURITY.md](../SECURITY.md)；版本退役时同步更新该入口的有效范围。

## Deprecation, end of maintenance, and retirement

Every IntelligenceDynamics version entering distribution or maintenance must use one of `maintained`, `deprecated`, `end-of-maintenance`, or `retired` in its version record. The project owner decides each transition and records the affected version, effective boundary, available replacement, and data-reading or export requirements.

- **Deprecation:** While an entry point remains available, it emits an explicit deprecation diagnostic and preserves its established result and error semantics. Version notes and the software guide identify the replacement and known data-conversion boundary.
- **End of maintenance:** Version notes identify the last maintained version and the categories of change no longer accepted. Existing run artifacts, logs, queue states, and checkpoints are not automatically deleted or rewritten by the status change.
- **Retired versions and old entries:** A retired CLI, module entry, or control endpoint does not accept new work and report success. An old entry still shipped for offline use returns an explicit nonzero or non-2xx retirement response. Readers reject incompatible data or checkpoints explicitly rather than guessing an upgrade.
- **Runs and queues:** Retirement stops new run creation and new queue claims. Accepted work reaches a real provider-confirmed completion, cancellation, or recoverable blocked state. Idempotency keys, leases, and terminal states remain intact, and an unknown remote result is not labelled successful.
- **Logs and checkpoints:** Before disposition, the data owner chooses the export, retention, or deletion scope and verifies run identity, format, access authorization, and completed exports. Exports retain the source data access permissions, and checkpoints are read with their corresponding codec.
- **Hosting, indexes, and credentials:** If hosted endpoints or package-index artifacts exist, an authorized operator stops the entry, prevents new installations or submissions, and records artifact status. If service identities, access tokens, or keys exist, they are revoked through their provider and checked for invalidation. Credentials do not enter source, logs, or retirement records.
- **Security reporting:** Use the private channel in [SECURITY.md](../SECURITY.md) and update its valid scope when a version retires.
