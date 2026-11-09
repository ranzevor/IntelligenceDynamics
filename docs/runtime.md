# 运行管理

[English](en/runtime.md) · [首页](../README.md)

## 本地命令

```bash
intelligence-dynamics run configs/local-command.json local-run
```

输出目录必须不存在。运行器保存 `request.json`、`stdout.log`、`stderr.log` 和 `run.json`。命令使用 argv 数组，不通过 shell 拼接。默认请求记录只保存命令摘要与参数数量；需要保留已审查参数时，设置 `record_command=true`。结果包含退出码、耗时、声明产物的长度和 SHA256，以及可获得的单进程与进程树 RSS 采样峰值。内存峰值取自采样期间的 RSS 观测，未采到时为 `null`。测量错误与程序退出状态分别保存。

产物格式支持 binary、JSON、测量 JSONL、SQLite、NPZ 和 LZMA。`Artifact` 可声明 JSON 必需字段、SQLite 必需表、NPZ 必需数组、轴长度和序列变更次数。压缩容器展开量受 `max_bytes` 限制，默认 256 MiB。LZMA 只检查容器，不反序列化其内容。SQLite 产物须先完成 checkpoint，且没有非空 WAL 或回滚日志，再通过只读 `quick_check` 检查；NPZ 拒绝对象数组和非有限数。路径必须是输出目录内的规范相对路径，不允许符号链接或 Windows 重解析点。格式检查与返回的长度、SHA-256 取自同一份有界快照；检查期间原件变化会被拒绝。

超时终止 POSIX 进程组；命令退出后也清理该进程组。Windows 终止直接子进程；需要进程树隔离时，配置宿主 Job Object 或容器。

运行环境（包版本、Python 版本、操作系统与架构）保存在 `request.json` 的 `runtime` 字段。任务状态依据命令退出、声明产物检查和进程、日志清理结果判定；其他工作文件不参与产物验收。`run.json` 保存失败时，`run_command` 抛出 `RunPersistenceError`，其 `result` 属性保留已完成的运行结果。

`run_command` 的 `environment` 参数用于非敏感实验环境覆盖。请求只保存覆盖键名与规范化内容摘要，不保存明文值；继承的系统环境不进入摘要。复现时需提供同样的命令配置和显式设置，并单独固定模型依赖的系统环境。运行器诊断隐藏已知的命令绝对路径、输出路径和用户主目录；程序自己的标准输出、标准错误及业务产物原样保存，不进行内容脱敏。

## 任务队列

`runtime.Queue` 使用 SQLite 保存请求、状态与转换历史。相同 request_id 和相同规范重复入队返回同一任务；相同 ID 配不同规范是错误。租约带随机 token，过期或被替换的 token 不能更新状态。

`enqueue(..., requires=[...])` 可声明已入队任务作为先决条件。只有这些任务全部成功才允许后续提交；失败的前置任务不会放行。

控制器每次 `tick` 处理一个可领取任务，使用持久化的轮询顺序；领取后移至队尾，新任务追加至队尾。首次取消和显式重试取消将任务前置一次，重复取消不反复插队。取消受阻、提交结果未知和清理等待继续参与轮询，不阻塞无依赖任务。旧数据库首次打开时在事务中补齐领取顺序，按原入队顺序回填，保留任务身份、状态、历史和租约。

```text
QUEUED → SUBMITTING → SUBMITTED / RUNNING
RUNNING → CLEANUP_WAIT → ARTIFACT_WAIT → SUCCEEDED / FAILED
取消：CANCEL_SUBMITTING → CANCEL_REQUESTED → CLEANUP_WAIT → CANCELLED
未接受：CANCEL_SUBMITTING → CANCEL_RETRY / CANCEL_BLOCKED
```

提交意图先持久化。提交响应未知时，下一次先按稳定 provider_id 查询，并核对规范摘要标签。未观察到任务且没有取消意图时，可用同一 ID 和幂等请求身份重试创建；提交结果在查询到提供方任务状态前保持未知。取消期间不再发起创建，继续查询迟到的提交结果。

取消调用先记录 `CANCEL_SUBMITTING` 意图和尝试次数。提供方接受后写入 `CANCEL_REQUESTED`，随后只查询状态与清理。超时、连接失败或可恢复 HTTP 错误进入 `CANCEL_RETRY`，每次重试前先查询远端；默认最多尝试三次。拒绝或重试耗尽后进入 `CANCEL_BLOCKED`，仍监测远端，不写入取消成功。处理失败原因后，可调用 `queue.retry_cancellation(request_id, now=...)` 重新允许有限次尝试。提供方必须支持针对稳定任务身份的幂等取消。

只有提交前取消，或提供方明确报告 `CANCELLED` 且资源检查为空，才写入本地 `CANCELLED`。已接受取消后暂时查无任务时继续查询。远端先完成时保留实际成功或失败结果。远端成功还必须通过资源清理和声明产物检查。产物默认最多检查三次，检查异常也计入次数；耗尽后标记失败。`execution_timeout` 从首次提交意图起计时，到期发起取消，最大七天。

`Controller` 显式接收 provider、resources_present 和 artifacts_valid。`resources_present` 查询任务资源是否仍存在，`artifacts_valid` 检查声明产物是否满足内容合同，两个回调均返回布尔值。租约时长需覆盖一次 provider 调用和产物检查；过期工作者的结果不会提交。

## Google Batch

`configs/batch-job.json` 是结构模板。先复制到机器私有配置，并把 `image` 替换为在运行时从镜像提供方取得的不可变引用，再生成请求：

```bash
intelligence-dynamics batch-spec local-config/batch-job.json batch-job.json
```

该命令只生成请求体。配置中的示例项目、桶和镜像必须换成实际资源。镜像必须使用 SHA256 digest，任务重试次数固定为 0，时限不超过七天。

`job_spec` 支持多个任务、并行度和逐任务环境变量。`estimate_timeout` 根据观测时长、工作量比例、余量和上限计算时限。

`BatchClient` 提供 get、create、cancel。构造实例不会取凭据或连接网络。每次请求从调用方 token_provider 取得短期令牌，使用固定 HTTPS Batch API；create 同时使用稳定 jobId 与 UUID requestId。HTTP 重定向会被拒绝。

`GoogleInspector` 按准确任务标签分页查询计算实例，并检查 JSON 对象的必要字段和规范化内容摘要。分页循环、查询失败及未解决的警告抛出错误并中断本次检查。

实际集成需提供执行身份、镜像仓库访问、输出桶、资源清理查询和产物验证。`resources_present` 查询目标任务创建的计算资源；`artifacts_valid` 检查预期对象的内容合同。

接口依据：[Batch v1 Job](https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs)、[Create job](https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs/create)、[Cloud Storage 桶名](https://docs.cloud.google.com/storage/docs/buckets)。

## 控制服务

`runtime.Application(controller, bearer_token=...)` 是 WSGI 应用。可挂载到组织使用的 WSGI 容器，不在导入时启动服务。

|接口|方法|输入|
|---|---|---|
|`/health`|GET|无|
|`/jobs`|POST|request_id、specification|
|`/describe`|POST|request_id|
|`/cancel`|POST|request_id|
|`/tick`|POST|无|

全部接口要求 bearer token。请求体上限 64 KiB。应用返回任务身份与状态，不回显令牌、内部异常或完整规范。部署通过 TLS 提供服务，并以服务管理器控制进程寿命、队列持久卷和资源配额。
