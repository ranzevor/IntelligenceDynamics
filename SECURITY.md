# 安全 / Security

## 输入与执行

JSON 接口限制记录大小并拒绝重复键及非有限数。NPZ 使用 `allow_pickle=False`。产物检查不反序列化 Python pickle。文件输出拒绝已存在目标和符号链接路径；输出目录由调用方独占管理。

`run` 执行配置中的程序。仅运行可信命令。默认 `request.json` 只记录命令摘要与参数数量；`record_command=true` 会保存完整参数。命令参数中的低熵值可通过枚举并重算摘要匹配；密钥须从参数中移除。子进程输出和声明产物原样保存，分享前检查其内容。通过操作系统或容器配置 CPU、内存和磁盘配额。

WSGI 控制端点要求至少 32 字符的 bearer token。部署时使用 TLS、独立服务身份和最小权限。令牌不写入配置样例、日志或队列。BatchClient 使用调用方提供的短期访问令牌，拒绝 HTTP 重定向。


安全漏洞可通过 [GitHub 私密漏洞报告](https://github.com/ranzevor/IntelligenceDynamics/security/advisories/new) 提交，或私密发送至 [ranzevor@gmail.com](mailto:ranzevor@gmail.com)。请提供最小合成复现、受影响版本和预期/实际行为。当前维护版本与范围见 [CHANGELOG.md](CHANGELOG.md)。不要在公开问题中附带访问令牌、私密数据、模型权重或完整运行日志。

## Input and execution

JSON interfaces bound record size and reject duplicate keys and non-finite numbers. NPZ loading uses `allow_pickle=False`. Artifact inspection does not deserialize Python pickle. File output rejects existing targets and symbolic-link paths; callers exclusively manage output directories.

`run` executes the configured program. Use trusted commands only. By default, `request.json` records the command digest and argument count; `record_command=true` saves full arguments. Low-entropy argument values can be matched by enumerating candidates and recomputing their digests. Remove secrets from arguments. Child-process output and declared artifacts are preserved verbatim; inspect their contents before sharing. Configure CPU, memory, and disk quotas through the operating system or container.

WSGI control endpoints require a bearer token of at least 32 characters. Deploy behind TLS with a dedicated service identity and least privilege. Keep tokens out of configuration examples, logs, and the queue. BatchClient receives short-lived access tokens from its caller and refuses HTTP redirects.


Report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/ranzevor/IntelligenceDynamics/security/advisories/new) or privately to [ranzevor@gmail.com](mailto:ranzevor@gmail.com). Include a minimal synthetic reproduction, affected version, and expected versus actual behavior. See [CHANGELOG.md](CHANGELOG.md) for maintained versions and scope. Do not include tokens, private data, model weights, or complete run logs in public reports.
