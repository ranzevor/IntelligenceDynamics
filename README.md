<p align="center"><img src="assets/brand.png" width="80" height="80" alt="IntelligenceDynamics 三节点标识"></p>

# IntelligenceDynamics

**面向持续适应系统的测量、学习规则实验与因果比较工具。**

[English](README.en.md) · [报告问题](https://github.com/ranzevor/IntelligenceDynamics/issues) · [安装与使用](docs/software.md) · [研究问题](docs/research.md) · [数据契约](docs/data.md) · [API](docs/api-research.md)

[![Python 3.10 及以上](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml) [![JSONL 协议 0.1](https://img.shields.io/badge/JSONL-0.1-blue?style=flat)](docs/data.md) [![Apache-2.0 许可](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE) [![CI main](https://img.shields.io/github/actions/workflow/status/ranzevor/IntelligenceDynamics/ci.yml?branch=main&style=flat&label=CI)](https://github.com/ranzevor/IntelligenceDynamics/actions/workflows/ci.yml)

![三条适应系统轨迹汇入动力学轨道的概念插画](assets/cover.png)

版本 **0.1.0**，内容见 [更新记录](CHANGELOG.md)。

## 用途

IntelligenceDynamics 研究一个问题：在部分可观察、环境变化和资源约束下，工程学习系统、进化系统与生物控制系统是否会形成可比较的适应机制？

工具将能力、计算、记忆和持续可塑性分别测量，并用配对干预比较机制作用。合成线索任务提供小规模、确定性的算法试验入口。

- **测量与交换**：类型化 JSONL 记录、缺失值、运行身份、严格解析与原子输出。
- **学习规则实验**：固定、持续更新、条件门控、累积、持续性、缓冲和随机门控策略。
- **实验比较**：训练与留出任务隔离、反馈扰动、状态干预、共同时间支持和显式判据。
- **数据接入**：DAL 演化 CSV、Forager 多采样率 NPZ、种群观测和环境接口检查。
- **原生研究接口**：DAL 个体诊断、Foragax 环境、DRQN 交互与更新调度、checkpoint 续跑和选择性干预。
- **运行管理**：本地命令与产物检查、事务任务队列、Google Batch 请求和可注入的控制服务。

## 快速开始

需要 Python 3.10 或更高版本。获取源码并进入项目目录，创建虚拟环境：

```bash
git clone https://github.com/ranzevor/IntelligenceDynamics.git
cd IntelligenceDynamics
python -m venv .venv
```

Linux 与 macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

安装核心工具并运行合成测量示例：

```bash
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation --no-deps .
intelligence-dynamics demo demo.jsonl
intelligence-dynamics validate demo.jsonl
intelligence-dynamics summarize demo.jsonl
```

`demo.jsonl` 包含 1 条运行身份记录和 48 条测量记录。输出路径必须尚不存在。更换文件名可再次运行。

运行带有训练、留出集和对照组的合成实验：

```bash
intelligence-dynamics study configs/cue-study.json study.json
```

本地构建的 wheel 安装、校验或卸载步骤见 [wheel 安装与恢复](docs/software.md#wheel-安装与恢复)。功能建议、缺陷和贡献入口见[贡献指南](CONTRIBUTING.md)。

## 文档

|主题|入口|
|---|---|
|安装、命令与示例|[软件指南](docs/software.md)|
|记录类型、缺失值与指标|[数据契约](docs/data.md)|
|科学问题、假设与实验设计|[研究设计](docs/research.md)|
|任务、策略、搜索与干预 API|[研究 API](docs/api-research.md)|
|原生模型、状态保存与续跑|[原生模型接口](docs/native.md)|
|本地运行、队列与云接口|[运行管理](docs/runtime.md)|
|维护与依赖更新|[贡献指南](CONTRIBUTING.md)|
|安全与报告问题|[安全说明](SECURITY.md)|
|开发接口、弃用与退役|[开发与维护契约](docs/standards.md)|
|许可与材料来源|[许可说明](LICENSE-NOTICE.md)|

## 引用

研究中使用本软件时，可使用 [CITATION.cff](CITATION.cff) 中的作者和版本信息引用。

## 许可

Copyright © 2026 Ranzevor。项目自有材料采用 [Apache-2.0](LICENSE)，适用范围与商标说明见 [许可说明](LICENSE-NOTICE.md)。第三方派生代码保留 [MIT 通知](THIRD_PARTY_NOTICES.md)，依赖和学术来源见 [材料说明](docs/materials.md)。
