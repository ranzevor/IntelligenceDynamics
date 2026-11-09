# 材料与依赖 / Materials and dependencies

## 项目自有材料 / Project-owned materials

项目自有代码、文档、配置、示例与图形采用 [Apache-2.0](../LICENSE)，署名 Ranzevor；许可范围及商标使用见 [许可说明](../LICENSE-NOTICE.md)。

Project-owned code, documentation, configurations, examples, and artwork are licensed under [Apache-2.0](../LICENSE), credited to Ranzevor. See the [license notice](../LICENSE-NOTICE.md) for scope and trademark terms.

## 软件依赖 / Software dependencies

|发行物 / Distribution|版本 / Version|用途 / Use|许可 / License|
|---|---|---|---|
|Python|3.10+|核心运行 / Core runtime|PSF License|
|NumPy|2.2.6|可选 NPZ 接口 / Optional NPZ interfaces|BSD-3-Clause；发行物内含各组件通知 / Component notices in distribution|
|setuptools|84.2.1|构建 / Build|MIT|
|wheel|0.48.0|构建工具 / Build tooling|MIT|
|packaging|26.3|wheel 的构建传递依赖 / Transitive build dependency of wheel|Apache-2.0 OR BSD-2-Clause|
|Ruff|0.17.4|格式与静态检查 / Formatting and lint|MIT|
|tomli|2.5.0|仅 Python 3.10 开发测试的 TOML 解析 / TOML parsing for development tests on Python 3.10 only|MIT|
|PyYAML|6.0.3|开发测试的 CI 与更新配置解析 / CI and update-configuration parsing in development tests|MIT|
|absl-py|2.5.1|原生状态测试支持 / Native-state test support|Apache-2.0|
|JAX / jaxlib|0.9.0|原生数组与计算后端 / Native array and compute backend|Apache-2.0|
|ml_dtypes|0.6.0|扩展数值类型 / Extended numeric dtypes|Apache-2.0|
|opt_einsum|3.4.0|张量表达式优化 / Tensor-expression optimization|MIT|
|Optax|0.2.8|优化器状态接口 / Optimizer-state interfaces|Apache-2.0|
|SciPy|1.18.1|JAX 数值依赖 / JAX numerical dependency|BSD-3-Clause|
|[actions/upload-artifact](https://github.com/actions/upload-artifact/releases/tag/v7.0.2)|v7.0.2|CI 验证资产 / CI verification artifacts|MIT|
|[actions/setup-python](https://github.com/actions/setup-python/releases/tag/v7.1.0)|v7.1.0|CI Python 环境 / CI Python environment|MIT|
|[actions/checkout](https://github.com/actions/checkout/releases/tag/v7.0.1)|v7.0.1|CI 源码检出 / CI source checkout|MIT|

Python 依赖从官方 PyPI 索引安装，依赖清单固定精确版本。CI 动作从表列官方 GitHub 仓库取得，完整提交摘要固定在[工作流配置](../.github/workflows/ci.yml)。源码包不内置上述依赖发行物；各发行物的 LICENSE 和 NOTICE 随其安装保留。

Python dependencies are installed from the official PyPI index with exact versions pinned in the dependency files. CI actions come from the official GitHub repositories listed above; full commit SHAs are pinned in the [workflow configuration](../.github/workflows/ci.yml). Dependency distributions are not bundled in this source package. Their installed LICENSE and NOTICE files retain their own terms.

## 原生集成 / Native integration

DAL 与 DRQN 适配器的派生实现保留 [MIT 版权与许可全文](../THIRD_PARTY_NOTICES.md)。对象接口与依赖准备见 [原生模型接口](native.md)。Foragax 环境、外部 agent、训练数据与 checkpoint 由调用方提供，不内置于本项目源码。

Derived portions of the DAL and DRQN adapters retain their [complete MIT notices](../THIRD_PARTY_NOTICES.md). See [native model interfaces](en/native.md) for object contracts and dependency preparation. Foragax environments, external agents, training data, and checkpoints are supplied by the caller and are not bundled in this source tree.

## 学术来源 / Academic sources

- [Forager](https://arxiv.org/abs/2605.01131)：部分可观察持续强化学习、状态构造与可塑性测量。
- [Domain-adapted learning](https://arxiv.org/abs/2404.12631)：进化选择、调制学习及学习组织方式。
- [Meta-learning plasticity rules](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910)：塑性规则搜索、稳定性和复杂度。
- [Distributed brain-and-cord control](https://www.nature.com/articles/s41586-026-10735-w)：脑与神经索的分布式控制结构。

这些文献提供研究问题和方法背景。研究问题、假设与实验设计见 [研究设计](research.md)。外部论文、图表、模型和数据的使用遵循 [许可说明](../LICENSE-NOTICE.md)。

These works provide research questions and methodological context. See [research design](en/research.md) for questions, hypotheses, and experiments. Use external papers, figures, models, and data under the terms described in the [license notice](../LICENSE-NOTICE.md).

## 图形 / Artwork

`assets/brand.png` 是项目标识，`assets/cover.png` 是项目概念图。封面表现三类适应系统的概念联系。README 徽章由 Shields.io 以 flat SVG 提供，分别链接 Python 要求、数据协议、许可与 CI 工作流；Python 徽章使用标准 Python 图标。

`assets/brand.png` is the project mark and `assets/cover.png` is project conceptual artwork. The cover illustrates conceptual relationships among adaptive systems. Shields.io provides the README badges as flat SVGs linking to Python requirements, the data protocol, licensing, and the CI workflow. The Python badge uses the standard Python icon.
