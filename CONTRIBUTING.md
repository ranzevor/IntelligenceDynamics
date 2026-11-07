# 开发与维护 / Contributing

## 贡献许可与署名

项目自有材料的贡献按 [Apache-2.0](LICENSE) 接收。引入第三方代码或材料时，注明来源、适用许可和改动，并保留必要的版权与许可通知；具体范围见 [许可说明](LICENSE-NOTICE.md)。

每个贡献提交须遵循 [Developer Certificate of Origin 1.1](https://developercertificate.org/)。使用 `git commit -s` 添加 `Signed-off-by` 行，确认你有权按适用许可提交该贡献。Git 使用你配置的署名和邮箱，这些信息将随提交进入公开历史。

## 问题与协作

使用 [Issue 表单](https://github.com/ranzevor/IntelligenceDynamics/issues/new/choose) 报告缺陷或建议功能。缺陷报告包含版本、环境、最小合成复现和预期/实际行为；安全漏洞使用 [SECURITY.md](SECURITY.md) 中的私密渠道。

外部贡献从 fork 的主题分支提交 PR 到 `main`。在 PR 中说明问题、变化、兼容影响与实际验证结果；讨论围绕可复现行为和具体证据。维护入口由 [Ranzevor](https://github.com/ranzevor) 管理。

## 开发流程

在项目根目录的虚拟环境中安装开发依赖与项目：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation --no-deps -e .
```

1. 为行为变更写一个能复现预期差异的合成测试。
2. 修改实现和对应中英文契约。
3. 执行 `python -B -m intelligence_dynamics check-source --workspace .`；使用 `python -B -m unittest discover -s tests -v` 运行测试，并分别加入 `-O`、`-OO` 执行另外两种模式。
4. 执行 `ruff check --no-cache .` 与 `ruff format --check --no-cache .`。
5. 按[构建指南](docs/software.md#构建)生成并严格检查源码副本，在独立目录构建 wheel，再在新的虚拟环境安装并运行相关命令。

开发环境支持 Python 3.10 及以上。Python 3.10 使用锁定的 tomli 解析项目配置，Python 3.11 及以上使用标准库 tomllib。符号链接案例仅在已确认系统不支持或缺少创建权限时单独跳过，其他检查仍执行；Linux 测试需在原生目录运行这些案例。

Python 3.12 CPU 环境的原生 JAX 状态测试：

```bash
python -m pip install -r requirements-native.txt
python -B -m unittest tests.test_native_state_codec_types -v
```

源码边界测试使用系统 Git 创建本地 `git init` 工作树和 linked worktree。

数据合同变更必须同时更新解析器、序列化器、负例和文档。新增指标要定义单位、方向、缺失值与分母为零时的处理。随机算法要注明种子、训练/留出划分和对照组。

依赖更新使用官方 PyPI 索引上的精确版本，同步修改构建声明与对应依赖清单，再测试核心无 NumPy 安装与完整开发安装。不要提交虚拟环境、缓存、构建物、数据集或凭据。

## CI 与依赖更新

CI 在 PR、`main` 推送及手动触发时运行，覆盖 Windows/Linux 的 Python 3.10、3.12；3.12 安装原生测试依赖。每种组合运行普通、`-O`、`-OO` 回归、源码边界与格式检查，另在隔离环境验证 wheel 的安装、核心命令、卸载和重装。配置检查入口是 `python -B tools/check_maintenance.py`。

Dependabot 每周一北京时间 09:00 检查 pip 依赖与 GitHub Actions，最多分别保留 5 个和 3 个更新 PR；次版本和补丁依赖更新按组提出。更新不自动合并。审查上游来源、兼容性和许可，保持构建声明与 lock 一致，再运行受影响检查与安装验证。GitHub Actions 使用允许的官方仓库与完整提交摘要；测试接受更新后的合法摘要，拒绝浮动引用及其他来源。

完整源码包包含 `.github` 配置与开发工具，wheel 仅提供可安装模块和必要许可材料。新文件需同步 `validation.PROJECT_FILES`。发行构建、标签及恢复步骤见[软件指南](docs/software.md#发行构建)。

## Contribution terms and sign-off

Contributions to project-owned materials are accepted under [Apache-2.0](LICENSE). Identify the source, applicable license, and changes when introducing third-party code or materials, and preserve required copyright and license notices. See the [license notice](LICENSE-NOTICE.md) for scope.

Each contribution commit must comply with the [Developer Certificate of Origin 1.1](https://developercertificate.org/). Use `git commit -s` to add a `Signed-off-by` line certifying your right to submit the contribution under its applicable license. Git uses your configured name and email, which become part of the public commit history.

## Development workflow

Install development dependencies and the project in a virtual environment from the project root:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation --no-deps -e .
```

1. Add a synthetic test that distinguishes the intended behavior.
2. Update the implementation and both language versions of its contract.
3. Run `python -B -m intelligence_dynamics check-source --workspace .` and `python -B -m unittest discover -s tests -v`; repeat the test command with `-O` and `-OO` for the other two modes.
4. Run `ruff check --no-cache .` and `ruff format --check --no-cache .`.
5. Follow the [build guide](docs/en/software.md#build) to prepare and strictly check a source copy, build its wheel from a separate copy, then install it in a fresh virtual environment and run affected commands.

Development supports Python 3.10 and later. Python 3.10 uses pinned tomli to parse project configuration; Python 3.11 and later use standard-library tomllib. Only confirmed lack of symbolic-link support or creation privileges skips an individual symbolic-link case; other checks still run. Linux testing runs those cases in a native filesystem directory.

Native JAX state tests in a Python 3.12 CPU environment:

```bash
python -m pip install -r requirements-native.txt
python -B -m unittest tests.test_native_state_codec_types -v
```

The source-boundary tests use the system Git executable to create a local `git init` work tree and linked worktree.

Data-contract changes include parsers, serializers, negative tests, and documentation. Define each metric's unit, direction, missing-value behavior, and zero-denominator handling. Document seeds, training/heldout splits, and controls for randomized algorithms.

Use exact dependency versions from the official PyPI index. Synchronize the build declarations and dependency files, then test both the NumPy-free core and the full development installation. Exclude virtual environments, caches, build artifacts, datasets, and credentials from commits.

## Issues and collaboration

Use the [issue forms](https://github.com/ranzevor/IntelligenceDynamics/issues/new/choose) for bugs and feature requests. Include version, environment, a minimal synthetic reproduction, and expected versus actual behavior. Use the private channels in [SECURITY.md](SECURITY.md) for vulnerabilities.

Send external contributions from a topic branch in a fork to `main`. Explain the problem, change, compatibility impact, and actual validation in the PR. Discuss reproducible behavior and concrete evidence. [Ranzevor](https://github.com/ranzevor) maintains the repository.

## CI and dependency updates

CI runs for PRs, pushes to `main`, and manual dispatch. It covers Python 3.10 and 3.12 on Windows/Linux, installing native test dependencies on 3.12. Each combination runs normal, `-O`, and `-OO` regressions, source checks, and formatting. A separate clean consumer verifies wheel installation, core commands, removal, and reinstallation. Run `python -B tools/check_maintenance.py` to check maintenance configuration.

Dependabot checks pip dependencies and GitHub Actions each Monday at 09:00 Asia/Shanghai, with at most 5 and 3 open update PRs respectively. Minor and patch dependency updates are grouped. Updates are not merged automatically. Review upstream source, compatibility, and license changes; synchronize build declarations with locks and run affected checks and installation verification. Actions use allowed official repositories and full commit SHAs. Tests accept valid updated SHAs and reject floating references and other sources.

The complete source archive includes `.github` configuration and development tools; the wheel includes installable modules and required license materials. Register new files in `validation.PROJECT_FILES`. See the [software guide](docs/en/software.md#release-builds) for release builds, tags, and recovery.
