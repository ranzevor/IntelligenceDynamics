<p align="center"><img src="assets/brand.png" width="80" height="80" alt="IntelligenceDynamics three-node mark"></p>

# IntelligenceDynamics

**Measurements, learning-rule experiments, and causal comparisons for continually adapting systems.**

[中文](README.md) · [Issues](https://github.com/ranzevor/IntelligenceDynamics/issues) · [Installation and use](docs/en/software.md) · [Research questions](docs/en/research.md) · [Data contract](docs/en/data.md) · [API](docs/en/api-research.md)

[![Python 3.10 and later](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](pyproject.toml) [![JSONL protocol 0.1](https://img.shields.io/badge/JSONL-0.1-blue?style=flat)](docs/en/data.md) [![Apache-2.0 license](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE) [![CI main](https://img.shields.io/github/actions/workflow/status/ranzevor/IntelligenceDynamics/ci.yml?branch=main&style=flat&label=CI)](https://github.com/ranzevor/IntelligenceDynamics/actions/workflows/ci.yml)

![Conceptual illustration of three adaptive-system trajectories converging into a dynamical orbit](assets/cover.png)

Version **0.1.0**. See the [version notes](CHANGELOG.md).

## Purpose

IntelligenceDynamics asks whether engineered learners, evolutionary systems, and biological controllers develop comparable adaptive mechanisms under partial observability, environmental change, and resource constraints.

The tools measure capability, computation, memory, and continuing plasticity separately, then use paired interventions to compare mechanisms. Synthetic cue tasks provide a small, deterministic entry point for algorithm experiments.

- **Measurement and exchange**: typed JSONL records, missing values, run identities, strict parsing, and atomic output.
- **Learning-rule experiments**: fixed, continuously updating, conditional, accumulator, persistence, buffered, and random-gated policies.
- **Experimental comparisons**: separate training and heldout tasks, feedback corruption, state interventions, common temporal support, and explicit criteria.
- **Data ingestion**: DAL evolution CSV, multirate Forager NPZ, population observations, and environment interface checks.
- **Native research interfaces**: DAL diagnostics, Foragax environments, DRQN interaction and update scheduling, checkpoint continuation, and selective interventions.
- **Execution management**: local commands and artifact checks, a transactional queue, Google Batch requests, and an injectable control service.

## Quick start

Python 3.10 or later is required. Get the source, enter the project directory, and create a virtual environment:

```bash
git clone https://github.com/ranzevor/IntelligenceDynamics.git
cd IntelligenceDynamics
python -m venv .venv
```

Linux and macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the core tools and run a synthetic measurement example:

```bash
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation --no-deps .
intelligence-dynamics demo demo.jsonl
intelligence-dynamics validate demo.jsonl
intelligence-dynamics summarize demo.jsonl
```

`demo.jsonl` contains one run identity and 48 measurement records. Output paths must not already exist. Use another filename to run the example again.

Run a synthetic experiment with training tasks, heldout tasks, and controls:

```bash
intelligence-dynamics study configs/cue-study.json study.json
```

For installation, verification, and removal of a locally built wheel, see [wheel installation and recovery](docs/en/software.md#wheel-installation-and-recovery). See [Contributing](CONTRIBUTING.md) for feature requests, bug reports, and changes.

## Documentation

|Topic|Entry|
|---|---|
|Installation, commands, and examples|[Software guide](docs/en/software.md)|
|Record types, missing values, and metrics|[Data contract](docs/en/data.md)|
|Scientific questions, hypotheses, and experimental design|[Research design](docs/en/research.md)|
|Task, policy, search, and intervention APIs|[Research API](docs/en/api-research.md)|
|Native models, checkpoint storage, and continuation|[Native model interfaces](docs/en/native.md)|
|Local execution, queues, and cloud interfaces|[Execution management](docs/en/runtime.md)|
|Maintenance and dependency updates|[Contributing](CONTRIBUTING.md)|
|Security and reporting issues|[Security](SECURITY.md)|
|Development interfaces, deprecation, and retirement|[Development and maintenance contracts](docs/standards.md)|
|Licensing and material origins|[License notice](LICENSE-NOTICE.md)|

## Citation

When using this software in research, cite the author and version recorded in [CITATION.cff](CITATION.cff).

## License

Copyright © 2026 Ranzevor. Project-owned materials are licensed under [Apache-2.0](LICENSE); see the [license notice](LICENSE-NOTICE.md) for scope and trademark terms. Third-party-derived code retains its [MIT notices](THIRD_PARTY_NOTICES.md). Dependencies and academic sources are listed in the [material notices](docs/materials.md).
