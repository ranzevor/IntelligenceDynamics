# Software guide

[中文](../software.md) · [Home](../../README.en.md)

## Installation

The core package uses the Python standard library and requires Python 3.10 or later. `requirements-build.txt` pins exact build-tool versions from the official PyPI index. When a lock includes `--require-hashes`, pip also verifies its complete SHA-256 values; obtain expected hashes from a trusted release source. NumPy 2.2.6 is optional for NPZ and native array-state interfaces; its distributions support Python 3.10–3.13.

```bash
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation --no-deps .
```

Enable NPZ interfaces:

```bash
python -m pip install -r requirements-arrays.txt
```

`python -m intelligence_dynamics` is equivalent to `intelligence-dynamics`. Run examples from the project root. Outputs refuse overwriting by default, and parent directories must already exist.

## Commands

|Command|Input|Output|
|---|---|---|
|`check-source ROOT [--workspace]`|Project directory and check mode|Source-boundary check result|
|`demo OUTPUT`|Optional `--seed`|Synthetic task measurements in JSONL|
|`validate INPUT`|Measurement JSONL|Record counts, run identity, and ordering checks|
|`summarize INPUT`|Measurement JSONL|Complete-observation capability mean and missing counts|
|`study CONFIG OUTPUT`|Synthetic study JSON configuration|Search, heldout evaluation, and controls in JSON|
|`convert FORMAT INPUT CONFIG OUTPUT`|Native data and metadata configuration|Measurement JSONL|
|`run CONFIG OUTPUT_DIR`|Command, deadline, and artifact declarations|Logs, exit status, and artifact hashes|
|`batch-spec CONFIG OUTPUT`|Cloud target, image digest, and resources|Google Batch JSON request body|

Successful commands return 0; command or data errors return 2. A failed local program, invalid artifact, or timeout returns 1. Structured results go to standard output and command errors to standard error. Success messages show output filenames without parent directories; errors hide known input paths and the home directory. Files are still written to the requested locations.

## Native data conversion

The DAL example contains synthetic values:

```bash
intelligence-dynamics convert dal-csv examples/dal.csv configs/dal-conversion.json dal.jsonl
intelligence-dynamics validate dal.jsonl
```

The first 11 columns describe generation, fitness, and update statistics; subsequent columns contain trial scores. Generation multiplied by task instances, trials, and steps per trial gives cumulative interactions. `None` is accepted only in the two ablation-fitness columns.

Create and convert a multirate NPZ:

```bash
python examples/make_array_fixture.py forager.npz
intelligence-dynamics convert forager-npz forager.npz configs/forager-conversion.json forager.jsonl
intelligence-dynamics validate forager.jsonl
```

`rewards` contains step samples; `total_loss` contains optimizer-update samples. Step samples must divide evenly into updates. Each update interval produces a mean reward while retaining gradients, static scalars, positions, and context information. `expected_rollout_steps` checks the sampling interval.

Use `array-npz` for equally sampled one-dimensional arrays. Its configuration accepts `capability_field` and `interval`. Conversion copies input in chunks to the system temporary directory and uses that copy for both SHA256 and measurement parsing, then removes it. The temporary directory needs space for one input copy. Converted metadata includes that copy’s SHA256 and the configuration digest. Object arrays, non-finite values, duplicate names, and unsupported shapes are rejected.

## Configuration

- `configs/cue-study.json`: training and heldout seeds, task dimensions, and search settings.
- `configs/dal-conversion.json`: run identity and generation-to-interaction scale.
- `configs/forager-conversion.json`: run identity and expected rollout interval.
- `configs/local-command.json`: a local command producing one JSON artifact.
- `configs/batch-job.json`: a Batch request structure template. Its `image` field requires an immutable image reference obtained at runtime and supplied in a private copy. The template embeds no concrete digest and cannot be submitted directly.

Configuration is strict UTF-8 JSON. Duplicate keys, NaN, and Infinity are rejected. Configuration is not executed as Python code. Keep machine-private configuration in `local-config/` and run outputs in `runs/`. These directories and the documented default generated files are excluded from version control. Add ignore rules for custom output paths.

## Build

Install development dependencies and the editable project as shown in the [README](../../README.en.md#development-and-tests). Run the following commands from the project root in Bash or PowerShell. The source-copy and build directories must not exist and must be outside the project.

```bash
python -B -m intelligence_dynamics check-source --workspace .
python -B tools/prepare_source.py ../IntelligenceDynamics-source
python -B -m intelligence_dynamics check-source ../IntelligenceDynamics-source
python -B -c "import shutil; shutil.copytree('../IntelligenceDynamics-source', '../IntelligenceDynamics-build')"
python -m pip wheel --no-build-isolation --no-deps ../IntelligenceDynamics-build --wheel-dir ../IntelligenceDynamics-wheels
```

`prepare_source.py` checks the workspace, copies only regular source files registered in `validation.PROJECT_FILES`, compares their contents, and strictly checks the copy. It rejects a target inside the project, an existing target, a missing parent, or traversal through a symbolic link or Windows reparse point (including a directory junction). Copy or verification failures return nonzero; inspect any failed destination and retry with a new path. The wheel is built in a separate copy, keeping source and build artifacts separate.

Default `check-source ROOT` accepts only registered source and a valid root `.git` management directory or file. When a root `.git` entry is present, system Git validates its worktree identity; source packages without `.git` do not require Git. Missing files, unexpected files or directories, nested `.git` entries, symbolic links, and Windows reparse points are rejected. `--workspace` uses a fixed allowlist for root `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `build/`, `dist/`, `runs/`, `local-config/`, `local-run/`, the project's root or `src/` egg-info, `__pycache__/` in registered source directories, root `.env`, `.env.*`, SQLite databases and sidecars, and the default output files in this guide and the README. New ignore rules do not expand this list; keep custom artifacts in an allowed output directory. Both modes check registered-source completeness and path types.

The wheel contains the `intelligence_dynamics` modules, type marker, full Apache-2.0 license, license notice, third-party notices, and command entry point. The source archive also provides tests, configurations, examples, and bilingual documentation. Verify installation in a fresh virtual environment, install the wheel, then run `demo`, `validate`, `summarize`, DAL conversion, and `study` from outside the project.

## Release installation and recovery

Choose a specific version on the [release page](https://github.com/ranzevor/IntelligenceDynamics/releases) and download the wheel, complete source archive, `release-manifest.json`, and `SHA256SUMS`. The manifest records the source commit, file lengths, and hashes. Compare downloaded files with that version's SHA-256 list. The complete source archive supplies `configs/` and `examples/` for the documented examples.

In a fresh virtual environment, install from the directory containing the wheel:

```bash
python -m pip install --no-index --no-deps intelligence_dynamics-0.1.0-py3-none-any.whl
intelligence-dynamics demo demo.jsonl
intelligence-dynamics validate demo.jsonl
intelligence-dynamics summarize demo.jsonl
```

The core wheel has no additional runtime dependencies. Install optional NPZ and native dependencies as described in this guide and the native-interface guide. Call the Python API with `import intelligence_dynamics` inside the environment; module and console commands are equivalent.

Remove it with `python -m pip uninstall intelligence-dynamics`. Installation and removal do not migrate or delete existing JSONL, queue databases, run logs, or checkpoints. Keep those files and the wheel you used. Recover by reinstalling the saved wheel in an isolated environment and validating existing inputs first. See the [data contract](data.md) and [native interfaces](native.md) for data and checkpoint compatibility.

## Release builds

Build from the exact Git commit intended for release. Install the locked build tools as shown in the README. The output directory must not exist and must be outside the project:

```bash
python -B tools/build_release.py ../IntelligenceDynamics-release
python -B tools/smoke_wheel.py ../IntelligenceDynamics-release/artifacts ../IntelligenceDynamics-consumer
```

The builder compares source bytes with the commit, exports registered source from that commit’s Git objects, and builds the wheel in a separate copy. `artifacts/` contains the wheel, complete source ZIP, source and wheel file manifests, build-tool versions, asset hashes, and `SHA256SUMS`. Build logs remain in the output directory.

Consumer verification first recomputes the Git tree from the source ZIP’s actual file bytes and checks `release-manifest.json` field `source.tree`, then installs the actual wheel in a temporary environment and runs the Python API, demo, validation, summary, DAL conversion, and synthetic study outside the project. It checks existing outputs, unknown commands, malformed input, removal, and reinstallation. Raw output and results remain in the separate consumer directory.

Use an annotated `v<version>` tag pointing to the checked commit. Confirm that the version is unused before upload, then read back the tag commit, release body, and asset identities and compare bytes from official downloads. Preserve published tags and assets; deliver corrections under a new version. Release notes describe runtime requirements, actual verification scope, installation and recovery, licensing, and compatibility changes.
