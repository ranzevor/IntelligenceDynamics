"""Validation for scalar measurements and JSON values."""

from __future__ import annotations

import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

PROJECT_FILES = frozenset(
    """
.gitattributes
.github/ISSUE_TEMPLATE/bug_report.yml
.github/ISSUE_TEMPLATE/config.yml
.github/ISSUE_TEMPLATE/feature_request.yml
.github/dependabot.yml
.github/pull_request_template.md
.github/workflows/ci.yml
.gitignore
CHANGELOG.md
CITATION.cff
CONTRIBUTING.md
LICENSE
LICENSE-NOTICE.md
README.en.md
README.md
SECURITY.md
THIRD_PARTY_NOTICES.md
assets/README.md
assets/brand.png
assets/cover.png
configs/batch-job.json
configs/cue-study.json
configs/dal-conversion.json
configs/forager-conversion.json
configs/local-command.json
configs/measurement-record.schema.json
docs/api-research.md
docs/data.md
docs/en/api-research.md
docs/en/data.md
docs/en/native.md
docs/en/research.md
docs/en/runtime.md
docs/en/software.md
docs/materials.md
docs/native.md
docs/research.md
docs/runtime.md
docs/software.md
docs/standards.md
examples/dal.csv
examples/make_array_fixture.py
pyproject.toml
requirements-arrays.txt
requirements-build.txt
requirements-dev.txt
requirements-lint.txt
requirements-native.txt
src/intelligence_dynamics/__init__.py
src/intelligence_dynamics/__main__.py
src/intelligence_dynamics/adapters.py
src/intelligence_dynamics/cli.py
src/intelligence_dynamics/experiments.py
src/intelligence_dynamics/metrics.py
src/intelligence_dynamics/native/__init__.py
src/intelligence_dynamics/native/_state.py
src/intelligence_dynamics/native/_values.py
src/intelligence_dynamics/native/checkpoint.py
src/intelligence_dynamics/native/dal.py
src/intelligence_dynamics/native/drqn.py
src/intelligence_dynamics/native/foragax.py
src/intelligence_dynamics/native/lifecycle.py
src/intelligence_dynamics/py.typed
src/intelligence_dynamics/records.py
src/intelligence_dynamics/research/__init__.py
src/intelligence_dynamics/research/_validation.py
src/intelligence_dynamics/research/causal.py
src/intelligence_dynamics/research/criteria.py
src/intelligence_dynamics/research/evaluation.py
src/intelligence_dynamics/research/evolution.py
src/intelligence_dynamics/research/policy.py
src/intelligence_dynamics/research/support.py
src/intelligence_dynamics/research/task.py
src/intelligence_dynamics/runtime/__init__.py
src/intelligence_dynamics/runtime/batch.py
src/intelligence_dynamics/runtime/inspection.py
src/intelligence_dynamics/runtime/local.py
src/intelligence_dynamics/runtime/queue.py
src/intelligence_dynamics/runtime/resources.py
src/intelligence_dynamics/runtime/service.py
src/intelligence_dynamics/schema.py
src/intelligence_dynamics/state.py
src/intelligence_dynamics/storage.py
src/intelligence_dynamics/validation.py
tests/__init__.py
tests/support.py
tests/test_cli_service.py
tests/test_experiments_arrays.py
tests/test_maintenance.py
tests/test_metrics_adapters.py
tests/test_native_adapters.py
tests/test_native_arrays.py
tests/test_native_configuration.py
tests/test_native_lifecycle.py
tests/test_native_state_codec_types.py
tests/test_queue_recovery.py
tests/test_records_storage.py
tests/test_release_tools.py
tests/test_research_causal_support.py
tests/test_research_criteria.py
tests/test_research_policy.py
tests/test_research_search.py
tests/test_research_tasks.py
tests/test_runtime.py
tests/test_runtime_contracts.py
tests/test_runtime_privacy.py
tests/test_source_preparation.py
tests/test_state.py
tests/test_windows_memory_contract.py
tools/build_release.py
tools/check_maintenance.py
tools/prepare_source.py
tools/smoke_wheel.py
""".split()
)


def _project_directories():
    result = set()
    for name in PROJECT_FILES:
        parts = name.split("/")
        result.update("/".join(parts[:index]) for index in range(1, len(parts)))
    return frozenset(result)


PROJECT_DIRECTORIES = _project_directories()

WORKSPACE_DIRECTORIES = frozenset(
    {
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "runs",
        "local-config",
        "local-run",
        "intelligence_dynamics.egg-info",
        "src/intelligence_dynamics.egg-info",
    }
)
WORKSPACE_FILES = frozenset(
    {"demo.jsonl", "study.json", "dal.jsonl", "forager.npz", "forager.jsonl", "batch-job.json"}
)


def _workspace_artifact(relative: str, *, directory: bool) -> bool:
    if directory:
        parent, _, name = relative.rpartition("/")
        return relative in WORKSPACE_DIRECTORIES or (
            name == "__pycache__" and parent in PROJECT_DIRECTORIES
        )
    return relative in WORKSPACE_FILES or (
        "/" not in relative
        and (
            relative == ".env"
            or relative.startswith(".env.")
            or relative.endswith((".sqlite", ".sqlite-journal", ".sqlite-wal", ".sqlite-shm"))
        )
    )


def _identity(path: Path):
    value = path.stat(follow_symlinks=False)
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        getattr(value, "st_file_attributes", 0),
    )


def _linked_identity(identity):
    return stat.S_ISLNK(identity[2]) or bool(
        identity[5] & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def is_linked_path(path: str | Path) -> bool:
    """Identify symbolic links and Windows reparse points without following them."""
    try:
        return _linked_identity(_identity(Path(path)))
    except FileNotFoundError:
        return False


def root_git_token(root: str | Path):
    """Check an optional root management entry using Git's worktree semantics."""
    root = Path(root)
    entry = root / ".git"
    try:
        identity = _identity(entry)
    except FileNotFoundError:
        return None
    if _linked_identity(identity):
        raise ValueError("root .git entry cannot be a symbolic link or reparse point")
    if not (stat.S_ISDIR(identity[2]) or stat.S_ISREG(identity[2])):
        raise ValueError("root .git entry must be a clone directory or worktree file")
    # A caller's Git environment must not redirect validation to another repository.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    try:
        result = subprocess.run(
            ["git", "-C", str(root.resolve(strict=True)), "rev-parse", "--show-toplevel"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError("Git is required to inspect the root .git entry") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Git inspection of the root .git entry timed out") from exc
    except OSError as exc:
        raise ValueError("Git could not inspect the root .git entry") from exc
    if result.returncode:
        raise ValueError("root .git does not identify this project as a Git work tree")
    top = result.stdout.removesuffix(b"\n")
    if Path(os.fsdecode(top)).resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("root .git does not identify this project as a Git work tree")
    if _identity(entry) != identity:
        raise ValueError("root .git entry changed during validation")
    return identity


def validate_project_files(root: str | Path, *, workspace: bool = False):
    """Validate registered source files, optionally excluding known workspace artifacts."""
    if type(workspace) is not bool:
        raise ValueError("workspace must be a boolean")
    root = Path(root)
    if any(is_linked_path(part) for part in root.parents):
        raise ValueError("project root cannot traverse symbolic links or reparse points")
    try:
        root_before = _identity(root)
    except FileNotFoundError as exc:
        raise ValueError("project root does not exist") from exc
    if _linked_identity(root_before) or not stat.S_ISDIR(root_before[2]):
        raise ValueError("project root must be a regular directory")
    git_before = root_git_token(root)
    files, directories = set(), set()
    stack = sorted(root.iterdir(), reverse=True)
    while stack:
        path = stack.pop()
        relative = path.relative_to(root).as_posix()
        if relative == ".git":
            continue
        if path.name == ".git":
            raise ValueError(f"nested .git entry is not allowed: {relative}")
        identity = _identity(path)
        if _linked_identity(identity):
            raise ValueError(
                f"project paths cannot be symbolic links or reparse points: {relative}"
            )
        if stat.S_ISDIR(identity[2]):
            if (
                workspace
                and relative not in PROJECT_DIRECTORIES
                and _workspace_artifact(relative, directory=True)
            ):
                continue
            directories.add(relative)
            stack.extend(sorted(path.iterdir(), reverse=True))
        elif stat.S_ISREG(identity[2]):
            if (
                workspace
                and relative not in PROJECT_FILES
                and _workspace_artifact(relative, directory=False)
            ):
                continue
            files.add(relative)
        else:
            raise ValueError(f"project path is not a regular file or directory: {relative}")
    missing_files = sorted(PROJECT_FILES - files)
    unexpected_files = sorted(files - PROJECT_FILES)
    missing_directories = sorted(PROJECT_DIRECTORIES - directories)
    unexpected_directories = sorted(directories - PROJECT_DIRECTORIES)
    errors = []
    for label, values in (
        ("missing files", missing_files),
        ("unexpected files", unexpected_files),
        ("missing directories", missing_directories),
        ("unexpected directories", unexpected_directories),
    ):
        if values:
            errors.append(f"{label}: {', '.join(values)}")
    if errors:
        raise ValueError("; ".join(errors))
    if _identity(root) != root_before or root_git_token(root) != git_before:
        raise ValueError("project root or root .git entry changed during validation")
    return tuple(sorted(files))


def text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    value.encode("utf-8")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} contains a control character")
    return value


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(
    value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the finite numeric range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def json_value(value: Any, name: str = "value", *, depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError(f"{name} exceeds the maximum JSON nesting depth")
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if type(value) in (int, float):
        number(value, name)
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(v, f"{name}[{i}]", depth=depth + 1) for i, v in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError(f"{name} requires string keys")
        return {
            text(k, f"{name} key"): json_value(v, f"{name}.{k}", depth=depth + 1)
            for k, v in value.items()
        }
    raise ValueError(f"{name} contains unsupported type {type(value).__name__}")
