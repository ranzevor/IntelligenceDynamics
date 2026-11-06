"""Bounded command execution and declared artifact inspection."""

from __future__ import annotations

import hashlib
import os
import platform
import signal
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

from ..storage import canonical_bytes, digest, loads, read_records, write_json
from ..validation import integer, is_linked_path, json_value, number, text
from .resources import process_tree_resident_bytes, resident_bytes


@dataclass(frozen=True)
class Artifact:
    path: str
    format: str = "binary"
    sha256: str | None = None
    max_bytes: int = 268435456
    required_fields: tuple[str, ...] = ()
    expected_fields: dict = field(default_factory=dict)
    required_tables: tuple[str, ...] = ()
    required_arrays: tuple[str, ...] = ()
    array_shapes: dict[str, tuple[int | None, ...]] = field(default_factory=dict)
    changes: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        text(self.path, "artifact path")
        integer(self.max_bytes, "max_bytes", minimum=1)
        if (
            ((self.required_fields or self.expected_fields) and self.format != "json")
            or (self.required_tables and self.format != "sqlite")
            or (
                (self.required_arrays or self.array_shapes or self.changes) and self.format != "npz"
            )
        ):
            raise ValueError("artifact requirements do not match its format")
        for name in ("required_fields", "required_tables", "required_arrays"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            for item in value:
                text(item, name)
            object.__setattr__(self, name, tuple(value))
        if not isinstance(self.array_shapes, dict) or not isinstance(self.changes, dict):
            raise ValueError("array_shapes and changes must be objects")
        if not isinstance(self.expected_fields, dict):
            raise ValueError("expected_fields must be an object")
        object.__setattr__(self, "expected_fields", json_value(self.expected_fields))
        shapes = {}
        for name, shape in self.array_shapes.items():
            text(name, "array name")
            if not isinstance(shape, (tuple, list)):
                raise ValueError("array shape must be a sequence")
            for dimension in shape:
                if dimension is not None:
                    integer(dimension, "array dimension")
            shapes[name] = tuple(shape)
        object.__setattr__(self, "array_shapes", shapes)
        for name, count in self.changes.items():
            text(name, "array name")
            integer(count, "change count")
        object.__setattr__(self, "changes", dict(self.changes))
        parts = self.path.split("/")
        if (
            self.path.startswith("/")
            or "\\" in self.path
            or any(p in {"", ".", ".."} for p in parts)
        ):
            raise ValueError("artifact path must be a canonical relative POSIX path")
        if ":" in self.path or self.format not in {
            "binary",
            "json",
            "jsonl",
            "sqlite",
            "npz",
            "lzma",
        }:
            raise ValueError("invalid artifact path or format")
        if self.sha256 is not None:
            import re

            if not isinstance(self.sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.sha256):
                raise ValueError("artifact sha256 must be 64 lowercase hex digits")


def _artifact_path(root: Path, path: Path) -> None:
    for part in (path, *path.parents):
        if is_linked_path(part):
            raise ValueError("artifact paths cannot traverse symbolic links or reparse points")
        if part == root:
            break
    if not path.is_file():
        raise ValueError("artifact is missing or not a regular file")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("artifact path escapes its root")


def _artifact_bytes(path: Path, maximum: int, target=None) -> tuple[str, int]:
    checksum = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1048576, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ValueError("artifact exceeds declared size bound")
            checksum.update(chunk)
            if target is not None:
                target.write(chunk)
    return checksum.hexdigest(), size


def _sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-journal"):
        sidecar = path.with_name(path.name + suffix)
        if is_linked_path(sidecar) or (
            sidecar.exists() and (not sidecar.is_file() or sidecar.stat().st_size)
        ):
            raise ValueError("SQLite artifact requires checkpointed data without pending journals")


def inspect_artifact(root: str | Path, declaration: Artifact) -> dict:
    root = Path(root)
    if any(is_linked_path(part) for part in (root, *root.parents)):
        raise ValueError("artifact root cannot traverse symbolic links or reparse points")
    root = root.resolve(strict=True)
    path = root.joinpath(*PurePosixPath(declaration.path).parts)
    _artifact_path(root, path)
    if declaration.format == "sqlite":
        _sqlite_sidecars(path)
    before = path.stat()
    if before.st_size > declaration.max_bytes:
        raise ValueError("artifact exceeds declared size bound")
    # Format checks and the reported identity consume the same private snapshot.
    with tempfile.TemporaryDirectory(prefix="id-artifact-") as directory:
        snapshot = Path(directory) / "artifact"
        with snapshot.open("wb") as target:
            actual, size = _artifact_bytes(path, declaration.max_bytes, target)
        _inspect_artifact_content(snapshot, declaration)
        _artifact_path(root, path)
        if declaration.format == "sqlite":
            _sqlite_sidecars(path)
        current, current_size = _artifact_bytes(path, declaration.max_bytes)
        after = path.stat()
        if (
            (before.st_size, before.st_mtime_ns, before.st_ino)
            != (after.st_size, after.st_mtime_ns, after.st_ino)
            or current != actual
            or current_size != size
        ):
            raise ValueError("artifact changed during inspection")
    if declaration.sha256 is not None and declaration.sha256 != actual:
        raise ValueError("artifact digest differs from declaration")
    return {
        "path": declaration.path,
        "format": declaration.format,
        "bytes": size,
        "sha256": actual,
    }


def _inspect_artifact_content(path: Path, declaration: Artifact) -> None:
    if declaration.format == "json":
        value = loads(path.read_bytes())
        if declaration.required_fields and (
            not isinstance(value, dict)
            or any(key not in value for key in declaration.required_fields)
        ):
            raise ValueError("JSON artifact is missing required fields")
        if declaration.expected_fields and (
            not isinstance(value, dict)
            or any(
                key not in value or canonical_bytes(value[key]) != canonical_bytes(expected)
                for key, expected in declaration.expected_fields.items()
            )
        ):
            raise ValueError("JSON artifact fields differ from declared values")
    elif declaration.format == "jsonl":
        if not sum(1 for _ in read_records(path)):
            raise ValueError("JSONL artifact contains no records")
    elif declaration.format == "sqlite":
        with path.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise ValueError("SQLite artifact has no database header")
        database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            if database.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite artifact integrity check failed")
            tables = {
                row[0]
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if not tables or not set(declaration.required_tables) <= tables:
                raise ValueError("SQLite artifact is missing user tables")
        except BaseException as primary:
            try:
                database.close()
            except BaseException as secondary:
                if hasattr(primary, "add_note"):
                    primary.add_note(f"database close failed: {secondary}")
                else:
                    primary.cleanup_errors = (str(secondary),)
            raise
        else:
            database.close()
    elif declaration.format == "npz":
        import zipfile

        import numpy as np

        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > declaration.max_bytes:
                raise ValueError("NPZ expanded size exceeds declared bound")
        with np.load(path, allow_pickle=False) as arrays:
            if not arrays.files or len(set(arrays.files)) != len(arrays.files):
                raise ValueError("NPZ artifact requires unique arrays")
            for name in arrays.files:
                array = arrays[name]
                if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                    raise ValueError("NPZ artifact requires finite numeric arrays")
            expected = (
                set(declaration.required_arrays)
                | declaration.array_shapes.keys()
                | declaration.changes.keys()
            )
            if not expected <= set(arrays.files):
                raise ValueError("NPZ artifact is missing required arrays")
            for name, shape in declaration.array_shapes.items():
                actual_shape = arrays[name].shape
                if len(shape) != len(actual_shape) or any(
                    expected is not None and expected != actual
                    for expected, actual in zip(shape, actual_shape)
                ):
                    raise ValueError(f"NPZ shape differs for {name}")
            for name, expected in declaration.changes.items():
                values = arrays[name]
                if values.ndim != 1 or int(np.count_nonzero(values[1:] != values[:-1])) != expected:
                    raise ValueError(f"NPZ change count differs for {name}")
    elif declaration.format == "lzma":
        import lzma

        with lzma.open(path, "rb") as stream:
            expanded = 0
            while chunk := stream.read(min(1048576, declaration.max_bytes + 1 - expanded)):
                expanded += len(chunk)
                if expanded > declaration.max_bytes:
                    raise ValueError("LZMA expanded size exceeds declared bound")
            if expanded == 0:
                raise ValueError("LZMA artifact contains no data")


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int | None
    elapsed_seconds: float
    artifacts: tuple[dict, ...]
    error: str | None = None
    peak_sampled_rss_bytes: int | None = None
    peak_sampled_tree_rss_bytes: int | None = None
    memory_errors: tuple[str, ...] = ()
    cleanup_errors: tuple[str, ...] = ()


def run_command(
    command: Sequence[str],
    output: str | Path,
    *,
    timeout: float,
    artifacts: Sequence[Artifact] = (),
    environment: dict[str, str] | None = None,
    record_command: bool = False,
) -> ProcessResult:
    """Run one command in a new directory and retain its logs and outcome."""
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty argument sequence")
    for argument in command:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError("command arguments must be strings without NUL")
    text(command[0], "command executable")
    if type(record_command) is not bool:
        raise ValueError("record_command must be a boolean")
    number(timeout, "timeout", minimum=0)
    if timeout == 0:
        raise ValueError("timeout must be positive")
    if any(not isinstance(item, Artifact) for item in artifacts):
        raise ValueError("artifacts must contain Artifact declarations")
    if len({item.path for item in artifacts}) != len(artifacts):
        raise ValueError("artifact paths must be unique")
    reserved = {
        "run.json",
        "request.json",
        "stdout.log",
        "stderr.log",
    }
    if any(item.path in reserved for item in artifacts):
        raise ValueError("artifact path is reserved for runner output")
    env = None
    if environment is not None:
        env = dict(os.environ)
        for key, value in environment.items():
            text(key, "environment key")
            if "=" in key or not isinstance(value, str) or "\x00" in value:
                raise ValueError("invalid environment override")
            env[key] = value
    target = Path(output)
    for part in (target, *target.parents):
        if is_linked_path(part):
            raise ValueError("output cannot traverse symbolic links or reparse points")
    target.mkdir(parents=False, exist_ok=False)
    private_paths = {str(target), str(target.absolute()), str(Path.home())}
    private_paths.update(
        arg for arg in command if Path(arg).is_absolute() or PureWindowsPath(arg).is_absolute()
    )

    def diagnostic(exc):
        message = str(exc)
        for private in sorted(private_paths, key=len, reverse=True):
            if len(private) > 1:
                message = message.replace(private, "<path>").replace(
                    private.replace("\\", "\\\\"), "<path>"
                )
        return f"{type(exc).__name__}: {message}"

    from .. import __version__

    write_json(
        target / "request.json",
        {
            "runtime": {
                "package_version": __version__,
                "python_version": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
            },
            "command": list(command) if record_command else None,
            "command_sha256": digest(list(command)),
            "argument_count": len(command),
            "timeout": timeout,
            "artifacts": [asdict(item) for item in artifacts],
            "environment_override_keys": sorted(environment) if environment is not None else [],
            "environment_overrides_sha256": digest(environment)
            if environment is not None
            else None,
        },
    )
    start = time.monotonic()
    status, error, returncode, checked = "FAILED", None, None, ()
    process = None
    stdout = stderr = None
    memory_samples, tree_samples, memory_errors, cleanup_errors = [], [], [], []
    primary = None
    try:
        stdout = (target / "stdout.log").open("wb")
        stderr = (target / "stderr.log").open("wb")
        process = subprocess.Popen(
            list(command),
            cwd=target,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name == "posix",
        )
        deadline = start + timeout
        while process.poll() is None:
            try:
                sample = resident_bytes(process.pid)
                tree_sample = process_tree_resident_bytes(process.pid)
                if sample is not None:
                    memory_samples.append(sample)
                if tree_sample is not None:
                    tree_samples.append(tree_sample)
            except OSError as exc:
                message = diagnostic(exc)
                if message not in memory_errors:
                    memory_errors.append(message)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "TIMED_OUT"
                _terminate(process)
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        returncode = process.wait()
        if status != "TIMED_OUT":
            if returncode == 0:
                checked = tuple(inspect_artifact(target, item) for item in artifacts)
                status = "SUCCEEDED"
            else:
                error = f"command exited with status {returncode}"
    except BaseException as exc:
        primary = exc
        error = diagnostic(exc)
    finally:

        def cleanup(label, operation):
            try:
                operation()
            except BaseException as exc:
                cleanup_errors.append(f"{label}: {diagnostic(exc)}")

        if process is not None:

            def terminate_owned():
                if os.name == "posix" or process.poll() is None:
                    _terminate(process)

            cleanup("process termination", terminate_owned)
            cleanup("process wait", lambda: process.wait(timeout=5))
        for label, handle in (("stdout close", stdout), ("stderr close", stderr)):
            if handle is not None:
                cleanup(label, handle.close)
    if cleanup_errors and status == "SUCCEEDED":
        status = "FAILED"
    if error is None and cleanup_errors:
        error = cleanup_errors[0]
    result = ProcessResult(
        status,
        returncode,
        time.monotonic() - start,
        checked,
        error,
        max(memory_samples) if memory_samples else None,
        max(tree_samples) if tree_samples else None,
        tuple(memory_errors),
        tuple(cleanup_errors),
    )
    try:
        write_json(target / "run.json", asdict(result))
    except Exception as persistence_error:
        raise RunPersistenceError(result, persistence_error, diagnostic(persistence_error)) from (
            primary or persistence_error
        )
    if primary is not None and not isinstance(primary, Exception):
        raise primary
    return result


class RunPersistenceError(RuntimeError):
    """Expose the completed process outcome when its result file cannot be saved."""

    def __init__(self, result, persistence_error, message=None):
        self.result = result
        self.persistence_error = persistence_error
        super().__init__(
            f"{result.error or result.status}; result persistence failed: {message or persistence_error}"
        )


def _terminate(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
