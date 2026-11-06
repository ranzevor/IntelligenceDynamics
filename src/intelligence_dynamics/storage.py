"""Strict JSON parsing and atomic file output."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .records import from_object, to_object
from .validation import is_linked_path, json_value

MAX_LINE_BYTES = 1024 * 1024


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"invalid JSON number: {value}")


def loads(data: str | bytes):
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        value = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    return json_value(value)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _check_path(path: Path) -> None:
    for part in (path, *path.parents):
        if is_linked_path(part):
            raise ValueError("output paths cannot traverse symbolic links or reparse points")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("output target must be a regular file")
    if not path.parent.is_dir():
        raise ValueError("output parent directory does not exist")


def atomic_write(path: str | Path, chunks: Iterable[bytes], *, overwrite: bool = False) -> None:
    target = Path(path)
    _check_path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(str(target))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".id-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("output chunks must be bytes")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        _check_path(target)
        if overwrite:
            if target.exists():
                os.chmod(temporary, target.stat().st_mode & 0o777)
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        temporary = None
    except BaseException:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    atomic_write(path, [canonical_bytes(value) + b"\n"], overwrite=overwrite)


def write_records(path: str | Path, records: Iterable, *, overwrite: bool = False) -> int:
    count = 0

    def chunks():
        nonlocal count
        for record in records:
            line = canonical_bytes(to_object(record)) + b"\n"
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("measurement record exceeds one MiB")
            count += 1
            yield line

    atomic_write(path, chunks(), overwrite=overwrite)
    return count


def read_records(path: str | Path):
    with Path(path).open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"line {line_number}: record exceeds one MiB")
            if not line.strip():
                raise ValueError(f"line {line_number}: empty record")
            try:
                yield from_object(loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_records(path: str | Path, records: Iterable) -> int:
    """Atomically extend one writer's validated record stream."""
    target = Path(path)
    _check_path(target)
    existing = target.exists()
    previous = list(read_records(target)) if existing else []
    incoming = list(records)
    write_records(target, (*previous, *incoming), overwrite=existing)
    return len(incoming)
