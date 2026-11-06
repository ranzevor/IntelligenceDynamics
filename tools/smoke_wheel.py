"""Exercise release assets from a fresh NumPy-free consumer environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from intelligence_dynamics.validation import is_linked_path  # noqa: E402
from tools.build_release import (  # noqa: E402
    external_destination,
    source_git_tree,
    source_manifest_digest,
    verify_wheel_source,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_archive(path: Path, entries: list[dict], *, prefix: str = "") -> None:
    require(isinstance(entries, list) and bool(entries), "invalid archive manifest entries")
    for row in entries:
        require(
            isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and type(row.get("bytes")) is int
            and row["bytes"] >= 0
            and isinstance(row.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "invalid archive manifest entry",
        )
    names = [row["path"] for row in entries]
    require(len(names) == len(set(names)), "duplicate archive manifest entry")
    for name in names:
        parsed = PurePosixPath(name)
        require(
            bool(parsed.parts)
            and not parsed.is_absolute()
            and name == parsed.as_posix()
            and ".." not in parsed.parts
            and "\\" not in name
            and ":" not in name,
            "invalid archive manifest path",
        )
    with zipfile.ZipFile(path) as archive:
        actual = archive.namelist()
        require(
            len(actual) == len(set(actual)) and set(actual) == {prefix + name for name in names},
            "archive file set differs from manifest",
        )
        for row in entries:
            info = archive.getinfo(prefix + row["path"])
            require(
                not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000,
                "archive entry is not a regular file",
            )
            data = archive.read(info)
            require(
                len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                "archive entry identity differs",
            )


def verify_assets(artifacts: Path) -> tuple[dict, Path, Path]:
    require(
        not any(is_linked_path(part) for part in (artifacts, *artifacts.parents)),
        "release path cannot traverse symbolic links or reparse points",
    )
    manifest_path = artifacts / "release-manifest.json"
    require(
        manifest_path.is_file() and not is_linked_path(manifest_path),
        "release manifest must be a regular file",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest["format"] == "intelligence-dynamics-release-v1"
        and manifest["project"] == "IntelligenceDynamics",
        "release manifest format differs",
    )
    version = manifest["version"]
    require(
        isinstance(version, str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version),
        "invalid release version",
    )
    wheel_name = f"intelligence_dynamics-{version}-py3-none-any.whl"
    source_name = f"IntelligenceDynamics-{version}-source.zip"
    expected = {wheel_name, source_name}
    names = [row["path"] for row in manifest["assets"]]
    require(
        len(names) == len(set(names)) and set(names) == expected,
        "release asset manifest must be unique and complete",
    )
    require(
        {p.name for p in artifacts.iterdir()} == expected | {"release-manifest.json", "SHA256SUMS"},
        "release directory contains missing or extra assets",
    )
    for path in artifacts.iterdir():
        require(path.is_file() and not is_linked_path(path), "release assets must be regular files")
    for row in manifest["assets"]:
        data = (artifacts / row["path"]).read_bytes()
        require(
            len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
            "asset identity differs",
        )
    checksums = {}
    for line in (artifacts / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        require(name not in checksums, "duplicate checksum entry")
        checksums[name] = value
    require(set(checksums) == expected | {"release-manifest.json"}, "checksum file set differs")
    for name, value in checksums.items():
        require(
            hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == value,
            "checksum identity differs",
        )
    require(
        isinstance(manifest.get("source_manifest_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest["source_manifest_sha256"])
        and manifest["source_manifest_sha256"] == source_manifest_digest(manifest["source_files"]),
        "source manifest identity differs",
    )
    wheel, source = artifacts / wheel_name, artifacts / source_name
    verify_archive(wheel, manifest["wheel_files"])
    verify_archive(source, manifest["source_files"], prefix=f"IntelligenceDynamics-{version}/")
    prefix = f"IntelligenceDynamics-{version}/"
    with zipfile.ZipFile(source) as archive:
        source_bytes = {
            row["path"]: archive.read(prefix + row["path"]) for row in manifest["source_files"]
        }
    origin = manifest.get("source")
    require(isinstance(origin, dict), "release source identity must be an object")
    tree = origin.get("tree")
    require(
        isinstance(tree, str)
        and re.fullmatch(r"[0-9a-f]{40}", tree)
        and tree == source_git_tree(source_bytes),
        "source Git tree differs from source archive",
    )
    verify_wheel_source(wheel, source_bytes, version)
    return manifest, wheel, source


def smoke(artifacts: Path, destination: Path) -> dict:
    manifest, wheel, archive = verify_assets(artifacts)
    artifacts = artifacts.resolve(strict=True)
    wheel, archive = artifacts / wheel.name, artifacts / archive.name
    destination = external_destination(ROOT, destination)
    destination.mkdir()
    work = destination / "work"
    work.mkdir()
    rows = []
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(label, command, expected=0):
        result = subprocess.run(
            [str(arg) for arg in command], cwd=work, env=environment, capture_output=True
        )
        prefix = f"{len(rows) + 1:02d}-{label}"
        (destination / f"{prefix}.stdout.log").write_bytes(result.stdout)
        (destination / f"{prefix}.stderr.log").write_bytes(result.stderr)
        rows.append(
            {
                "label": label,
                "argv": [str(arg) for arg in command],
                "exit": result.returncode,
                "expected_exit": expected,
                "stdout": f"{prefix}.stdout.log",
                "stderr": f"{prefix}.stderr.log",
            }
        )
        (destination / "commands.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        require(
            result.returncode == expected, f"{label} exit {result.returncode}, expected {expected}"
        )
        return result.stdout

    with zipfile.ZipFile(archive) as source:
        prefix = f"IntelligenceDynamics-{manifest['version']}/"
        for name in ("examples/dal.csv", "configs/dal-conversion.json", "configs/cue-study.json"):
            data = source.read(prefix + name)
            expected = next(row for row in manifest["source_files"] if row["path"] == name)
            require(
                hashlib.sha256(data).hexdigest() == expected["sha256"],
                "fixture source identity differs",
            )
            (work / Path(name).name).write_bytes(data)
    with tempfile.TemporaryDirectory(prefix="intelligence-dynamics-consumer-") as temporary:
        venv.EnvBuilder(with_pip=True).create(temporary)
        python = Path(temporary) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        cli = Path(temporary) / (
            "Scripts/intelligence-dynamics.exe" if os.name == "nt" else "bin/intelligence-dynamics"
        )
        run("install", [python, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
        probe = "import importlib.util,json,intelligence_dynamics as p; print(json.dumps({'version':p.__version__,'numpy':importlib.util.find_spec('numpy') is not None}))"
        state = json.loads(run("python-api", [python, "-I", "-B", "-c", probe]))
        require(
            state == {"version": manifest["version"], "numpy": False},
            "core environment identity differs",
        )
        run("demo", [cli, "demo", "demo.jsonl"])
        original = (work / "demo.jsonl").read_bytes()
        require(
            len(original.splitlines()) == 49,
            "demo must contain one run identity and 48 measurements",
        )
        run(
            "validate",
            [python, "-I", "-B", "-m", "intelligence_dynamics", "validate", "demo.jsonl"],
        )
        summary = json.loads(run("summarize", [cli, "summarize", "demo.jsonl"]))
        require(bool(summary), "summary must contain measurements")
        run("dal", [cli, "convert", "dal-csv", "dal.csv", "dal-conversion.json", "dal.jsonl"])
        run("dal-validate", [cli, "validate", "dal.jsonl"])
        run("study", [cli, "study", "cue-study.json", "study.json"])
        require(
            bool(json.loads((work / "study.json").read_text(encoding="utf-8"))),
            "study result is empty",
        )
        run("overwrite-refused", [cli, "demo", "demo.jsonl"], 2)
        require((work / "demo.jsonl").read_bytes() == original, "existing data changed")
        run("unknown-command", [cli, "unknown-command"], 2)
        (work / "invalid.jsonl").write_text("{invalid}\n", encoding="utf-8")
        run("malformed-input", [cli, "validate", "invalid.jsonl"], 2)
        run("recover-new-output", [cli, "demo", "recovered.jsonl"])
        run("uninstall", [python, "-m", "pip", "uninstall", "--yes", "intelligence-dynamics"])
        absent = run(
            "uninstalled",
            [
                python,
                "-I",
                "-B",
                "-c",
                "import importlib.util,json; print(json.dumps(importlib.util.find_spec('intelligence_dynamics') is None))",
            ],
        )
        require(json.loads(absent) is True, "uninstalled package remains importable")
        require((work / "demo.jsonl").read_bytes() == original, "uninstall modified user data")
        run("reinstall", [python, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
        run("reinstalled-validate", [cli, "validate", "demo.jsonl"])
        run("reinstalled-demo", [cli, "demo", "reinstalled.jsonl"])
    result = {
        "result": "PASS",
        "version": manifest["version"],
        "source_commit": manifest["source"]["commit"],
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python": sys.version,
        "commands": len(rows),
        "numpy_installed": False,
        "passed": len(rows),
        "failed": 0,
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.artifacts, args.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
