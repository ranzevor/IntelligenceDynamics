"""Validate dependency locks and GitHub maintenance configuration."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ALLOWED_ACTIONS = frozenset({"actions/checkout", "actions/setup-python", "actions/upload-artifact"})


def requirement(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+!-]*)", text)
    if match is None:
        raise ValueError("dependency must use an exact version without a URL or wildcard")
    return re.sub(r"[-_.]+", "-", match[1]).lower(), match[2]


def requirement_lines(text: str):
    pending = ""
    for raw in text.splitlines():
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        yield pending + line
        pending = ""
    if pending:
        raise ValueError("incomplete dependency continuation")


def split_hash_options(line: str) -> tuple[str, bool]:
    hashes = []

    def remove(match):
        value = match[1]
        if not re.fullmatch(r"sha256:[0-9A-Fa-f]{64}", value):
            raise ValueError("dependency hash must be a complete SHA-256")
        hashes.append(value)
        return " "

    value = re.sub(r"(?<!\S)--hash(?:=|\s+)(\S+)", remove, line)
    return value.strip(), bool(hashes)


def locked_requirement(line: str) -> tuple[str, str, bool]:
    value, hashed = split_hash_options(line)
    name, version = requirement(value)
    return name, version, hashed


def parse_lock(text: str) -> dict[str, str]:
    result = {}
    hashed = set()
    official_index = False
    require_hashes = False
    for line in requirement_lines(text):
        if line in {
            "--index-url https://pypi.org/simple",
            "--index-url=https://pypi.org/simple",
        }:
            official_index = True
            continue
        if line.startswith("--index-url"):
            raise ValueError("lock must select the official PyPI index")
        if line == "--require-hashes":
            require_hashes = True
            continue
        name, version, has_hash = locked_requirement(line)
        if name in result:
            raise ValueError("duplicate dependency in lock")
        result[name] = version
        if has_hash:
            hashed.add(name)
    if not official_index:
        raise ValueError("lock must select the official PyPI index")
    if not result:
        raise ValueError("empty dependency lock")
    if (require_hashes or hashed) and hashed != result.keys():
        raise ValueError("hash-required lock has a dependency without a hash")
    return result


def included_requirement(line: str) -> str | None:
    match = re.fullmatch(r"(?:-r\s*|--requirement(?:=|\s+))(\S+)", line)
    return match[1] if match else None


def validate_action(reference: str) -> None:
    name, separator, revision = reference.partition("@")
    if name not in ALLOWED_ACTIONS or not separator:
        raise ValueError("action source is not allowed")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision == "0" * 40:
        raise ValueError("action must use a full commit SHA")


def validate_dependencies(root: Path) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock_texts = {p.name: p.read_text(encoding="utf-8") for p in root.glob("requirements-*.lock")}
    locks = {name: parse_lock(value) for name, value in lock_texts.items()}
    hash_modes = {
        name: any(
            token == "--require-hashes" or token == "--hash" or token.startswith("--hash=")
            for line in requirement_lines(value)
            for token in line.split()
        )
        for name, value in lock_texts.items()
    }
    build = dict(requirement(row) for row in project["build-system"]["requires"])
    arrays = dict(requirement(row) for row in project["project"]["optional-dependencies"]["arrays"])
    if build != locks["requirements-build.lock"] or arrays != locks["requirements-arrays.lock"]:
        raise ValueError("project dependency declarations differ from locks")
    if locks["requirements-native.lock"]["numpy"] != arrays["numpy"]:
        raise ValueError("native and optional NumPy versions differ")
    includes = set()
    packages = {}
    hashed_packages = set()
    development_hash_mode = False
    for line in requirement_lines((root / "requirements-dev.txt").read_text(encoding="utf-8")):
        if line == "--require-hashes":
            development_hash_mode = True
            continue
        included = included_requirement(line)
        if included is not None:
            if included in includes:
                raise ValueError("duplicate development dependency include")
            includes.add(included)
            continue
        value, has_hash = split_hash_options(line)
        value, _, marker = value.partition(";")
        name, version = requirement(value.strip())
        if name in packages or (
            not re.fullmatch(r"python_version\s*<\s*(['\"])3\.11\1", marker.strip())
            if name == "tomli"
            else bool(marker)
        ):
            raise ValueError("development dependency marker or identity differs")
        packages[name] = version
        if has_hash:
            hashed_packages.add(name)
    if includes != {
        "requirements-build.lock",
        "requirements-arrays.lock",
        "requirements-lint.lock",
    }:
        raise ValueError("development dependency includes differ")
    if set(packages) != {"tomli", "pyyaml"}:
        raise ValueError("development parser dependencies differ")
    if (
        development_hash_mode or hashed_packages or any(hash_modes[name] for name in includes)
    ) and (hashed_packages != packages.keys() or not all(hash_modes[name] for name in includes)):
        raise ValueError("hash-required development dependencies include unhashed requirements")
    native = list(requirement_lines((root / "requirements-native.txt").read_text(encoding="utf-8")))
    if len(native) != 1 or included_requirement(native[0]) != "requirements-native.lock":
        raise ValueError("native dependency manifest must include its lock")


def validate_configuration(root: Path) -> None:
    import yaml

    workflow = yaml.load(
        (root / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    if set(workflow["on"]) != {"pull_request", "push", "workflow_dispatch"}:
        raise ValueError("CI trigger policy differs")
    if workflow["permissions"] != {"contents": "read"}:
        raise ValueError("CI requires read-only repository permissions")
    for job in workflow["jobs"].values():
        if "permissions" in job and job["permissions"] != {"contents": "read"}:
            raise ValueError("job permissions exceed the workflow policy")
        for step in job.get("steps", []):
            if "uses" in step:
                validate_action(step["uses"])
                if (
                    step["uses"].startswith("actions/checkout@")
                    and step.get("with", {}).get("persist-credentials") != "false"
                ):
                    raise ValueError("checkout must not persist credentials")
    config = yaml.load(
        (root / ".github/dependabot.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    if config["version"] != "2" or {row["package-ecosystem"] for row in config["updates"]} != {
        "pip",
        "github-actions",
    }:
        raise ValueError("dependency update ecosystems differ")
    for row in config["updates"]:
        if row["directory"] != "/" or row["schedule"] != {
            "interval": "weekly",
            "day": "monday",
            "time": "09:00",
            "timezone": "Asia/Shanghai",
        }:
            raise ValueError("dependency update location or schedule differs")
        if not 1 <= int(row["open-pull-requests-limit"]) <= 5:
            raise ValueError("dependency update concurrency is outside policy")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    validate_dependencies(args.root)
    validate_configuration(args.root)
    print(
        json.dumps(
            {"result": "PASS", "checks": ["dependency-locks", "CI-actions", "update-schedule"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
