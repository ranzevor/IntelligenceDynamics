"""Build a wheel, complete source archive, and source-bound release manifest."""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from intelligence_dynamics import __version__  # noqa: E402
from intelligence_dynamics.storage import digest  # noqa: E402
from intelligence_dynamics.validation import (  # noqa: E402
    is_linked_path,
    validate_project_files,
)


def external_destination(root: Path, destination: Path) -> Path:
    for part in (destination, *destination.parents):
        if is_linked_path(part):
            raise ValueError("output path cannot traverse symbolic links or reparse points")
    destination = destination.resolve()
    if destination == root.resolve() or destination.is_relative_to(root.resolve()):
        raise ValueError("output must be outside the source project")
    if destination.exists():
        raise FileExistsError("output destination already exists")
    if not destination.parent.is_dir():
        raise ValueError("output parent does not exist")
    return destination


def file_entry(path: Path, name: str) -> dict:
    data = path.read_bytes()
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def source_manifest_digest(entries: list[dict]) -> str:
    """Hash the sorted path/content identities independently of archive metadata."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest entries must be a nonempty list")
    identities = []
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("invalid source manifest entry")
        name = row["path"]
        parsed = PurePosixPath(name)
        if (
            not parsed.parts
            or parsed.is_absolute()
            or name != parsed.as_posix()
            or ".." in parsed.parts
            or "\\" in name
            or ":" in name
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("invalid source manifest identity")
        identities.append({"path": name, "sha256": row["sha256"]})
    if len({row["path"] for row in identities}) != len(identities):
        raise ValueError("duplicate source manifest entry")
    return digest(sorted(identities, key=lambda row: row["path"]))


def source_git_tree(source: dict[str, bytes]) -> str:
    """Compute the SHA-1 Git tree for the release's regular 100644 source files."""
    tree = {}
    for name, data in source.items():
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or name != path.as_posix()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or not isinstance(data, bytes)
        ):
            raise ValueError("invalid Git source member")
        parent = tree
        for part in path.parts[:-1]:
            parent = parent.setdefault(part, {})
            if not isinstance(parent, dict):
                raise ValueError("Git source has a file/directory collision")
        if path.name in parent:
            raise ValueError("Git source has a file/directory collision")
        parent[path.name] = data

    def object_id(kind, data):
        return hashlib.sha1(kind + b" " + str(len(data)).encode() + b"\0" + data).digest()

    def tree_id(branch):
        entries = []
        for name, value in branch.items():
            encoded = name.encode("utf-8")
            directory = isinstance(value, dict)
            identity = tree_id(value) if directory else object_id(b"blob", value)
            mode = b"40000" if directory else b"100644"
            entries.append(
                (encoded + (b"/" if directory else b""), mode + b" " + encoded + b"\0" + identity)
            )
        return object_id(b"tree", b"".join(body for _, body in sorted(entries)))

    return tree_id(tree).hex()


def _requirement_identity(value: str) -> tuple:
    requirement, separator, marker = value.partition(";")
    match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)(?:\[([^]]+)\])?\s*(.*?)\s*", requirement)
    if match is None:
        raise ValueError("invalid wheel dependency declaration")
    name = re.sub(r"[-_.]+", "-", match[1]).lower()
    extras = tuple(
        sorted(
            re.sub(r"[-_.]+", "-", x.strip()).lower()
            for x in (match[2] or "").split(",")
            if x.strip()
        )
    )
    specifiers = tuple(sorted(re.sub(r"\s+", "", match[3].strip("()")).split(",")))
    try:
        condition = (
            ast.dump(ast.parse(marker.strip(), mode="eval"), include_attributes=False)
            if separator
            else ""
        )
    except SyntaxError as exc:
        raise ValueError("invalid wheel dependency marker") from exc
    return name, extras, specifiers, condition


def _verify_dependency_metadata(metadata, project: dict, source: dict[str, bytes]) -> None:
    optional = project.get("optional-dependencies", {})
    expected = list(project.get("dependencies", []))
    for extra, dependencies in optional.items():
        for dependency in dependencies:
            requirement, separator, marker = dependency.partition(";")
            condition = f"({marker}) and " if separator else ""
            expected.append(f"{requirement}; {condition}extra == {json.dumps(extra)}")
    declared = sorted(_requirement_identity(value) for value in expected)
    actual = sorted(_requirement_identity(value) for value in metadata.get_all("Requires-Dist", []))

    def normalize_extra(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    if declared != actual or sorted(normalize_extra(name) for name in optional) != sorted(
        normalize_extra(name) for name in metadata.get_all("Provides-Extra", [])
    ):
        raise ValueError("wheel dependencies differ from source declarations")
    license_value = project.get("license")
    if isinstance(license_value, dict):
        expected_license = license_value.get("text")
        if expected_license is None and "file" in license_value:
            expected_license = source[license_value["file"]].decode("utf-8")
        actual_license = metadata["License"]
        if metadata["License-Expression"]:
            raise ValueError("wheel license metadata conflicts with source declarations")
    elif isinstance(license_value, str):
        expected_license, actual_license = license_value, metadata["License-Expression"]
    else:
        expected_license = actual_license = None
    if (" ".join((expected_license or "").split()) != " ".join((actual_license or "").split())) or (
        license_value is None and (metadata["License"] or metadata["License-Expression"])
    ):
        raise ValueError("wheel license metadata differs from source declarations")


def verify_wheel_source(wheel_path: Path, source: dict[str, bytes], version: str) -> None:
    """Check promised package content against source and its packaging declarations."""
    try:
        project_file = tomllib.loads(source["pyproject.toml"].decode("utf-8"))
        project = project_file["project"]
        settings = project_file["tool"]["setuptools"]
        discovery = settings["packages"]["find"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("invalid source packaging declarations") from exc
    if project["name"] != "intelligence-dynamics" or project["version"] != version:
        raise ValueError("source project identity differs from release")
    packages = {}
    for where in discovery.get("where", ["."]):
        prefix = where.rstrip("/") + "/" if where != "." else ""
        for name in source:
            if not name.startswith(prefix):
                continue
            relative = PurePosixPath(name[len(prefix) :])
            for parent in relative.parents:
                if not parent.parts or not all(part.isidentifier() for part in parent.parts):
                    continue
                package = ".".join(parent.parts)
                if not discovery.get("namespaces", True) and (
                    prefix + parent.as_posix() + "/__init__.py" not in source
                ):
                    continue
                if not any(
                    fnmatch.fnmatchcase(package, p) for p in discovery.get("include", ["*"])
                ):
                    continue
                if any(fnmatch.fnmatchcase(package, p) for p in discovery.get("exclude", [])):
                    continue
                packages[package] = prefix + parent.as_posix() + "/"
    expected = {}
    for package, prefix in packages.items():
        target = package.replace(".", "/") + "/"
        for name, data in source.items():
            if name.startswith(prefix) and "/" not in name[len(prefix) :] and name.endswith(".py"):
                expected[target + name[len(prefix) :]] = data
        patterns = settings.get("package-data", {}).get("", []) + settings.get(
            "package-data", {}
        ).get(package, [])
        for pattern in patterns:
            if not any(
                name.startswith(prefix) and fnmatch.fnmatchcase(name[len(prefix) :], pattern)
                for name in source
            ):
                raise ValueError("declared package data is missing from source")
        # Setuptools includes PEP 561 markers even without an explicit package-data row.
        patterns = [*patterns, "py.typed", "*.pyi"]
        excluded = settings.get("exclude-package-data", {}).get("", []) + settings.get(
            "exclude-package-data", {}
        ).get(package, [])
        for name, data in source.items():
            if name.startswith(prefix):
                relative = name[len(prefix) :]
                if any(fnmatch.fnmatchcase(relative, p) for p in patterns) and not any(
                    fnmatch.fnmatchcase(relative, p) for p in excluded
                ):
                    expected[target + relative] = data
    if not expected:
        raise ValueError("source declares no package content")
    dist_info = f"intelligence_dynamics-{version}.dist-info/"
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        actual = set(names)
        if len(names) != len(actual):
            raise ValueError("duplicate wheel archive entry")
        licenses = set()
        for pattern in settings.get("license-files", []):
            matches = [name for name in source if fnmatch.fnmatchcase(name, pattern)]
            if not matches:
                raise ValueError("declared license file is missing from source")
            licenses.update(matches)
        for name in licenses:
            candidates = {dist_info + PurePosixPath(name).name, dist_info + "licenses/" + name}
            matches = candidates & actual
            if len(matches) != 1:
                raise ValueError("wheel license mapping differs from source")
            expected[matches.pop()] = source[name]
        generated = {dist_info + name for name in ("METADATA", "WHEEL", "RECORD")}
        scripts = project.get("scripts", {})
        if scripts:
            generated.add(dist_info + "entry_points.txt")
        optional = {dist_info + "top_level.txt"}
        if not generated <= actual or actual - optional != set(expected) | generated:
            raise ValueError("wheel package members differ from source declarations")
        for name, data in expected.items():
            if wheel.read(name) != data:
                raise ValueError("wheel package content differs from source")
        metadata = BytesParser().parsebytes(wheel.read(dist_info + "METADATA"))
        if (
            metadata["Name"] != project["name"]
            or metadata["Version"] != version
            or metadata["Requires-Python"] != project.get("requires-python")
            or set(metadata.get_all("License-File", [])) != licenses
        ):
            raise ValueError("wheel metadata differs from source declarations")
        _verify_dependency_metadata(metadata, project, source)
        wheel_metadata = BytesParser().parsebytes(wheel.read(dist_info + "WHEEL"))
        if wheel_metadata["Root-Is-Purelib"] != "true" or wheel_metadata.get_all("Tag") != [
            "py3-none-any"
        ]:
            raise ValueError("wheel platform contract differs")
        if scripts:
            entry_points = configparser.ConfigParser(interpolation=None)
            entry_points.optionxform = str
            entry_points.read_string(wheel.read(dist_info + "entry_points.txt").decode("utf-8"))
            if (
                set(entry_points.sections()) != {"console_scripts"}
                or dict(entry_points["console_scripts"]) != scripts
            ):
                raise ValueError("wheel entry points differ from source declarations")
        top_level = dist_info + "top_level.txt"
        if top_level in actual and set(wheel.read(top_level).decode().splitlines()) != {
            name.split(".")[0] for name in packages
        }:
            raise ValueError("wheel top-level packages differ from source")
        record_name = dist_info + "RECORD"
        records = list(csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8"))))
        if (
            any(len(row) != 3 for row in records)
            or len({row[0] for row in records}) != len(records)
            or {row[0] for row in records} != actual
        ):
            raise ValueError("wheel RECORD file set differs")
        for name, checksum, length in records:
            if name == record_name:
                if checksum or length:
                    raise ValueError("wheel RECORD self-entry differs")
                continue
            data = wheel.read(name)
            value = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if checksum != "sha256=" + value or length != str(len(data)):
                raise ValueError("wheel RECORD identity differs")


def git_identity(root: Path, names: list[str]) -> dict:
    def git(*args):
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        if top != root.resolve():
            raise FileNotFoundError("source has no root repository")
        head = git("rev-parse", "HEAD").decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "tree": None, "base_commit": None, "source_date_epoch": 946684800}
    epoch = int(git("show", "-s", "--format=%ct", head).decode())
    objects = {}
    for entry in git("ls-tree", "-r", "-z", head).split(b"\0"):
        if not entry:
            continue
        identity, name = entry.split(b"\t", 1)
        mode, kind, blob = identity.split()
        objects[name.decode("utf-8")] = (mode, kind, blob.decode())
    clean = set(objects) == set(names)
    for name in names:
        data = (root / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if objects.get(name) != (b"100644", b"blob", blob):
            clean = False
    return {
        "commit": head if clean else None,
        "tree": git("rev-parse", f"{head}^{{tree}}").decode().strip() if clean else None,
        "base_commit": head,
        "source_date_epoch": epoch,
    }


def export_commit(root: Path, commit: str, names: list[str], destination: Path) -> list[dict]:
    def git(*args, input=None):
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *args],
            input=input,
            capture_output=True,
            check=True,
        ).stdout

    objects = {}
    for entry in git("ls-tree", "-r", "-z", commit).split(b"\0"):
        if not entry:
            continue
        identity, name = entry.split(b"\t", 1)
        mode, kind, blob = identity.split()
        if mode != b"100644" or kind != b"blob":
            raise ValueError("commit source entries must be regular files")
        objects[name.decode("utf-8")] = blob
    if set(objects) != set(names):
        raise ValueError("commit source boundary differs from registered files")
    payload = git("cat-file", "--batch", input=b"\n".join(objects[name] for name in names) + b"\n")
    destination.mkdir()
    entries = []
    offset = 0
    for name in names:
        boundary = payload.find(b"\n", offset)
        if boundary < 0:
            raise ValueError("Git object response is incomplete")
        header = payload[offset:boundary].split()
        if len(header) != 3 or header[0] != objects[name] or header[1] != b"blob":
            raise ValueError("Git object identity differs")
        length = int(header[2])
        if length < 0:
            raise ValueError("Git object size is invalid")
        offset = boundary + 1
        data = payload[offset : offset + length]
        offset += length
        if len(data) != length or payload[offset : offset + 1] != b"\n":
            raise ValueError("Git object content is incomplete")
        offset += 1
        blob = hashlib.sha1(b"blob " + str(length).encode() + b"\0" + data).hexdigest()
        if blob.encode() != objects[name]:
            raise ValueError("Git object content differs from its blob identity")
        entries.append({"path": name, "bytes": length, "sha256": hashlib.sha256(data).hexdigest()})
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    if offset != len(payload):
        raise ValueError("Git object response has unexpected data")
    return entries


def write_source_archive(source: Path, names: list[str], output: Path, epoch: int) -> None:
    stamp = time.gmtime(max(315532800, min(epoch, 4354819198)))[:6]
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            info = zipfile.ZipInfo(f"IntelligenceDynamics-{__version__}/{name}", stamp)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (source / name).read_bytes())


def build_release(root: Path, destination: Path) -> dict:
    names = validate_project_files(root, workspace=True)
    destination = external_destination(root, destination)
    identity = git_identity(root, names)
    if identity["commit"] is None:
        raise ValueError("release requires source bytes from an exact Git commit")
    if os.environ.get("GITHUB_SHA") and identity["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("checked-out source differs from the CI event commit")
    destination.mkdir()
    source, build = destination / "source", destination / "build"
    before = export_commit(root, identity["commit"], names, source)
    if before != [file_entry(source / name, name) for name in validate_project_files(source)]:
        raise ValueError("exported source differs from the captured Git objects")
    if before != [
        file_entry(root / name, name) for name in validate_project_files(root, workspace=True)
    ]:
        raise ValueError("working source differs from the captured Git commit")
    shutil.copytree(source, build)
    if before != [file_entry(build / name, name) for name in names]:
        raise ValueError("build copy differs from the captured Git objects")
    artifacts = destination / "artifacts"
    artifacts.mkdir()
    environment = dict(os.environ, SOURCE_DATE_EPOCH=str(identity["source_date_epoch"]))
    command = [
        sys.executable,
        "-B",
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        str(build),
        "--wheel-dir",
        str(artifacts),
    ]
    (destination / "build-command.json").write_text(
        json.dumps({"argv": command, "source_date_epoch": identity["source_date_epoch"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(command, capture_output=True, env=environment)
    (destination / "build.stdout.log").write_bytes(result.stdout)
    (destination / "build.stderr.log").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"wheel build failed with exit {result.returncode}; inspect build logs")
    wheels = list(artifacts.glob("*.whl"))
    if (
        len(wheels) != 1
        or wheels[0].name != f"intelligence_dynamics-{__version__}-py3-none-any.whl"
    ):
        raise ValueError("wheel identity differs from the project version")
    source_archive = artifacts / f"IntelligenceDynamics-{__version__}-source.zip"
    write_source_archive(source, names, source_archive, identity["source_date_epoch"])
    source_bytes = {name: (source / name).read_bytes() for name in names}
    if source_git_tree(source_bytes) != identity["tree"]:
        raise ValueError("exported source tree differs from the captured Git commit")
    verify_wheel_source(wheels[0], source_bytes, __version__)
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_files = [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name in sorted(wheel.namelist())
            for data in [wheel.read(name)]
        ]
    if (
        before
        != [file_entry(root / name, name) for name in validate_project_files(root, workspace=True)]
        or before != [file_entry(source / name, name) for name in validate_project_files(source)]
        or before != [file_entry(build / name, name) for name in names]
    ):
        raise ValueError("source content changed during the build")
    manifest = {
        "format": "intelligence-dynamics-release-v1",
        "project": "IntelligenceDynamics",
        "version": __version__,
        "source": {key: identity[key] for key in ("commit", "tree", "source_date_epoch")},
        "source_manifest_sha256": source_manifest_digest(before),
        "source_files": before,
        "wheel_files": wheel_files,
        "build": {
            "python": sys.version,
            "platform": sys.platform,
            "tools": {
                name: importlib.metadata.version(name)
                for name in ("setuptools", "wheel", "packaging")
            },
        },
        "assets": [file_entry(p, p.name) for p in sorted(artifacts.iterdir())],
    }
    (artifacts / "release-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{file_entry(p, p.name)['sha256']}  {p.name}\n" for p in sorted(artifacts.iterdir())
    ]
    (artifacts / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    return {
        "result": "PASS",
        "version": __version__,
        "commit": identity["commit"],
        "source_files": len(names),
        "assets": [p.name for p in sorted(artifacts.iterdir())],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_release(ROOT, args.destination)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"build-release: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
