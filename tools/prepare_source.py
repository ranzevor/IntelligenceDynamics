"""Copy registered source files without development or run artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def prepare_source(root: str | Path, destination: str | Path) -> dict:
    from intelligence_dynamics.storage import digest, file_digest
    from intelligence_dynamics.validation import is_linked_path, validate_project_files

    root, destination = Path(root), Path(destination)
    names = validate_project_files(root, workspace=True)
    root = root.resolve(strict=True)
    for part in (destination, *destination.parents):
        if is_linked_path(part):
            raise ValueError("candidate path cannot traverse symbolic links or reparse points")
    destination = destination.resolve(strict=False)
    if destination == root or destination.is_relative_to(root):
        raise ValueError("candidate must be outside the source project")
    if destination.exists():
        raise FileExistsError("candidate destination already exists")
    if not destination.parent.is_dir():
        raise ValueError("candidate parent directory does not exist")
    manifest = [{"path": name, "sha256": file_digest(root / name)} for name in names]
    destination.mkdir()
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    if validate_project_files(destination) != names:
        raise ValueError("candidate source boundary differs")
    if validate_project_files(root, workspace=True) != names:
        raise ValueError("source boundary changed during preparation")
    for item in manifest:
        if (
            file_digest(root / item["path"]) != item["sha256"]
            or file_digest(destination / item["path"]) != item["sha256"]
        ):
            raise ValueError("source content changed during preparation")
    return {
        "candidate": destination.name,
        "files": len(names),
        "source_manifest_sha256": digest(manifest),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    try:
        result = prepare_source(root, args.destination)
    except (OSError, ValueError) as exc:
        print(f"prepare-source: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
