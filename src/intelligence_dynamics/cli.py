"""Command line workflows for measurements and synthetic experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from itertools import chain
from pathlib import Path

from . import __version__
from .adapters import DalScale, dal_rows, forager_npz_records, npz_records
from .metrics import observations
from .records import ComputationCounters, RunMetadata, StepRecord
from .research import (
    LearningRule,
    SearchConfig,
    TabularPolicy,
    TaskConfig,
    make_bank,
    run_episode,
    run_study,
)
from .runtime import Artifact, BatchTarget, job_spec, run_command
from .storage import (
    canonical_bytes,
    digest,
    file_digest,
    loads,
    read_records,
    write_json,
    write_records,
)
from .validation import validate_project_files


def configuration(path, allowed, required=()):
    value = loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise ValueError("configuration has unknown or missing fields")
    return value


@contextmanager
def _conversion_input(path: Path):
    """Bind conversion and its raw input identity to one disk-backed snapshot."""
    with tempfile.TemporaryDirectory(prefix="id-convert-") as directory:
        snapshot = Path(directory) / "input"
        checksum = hashlib.sha256()
        with path.open("rb") as source, snapshot.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                checksum.update(chunk)
                target.write(chunk)
        yield snapshot, checksum.hexdigest()


def demo_records(seed=7):
    config = TaskConfig(steps=48, cue_count=4, delay=1, switches=(16, 32), seed=seed)
    rule = LearningRule(priors=(0.0, 0.0, 0.0, 0.0))
    result = run_episode(config, TabularPolicy(rule, mode="buffered"), seed=seed)
    run_id = f"cue-demo-{seed}"
    yield RunMetadata(
        run_id,
        "B",
        "buffered-tabular-policy",
        "switching-cue-task",
        seed,
        f"intelligence-dynamics-{__version__}",
        digest(asdict(config)),
        notes={"dataset": "synthetic", "task": asdict(config), "rule": asdict(rule)},
    )
    updates = {delivery.step: len(delivery.committed_ids) for delivery in result.deliveries}
    for row in result.records:
        yield StepRecord(
            run_id,
            row.step,
            float(row.step),
            capability=row.reward,
            reward_or_return=row.reward,
            computation=ComputationCounters(learning_updates=updates.get(row.step, 0)),
            context_id=f"regime:{row.regime}",
            extras={"cue": row.cue, "action": row.action, "target": row.target},
        )


def validate_stream(records):
    runs, previous, counts = {}, {}, {}
    for record in records:
        name = type(record).__name__
        counts[name] = counts.get(name, 0) + 1
        if isinstance(record, RunMetadata):
            if record.run_id in runs:
                raise ValueError("run metadata must be unique")
            runs[record.run_id] = record
        elif isinstance(record, StepRecord):
            if record.run_id not in runs:
                raise ValueError("step requires preceding run metadata")
            if record.run_id in previous:
                step, lifetime = previous[record.run_id]
                if record.step <= step or record.lifetime < lifetime:
                    raise ValueError(
                        "steps must increase and lifetimes cannot decrease within a run"
                    )
            previous[record.run_id] = record.step, record.lifetime
        elif hasattr(record, "run_id") and record.run_id not in runs:
            raise ValueError("record references an unknown run")
    if not runs:
        raise ValueError("stream requires run metadata")
    return {"runs": len(runs), "records": sum(counts.values()), "record_types": counts}


def _parser():
    parser = argparse.ArgumentParser(
        prog="intelligence-dynamics",
        description="Adaptive-system measurements and synthetic experiments",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check-source", help="validate the exact source-project file boundary"
    )
    check.add_argument("root", nargs="?", type=Path, default=Path("."))
    check.add_argument(
        "--workspace", action="store_true", help="exclude known development and run artifacts"
    )
    demo = commands.add_parser("demo", help="write a deterministic synthetic measurement stream")
    demo.add_argument("output", type=Path)
    demo.add_argument("--seed", type=int, default=7)
    for command in ("validate", "summarize"):
        item = commands.add_parser(command, help=command + " a measurement stream")
        item.add_argument("input", type=Path)
    study = commands.add_parser("study", help="run a configured synthetic learning-rule study")
    study.add_argument("config", type=Path)
    study.add_argument("output", type=Path)
    convert = commands.add_parser("convert", help="convert a native measurement file")
    convert.add_argument("format", choices=("dal-csv", "forager-npz", "array-npz"))
    convert.add_argument("input", type=Path)
    convert.add_argument("config", type=Path)
    convert.add_argument("output", type=Path)
    local = commands.add_parser("run", help="run a declared local command in a new directory")
    local.add_argument("config", type=Path)
    local.add_argument("output", type=Path)
    batch = commands.add_parser("batch-spec", help="write a Google Batch job specification")
    batch.add_argument("config", type=Path)
    batch.add_argument("output", type=Path)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check-source":
            files = validate_project_files(args.root, workspace=args.workspace)
            result = {"files": len(files), "result": "PASS"}
            if args.workspace:
                result["mode"] = "workspace"
        elif args.command == "demo":
            count = write_records(args.output, demo_records(args.seed))
            result = {
                "output": args.output.name,
                "records": count,
                "sha256": file_digest(args.output),
            }
        elif args.command in {"validate", "summarize"}:
            records = list(read_records(args.input))
            result = validate_stream(records)
            if args.command == "summarize":
                result["capability"] = observations(
                    record.capability for record in records if isinstance(record, StepRecord)
                )
        elif args.command == "study":
            config = configuration(
                args.config,
                {
                    "mode",
                    "seed",
                    "train_seed",
                    "heldout_seed",
                    "pairs",
                    "steps",
                    "cue_count",
                    "delay",
                    "search",
                },
                {"train_seed", "heldout_seed"},
            )
            bank_options = {
                key: config[key]
                for key in ("pairs", "steps", "cue_count", "delay")
                if key in config
            }
            training = make_bank(partition="train", seed=config["train_seed"], **bank_options)
            testing = make_bank(partition="heldout", seed=config["heldout_seed"], **bank_options)
            result = {
                "configuration": config,
                "configuration_sha256": digest(config),
                "dataset": "synthetic",
                "study": asdict(
                    run_study(
                        training,
                        testing,
                        mode=config.get("mode", "buffered"),
                        seed=config.get("seed", 0),
                        config=SearchConfig(**config.get("search", {})),
                    )
                ),
            }
            write_json(args.output, result)
            result = {"output": args.output.name, "sha256": file_digest(args.output)}
        elif args.command == "convert":
            config = configuration(
                args.config,
                {"metadata", "scale", "expected_rollout_steps", "capability_field", "interval"},
                {"metadata"},
            )
            with _conversion_input(args.input) as (native_input, native_sha256):
                metadata_fields = dict(config["metadata"])
                metadata_fields["config_digest"] = digest(config)
                notes = dict(metadata_fields.get("notes", {}))
                notes["native_sha256"] = native_sha256
                metadata_fields["notes"] = notes
                metadata = RunMetadata(**metadata_fields)
                if args.format == "dal-csv":
                    scale = DalScale(**config["scale"])
                    with native_input.open(encoding="utf-8", newline="") as source:
                        count = write_records(
                            args.output,
                            chain(
                                [metadata], dal_rows(source, run_id=metadata.run_id, scale=scale)
                            ),
                        )
                else:
                    if args.format == "forager-npz":
                        rows = forager_npz_records(
                            native_input,
                            run_id=metadata.run_id,
                            expected_rollout_steps=config.get("expected_rollout_steps"),
                        )
                    else:
                        rows = npz_records(
                            native_input,
                            run_id=metadata.run_id,
                            capability_field=config.get("capability_field", "rewards"),
                            interval=config.get("interval", 1),
                        )
                    count = write_records(args.output, chain([metadata], rows))
            result = {
                "output": args.output.name,
                "records": count,
                "sha256": file_digest(args.output),
            }
        elif args.command == "run":
            config = configuration(
                args.config,
                {"command", "timeout", "artifacts", "record_command"},
                {"command", "timeout"},
            )
            result = asdict(
                run_command(
                    config["command"],
                    args.output,
                    timeout=config["timeout"],
                    artifacts=[Artifact(**item) for item in config.get("artifacts", [])],
                    record_command=config.get("record_command", False),
                )
            )
            print(canonical_bytes(result).decode("utf-8"))
            return 0 if result["status"] == "SUCCEEDED" else 1
        else:
            config = configuration(
                args.config,
                {
                    "target",
                    "image",
                    "arguments",
                    "timeout_seconds",
                    "cpu_milli",
                    "memory_mib",
                    "machine_type",
                    "task_count",
                    "parallelism",
                    "task_environments",
                },
                {"target", "image", "arguments"},
            )
            target = BatchTarget(**config.pop("target"))
            result = job_spec(target, **config)
            write_json(args.output, result)
            result = {"output": args.output.name, "sha256": file_digest(args.output)}
        print(canonical_bytes(result).decode("utf-8"))
        return 0
    except (ValueError, TypeError, KeyError, OSError, csv.Error, ImportError) as exc:
        message = str(exc)
        paths = {str(Path.home())}
        for value in vars(args).values():
            if isinstance(value, Path):
                paths.update((str(value), str(value.absolute())))
        for private in sorted(paths, key=len, reverse=True):
            if len(private) > 1:
                message = message.replace(private, "<path>").replace(
                    private.replace("\\", "\\\\"), "<path>"
                )
        print(f"intelligence-dynamics: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
