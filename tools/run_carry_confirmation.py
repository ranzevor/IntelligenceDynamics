"""Run and analyze a fixed-sample native carry confirmation campaign."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from intelligence_dynamics.research.confirmation import (  # noqa: E402
    confirm_carry_effects,
    summarize_confirmation_pair,
)
from intelligence_dynamics.storage import file_digest, loads, write_json  # noqa: E402
from tools import run_carry_study as runner  # noqa: E402

GIB = 1024**3
CHECKPOINT_READY = 10


def execution_profile():
    if sys.platform != "linux":
        raise ValueError("carry confirmation requires Linux")
    cpus = sorted(os.sched_getaffinity(0))[:8]
    os.sched_setaffinity(0, cpus)
    os.environ.update(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    return {"logical_cpus": cpus, "omp_threads": 8, "openblas_threads": 1}


def stage_progress(directory, config, identity):
    directory = runner.safe_path(directory)
    progress = []
    for phase, endpoint in (
        ("common", config["boundary"]),
        ("continued", config["endpoint"]),
        ("reset", config["endpoint"]),
    ):
        stage = runner.latest_stage(directory, phase)
        if stage is None:
            progress.append(0)
            continue
        stage = runner.safe_path(stage)
        record = loads(runner.safe_path(stage / "stage.json").read_bytes())
        step = record["step"]
        if (
            record["run_identity"] != identity
            or record["phase"] != phase
            or type(step) is not int
            or not 0 < step <= endpoint
            or stage.name != f"{phase}-{step:09d}"
            or not runner.safe_path(stage / "native/manifest.json").is_file()
        ):
            raise ValueError("saved stage has an inconsistent identity or step")
        if file_digest(runner.safe_path(stage / "traces.npz")) != record["traces_file_sha256"]:
            raise ValueError("saved stage trace bytes differ")
        progress.append(step)
    return tuple(progress)


def completed_index(directory, index, config, identity):
    path = directory / "summary.json"
    if not path.exists():
        return None
    result = loads(runner.safe_path(path).read_bytes())
    if (
        result["index"] != index
        or result["run_identity"] != identity
        or result["boundary"] != config["boundary"]
        or result["endpoint"] != config["endpoint"]
        or stage_progress(directory, config, identity)
        != (config["boundary"], config["endpoint"], config["endpoint"])
    ):
        raise ValueError("completed index differs from the campaign")
    return result


def run_campaign(args, config, upstream, output, identity):
    execution = runner.safe_path(output / "execution")
    execution.mkdir(exist_ok=True)
    for index in config["indices"]:
        directory = runner.safe_path(output / f"index-{index}")
        while completed_index(directory, index, config, identity) is None:
            if shutil.disk_usage(output).free < 2 * config["checkpoint_max_bytes"]:
                raise ValueError("insufficient free space for the next checkpoint")
            before = stage_progress(directory, config, identity)
            attempt = runner.safe_path(tempfile.mkdtemp(prefix=f"index-{index}-", dir=execution))
            command = [
                "/usr/bin/time",
                "-v",
                "-o",
                str(attempt / "resources.log"),
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--upstream",
                str(upstream),
                "--config",
                str(args.config),
                "--source-manifest",
                str(args.source_manifest),
                "--output",
                str(output),
                "--resume",
                "--worker-index",
                str(index),
            ]
            record = {
                "argv": command,
                "cwd": str(Path.cwd()),
                "started": time.time(),
                "state": "starting",
                "progress_before": list(before),
            }
            record_path = attempt / "process.json"
            write_json(record_path, record)
            process = None
            try:
                with (attempt / "stdout.log").open("wb") as stdout:
                    with (attempt / "stderr.log").open("wb") as stderr:
                        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                        record.update(state="running", pid=process.pid)
                        write_json(record_path, record, overwrite=True)
                        returncode = process.wait()
            except BaseException as error:
                record.update(
                    state="launch_failed" if process is None else "outcome_unknown",
                    finished=time.time(),
                    error={"type": type(error).__name__, "message": str(error)},
                )
                write_json(record_path, record, overwrite=True)
                raise
            record.update(state="exited", finished=time.time(), exit_code=returncode)
            write_json(record_path, record, overwrite=True)
            if returncode not in (0, CHECKPOINT_READY):
                raise RuntimeError(f"index {index} failed; inspect execution/{attempt.name}")
            after = stage_progress(directory, config, identity)
            if returncode == CHECKPOINT_READY:
                if not all(new >= old for old, new in zip(before, after)) or after == before:
                    raise ValueError("worker reported a checkpoint without advancing")
            elif completed_index(directory, index, config, identity) is None:
                raise ValueError("worker returned success without a complete index")
            print(f"index {index}: {after}", flush=True)


def read_final_traces(directory, phase, endpoint, config, identity):
    import numpy as np

    stage = runner.safe_path(directory / f"{phase}-{endpoint:09d}")
    record = loads(runner.safe_path(stage / "stage.json").read_bytes())
    if (record["run_identity"], record["phase"], record["step"]) != (identity, phase, endpoint):
        raise ValueError("final stage differs from the campaign")
    native = stage / "native"
    manifest = loads(runner.safe_path(native / "manifest.json").read_bytes())
    payload = runner.safe_path(native / "payload.bin")
    if (
        set(manifest) != {"format", "codec", "bytes", "payload_sha256", "checkpoint_sha256"}
        or type(manifest["bytes"]) is not int
        or manifest["bytes"] <= 0
        or manifest["format"] != "native-checkpoint-v1"
        or manifest["codec"] != "local-pickle5-zlib6"
        or manifest["bytes"] > config["checkpoint_max_bytes"]
        or payload.stat().st_size != manifest["bytes"]
        or file_digest(payload) != manifest["payload_sha256"]
    ):
        raise ValueError("final checkpoint bytes differ from their manifest")
    traces_path = runner.safe_path(stage / "traces.npz")
    if file_digest(traces_path) != record["traces_file_sha256"]:
        raise ValueError("final trace bytes differ from their stage record")
    with np.load(traces_path, allow_pickle=False) as archive:
        traces = {name: archive[name] for name in runner.TRACE_DTYPES}
    if runner.trace_identity(traces) != record["traces"]:
        raise ValueError("final typed traces differ from their stage record")
    for name, dtype in runner.TRACE_DTYPES.items():
        if (
            traces[name].shape != (endpoint,)
            or traces[name].dtype != np.dtype(dtype)
            or not np.isfinite(traces[name]).all()
        ):
            raise ValueError("final trace shape, dtype or values differ from the campaign")
    return traces, record, manifest


def analyze_campaign(output, config, identity):
    pairs = []
    for index in config["indices"]:
        directory = output / f"index-{index}"
        original = completed_index(directory, index, config, identity)
        if original is None:
            raise ValueError("confirmation requires every configured index")
        common_traces, common_record, common = read_final_traces(
            directory, "common", config["boundary"], config, identity
        )
        continued, control, _ = read_final_traces(
            directory, "continued", config["endpoint"], config, identity
        )
        reset, intervention, _ = read_final_traces(
            directory, "reset", config["endpoint"], config, identity
        )
        for record, allowed_paths in (
            (control, [[]]),
            (intervention, [[], [["agent_state", "carry"]]]),
        ):
            metadata = record["metadata"]
            if (
                metadata["common_payload_sha256"] != common["payload_sha256"]
                or metadata["intervention_paths"] not in allowed_paths
            ):
                raise ValueError("branch intervention or common checkpoint differs")
        if (
            common_record["metadata"]["index"] != index
            or common_record["metadata"]["seed"] != original["seed"]
            or control["metadata"]["common_state_sha256"]
            != intervention["metadata"]["common_state_sha256"]
        ):
            raise ValueError("branch identity differs from the common history")
        for name, values in common_traces.items():
            prefix = continued[name][: config["boundary"]]
            if prefix.dtype != values.dtype or prefix.tobytes() != values.tobytes():
                raise ValueError("branch prefix differs from the saved common trace")
        pair = summarize_confirmation_pair(
            continued, reset, boundary=config["boundary"], endpoint=config["endpoint"]
        )
        if any(
            pair[key] != value
            for key, value in original.items()
            if key not in ("index", "seed", "run_identity")
        ):
            raise ValueError("recomputed pair differs from the completed index")
        pair.update(index=index, seed=original["seed"], run_identity=identity)
        pair["intervention_applied"] = bool(intervention["metadata"]["intervention_paths"])
        pair["carry"] = {
            name: intervention["metadata"][name]
            for name in ("carry_before", "carry_after_intervention", "carry_after_first_step")
        }
        pairs.append(pair)
    result = confirm_carry_effects(pairs, indices=config["indices"])
    result["indices"] = pairs
    result["scientific_interactions"] = len(pairs) * (2 * config["endpoint"] - config["boundary"])
    target = output / "summary.json"
    if target.exists():
        if loads(target.read_bytes()) != result:
            raise ValueError("existing confirmation differs from recomputed results")
    else:
        write_json(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path, default=ROOT / "configs/carry-upstream.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/carry-confirmation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = runner.read_config(args.config)
    if config["endpoint"] - config["boundary"] <= 1024:
        raise ValueError("confirmation requires more than 1024 post-boundary interactions")
    if args.worker_index is not None and args.worker_index not in config["indices"]:
        raise ValueError("worker index is outside the configured campaign")
    profile = execution_profile()
    if args.worker_index is None and not args.resume:
        parent = runner.safe_path(args.output).parent
        if shutil.disk_usage(parent).free < 50 * GIB:
            raise ValueError("confirmation requires 50 GiB of free disk space")
    if args.worker_index is None and not Path("/usr/bin/time").is_file():
        raise ValueError("confirmation requires /usr/bin/time")
    upstream, output, identity = runner.prepare_study(
        args.upstream,
        args.source_manifest,
        config,
        args.output,
        resume=args.resume,
        execution_profile=profile,
    )
    if args.worker_index is not None:
        with runner.study_lock(output):
            try:
                runner.run_index(
                    args.worker_index,
                    output / f"index-{args.worker_index}",
                    upstream,
                    config,
                    identity,
                    stop_after_checkpoint=True,
                )
            except runner.CheckpointReady:
                return CHECKPOINT_READY
        return 0
    with runner.study_lock(output, name=".campaign-lock"):
        run_campaign(args, config, upstream, output, identity)
        analyze_campaign(output, config, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
