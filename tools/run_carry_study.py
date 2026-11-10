"""Run paired native DRQN continuations with one live-carry reset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import math
import pickle
import platform
import re
import sys
import tempfile
import time
import zlib
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligence_dynamics.native import (  # noqa: E402
    CheckpointCodec,
    CheckpointStore,
    DRQNExecutor,
    NativeCheckpoint,
    NativeLifecycle,
    execution_state,
    native_state_snapshot,
)
from intelligence_dynamics.native._state import state_differences  # noqa: E402
from intelligence_dynamics.storage import (  # noqa: E402
    atomic_write,
    digest,
    file_digest,
    loads,
    write_json,
)
from intelligence_dynamics.validation import (  # noqa: E402
    integer,
    is_linked_path,
    validate_project_files,
)

TRACE_DTYPES = {
    "rewards": "float16",
    "biome_id": "int32",
    "loss": "float16",
    "abs_td_error": "float16",
    "weight_change": "float16",
}
SCALARS = ("num_episodes", "num_steps", "total_reward", "total_steps")


def safe_path(path):
    path = Path(path).absolute()
    if any(is_linked_path(part) for part in (path, *path.parents)):
        raise ValueError("study paths cannot traverse links or reparse points")
    return path.resolve()


def read_config(path):
    config = loads(Path(path).read_bytes())
    expected = {
        "upstream_commit",
        "lock_blob",
        "foragax_version",
        "experiment",
        "indices",
        "boundary",
        "endpoint",
        "chunk_size",
        "checkpoint_interval",
        "checkpoint_max_bytes",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("study configuration has unexpected or missing fields")
    for name in ("upstream_commit", "lock_blob"):
        if not isinstance(config[name], str) or re.fullmatch(r"[0-9a-f]{40}", config[name]) is None:
            raise ValueError(f"{name} must be a full Git object ID")
    relative = PurePosixPath(config["experiment"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in config["experiment"]:
        raise ValueError("experiment must be a relative POSIX path")
    indices = config["indices"]
    if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)):
        raise ValueError("indices must be a nonempty unique list")
    for index in indices:
        integer(index, "index")
    for name in (
        "boundary",
        "endpoint",
        "chunk_size",
        "checkpoint_interval",
        "checkpoint_max_bytes",
    ):
        integer(config[name], name, minimum=1)
    if config["endpoint"] <= config["boundary"]:
        raise ValueError("endpoint must follow the branch boundary")
    if config["endpoint"] - config["boundary"] < 2:
        raise ValueError("each continuation needs at least two interactions")
    return config


def verify_upstream(upstream, manifest_path, config):
    manifest = loads(Path(manifest_path).read_bytes())
    if (
        manifest["commit"] != config["upstream_commit"]
        or manifest["lock_blob"] != config["lock_blob"]
    ):
        raise ValueError("upstream manifest does not match the study revision")
    checked = {}
    for row in manifest["files"]:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
            raise ValueError("upstream manifest contains an invalid path")
        source = safe_path(upstream / row["path"])
        if not source.is_relative_to(upstream) or row["path"] in checked:
            raise ValueError("upstream manifest contains an escaping or duplicate path")
        data = source.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != row["blob"] or len(data) != row["bytes"]:
            raise ValueError(f"upstream source differs: {row['path']}")
        checked[row["path"]] = {"blob": blob, "sha256": hashlib.sha256(data).hexdigest()}
    if checked.get("uv.lock", {}).get("blob") != config["lock_blob"]:
        raise ValueError("the pinned dependency lock is absent")
    if config["experiment"] not in checked:
        raise ValueError("the experiment configuration is absent")
    for source in (upstream / "src").rglob("*.py"):
        if source.relative_to(upstream).as_posix() not in checked:
            raise ValueError("unregistered Python source in upstream import directory")
    return checked


@contextmanager
def study_lock(output):
    import fcntl

    with (output / ".run-lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is using this study output") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def trace_identity(traces):
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in traces.items()
    }


def checkpoint_store(directory, config):
    limit = config["checkpoint_max_bytes"]

    def encode(value):
        payload = pickle.dumps(value, protocol=5)
        if len(payload) > limit:
            raise ValueError("uncompressed checkpoint exceeds the configured byte limit")
        return zlib.compress(payload, level=6)

    def decode(payload):
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, limit + 1)
        if len(raw) > limit or not decoder.eof or decoder.unused_data:
            raise ValueError("checkpoint compression stream exceeds its contract")
        return pickle.loads(raw)

    return CheckpointStore(
        directory,
        CheckpointCodec("local-pickle5-zlib6", encode, decode),
        max_bytes=limit,
    )


def scalar_types(state):
    return {
        name: {
            "dtype": str(getattr(state, name).dtype),
            "weak_type": bool(getattr(state, name).weak_type),
        }
        for name in SCALARS
    }


def native_state(state, types):
    import jax.numpy as jnp
    from utils.rlglue.rl_glue import GlueState

    values = {field.name: getattr(state, field.name) for field in fields(GlueState)}
    for name, spec in types.items():
        values[name] = (
            jnp.asarray(values[name])
            if spec["weak_type"]
            else jnp.asarray(values[name], dtype=spec["dtype"])
        )
        if (
            str(values[name].dtype) != spec["dtype"]
            or bool(values[name].weak_type) != spec["weak_type"]
        ):
            raise ValueError(f"restored {name} dtype or weak typing differs from the checkpoint")
    return GlueState(**values)


def save_stage(directory, phase, state, traces, config, identity, metadata):
    import numpy as np

    step = int(state.total_steps)
    target = directory / f"{phase}-{step:09d}"
    # Reward observations retain one entry per interaction in the lifecycle checkpoint.
    # The five original typed arrays are committed alongside it as traces.npz.
    checkpoint = NativeCheckpoint(execution_state(state), tuple(traces["rewards"].tolist()))
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=directory) as temporary:
        staging = Path(temporary)
        checkpoint_store(staging, config).save("native", checkpoint)
        data = io.BytesIO()
        np.savez_compressed(data, **traces)
        atomic_write(staging / "traces.npz", [data.getvalue()])
        record = {
            "phase": phase,
            "step": step,
            "run_identity": identity,
            "scalar_types": scalar_types(state),
            "traces": trace_identity(traces),
            "traces_file_sha256": file_digest(staging / "traces.npz"),
            "metadata": metadata,
        }
        write_json(staging / "stage.json", record)
        if target.exists():
            raise FileExistsError("study stage already exists")
        staging.rename(target)
    return target


def load_stage(directory, config, identity):
    import numpy as np

    record = loads((directory / "stage.json").read_bytes())
    if record["run_identity"] != identity:
        raise ValueError("saved stage belongs to a different study input")
    if file_digest(directory / "traces.npz") != record["traces_file_sha256"]:
        raise ValueError("saved trace file differs from its stage record")
    with np.load(directory / "traces.npz", allow_pickle=False) as archive:
        traces = {name: archive[name] for name in TRACE_DTYPES}
    if trace_identity(traces) != record["traces"]:
        raise ValueError("saved typed traces differ from their stage record")
    checkpoint = checkpoint_store(directory, config).load("native")
    if checkpoint.state.total_steps != record["step"]:
        raise ValueError("checkpoint and trace stage steps differ")
    for name, dtype in TRACE_DTYPES.items():
        if traces[name].shape != (record["step"],) or traces[name].dtype != np.dtype(dtype):
            raise ValueError("saved trace shape or dtype differs")
        if not np.isfinite(traces[name]).all():
            raise ValueError("saved traces contain nonfinite values")
    if tuple(traces["rewards"].tolist()) != checkpoint.history:
        raise ValueError("checkpoint reward history differs from the typed trace")
    return native_state(checkpoint.state, record["scalar_types"]), traces, record["metadata"]


def latest_stage(directory, phase):
    candidates = sorted(directory.glob(f"{phase}-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"))
    return candidates[-1] if candidates else None


def compiled_steps(glue, count):
    import jax
    import jax.numpy as jnp

    def step(state, _):
        state, interaction = glue._step(state)
        row = {
            "rewards": jnp.asarray(interaction.reward, dtype=jnp.float16),
            "biome_id": jnp.asarray(interaction.extra["biome_id"], dtype=jnp.int32),
            **{
                name: jnp.asarray(state.agent_state.metrics[name], dtype=jnp.float16)
                for name in ("loss", "abs_td_error", "weight_change")
            },
        }
        return state, (
            row,
            interaction.term | interaction.trunc,
            state.agent_state.steps,
            state.agent_state.updates,
        )

    return jax.jit(lambda state: jax.lax.scan(step, state, None, length=count))


def advance(glue, directory, phase, state, traces, endpoint, config, identity, metadata):
    import jax
    import numpy as np

    runners = {}
    saved_at = int(state.total_steps)
    while int(state.total_steps) < endpoint:
        start = int(state.total_steps)
        count = min(
            config["chunk_size"],
            endpoint - start,
            config["checkpoint_interval"] - start % config["checkpoint_interval"],
        )
        if count not in runners:
            runners[count] = compiled_steps(glue, count)
        runner = runners[count]
        began = time.monotonic()
        state, (chunk, terminated, agent_steps, updates) = runner(state)
        chunk, terminated, agent_steps, updates = jax.device_get(
            (chunk, terminated, agent_steps, updates)
        )
        if terminated.any():
            raise ValueError("the continuing study encountered a terminal or truncated interaction")
        for name, value in chunk.items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite native trace: {name}")
            traces[name] = np.concatenate((traces[name], value))
        if "first_step_audit" in metadata and metadata["first_update"] is None:
            positions = np.flatnonzero(updates > metadata["first_step_audit"]["updates_after"])
            if positions.size:
                position = int(positions[0])
                metadata["first_update"] = {
                    "environment_step": start + position + 1,
                    "agent_steps": int(agent_steps[position]),
                    "updates": int(updates[position]),
                }
        step = int(state.total_steps)
        metadata["segments"].append(
            {"start": start, "stop": step, "seconds": time.monotonic() - began}
        )
        print(f"{phase} {step}/{endpoint}", flush=True)
        if step == endpoint or step % config["checkpoint_interval"] == 0:
            saved = time.monotonic()
            save_stage(directory, phase, state, traces, config, identity, metadata)
            print(f"saved {phase} {step} in {time.monotonic() - saved:.2f}s", flush=True)
            saved_at = step
    if saved_at != endpoint:
        raise ValueError("endpoint has no committed stage")
    return state, traces


def branch_start(executor, common, traces, phase, common_payload_sha256):
    import numpy as np

    checkpoint = NativeCheckpoint(execution_state(common), tuple(traces["rewards"].tolist()))
    source_state = checkpoint.state
    before = native_state_snapshot(checkpoint.state)
    if phase == "reset":
        checkpoint = NativeLifecycle(executor).intervene(
            checkpoint, {("agent_state", "carry"): None}
        )
    intervened = native_state_snapshot(checkpoint.state)
    state, transition = executor.step(checkpoint.state)
    if transition.terminated or transition.truncated:
        raise ValueError("the branch first interaction ended the continuing environment")
    row = {
        "rewards": transition.reward,
        "biome_id": transition.extra["biome_id"],
        **{name: transition.metrics[name] for name in ("loss", "abs_td_error", "weight_change")},
    }
    result = {
        name: np.append(value, np.asarray(row[name], dtype=TRACE_DTYPES[name]))
        for name, value in traces.items()
    }
    metadata = {
        "segments": [],
        "boundary": checkpoint.state.total_steps,
        "common_state_sha256": before["sha256"],
        "common_payload_sha256": common_payload_sha256,
        "intervention_paths": [
            list(path) for path in state_differences(source_state, checkpoint.state)
        ],
        "intervened_state_sha256": intervened["sha256"],
        "first_step_audit": asdict(transition.audit),
        "first_update": {
            "environment_step": state.total_steps,
            "agent_steps": transition.audit.agent_steps_after,
            "updates": transition.audit.updates_after,
        }
        if transition.audit.learned
        else None,
        "carry_before": native_state_snapshot(common.agent_state.carry),
        "carry_after_intervention": native_state_snapshot(checkpoint.state.agent_state.carry),
        "carry_after_first_step": native_state_snapshot(state.agent_state.carry),
    }
    return native_state(state, scalar_types(common)), result, metadata


def run_index(index, directory, upstream, config, identity):
    import numpy as np
    from experiment.ExperimentModel import load
    from ml_instrumentation.Collector import Collector
    from ml_instrumentation.Sampler import Ignore
    from problems.registry import getProblem
    from utils.rlglue import RlGlue

    from intelligence_dynamics.research.carry import summarize_carry_pair

    directory.mkdir(exist_ok=True)
    experiment = load(str(upstream / config["experiment"]))
    collector = Collector(config={}, default=Ignore())
    collector.set_experiment_id(index)
    problem = getProblem(experiment.problem)(experiment, index, collector)
    agent, environment = problem.getAgent(), problem.getEnvironment()
    glue = RlGlue(agent, environment)
    executor = DRQNExecutor(agent, environment)
    stage = latest_stage(directory, "common")
    if stage is None:
        state, _ = glue._start(glue.state)
        traces = {name: np.empty(0, dtype=dtype) for name, dtype in TRACE_DTYPES.items()}
        metadata = {
            "segments": [],
            "index": index,
            "seed": problem.seed,
            "agent_total_steps": experiment.total_steps,
        }
    else:
        state, traces, metadata = load_stage(stage, config, identity)
    advance(
        glue, directory, "common", state, traces, config["boundary"], config, identity, metadata
    )
    common_path = latest_stage(directory, "common")
    arms = {}
    for phase in ("continued", "reset"):
        stage = latest_stage(directory, phase)
        if stage is None:
            common, common_traces, _ = load_stage(common_path, config, identity)
            payload = loads((common_path / "native" / "manifest.json").read_bytes())
            state, traces, metadata = branch_start(
                executor, common, common_traces, phase, payload["payload_sha256"]
            )
            del common, common_traces
        else:
            state, traces, metadata = load_stage(stage, config, identity)
        advance(
            glue, directory, phase, state, traces, config["endpoint"], config, identity, metadata
        )
        arms[phase] = traces
    result = summarize_carry_pair(
        arms["continued"], arms["reset"], boundary=config["boundary"], endpoint=config["endpoint"]
    )
    result.update(index=index, seed=problem.seed, run_identity=identity)
    target = directory / "summary.json"
    if target.exists():
        if loads(target.read_bytes()) != result:
            raise ValueError("existing summary differs from the resumed result")
    else:
        write_json(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument(
        "--source-manifest", type=Path, default=ROOT / "configs/carry-upstream.json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/carry-study.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    upstream, output = safe_path(args.upstream), safe_path(args.output)
    if output.is_relative_to(ROOT) or output.is_relative_to(upstream):
        raise ValueError("study output must be outside the project and upstream source")
    checked = verify_upstream(upstream, args.source_manifest, config)
    sys.path.insert(0, str(upstream / "src"))
    import jax

    if platform.system() != "Linux" or jax.default_backend() != "cpu":
        raise ValueError("this native carry study requires Linux with the JAX CPU backend")
    if importlib.metadata.version("continual-foragax") != config["foragax_version"]:
        raise ValueError("installed Foragax version differs from the study configuration")
    source = {
        name: file_digest(ROOT / name) for name in validate_project_files(ROOT, workspace=True)
    }
    contract = {
        "config": config,
        "upstream": checked,
        "project_sources": source,
        "python": sys.version,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "packages": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
    }
    identity = digest(contract)
    if args.resume:
        previous = loads((output / "run.json").read_bytes())
        if previous["identity"] != identity:
            raise ValueError("resume requires identical source, configuration and environment")
    else:
        output.mkdir(parents=False, exist_ok=False)
        write_json(
            output / "run.json", {"identity": identity, "contract": contract, "command": sys.argv}
        )
    with study_lock(output):
        results = [
            run_index(index, output / f"index-{index}", upstream, config, identity)
            for index in config["indices"]
        ]
        effects = [result["R_i_reset_minus_continued_post_mean_reward"] for result in results]
        write_json(
            output / "summary.json",
            {
                "run_identity": identity,
                "indices": results,
                "mean_paired_reward_effect": math.fsum(effects) / len(effects),
                "scientific_interactions": len(results)
                * (2 * config["endpoint"] - config["boundary"]),
            },
            overwrite=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
