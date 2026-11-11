import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from intelligence_dynamics.native import ExecutionState
from intelligence_dynamics.research.carry import summarize_carry_pair
from intelligence_dynamics.storage import write_json
from tests.support import create_junction, create_symlink
from tools import run_carry_confirmation as campaign
from tools import run_carry_study as runner

try:
    import numpy as np
except ImportError:
    np = None


class CampaignSchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {"indices": [25], "boundary": 2, "endpoint": 4, "checkpoint_max_bytes": 1024}
        self.args = SimpleNamespace(
            config=Path("config.json"), source_manifest=Path("sources.json")
        )

    def test_command_help_imports_source_with_site_packages_disabled(self):
        script = Path(__file__).resolve().parents[1] / "tools/run_carry_confirmation.py"
        for flags in ([], ["-O"], ["-OO"]):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-B", *flags, str(script), "--help"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--upstream", result.stdout)

    def stage(self, phase, step):
        stage = self.root / "index-25" / f"{phase}-{step:09d}"
        (stage / "native").mkdir(parents=True)
        (stage / "native/manifest.json").write_text("{}", encoding="utf-8")
        payload = b"trace"
        (stage / "traces.npz").write_bytes(payload)
        write_json(
            stage / "stage.json",
            {
                "phase": phase,
                "step": step,
                "run_identity": "run",
                "traces_file_sha256": hashlib.sha256(payload).hexdigest(),
            },
        )

    def completed(self):
        write_json(
            self.root / "index-25/summary.json",
            {
                "index": 25,
                "seed": 25,
                "run_identity": "run",
                "boundary": 2,
                "endpoint": 4,
            },
        )

    def test_workers_advance_then_finish_and_resume_skips_completed_index(self):
        calls = []
        steps = iter((("common", 2), ("continued", 4), ("reset", 4), None))

        def work(command, **kwargs):
            calls.append(command)
            stage = next(steps)
            if stage is None:
                self.completed()
                return SimpleNamespace(pid=1, wait=lambda: 0)
            self.stage(*stage)
            return SimpleNamespace(pid=1, wait=lambda: campaign.CHECKPOINT_READY)

        with patch.object(campaign.subprocess, "Popen", side_effect=work):
            campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("--resume" in command for command in calls))
        with patch.object(campaign.subprocess, "Popen") as child:
            campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
            child.assert_not_called()
        records = list((self.root / "execution").glob("*/process.json"))
        self.assertEqual(len(records), 4)

    def test_no_progress_and_premature_success_are_rejected(self):
        for code in (0, campaign.CHECKPOINT_READY):
            with self.subTest(code=code):
                with patch.object(
                    campaign.subprocess,
                    "Popen",
                    return_value=SimpleNamespace(pid=1, wait=lambda: code),
                ):
                    with self.assertRaises(ValueError):
                        campaign.run_campaign(self.args, self.config, self.root, self.root, "run")

    def test_failed_worker_preserves_its_exit_record(self):
        with patch.object(
            campaign.subprocess, "Popen", return_value=SimpleNamespace(pid=1, wait=lambda: 7)
        ):
            with self.assertRaisesRegex(RuntimeError, "index 25 failed"):
                campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
        record = next((self.root / "execution").glob("*/process.json"))
        self.assertEqual(json.loads(record.read_text())["exit_code"], 7)

    def test_launch_failure_preserves_command_directory_and_exception(self):
        with patch.object(
            campaign.subprocess, "Popen", side_effect=OSError("synthetic launch error")
        ):
            with self.assertRaisesRegex(OSError, "synthetic launch"):
                campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
        path = next((self.root / "execution").glob("*/process.json"))
        record = json.loads(path.read_text())
        self.assertEqual(record["state"], "launch_failed")
        self.assertEqual(record["error"]["type"], "OSError")
        self.assertEqual(record["cwd"], str(Path.cwd()))
        self.assertIn("--worker-index", record["argv"])

    def test_interrupted_wait_records_unknown_outcome(self):
        def interrupted():
            raise KeyboardInterrupt

        with patch.object(
            campaign.subprocess, "Popen", return_value=SimpleNamespace(pid=42, wait=interrupted)
        ):
            with self.assertRaises(KeyboardInterrupt):
                campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
        path = next((self.root / "execution").glob("*/process.json"))
        record = json.loads(path.read_text())
        self.assertEqual(record["state"], "outcome_unknown")
        self.assertEqual(record["pid"], 42)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_execution_junction_is_rejected_before_side_effects(self):
        external = self.root / "outside"
        external.mkdir()
        create_junction(self.root / "execution", external)
        with patch.object(campaign.subprocess, "Popen") as child:
            with self.assertRaisesRegex(ValueError, "links or reparse points"):
                campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
            child.assert_not_called()
        self.assertEqual(list(external.iterdir()), [])

    def test_execution_symlink_is_rejected_before_side_effects(self):
        external = self.root / "outside"
        external.mkdir()
        create_symlink(self, self.root / "execution", external, directory=True)
        with patch.object(campaign.subprocess, "Popen") as child:
            with self.assertRaisesRegex(ValueError, "links or reparse points"):
                campaign.run_campaign(self.args, self.config, self.root, self.root, "run")
            child.assert_not_called()
        self.assertEqual(list(external.iterdir()), [])

    def test_partial_or_wrong_identity_stage_is_rejected(self):
        stage = self.root / "index-25/common-000000002"
        stage.mkdir(parents=True)
        with self.assertRaises(OSError):
            campaign.stage_progress(stage.parent, self.config, "run")
        write_json(stage / "stage.json", {"run_identity": "other", "phase": "common", "step": 2})
        with self.assertRaises(ValueError):
            campaign.stage_progress(stage.parent, self.config, "run")

    @unittest.skipUnless(sys.platform == "linux", "Linux campaign lock")
    def test_concurrent_campaign_lock_rejects_another_process(self):
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            "from tools.run_carry_study import study_lock\n"
            "with study_lock(Path(sys.argv[1]), name='.campaign-lock'):\n"
            "    print('acquired')\n"
        )
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        command = [sys.executable, "-B", "-c", script, str(self.root)]
        with runner.study_lock(self.root, name=".campaign-lock"):
            blocked = subprocess.run(command, env=environment, capture_output=True, text=True)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("another process", blocked.stderr)
        released = subprocess.run(command, env=environment, capture_output=True, text=True)
        self.assertEqual(released.returncode, 0, released.stderr)
        self.assertEqual(released.stdout.strip(), "acquired")


@unittest.skipIf(np is None, "NumPy is required for carry-study traces")
class CampaignAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {
            "indices": list(range(25, 33)),
            "boundary": 1024,
            "endpoint": 4096,
            "checkpoint_max_bytes": 1024,
        }

    def traces(self, length, reward=0):
        return {
            name: np.full(
                length,
                reward if name == "rewards" else 0,
                dtype=np.int32 if name == "biome_id" else np.float16,
            )
            for name in runner.TRACE_DTYPES
        }

    def stage(self, index, phase, traces, metadata):
        step = len(traces["rewards"])
        stage = self.root / f"index-{index}" / f"{phase}-{step:09d}"
        (stage / "native").mkdir(parents=True)
        payload = b"synthetic checkpoint"
        (stage / "native/payload.bin").write_bytes(payload)
        manifest = {
            "format": "native-checkpoint-v1",
            "codec": "local-pickle5-zlib6",
            "bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_sha256": "a" * 64,
        }
        write_json(stage / "native/manifest.json", manifest)
        np.savez_compressed(stage / "traces.npz", **traces)
        write_json(
            stage / "stage.json",
            {
                "phase": phase,
                "step": step,
                "run_identity": "run",
                "traces_file_sha256": campaign.file_digest(stage / "traces.npz"),
                "traces": runner.trace_identity(traces),
                "metadata": metadata,
            },
        )
        return manifest["payload_sha256"]

    def populate(self):
        for index in self.config["indices"]:
            common = self.traces(1024)
            payload = self.stage(index, "common", common, {"index": index, "seed": index})
            left, right = self.traces(4096), self.traces(4096)
            right["rewards"][1024:] = 1
            base = {
                "common_payload_sha256": payload,
                "common_state_sha256": "same",
                "carry_before": {"tensor_bytes": 4},
                "carry_after_intervention": {"tensor_bytes": 0},
                "carry_after_first_step": {"tensor_bytes": 4},
            }
            self.stage(index, "continued", left, {**base, "intervention_paths": []})
            self.stage(
                index, "reset", right, {**base, "intervention_paths": [["agent_state", "carry"]]}
            )
            summary = summarize_carry_pair(left, right, boundary=1024, endpoint=4096)
            summary.update(index=index, seed=index, run_identity="run")
            write_json(self.root / f"index-{index}/summary.json", summary)

    def test_all_raw_pairs_are_recomputed_and_summary_is_idempotent(self):
        self.populate()
        result = campaign.analyze_campaign(self.root, self.config, "run")
        self.assertTrue(result["criterion_met"])
        self.assertEqual(result["mean_paired_reward_effect"], 1)
        self.assertEqual(result["scientific_interactions"], 8 * (8192 - 1024))
        self.assertEqual(campaign.analyze_campaign(self.root, self.config, "run"), result)

    def test_missing_index_and_corrupt_checkpoint_fail(self):
        with self.assertRaisesRegex(ValueError, "every configured"):
            campaign.analyze_campaign(self.root, self.config, "run")
        self.populate()
        with (self.root / "index-25/reset-000004096/native/payload.bin").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            campaign.analyze_campaign(self.root, self.config, "run")


@unittest.skipIf(np is None, "NumPy is required for carry-study traces")
class CheckpointYieldTests(unittest.TestCase):
    def test_staged_execution_matches_continuous_execution(self):
        config = {"chunk_size": 2, "checkpoint_interval": 2, "checkpoint_max_bytes": 1024 * 1024}
        state = ExecutionState({}, {}, 0)

        def compile_steps(glue, count):
            def advance(value):
                stop = value.total_steps + count
                numbers = np.arange(value.total_steps, stop, dtype=np.float16)
                traces = {
                    name: numbers.astype(np.int32) if name == "biome_id" else numbers
                    for name in runner.TRACE_DTYPES
                }
                return replace(value, total_steps=stop), (
                    traces,
                    np.zeros(count, dtype=bool),
                    np.arange(stop - count, stop),
                    np.arange(stop - count, stop),
                )

            return advance

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = (
                patch.dict(sys.modules, {"jax": SimpleNamespace(device_get=lambda value: value)}),
                patch.object(runner, "compiled_steps", side_effect=compile_steps),
                patch.object(runner, "scalar_types", return_value={}),
                patch.object(runner, "native_state", side_effect=lambda value, _: value),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                continuous, staged = root / "continuous", root / "staged"
                continuous.mkdir()
                staged.mkdir()
                traces = {
                    name: np.empty(0, dtype=dtype) for name, dtype in runner.TRACE_DTYPES.items()
                }
                final, expected = runner.advance(
                    None, continuous, "common", state, traces, 6, config, "run", {"segments": []}
                )
                current = state
                observed = {
                    name: np.empty(0, dtype=dtype) for name, dtype in runner.TRACE_DTYPES.items()
                }
                metadata = {"segments": []}
                for step in (2, 4, 6):
                    with self.assertRaises(runner.CheckpointReady):
                        runner.advance(
                            None,
                            staged,
                            "common",
                            current,
                            observed,
                            6,
                            config,
                            "run",
                            metadata,
                            stop_after_checkpoint=True,
                        )
                    saved = runner.latest_stage(staged, "common")
                    current, observed, metadata = runner.load_stage(saved, config, "run")
                    self.assertEqual(current.total_steps, step)
                self.assertEqual(final, current)
                for name in runner.TRACE_DTYPES:
                    self.assertEqual(expected[name].tobytes(), observed[name].tobytes())

                failed = root / "failed"
                failed.mkdir()
                empty = {
                    name: np.empty(0, dtype=dtype) for name, dtype in runner.TRACE_DTYPES.items()
                }
                with patch.object(runner, "save_stage", side_effect=OSError("write interrupted")):
                    with self.assertRaisesRegex(OSError, "write interrupted"):
                        runner.advance(
                            None,
                            failed,
                            "common",
                            state,
                            empty,
                            2,
                            config,
                            "run",
                            {"segments": []},
                            stop_after_checkpoint=True,
                        )
                self.assertIsNone(runner.latest_stage(failed, "common"))


if __name__ == "__main__":
    unittest.main()
