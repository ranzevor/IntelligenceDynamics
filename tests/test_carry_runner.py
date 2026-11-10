import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from intelligence_dynamics.native import ExecutionState
from tests.support import create_junction, create_symlink
from tools.run_carry_study import (
    latest_stage,
    load_stage,
    native_state,
    read_config,
    safe_path,
    save_stage,
    study_lock,
    verify_upstream,
)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import jax
except ImportError:
    jax = None


@unittest.skipIf(jax is None, "requires JAX scalar types")
class CarryScalarTypeTests(unittest.TestCase):
    def setUp(self):
        self.previous_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", False)
        self.state = ExecutionState({}, {}, 0, total_reward=1.5)
        self.types = {
            name: {"dtype": "int32", "weak_type": True}
            for name in ("num_episodes", "num_steps", "total_steps")
        }
        self.types["total_reward"] = {"dtype": "float32", "weak_type": False}
        module = ModuleType("utils.rlglue.rl_glue")
        module.GlueState = ExecutionState
        self.modules = patch.dict(
            sys.modules,
            {
                "utils": ModuleType("utils"),
                "utils.rlglue": ModuleType("utils.rlglue"),
                "utils.rlglue.rl_glue": module,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(jax.config.update, "jax_enable_x64", self.previous_x64)

    def test_scalar_types_survive_compatible_restore(self):
        state = native_state(self.state, self.types)
        for name, expected in self.types.items():
            value = getattr(state, name)
            self.assertEqual(str(value.dtype), expected["dtype"])
            self.assertEqual(value.weak_type, expected["weak_type"])

    def test_x64_change_cannot_silently_widen_weak_scalars(self):
        jax.config.update("jax_enable_x64", True)
        with self.assertRaisesRegex(ValueError, "dtype or weak typing differs"):
            native_state(self.state, self.types)


class CarryConfigurationTests(unittest.TestCase):
    def test_configuration_rejects_missing_unknown_and_invalid_fields(self):
        source = Path(__file__).resolve().parents[1] / "configs/carry-study.json"
        config = read_config(source)
        self.assertEqual(config["indices"], [23, 24])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.json"
            invalid = [
                {**config, "endpoint": config["boundary"]},
                {**config, "indices": [23, 23]},
                {**config, "indices": [True]},
                {**config, "chunk_size": 0},
                {**config, "experiment": "../external.json"},
                {**config, "extra": 1},
                {key: value for key, value in config.items() if key != "boundary"},
            ]
            for value in invalid:
                with self.subTest(value=value):
                    target.write_text(json.dumps(value), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_config(target)

    def test_safe_path_preserves_a_regular_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(safe_path(directory), Path(directory).resolve())

    @unittest.skipUnless(sys.platform == "linux", "Linux study lock")
    def test_concurrent_writer_is_rejected_and_lock_is_released(self):
        root = Path(__file__).resolve().parents[1]
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            "from tools.run_carry_study import study_lock\n"
            "with study_lock(Path(sys.argv[1])):\n"
            "    print('acquired')\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-B", "-c", script, directory]
            environment = {**os.environ, "PYTHONPATH": str(root)}
            with study_lock(Path(directory)):
                blocked = subprocess.run(
                    command, env=environment, capture_output=True, text=True, timeout=10
                )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("another process", blocked.stderr)
            released = subprocess.run(
                command, env=environment, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(released.returncode, 0, released.stderr)
            self.assertEqual(released.stdout.strip(), "acquired")


class CarryUpstreamTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="carry-upstream-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "src").mkdir()
        self.files = {"src/entry.py": b"VALUE = 1\n", "config.json": b"{}", "uv.lock": b"lock"}
        self.rows = []
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            self.rows.append({"path": name, "bytes": len(data), "blob": blob})
        self.config = {
            "upstream_commit": "a" * 40,
            "lock_blob": self.rows[-1]["blob"],
            "experiment": "config.json",
        }
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {"commit": "a" * 40, "lock_blob": self.config["lock_blob"], "files": self.rows}
            ),
            encoding="utf-8",
        )

    def test_pinned_input_content_and_unregistered_python(self):
        for root in (self.root, self.root / "src" / ".."):
            with self.subTest(root=root):
                self.assertEqual(
                    set(verify_upstream(root, self.manifest, self.config)), set(self.files)
                )
                extra = self.root / "src/extra.py"
                extra.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    verify_upstream(root, self.manifest, self.config)
                extra.unlink()
                (self.root / "src/entry.py").write_text("VALUE = 3\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_upstream(root, self.manifest, self.config)
                (self.root / "src/entry.py").write_bytes(self.files["src/entry.py"])

    @unittest.skipUnless(os.name == "nt", "Windows short paths")
    def test_windows_short_root_resolves_to_registered_files(self):
        import ctypes

        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short_path.restype = ctypes.c_uint
        size = get_short_path(str(self.root), None, 0)
        if not size:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(size)
        if not get_short_path(str(self.root), buffer, size):
            raise ctypes.WinError(ctypes.get_last_error())
        root = Path(buffer.value)
        if root == self.root.resolve():
            self.skipTest("filesystem does not provide a distinct short path")
        self.assertEqual(set(verify_upstream(root, self.manifest, self.config)), set(self.files))

    def test_invalid_and_duplicate_manifest_paths_are_rejected(self):
        paths = ("../entry.py", "/entry.py", "src\\entry.py", "src/entry.py")
        root = self.root / "src" / ".."
        for path in paths:
            with self.subTest(path=path):
                message = (
                    "escaping or duplicate path" if path == "src/entry.py" else "an invalid path"
                )
                rows = [*self.rows, {**self.rows[0], "path": path}]
                self.manifest.write_text(
                    json.dumps(
                        {"commit": "a" * 40, "lock_blob": self.config["lock_blob"], "files": rows}
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    verify_upstream(root, self.manifest, self.config)

    def test_linked_root_is_rejected(self):
        alias = self.root / "linked-root"
        create_symlink(self, alias, self.root, directory=True)
        with self.assertRaisesRegex(ValueError, "links or reparse points"):
            verify_upstream(alias, self.manifest, self.config)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_junction_root_is_rejected(self):
        alias = self.root / "junction-root"
        create_junction(alias, self.root)
        with self.assertRaisesRegex(ValueError, "links or reparse points"):
            verify_upstream(alias, self.manifest, self.config)


@unittest.skipIf(np is None, "requires NumPy arrays")
class CarryStageTests(unittest.TestCase):
    def setUp(self):
        self.config = {"checkpoint_max_bytes": 1024 * 1024}
        self.state = ExecutionState(
            agent_state={"carry": [1.0]},
            env_state={"step": 2},
            last_action=1,
            num_episodes=0,
            num_steps=2,
            total_reward=3.0,
            total_steps=2,
        )
        self.traces = {
            "rewards": np.array([1.0, 2.0], dtype="float16"),
            "biome_id": np.array([0, 1], dtype="int32"),
            **{
                name: np.array([0.0, 0.5], dtype="float16")
                for name in ("loss", "abs_td_error", "weight_change")
            },
        }

    def save(self, root):
        with patch("tools.run_carry_study.scalar_types", return_value={}):
            return save_stage(
                root, "common", self.state, self.traces, self.config, "run-1", {"segments": []}
            )

    def test_stage_roundtrip_and_existing_stage_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = self.save(root)
            original = (stage / "stage.json").read_bytes()
            self.assertEqual(latest_stage(root, "common"), stage)
            with patch(
                "tools.run_carry_study.native_state", side_effect=lambda state, types: state
            ):
                restored, traces, metadata = load_stage(stage, self.config, "run-1")
            self.assertEqual(restored.total_steps, 2)
            self.assertEqual(metadata, {"segments": []})
            for name in self.traces:
                self.assertEqual(traces[name].dtype, self.traces[name].dtype)
                self.assertEqual(traces[name].tobytes(), self.traces[name].tobytes())
            with self.assertRaises(FileExistsError):
                self.save(root)
            self.assertEqual((stage / "stage.json").read_bytes(), original)
            self.assertFalse(list(root.glob(".stage-*")))

    def test_corrupt_trace_and_wrong_run_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = self.save(Path(directory))
            with self.assertRaisesRegex(ValueError, "different study"):
                load_stage(stage, self.config, "run-2")
            with (stage / "traces.npz").open("ab") as stream:
                stream.write(b"corruption")
            with patch("tools.run_carry_study.checkpoint_store") as store:
                with self.assertRaisesRegex(ValueError, "trace file differs"):
                    load_stage(stage, self.config, "run-1")
                store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
