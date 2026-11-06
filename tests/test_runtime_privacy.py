import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intelligence_dynamics.runtime import run_command
from intelligence_dynamics.storage import digest


class RuntimePrivacyTests(unittest.TestCase):
    def test_default_command_digest_preserves_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            private_value = "/home/synthetic-private-owner/work/input.json"
            command = [sys.executable, "-c", "import sys;print(len(sys.argv[1]))", private_value]
            result = run_command(command, root, timeout=2)
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual(
                int((root / "stdout.log").read_text(encoding="utf-8")), len(private_value)
            )
            request = json.loads((root / "request.json").read_text(encoding="utf-8"))
            self.assertIsNone(request["command"])
            self.assertEqual(request["command_sha256"], digest(command))
            self.assertEqual(request["argument_count"], 4)
            for path in root.glob("*.json"):
                self.assertNotIn(private_value, path.read_text(encoding="utf-8"))
                self.assertNotIn(sys.executable, path.read_text(encoding="utf-8"))

    def test_command_recording_requires_boolean_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-c", "pass"]
            self.assertEqual(
                run_command(command, root / "explicit", timeout=2, record_command=True).status,
                "SUCCEEDED",
            )
            request = json.loads((root / "explicit/request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["command"], command)
            for value in (1, "true", None):
                with self.assertRaises(ValueError):
                    run_command(command, root / "invalid", timeout=2, record_command=value)
                self.assertFalse((root / "invalid").exists())

    def test_missing_executable_path_not_stored_in_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            private_path = str(Path(directory) / "private-owner" / "missing-executable")
            result = run_command([private_path], root, timeout=2)
            self.assertEqual(result.status, "FAILED")
            self.assertIn("FileNotFoundError", result.error)
            self.assertNotIn(private_path, result.error)
            for path in root.glob("*.json"):
                self.assertNotIn(private_path, path.read_text(encoding="utf-8"))

    def test_error_keeps_primary_kind_but_removes_known_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            command = ["C:\\Users\\synthetic-private-owner\\program.exe"]
            with patch(
                "intelligence_dynamics.runtime.local.subprocess.Popen",
                side_effect=OSError(f"synthetic failure: {command[0]!r}"),
            ):
                result = run_command(command, root, timeout=2)
            self.assertIn("OSError: synthetic failure", result.error)
            self.assertNotIn("synthetic-private-owner", result.error)
            self.assertIn("<path>", result.error)
