import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intelligence_dynamics.cli import main
from intelligence_dynamics.validation import PROJECT_FILES, validate_project_files
from tests.support import create_junction, create_symlink
from tools.prepare_source import main as prepare_main
from tools.prepare_source import prepare_source


class SourcePreparationTests(unittest.TestCase):
    project = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace"
        self.root.mkdir()
        for name in PROJECT_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project / name, target)
        self.destination = self.root.parent / "candidate"

    def artifact(self, name):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic workspace artifact\n", encoding="utf-8")

    def test_clean_source_passes_both_modes_and_cli_formats(self):
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                self.assertEqual(
                    set(validate_project_files(self.root, workspace=workspace)), PROJECT_FILES
                )
                output, error = io.StringIO(), io.StringIO()
                args = ["check-source", str(self.root)]
                if workspace:
                    args.append("--workspace")
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                    code = main(args)
                self.assertEqual(code, 0, error.getvalue())
                expected = {"result": "PASS", "files": len(PROJECT_FILES)}
                if workspace:
                    expected["mode"] = "workspace"
                self.assertEqual(json.loads(output.getvalue()), expected)

    def test_declared_artifacts_pass_only_workspace_mode(self):
        for name in (
            ".venv/pyvenv.cfg",
            ".pytest_cache/state",
            ".ruff_cache/state",
            "__pycache__/module.pyc",
            "src/intelligence_dynamics/__pycache__/module.pyc",
            "tests/__pycache__/module.pyc",
            "tools/__pycache__/prepare_source.pyc",
            "build/lib/generated.py",
            "dist/package.whl",
            "runs/synthetic/data.json",
            "local-config/private.json",
            "local-run/stdout.log",
            "intelligence_dynamics.egg-info/PKG-INFO",
            "src/intelligence_dynamics.egg-info/PKG-INFO",
            ".env",
            ".env.synthetic",
            "jobs.sqlite",
            "jobs.sqlite-journal",
            "jobs.sqlite-wal",
            "jobs.sqlite-shm",
            "demo.jsonl",
            "study.json",
            "dal.jsonl",
            "forager.npz",
            "forager.jsonl",
            "batch-job.json",
        ):
            self.artifact(name)
        self.assertEqual(set(validate_project_files(self.root, workspace=True)), PROJECT_FILES)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_project_files(self.root)

    def test_unknown_source_is_rejected_even_if_added_to_gitignore(self):
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("\nunregistered.py\n")
        for name in ("unregistered.py", "src/intelligence_dynamics/unregistered.py"):
            self.artifact(name)
            for workspace in (False, True):
                with self.subTest(name=name, workspace=workspace):
                    with self.assertRaisesRegex(ValueError, "unexpected files"):
                        validate_project_files(self.root, workspace=workspace)
            (self.root / name).unlink()

    def test_missing_registered_source_is_rejected_in_both_modes(self):
        (self.root / "src/intelligence_dynamics/records.py").unlink()
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                with self.assertRaisesRegex(ValueError, "missing files"):
                    validate_project_files(self.root, workspace=workspace)

    def test_wrong_source_and_artifact_path_types_are_rejected(self):
        (self.root / ".gitignore").unlink()
        (self.root / ".gitignore").mkdir()
        (self.root / "demo.jsonl").mkdir()
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                with self.assertRaisesRegex(ValueError, "missing files: .gitignore"):
                    validate_project_files(self.root, workspace=workspace)
        with self.assertRaises(ValueError):
            validate_project_files(self.project, workspace=1)

    def test_workspace_directory_symlink_is_rejected(self):
        external = self.root.parent / "external"
        external.mkdir()
        create_symlink(self, self.root / ".venv", external, directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            validate_project_files(self.root, workspace=True)

    def test_candidate_copies_only_registered_source_and_preserves_bytes(self):
        self.artifact(".venv/pyvenv.cfg")
        self.artifact("local-config/private.json")
        self.artifact("demo.jsonl")
        result = prepare_source(self.root, self.destination)
        self.assertEqual(result["files"], len(PROJECT_FILES))
        self.assertEqual(result["candidate"], self.destination.name)
        self.assertEqual(set(validate_project_files(self.destination)), PROJECT_FILES)
        for name in PROJECT_FILES:
            self.assertEqual(
                (self.root / name).read_bytes(), (self.destination / name).read_bytes()
            )
        self.assertFalse((self.destination / ".venv").exists())
        self.assertFalse((self.destination / "local-config").exists())
        self.assertFalse((self.destination / "demo.jsonl").exists())

    def test_existing_destination_is_unchanged(self):
        self.destination.mkdir()
        marker = self.destination / "retained"
        marker.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            prepare_source(self.root, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_candidate_must_be_outside_source_with_existing_parent(self):
        for target in (self.root, self.root / "candidate", self.root.parent / "absent/candidate"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    prepare_source(self.root, target)
        self.assertFalse((self.root / "candidate").exists())
        self.assertFalse((self.root.parent / "absent").exists())

    def test_candidate_cannot_traverse_symlink(self):
        external = self.root.parent / "external"
        external.mkdir()
        alias = self.root.parent / "alias"
        create_symlink(self, alias, external, directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            prepare_source(self.root, alias / "candidate")
        self.assertEqual(list(external.iterdir()), [])

    def test_copy_failure_propagates_and_cli_returns_nonzero(self):
        with patch(
            "tools.prepare_source.shutil.copy2", side_effect=OSError("synthetic copy failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic copy failure"):
                prepare_source(self.root, self.destination)
        output, error = io.StringIO(), io.StringIO()
        with (
            patch(
                "tools.prepare_source.prepare_source", side_effect=OSError("synthetic copy failure")
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            code = prepare_main([str(self.destination)])
        self.assertEqual(code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("OSError", error.getvalue())

    def test_changed_candidate_content_cannot_pass(self):
        original = shutil.copy2
        corrupted = False

        def corrupt(source, target):
            nonlocal corrupted
            original(source, target)
            if not corrupted:
                Path(target).write_bytes(b"synthetic corruption")
                corrupted = True

        with patch("tools.prepare_source.shutil.copy2", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "source content changed"):
                prepare_source(self.root, self.destination)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_source_directory_junction_is_rejected(self):
        external = self.root.parent / "external-docs"
        (self.root / "docs").rename(external)
        create_junction(self.root / "docs", external)
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                with self.assertRaisesRegex(ValueError, "reparse points"):
                    validate_project_files(self.root, workspace=workspace)
        with self.assertRaisesRegex(ValueError, "reparse points"):
            prepare_source(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_workspace_junction_is_rejected_before_exclusion(self):
        external = self.root.parent / "external-environment"
        external.mkdir()
        create_junction(self.root / ".venv", external)
        with self.assertRaisesRegex(ValueError, "reparse points"):
            validate_project_files(self.root, workspace=True)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_source_root_junction_is_rejected(self):
        alias = self.root.parent / "source-alias"
        create_junction(alias, self.root)
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                with self.assertRaises(ValueError):
                    validate_project_files(alias, workspace=workspace)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_candidate_cannot_traverse_junction(self):
        external = self.root.parent / "external-candidates"
        external.mkdir()
        alias = self.root.parent / "candidate-alias"
        create_junction(alias, external)
        with self.assertRaisesRegex(ValueError, "reparse points"):
            prepare_source(self.root, alias / "candidate")
        self.assertEqual(list(external.iterdir()), [])
