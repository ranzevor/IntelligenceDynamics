import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import unquote

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from intelligence_dynamics import __version__
from intelligence_dynamics.runtime import (
    Artifact,
    BatchTarget,
    Controller,
    Queue,
    inspect_artifact,
    job_spec,
    run_command,
)
from intelligence_dynamics.runtime.batch import estimate_timeout
from intelligence_dynamics.storage import digest
from intelligence_dynamics.validation import PROJECT_FILES, validate_project_files
from tests.support import create_symlink


def synthetic_image_reference(name):
    return "example.invalid/image@sha256:" + digest({"synthetic_image": name})


class SourcePackageContractTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def copy_project(self, destination):
        for name in PROJECT_FILES:
            source = self.root / name
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def git(self, root, *arguments, check=True):
        self.assertIsNotNone(shutil.which("git"), "Git is required for repository-boundary tests")
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        }
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            env=environment,
            check=check,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def check_modes(self, root):
        for workspace in (False, True):
            with self.subTest(workspace=workspace):
                self.assertEqual(
                    set(validate_project_files(root, workspace=workspace)), PROJECT_FILES
                )

    def test_clean_source_boundary(self):
        self.assertEqual(set(validate_project_files(self.root, workspace=True)), PROJECT_FILES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_project(root)
            self.check_modes(root)
            (root / ".gitignore").rename(root / "unexpected.txt")
            with self.assertRaisesRegex(ValueError, "missing files: .gitignore"):
                validate_project_files(root)
            with self.assertRaisesRegex(ValueError, "unexpected files: unexpected.txt"):
                validate_project_files(root)
            (root / "unexpected.txt").rename(root / ".gitignore")
            (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected files: unexpected.txt"):
                validate_project_files(root)
            (root / "unexpected.txt").unlink()
            (root / "docs" / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "nested .git"):
                validate_project_files(root)

    def test_git_accepted_configuration_forms(self):
        configurations = (
            None,
            '[core]\nrepositoryformatversion = "0"\nbare = "false"\n',
            "",
            "[core]\nrepositoryformatversion = 1\nbare = off\n",
        )
        for configuration in configurations:
            with (
                self.subTest(configuration=configuration),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.copy_project(root)
                self.git(root, "init", "--quiet")
                if configuration is not None:
                    (root / ".git/config").write_text(configuration, encoding="utf-8")
                self.assertEqual(
                    Path(
                        os.fsdecode(self.git(root, "rev-parse", "--show-toplevel").stdout).strip()
                    ).resolve(),
                    root.resolve(),
                )
                self.check_modes(root)

    def test_git_rejected_configuration_forms(self):
        for configuration in (
            "[core]\nrepositoryformatversion = 99\nbare = false\n",
            "[core]\nrepositoryformatversion = 0\nbare = true\n",
            "[core\n",
        ):
            with (
                self.subTest(configuration=configuration),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.copy_project(root)
                self.git(root, "init", "--quiet")
                (root / ".git/config").write_text(configuration, encoding="utf-8")
                self.assertNotEqual(
                    self.git(root, "rev-parse", "--show-toplevel", check=False).returncode, 0
                )
                for workspace in (False, True):
                    with self.assertRaisesRegex(ValueError, "does not identify this project"):
                        validate_project_files(root, workspace=workspace)

    def test_git_linked_worktree_and_separate_management_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            worktree = Path(directory) / "linked-worktree"
            repository.mkdir()
            self.copy_project(repository)
            self.git(repository, "init", "--quiet")
            self.git(repository, "config", "core.autocrlf", "false")
            self.git(repository, "add", "--all")
            self.git(
                repository,
                "-c",
                "user.name=Boundary Test",
                "-c",
                "user.email=boundary@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=disabled-hooks",
                "commit",
                "--quiet",
                "-m",
                "initial",
            )
            self.git(repository, "worktree", "add", "--quiet", "--detach", str(worktree), "HEAD")
            self.check_modes(worktree)
            source = Path(directory) / "separate-source"
            source.mkdir()
            self.copy_project(source)
            self.git(
                source, "init", "--quiet", "--separate-git-dir", str(Path(directory) / "metadata")
            )
            self.check_modes(source)

    def test_git_environment_cannot_redirect_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            other = Path(directory) / "other"
            root.mkdir()
            other.mkdir()
            self.copy_project(root)
            self.git(root, "init", "--quiet")
            self.git(other, "init", "--quiet")
            overrides = {
                "GIT_DIR": str(other / ".git"),
                "GIT_COMMON_DIR": str(other / ".git"),
                "GIT_WORK_TREE": str(other),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.bare",
                "GIT_CONFIG_VALUE_0": "true",
            }
            with patch.dict(os.environ, overrides):
                self.check_modes(root)
                (root / ".git/HEAD").write_text("invalid-head\n", encoding="utf-8")
                for workspace in (False, True):
                    with self.assertRaisesRegex(ValueError, "does not identify this project"):
                        validate_project_files(root, workspace=workspace)

    def test_invalid_git_entry_cannot_fall_back_to_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            ancestor = Path(directory)
            self.git(ancestor, "init", "--quiet")
            root = ancestor / "source"
            root.mkdir()
            self.copy_project(root)
            (root / ".git").mkdir()
            (root / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            for workspace in (False, True):
                with self.assertRaisesRegex(ValueError, "does not identify this project"):
                    validate_project_files(root, workspace=workspace)
            shutil.rmtree(root / ".git")
            (root / ".git").write_text(
                "gitdir: /definitely/not/a/git-management-directory\n", encoding="utf-8"
            )
            for workspace in (False, True):
                with self.assertRaisesRegex(ValueError, "does not identify this project"):
                    validate_project_files(root, workspace=workspace)

    def test_plain_source_does_not_require_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_project(root)
            with patch(
                "intelligence_dynamics.validation.subprocess.run", side_effect=FileNotFoundError
            ):
                self.check_modes(root)
                (root / ".git").mkdir()
                with self.assertRaisesRegex(ValueError, "Git is required"):
                    validate_project_files(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX named pipes")
    def test_root_git_fifo_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_project(root)
            os.mkfifo(root / ".git")
            for workspace in (False, True):
                with self.assertRaisesRegex(ValueError, "clone directory or worktree file"):
                    validate_project_files(root, workspace=workspace)

    def test_root_git_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_project(root)
            create_symlink(self, root / ".git", ".gitignore")
            for workspace in (False, True):
                with self.subTest(workspace=workspace):
                    with self.assertRaisesRegex(ValueError, "symbolic link"):
                        validate_project_files(root, workspace=workspace)

    def test_registered_source_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_project(root)
            (root / ".gitignore").unlink()
            create_symlink(self, root / ".gitignore", "README.md")
            for workspace in (False, True):
                with self.subTest(workspace=workspace):
                    with self.assertRaisesRegex(ValueError, "symbolic links"):
                        validate_project_files(root, workspace=workspace)

    def test_official_index_exact_version_locks(self):
        from tools.check_maintenance import validate_dependencies

        validate_dependencies(self.root)

    def test_local_markdown_links_resolve(self):
        pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        for name in sorted(PROJECT_FILES):
            if not name.endswith(".md"):
                continue
            document = self.root / name
            for raw in pattern.findall(document.read_text(encoding="utf-8")):
                target = raw.strip().split()[0].strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not path_text:
                    continue
                resolved = (document.parent / path_text).resolve()
                self.assertTrue(resolved.is_relative_to(self.root), (document, target))
                self.assertTrue(resolved.exists(), (document, target))

    def test_package_metadata_and_license_contract(self):
        project = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["license"]["text"], "Apache-2.0")
        self.assertEqual(
            set(project["tool"]["setuptools"]["license-files"]),
            {"LICENSE", "LICENSE-NOTICE.md", "THIRD_PARTY_NOTICES.md"},
        )
        version = project["project"]["version"]
        self.assertEqual(project["project"]["name"], "intelligence-dynamics")
        self.assertEqual(__version__, version)
        self.assertEqual(
            project["project"]["scripts"]["intelligence-dynamics"],
            "intelligence_dynamics.cli:main",
        )
        changelog = (self.root / "CHANGELOG.md").read_text(encoding="utf-8")
        versions = re.findall(
            r"^##\s+\[?v?([0-9][0-9A-Za-z.!+_-]*)\]?(?=\s|[（(]|$)",
            changelog,
            flags=re.MULTILINE,
        )
        self.assertTrue(versions)
        self.assertEqual(versions[0], version)


class ArtifactContractTests(unittest.TestCase):
    def test_database_schema_and_error_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.sqlite"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("VACUUM")
            with self.assertRaisesRegex(ValueError, "user tables"):
                inspect_artifact(root, Artifact(path.name, "sqlite"))
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("CREATE TABLE measurements(value REAL)")
            self.assertGreater(
                inspect_artifact(
                    root, Artifact(path.name, "sqlite", required_tables=["measurements"])
                )["bytes"],
                0,
            )
            connection = Mock()
            connection.execute.side_effect = sqlite3.OperationalError("primary-query-error")
            connection.close.side_effect = sqlite3.OperationalError("secondary-close-error")
            with patch(
                "intelligence_dynamics.runtime.local.sqlite3.connect", return_value=connection
            ):
                with self.assertRaisesRegex(sqlite3.OperationalError, "primary-query-error"):
                    inspect_artifact(root, Artifact(path.name, "sqlite"))
            self.assertTrue(connection.close.called)

    def test_npz_explicit_contract(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("arrays dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "trace.npz", rewards=[1.0, 2.0, 3.0, 4.0], biome=[0, 0, 1, 1])
            contract = Artifact(
                "trace.npz",
                "npz",
                required_arrays=["rewards"],
                array_shapes={"rewards": [4]},
                changes={"biome": 1},
            )
            self.assertTrue(inspect_artifact(root, contract))
            for invalid in (
                Artifact("trace.npz", "npz", required_arrays=["missing"]),
                Artifact("trace.npz", "npz", array_shapes={"rewards": [3]}),
                Artifact("trace.npz", "npz", changes={"biome": 0}),
                Artifact("trace.npz", "npz", max_bytes=10),
            ):
                with self.assertRaises(ValueError):
                    inspect_artifact(root, invalid)

    def test_lzma_container(self):
        import lzma

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.xz").write_bytes(lzma.compress(b"synthetic-container"))
            self.assertTrue(inspect_artifact(root, Artifact("state.xz", "lzma")))
            (root / "bad.xz").write_bytes(b"garbage")
            with self.assertRaises(lzma.LZMAError):
                inspect_artifact(root, Artifact("bad.xz", "lzma"))


class ProcessContractTests(unittest.TestCase):
    def test_cleanup_preserves_primary_and_outcome(self):
        process = Mock(pid=1)
        process.poll.return_value = None
        process.wait.side_effect = RuntimeError("primary-wait-error")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("intelligence_dynamics.runtime.local.subprocess.Popen", return_value=process),
            patch("intelligence_dynamics.runtime.local.resident_bytes", return_value=None),
            patch(
                "intelligence_dynamics.runtime.local.process_tree_resident_bytes", return_value=None
            ),
            patch(
                "intelligence_dynamics.runtime.local._terminate",
                side_effect=PermissionError("secondary-cleanup-error"),
            ),
        ):
            root = Path(directory) / "run"
            result = run_command(["synthetic-never-launched"], root, timeout=1)
            self.assertIn("primary-wait-error", result.error)
            self.assertTrue(
                any("secondary-cleanup-error" in item for item in result.cleanup_errors)
            )
            self.assertEqual(json.loads((root / "run.json").read_text())["error"], result.error)

    def test_argument_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            value = "a b;中文"
            result = run_command(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-c",
                    "import sys;print(repr(sys.argv[1:]))",
                    value,
                    "",
                ],
                root,
                timeout=2,
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual(
                (root / "stdout.log").read_text(encoding="utf-8").strip(), repr([value, ""])
            )


class QueueContractTests(unittest.TestCase):
    def provider(self):
        class Provider:
            remote = None
            cancellations = 0

            def get(self, job_id):
                return self.remote

            def create(self, job_id, spec):
                self.remote = {"labels": spec["labels"], "status": {"state": "RUNNING"}}
                return self.remote

            def cancel(self, job_id):
                self.cancellations += 1
                self.remote["status"]["state"] = "CANCELLATION_IN_PROGRESS"

        return Provider()

    def test_async_cancellation_once_and_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Queue(Path(directory) / "queue.sqlite")
            queue.enqueue("r", {"taskGroups": [1]})
            provider = self.provider()
            now = [100.0]
            controller = Controller(
                queue,
                provider,
                resources_present=lambda *a: False,
                artifacts_valid=lambda *a: True,
                clock=lambda: now[0],
                execution_timeout=5,
            )
            self.assertEqual(controller.tick().state, "RUNNING")
            now[0] = 106.0
            self.assertEqual(controller.tick().state, "CANCEL_REQUESTED")
            self.assertEqual(controller.tick().state, "CLEANUP_WAIT")
            self.assertEqual(provider.cancellations, 1)
            provider.remote["status"]["state"] = "CANCELLED"
            self.assertEqual(controller.tick().state, "CANCELLED")

    def test_artifact_exceptions_exhaust_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Queue(Path(directory) / "queue.sqlite")
            queue.enqueue("r", {"taskGroups": [1]})
            provider = self.provider()
            validator = Mock(side_effect=ValueError("synthetic corrupt artifact"))
            controller = Controller(
                queue,
                provider,
                resources_present=lambda *a: False,
                artifacts_valid=validator,
                clock=lambda: 100.0,
            )
            controller.tick()
            provider.remote["status"]["state"] = "SUCCEEDED"
            results = [controller.tick() for _ in range(3)]
            self.assertEqual(
                [item.state for item in results], ["ARTIFACT_WAIT", "ARTIFACT_WAIT", "FAILED"]
            )
            self.assertEqual([item.artifact_attempts for item in results], [1, 2, 3])
            self.assertIsNone(controller.tick())

    def test_prerequisite_success_required(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Queue(Path(directory) / "queue.sqlite")
            queue.enqueue("preflight", {"task": 1})
            queue.enqueue("experiment", {"task": 2}, requires=["preflight"])
            job = queue.claim(now=1)
            self.assertEqual(job.request_id, "preflight")
            self.assertIsNone(queue.claim(now=2))
            queue.transition(job, "FAILED", now=3)
            self.assertIsNone(queue.claim(now=4))
            queue.cancel("experiment")
            self.assertEqual(queue.claim(now=5).request_id, "experiment")
            with self.assertRaises(ValueError):
                queue.enqueue("unknown", {"task": 3}, requires=["absent"])


class BatchContractTests(unittest.TestCase):
    def test_multiple_task_environments(self):
        target = BatchTarget(
            "sample-project",
            "us-central1",
            "worker@sample-project.iam.gserviceaccount.com",
            "sample-bucket",
        )
        result = job_spec(
            target,
            image=synthetic_image_reference("multiple-task-environments"),
            arguments=["python"],
            task_count=2,
            parallelism=1,
            task_environments=[{"SEED": "1"}, {"SEED": "2"}],
        )
        group = result["taskGroups"][0]
        self.assertEqual(group["taskCount"], "2")
        self.assertEqual(group["taskEnvironments"][1]["variables"]["SEED"], "2")
        with self.assertRaises(ValueError):
            job_spec(
                target,
                image=synthetic_image_reference("invalid-environment-count"),
                arguments=["python"],
                task_count=2,
                task_environments=[{}],
            )

    def test_timeout_estimate(self):
        self.assertEqual(
            estimate_timeout(
                observed_seconds=10, observed_units=5, target_units=100, slack=2, grace_seconds=10
            ),
            410,
        )
        self.assertEqual(
            estimate_timeout(observed_seconds=10000, observed_units=1, target_units=100), 604800
        )
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                estimate_timeout(observed_seconds=value, observed_units=1, target_units=1)


class RunOutputTests(unittest.TestCase):
    def test_request_records_runtime_and_result_identifies_declared_artifacts(self):
        import platform
        from dataclasses import asdict

        from intelligence_dynamics.storage import file_digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    'from pathlib import Path;Path("value.json").write_text(\'{"answer":42}\')',
                ],
                root,
                timeout=2,
                artifacts=[Artifact("value.json", "json", expected_fields={"answer": 42})],
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {"request.json", "run.json", "stdout.log", "stderr.log", "value.json"},
            )
            request = json.loads((root / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(
                request["runtime"],
                {
                    "package_version": __version__,
                    "python_version": platform.python_version(),
                    "platform": f"{platform.system()} {platform.machine()}",
                },
            )
            self.assertEqual(result.artifacts[0]["sha256"], file_digest(root / "value.json"))
            self.assertEqual(result.artifacts[0]["bytes"], (root / "value.json").stat().st_size)
            self.assertEqual(
                json.loads((root / "run.json").read_text(encoding="utf-8")),
                json.loads(json.dumps(asdict(result))),
            )

    def test_undeclared_working_files_do_not_expand_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    'from pathlib import Path; Path("result.json").write_text(\'{"answer":42}\');'
                    '[Path(name).write_text("not-json") for name in '
                    '("execution.json", "provenance.json", "output-manifest.json")]',
                ],
                root,
                timeout=2,
                artifacts=[Artifact("result.json", "json", expected_fields={"answer": 42})],
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual([item["path"] for item in result.artifacts], ["result.json"])

    def test_undeclared_symlink_does_not_change_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_symlink(self, root / "capability-probe", "absent-target")
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    'from pathlib import Path; Path("result.json").write_text(\'{"answer":42}\');'
                    'Path("result-link.json").symlink_to("result.json")',
                ],
                root / "run",
                timeout=2,
                artifacts=[Artifact("result.json", "json", expected_fields={"answer": 42})],
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertTrue((root / "run/result-link.json").is_symlink())
            self.assertEqual(
                json.loads((root / "run/run.json").read_text(encoding="utf-8"))["status"],
                "SUCCEEDED",
            )

    def test_declared_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_symlink(self, root / "capability-probe", "absent-target")
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    'from pathlib import Path; Path("result.json").write_text(\'{"answer":42}\');'
                    'Path("result-link.json").symlink_to("result.json")',
                ],
                root / "run",
                timeout=2,
                artifacts=[Artifact("result-link.json", "json")],
            )
            self.assertEqual(result.status, "FAILED")
            self.assertIn("symbolic links", result.error)
            self.assertEqual(result.returncode, 0)

    def test_result_persistence_error_retains_completed_outcome(self):
        import intelligence_dynamics.runtime.local as local

        original = local.write_json
        failure = OSError("synthetic result storage failure")

        def failing(path, value, **kwargs):
            if path.name == "run.json":
                raise failure
            return original(path, value, **kwargs)

        for code, expected in ((0, "SUCCEEDED"), (3, "FAILED")):
            with (
                self.subTest(code=code),
                tempfile.TemporaryDirectory() as directory,
                patch.object(local, "write_json", side_effect=failing),
            ):
                root = Path(directory) / "run"
                with self.assertRaises(local.RunPersistenceError) as raised:
                    run_command(
                        [sys.executable, "-c", f"import sys;print('completed');sys.exit({code})"],
                        root,
                        timeout=2,
                    )
                self.assertEqual(raised.exception.result.status, expected)
                self.assertEqual(raised.exception.result.returncode, code)
                self.assertIs(raised.exception.persistence_error, failure)
                self.assertEqual((root / "stdout.log").read_text(encoding="utf-8"), "completed\n")


class ProviderCancellationTests(unittest.TestCase):
    def test_cancel_uses_cancel_operation_not_deletion(self):
        from intelligence_dynamics.runtime import BatchClient

        target = BatchTarget(
            "sample-project",
            "us-central1",
            "worker@sample-project.iam.gserviceaccount.com",
            "sample-bucket",
        )
        transport = Mock(return_value={})
        client = BatchClient(target, lambda: "synthetic-token", transport=transport)
        client.cancel("sample-job")
        method, url, headers, body = transport.call_args.args
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/jobs/sample-job:cancel"))
        self.assertIn("requestId", json.loads(body))

    def test_remote_cancelled_and_success_race(self):
        for state, expected in [
            ("CANCELLED", "CANCELLED"),
            ("SUCCEEDED", "SUCCEEDED"),
            ("FAILED", "FAILED"),
        ]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                queue = Queue(Path(directory) / "queue.sqlite")
                queue.enqueue("r", {"taskGroups": [1]})
                provider = QueueContractTests().provider()
                controller = Controller(
                    queue,
                    provider,
                    resources_present=lambda *a: False,
                    artifacts_valid=lambda *a: True,
                    clock=lambda: 100.0,
                )
                controller.tick()
                queue.cancel("r")
                provider.remote["status"]["state"] = state
                self.assertEqual(controller.tick().state, expected)
                self.assertEqual(provider.cancellations, 0)


class MetadataContractTests(unittest.TestCase):
    def test_checkpoint_metadata_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint.json").write_text('{"step":12,"state_file":"state.npz"}')
            self.assertTrue(
                inspect_artifact(
                    root, Artifact("checkpoint.json", "json", expected_fields={"step": 12})
                )
            )
            for expected in (13, True, "12"):
                with self.assertRaises(ValueError):
                    inspect_artifact(
                        root,
                        Artifact("checkpoint.json", "json", expected_fields={"step": expected}),
                    )
            with self.assertRaises(ValueError):
                Artifact("state.npz", "npz", expected_fields={"step": 12})


class QueueConnectionTests(unittest.TestCase):
    def test_close_failure_does_not_mask_transaction_error(self):
        from unittest.mock import MagicMock

        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__exit__.return_value = False
        connection.close.side_effect = sqlite3.OperationalError("secondary-close-error")
        queue = Queue.__new__(Queue)
        queue.path = Path("synthetic.sqlite")
        with patch("intelligence_dynamics.runtime.queue.sqlite3.connect", return_value=connection):
            with self.assertRaisesRegex(sqlite3.OperationalError, "primary-transaction-error"):
                with queue._connect():
                    raise sqlite3.OperationalError("primary-transaction-error")
        self.assertTrue(connection.close.called)


class EnvironmentIdentityTests(unittest.TestCase):
    def test_override_identity_without_plaintext_values(self):
        from intelligence_dynamics.storage import digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {"EXPERIMENT_MODE": "synthetic-observation-value"}
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    'import os; assert os.environ["EXPERIMENT_MODE"].startswith("synthetic")',
                ],
                root / "first",
                timeout=2,
                environment=values,
            )
            self.assertEqual(result.status, "SUCCEEDED")
            request = json.loads((root / "first/request.json").read_text())
            self.assertEqual(request["environment_override_keys"], ["EXPERIMENT_MODE"])
            self.assertEqual(request["environment_overrides_sha256"], digest(values))
            self.assertNotIn(
                "synthetic-observation-value", (root / "first/request.json").read_text()
            )
            run_command(
                [sys.executable, "-c", "pass"],
                root / "second",
                timeout=2,
                environment={"EXPERIMENT_MODE": "another-value"},
            )
            second = json.loads((root / "second/request.json").read_text())
            self.assertNotEqual(
                request["environment_overrides_sha256"], second["environment_overrides_sha256"]
            )
