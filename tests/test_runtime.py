import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from intelligence_dynamics.runtime import (
    Artifact,
    BatchClient,
    BatchTarget,
    Controller,
    ProviderError,
    Queue,
    bucket_name,
    inspect_artifact,
    job_spec,
    run_command,
)
from intelligence_dynamics.storage import digest, loads
from tests.support import create_junction, create_symlink


def synthetic_image_reference(name):
    return "registry.example/model@sha256:" + digest({"synthetic_image": name})


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_success(self):
        output = self.root / "run"
        result = run_command(
            [
                sys.executable,
                "-c",
                'from pathlib import Path; Path("data.json").write_text("{\\"value\\":3}"); print("done")',
            ],
            output,
            timeout=5,
            artifacts=[Artifact("data.json", "json")],
        )
        self.assertEqual(result.status, "SUCCEEDED")
        self.assertEqual(len(result.artifacts), 1)
        self.assertEqual((output / "stdout.log").read_text().strip(), "done")
        with self.assertRaises(FileExistsError):
            run_command([sys.executable, "-V"], output, timeout=5)

    def test_failure_missing_artifact_and_timeout_preserved(self):
        for name, command, timeout, artifacts, expected in [
            (
                "badexit",
                [sys.executable, "-c", 'import sys; print("failure",file=sys.stderr);sys.exit(2)'],
                5,
                [],
                "FAILED",
            ),
            ("missing", [sys.executable, "-c", "pass"], 5, [Artifact("missing")], "FAILED"),
            (
                "timeout",
                [sys.executable, "-c", "import time;time.sleep(10)"],
                0.03,
                [],
                "TIMED_OUT",
            ),
            ("spawn", ["/does-not-exist"], 5, [], "FAILED"),
        ]:
            with self.subTest(name=name):
                result = run_command(
                    command, self.root / name, timeout=timeout, artifacts=artifacts
                )
                self.assertEqual(result.status, expected)
                self.assertEqual(
                    loads((self.root / name / "run.json").read_bytes())["status"], expected
                )
                self.assertTrue((self.root / name / "request.json").exists())

    def test_artifact_paths(self):
        for path in ("../x", "/x", "a//x", "a/./x", "C:x", "a\\x"):
            with self.assertRaises(ValueError):
                Artifact(path)
        with self.assertRaises(ValueError):
            inspect_artifact(self.root, Artifact("missing"))
        (self.root / "data").write_bytes(b"a")
        with self.assertRaises(ValueError):
            inspect_artifact(
                self.root, Artifact("data", sha256=digest({"synthetic_mismatch": "data"}))
            )
        with self.assertRaises(ValueError):
            run_command("echo hi", self.root / "x", timeout=1)

    def test_artifact_symlink_is_rejected(self):
        (self.root / "data").write_bytes(b"a")
        create_symlink(self, self.root / "link", self.root / "data")
        with self.assertRaises(ValueError):
            inspect_artifact(self.root, Artifact("link"))

    def test_artifact_snapshot_identity_and_unicode_path(self):
        folder = self.root / "测量结果"
        folder.mkdir()
        path = folder / "结果.json"
        data = b'{"status":"done"}\n'
        path.write_bytes(data)
        checksum = hashlib.sha256(data).hexdigest()
        declaration = Artifact(
            "测量结果/结果.json",
            "json",
            sha256=checksum,
            expected_fields={"status": "done"},
            max_bytes=len(data),
        )
        result = inspect_artifact(self.root, declaration)
        self.assertEqual(result["sha256"], checksum)
        self.assertEqual(result["bytes"], len(data))
        with self.assertRaises(ValueError):
            inspect_artifact(self.root, Artifact(declaration.path, max_bytes=len(data) - 1))
        path.write_bytes(b'{"status":"fail"}\n')
        with self.assertRaisesRegex(ValueError, "fields differ"):
            inspect_artifact(self.root, declaration)

    def test_artifact_replacement_after_parse_is_rejected(self):
        from intelligence_dynamics.runtime import local

        path = self.root / "result.json"
        data = b'{"status":"done"}\n'
        replacement = b'{"status":"fail"}\n'
        self.assertEqual(len(data), len(replacement))
        path.write_bytes(data)
        before = path.stat()
        parse = local.loads

        def change_after_parse(body):
            value = parse(body)
            path.write_bytes(replacement)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return value

        with patch.object(local, "loads", side_effect=change_after_parse):
            with self.assertRaisesRegex(ValueError, "changed during inspection"):
                inspect_artifact(
                    self.root, Artifact("result.json", "json", expected_fields={"status": "done"})
                )
        self.assertEqual(path.read_bytes(), replacement)

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_artifact_directory_junction_is_rejected(self):
        external = self.root / "external"
        external.mkdir()
        (external / "result.json").write_bytes(b'{"status":"done"}\n')
        run = self.root / "run"
        run.mkdir()
        create_junction(run / "bridge", external)
        with self.assertRaisesRegex(ValueError, "reparse points"):
            inspect_artifact(run, Artifact("bridge/result.json", "json"))

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_artifact_root_junction_is_rejected(self):
        run = self.root / "run"
        run.mkdir()
        (run / "result.json").write_bytes(b'{"status":"done"}\n')
        alias = self.root / "run-alias"
        create_junction(alias, run)
        with self.assertRaisesRegex(ValueError, "reparse points"):
            inspect_artifact(alias, Artifact("result.json", "json"))

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_writer_paths_reject_junctions_before_side_effects(self):
        from intelligence_dynamics.native.checkpoint import CheckpointCodec, CheckpointStore
        from intelligence_dynamics.storage import write_json

        external = self.root / "external"
        external.mkdir()
        alias = self.root / "alias"
        create_junction(alias, external)
        codec = CheckpointCodec("synthetic", lambda value: b"", lambda value: None)
        operations = (
            lambda: write_json(alias / "result.json", {"value": 1}),
            lambda: Queue(alias / "jobs.sqlite"),
            lambda: CheckpointStore(alias, codec),
            lambda: run_command([sys.executable, "-c", "pass"], alias / "run", timeout=2),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "reparse points"):
                    operation()
                self.assertEqual(list(external.iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Windows directory junctions")
    def test_generated_junction_only_affects_declared_artifacts(self):
        external = self.root / "external"
        external.mkdir()
        private = external / "private.json"
        private.write_bytes(b"{}")
        for declared, expected in ((False, "SUCCEEDED"), (True, "FAILED")):
            with self.subTest(declared=declared):
                result = run_command(
                    [
                        sys.executable,
                        "-c",
                        "import _winapi,sys; _winapi.CreateJunction(sys.argv[1], 'bridge')",
                        str(external),
                    ],
                    self.root / f"run-{declared}",
                    timeout=5,
                    artifacts=[Artifact("bridge/private.json", "json")] if declared else [],
                )
                self.assertEqual(result.status, expected)
                if declared:
                    self.assertIn("reparse points", result.error)
                self.assertEqual(private.read_bytes(), b"{}")

    def test_sqlite_artifact_requires_checkpointed_wal(self):
        path = self.root / "result.sqlite"
        declaration = Artifact(path.name, "sqlite", required_tables=("measurements",))
        with closing(sqlite3.connect(path)) as database:
            self.assertEqual(database.execute("PRAGMA journal_mode=WAL").fetchone(), ("wal",))
            with database:
                database.execute("CREATE TABLE measurements (value INTEGER)")
                database.execute("INSERT INTO measurements VALUES (3)")
            self.assertGreater(path.with_name(path.name + "-wal").stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "checkpointed data"):
                inspect_artifact(self.root, declaration)
            self.assertEqual(database.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0], 0)
            result = inspect_artifact(self.root, declaration)
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


class BatchTests(unittest.TestCase):
    def target(self, **values):
        base = dict(
            project="sample-project",
            region="us-central1",
            service_account="worker@sample-project.iam.gserviceaccount.com",
            bucket="sample-bucket",
        )
        return BatchTarget(**(base | values))

    def test_bucket_boundaries(self):
        for value in (
            "a" * 64,
            "a" * 64 + ".abc",
            "192.168.5.4",
            "goog-review",
            "my_google_bucket",
            "my_g00gle_bucket",
            "a..b",
            ".abc",
            "a_b_",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bucket_name(value)
        for value in ("abc", "a" * 63, "a" * 63 + "." + "b" * 63, "a_b-c"):
            self.assertEqual(bucket_name(value), value)

    def test_account_and_spec(self):
        for length in (5, 31):
            with self.assertRaises(ValueError):
                self.target(
                    service_account="a" * length + "@sample-project.iam.gserviceaccount.com"
                )
        for length in (6, 30):
            self.target(service_account="a" * length + "@sample-project.iam.gserviceaccount.com")
        spec = job_spec(
            self.target(),
            image=synthetic_image_reference("account-and-spec"),
            arguments=["python", "-V"],
        )
        self.assertEqual(spec["taskGroups"][0]["taskSpec"]["maxRetryCount"], 0)
        with self.assertRaises(ValueError):
            job_spec(self.target(), image="model:latest", arguments=["python"])

    def test_transport_not_called_before_validation(self):
        transport = Mock(return_value={})
        token = Mock(return_value="synthetic-token")
        client = BatchClient(self.target(), token, transport=transport)
        self.assertFalse(token.called)
        with self.assertRaises(ValueError):
            client.get("../other")
        self.assertFalse(token.called)
        client.create("valid-job", {"taskGroups": []})
        args = transport.call_args.args
        self.assertTrue(
            args[1].startswith(
                "https://batch.googleapis.com/v1/projects/sample-project/locations/us-central1/jobs?"
            )
        )
        self.assertIn("requestId=", args[1])
        self.assertEqual(args[2]["Authorization"], "Bearer synthetic-token")
        transport.side_effect = ProviderError(404, "missing")
        self.assertIsNone(client.get("valid-job"))
        transport.side_effect = ProviderError(403, "denied")
        with self.assertRaises(ProviderError):
            client.get("valid-job")


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.queue = Queue(Path(self.temp.name) / "jobs.sqlite")
        self.now = 100.0

    def tearDown(self):
        self.temp.cleanup()

    def test_idempotency_and_fencing(self):
        first = self.queue.enqueue("request", {"value": 1})
        self.assertEqual(first, self.queue.enqueue("request", {"value": 1}))
        with self.assertRaises(ValueError):
            self.queue.enqueue("request", {"value": 2})
        old = self.queue.claim(now=100, lease_seconds=5)
        self.assertIsNone(self.queue.claim(now=101))
        new = self.queue.claim(now=106)
        with self.assertRaises(RuntimeError):
            self.queue.transition(old, "RUNNING", now=107)
        self.queue.transition(new, "RUNNING", now=107)
        self.assertEqual(self.queue.get("request").state, "RUNNING")

    def controller(self, provider, *, resources=False, artifacts=True):
        return Controller(
            self.queue,
            provider,
            resources_present=lambda *args: resources,
            artifacts_valid=lambda *args: artifacts,
            clock=lambda: self.now,
        )

    def provider(self):
        class Provider:
            remote = None
            creates = 0
            deletes = 0
            timeout_after_create = False

            def get(self, job_id):
                return self.remote

            def create(self, job_id, spec):
                self.creates += 1
                self.remote = {"labels": spec["labels"], "status": {"state": "RUNNING"}}
                if self.timeout_after_create:
                    raise TimeoutError("synthetic timeout")
                return self.remote

            def cancel(self, job_id):
                self.deletes += 1
                self.remote["status"]["state"] = "CANCELLED"

        return Provider()

    def test_unknown_submission_reconciles_without_duplicate(self):
        self.queue.enqueue("r", {"taskGroups": [1]})
        provider = self.provider()
        provider.timeout_after_create = True
        controller = self.controller(provider)
        self.assertEqual(controller.tick().state, "SUBMITTING")
        self.assertEqual(controller.tick().state, "RUNNING")
        self.assertEqual(provider.creates, 1)
        self.assertTrue(any(row["state"] == "SUBMITTING" for row in self.queue.history("r")))

    def test_cleanup_and_artifact_gates(self):
        self.queue.enqueue("r", {"taskGroups": [1]})
        provider = self.provider()
        self.controller(provider).tick()
        provider.remote["status"]["state"] = "SUCCEEDED"
        self.assertEqual(self.controller(provider, resources=True).tick().state, "CLEANUP_WAIT")
        self.assertEqual(self.controller(provider, artifacts=False).tick().state, "ARTIFACT_WAIT")
        self.assertEqual(self.controller(provider).tick().state, "SUCCEEDED")

    def test_artifact_failure_and_collision(self):
        self.queue.enqueue("r", {"taskGroups": [1]})
        provider = self.provider()
        self.controller(provider).tick()
        provider.remote["status"]["state"] = "SUCCEEDED"
        controller = self.controller(provider, artifacts=False)
        self.assertEqual(
            [controller.tick().state for _ in range(3)],
            ["ARTIFACT_WAIT", "ARTIFACT_WAIT", "FAILED"],
        )
        self.queue.enqueue("r2", {"taskGroups": [2]})
        self.assertEqual(self.controller(provider).tick().state, "FAILED")

    def test_cancel_pre_and_post_submission(self):
        self.queue.enqueue("r", {"taskGroups": [1]})
        self.queue.cancel("r")
        provider = self.provider()
        self.assertEqual(self.controller(provider).tick().state, "CANCELLED")
        self.assertEqual(provider.creates, 0)
        self.queue.enqueue("r2", {"taskGroups": [1]})
        self.controller(provider).tick()
        self.queue.cancel("r2")
        self.assertEqual(self.controller(provider).tick().state, "CANCEL_REQUESTED")
        self.assertEqual(self.controller(provider, resources=True).tick().state, "CLEANUP_WAIT")
        self.assertEqual(self.controller(provider).tick().state, "CANCELLED")


class ResourceTests(unittest.TestCase):
    def test_current_process_memory(self):
        import os

        from intelligence_dynamics.runtime.resources import resident_bytes

        value = resident_bytes(os.getpid())
        self.assertTrue(value is None or value > 0)
        with self.assertRaises(ValueError):
            resident_bytes(True)
        self.assertIsNone(resident_bytes(2147483647))

    def test_memory_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_command(
                [sys.executable, "-c", "import time; x=bytearray(1000000); time.sleep(.1)"],
                Path(directory) / "run",
                timeout=2,
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertTrue(
                result.peak_sampled_rss_bytes is None or result.peak_sampled_rss_bytes > 0
            )


class InspectionTests(unittest.TestCase):
    def setUp(self):
        from intelligence_dynamics.runtime import GoogleInspector

        self.target = BatchTarget(
            "sample-project",
            "us-central1",
            "worker@sample-project.iam.gserviceaccount.com",
            "sample-bucket",
        )
        self.transport = Mock()
        self.inspector = GoogleInspector(
            self.target, lambda: "synthetic-token", transport=self.transport
        )
        self.job = Mock(provider_id="id-synthetic")

    def test_empty_then_exact_resource(self):
        self.transport.side_effect = [
            {"items": {}, "nextPageToken": "next"},
            {"items": {"zone": {"instances": [{"labels": {"id-job": "id-synthetic"}}]}}},
        ]
        self.assertTrue(self.inspector.resources_present(self.job))
        self.assertEqual(self.transport.call_count, 2)
        self.assertIn("pageToken=next", self.transport.call_args.args[1])

    def test_invalid_cycle_or_failed_query_not_clean(self):
        for token in ([], 2, " "):
            self.transport.side_effect = None
            self.transport.return_value = {"nextPageToken": token}
            with self.assertRaises(ValueError):
                self.inspector.resources_present(self.job)
        self.transport.side_effect = [{"nextPageToken": "cycle"}, {"nextPageToken": "cycle"}]
        with self.assertRaises(ValueError):
            self.inspector.resources_present(self.job)
        self.transport.side_effect = ProviderError(403, "denied")
        with self.assertRaises(ProviderError):
            self.inspector.resources_present(self.job)

    def test_other_label_and_missing_object(self):
        self.transport.return_value = {
            "items": {"zone": {"instances": [{"labels": {"id-job": "other"}}]}}
        }
        self.assertFalse(self.inspector.resources_present(self.job))
        self.transport.side_effect = ProviderError(404, "missing")
        self.assertIsNone(self.inspector.json_artifact("run/summary.json"))
        self.transport.side_effect = None
        self.transport.return_value = {"score": 0.5}
        self.assertEqual(
            self.inspector.json_artifact(
                "run/a b.json", expected_digest=digest({"score": 0.5}), required_fields=["score"]
            ),
            {"score": 0.5},
        )
        self.assertIn("run%2Fa%20b.json", self.transport.call_args.args[1])
        with self.assertRaises(ValueError):
            self.inspector.json_artifact("summary.json", required_fields=["missing"])
        with self.assertRaises(ValueError):
            self.inspector.json_artifact(
                "summary.json", expected_digest=digest({"synthetic_mismatch": "summary"})
            )


class ProcessTreeTests(unittest.TestCase):
    def test_family_excludes_unrelated_and_handles_cycles(self):
        from intelligence_dynamics.runtime.resources import descendant_ids

        self.assertEqual(descendant_ids({1: 0, 2: 1, 3: 2, 4: 0}, 1), (1, 2, 3))
        self.assertEqual(descendant_ids({1: 2, 2: 1}, 1), (1, 2))

    def test_observation_error_does_not_change_exit(self):
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "intelligence_dynamics.runtime.local.process_tree_resident_bytes",
                side_effect=PermissionError("synthetic observation denied"),
            ),
        ):
            result = run_command(
                [sys.executable, "-c", "import time;time.sleep(.1)"],
                Path(directory) / "run",
                timeout=2,
            )
            self.assertEqual(result.status, "SUCCEEDED")
            self.assertEqual(len(result.memory_errors), 1)
            self.assertIsNone(result.peak_sampled_tree_rss_bytes)
