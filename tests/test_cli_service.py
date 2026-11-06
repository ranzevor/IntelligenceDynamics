import contextlib
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from intelligence_dynamics import cli
from intelligence_dynamics.cli import demo_records, main, validate_stream
from intelligence_dynamics.runtime import Application, Queue
from intelligence_dynamics.storage import (
    append_records,
    canonical_bytes,
    digest,
    read_records,
    write_records,
)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            result = main(list(args))
        return result, output.getvalue(), errors.getvalue()

    def test_demo_validate_summary_refuse_overwrite(self):
        project = Path(__file__).resolve().parents[1]
        from intelligence_dynamics.validation import PROJECT_FILES

        code, output, error = self.invoke("check-source", "--workspace", str(project))
        self.assertEqual(code, 0, error)
        self.assertEqual(
            json.loads(output), {"files": len(PROJECT_FILES), "result": "PASS", "mode": "workspace"}
        )
        path = self.root / "demo.jsonl"
        self.assertEqual(self.invoke("demo", str(path))[0], 0)
        before = path.read_bytes()
        self.assertEqual(self.invoke("validate", str(path))[0], 0)
        self.assertEqual(self.invoke("summarize", str(path))[0], 0)
        self.assertEqual(self.invoke("demo", str(path))[0], 2)
        self.assertEqual(path.read_bytes(), before)

    def test_demo_updates_use_environment_steps(self):
        from intelligence_dynamics.research import (
            LearningRule,
            TabularPolicy,
            TaskConfig,
            run_episode,
        )

        for seed in (0, 1, 7, 23):
            with self.subTest(seed=seed):
                episode = run_episode(
                    TaskConfig(steps=48, cue_count=4, delay=1, switches=(16, 32), seed=seed),
                    TabularPolicy(LearningRule(priors=(0.0,) * 4), mode="buffered"),
                    seed=seed,
                )
                expected = {row.step: len(row.committed_ids) for row in episode.deliveries}
                actual = [row.computation.learning_updates for row in list(demo_records(seed))[1:]]
                self.assertEqual(actual, [expected.get(step, 0) for step in range(48)])
                self.assertEqual(sum(actual), len(episode.updates))
                self.assertEqual(actual[:2], [0, 0])

    def test_output_messages_do_not_include_private_parent_directories(self):
        import json

        destination = self.root / "demo.jsonl"
        code, output, error = self.invoke("demo", str(destination))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["output"], "demo.jsonl")
        self.assertNotIn(str(self.root), output + error)
        code, output, error = self.invoke("validate", str(self.root / "missing.jsonl"))
        self.assertEqual(code, 2)
        self.assertIn("FileNotFoundError", error)
        self.assertNotIn(str(self.root), output + error)

    def test_stream_run_order(self):
        records = list(demo_records())
        self.assertEqual(validate_stream(records)["records"], 49)
        for bad in (records[1:], records + records, [records[0], records[2], records[1]]):
            with self.assertRaises(ValueError):
                validate_stream(bad)

    def test_append_and_failed_append(self):
        records = list(demo_records())
        path = self.root / "data.jsonl"
        write_records(path, records[:3])
        self.assertEqual(append_records(path, records[3:]), 46)
        self.assertEqual(list(read_records(path)), records)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            append_records(path, [object()])
        self.assertEqual(path.read_bytes(), before)

    def test_bad_configuration_does_not_write(self):
        config = self.root / "study.json"
        config.write_text('{"train_seed":1,"heldout_seed":1}')
        path = self.root / "study-result.json"
        self.assertEqual(self.invoke("study", str(config), str(path))[0], 2)
        self.assertFalse(path.exists())

    def conversion_fixture(self, kind, *, compressed=False):
        config = {
            "metadata": {
                "run_id": "synthetic",
                "track": "B",
                "system_name": "synthetic",
                "environment_name": "synthetic",
                "seed": 7,
                "source_revision": "synthetic",
            }
        }
        if kind == "dal-csv":
            original = b"0,0.1,0.1,0.05,0.2,0.3,0.01,0.1,0.4,None,None,0.0,0.1\n"
            replacement = original.replace(b"0,0.1,", b"0,0.9,", 1)
            config["scale"] = {"task_instances": 8, "trials": 5, "trial_steps": 5}
            return original, replacement, config, 0.1, "dal_rows"
        import numpy as np

        def archive(reward):
            stream = io.BytesIO()
            arrays = {"rewards": np.array([reward, 2.5], dtype=np.float64)}
            if kind == "forager-npz":
                arrays["total_loss"] = np.array([0.25])
            save = np.savez_compressed if compressed else np.savez
            save(stream, **arrays)
            return stream.getvalue()

        if kind == "forager-npz":
            config["expected_rollout_steps"] = 2
            expected, reader = 1.875, "forager_npz_records"
        else:
            config["interval"] = 1
            expected, reader = 1.25, "npz_records"
        return archive(1.25), archive(9.75), config, expected, reader

    def check_conversion_identity(self, kind, *, compressed=False):
        original, replacement, config, expected, reader = self.conversion_fixture(
            kind, compressed=compressed
        )
        if kind == "dal-csv" and compressed:
            original = original.replace(b"\n", b"\r\n")
        for mutate in (False, True) if not compressed else (False,):
            with self.subTest(format=kind, mutate=mutate, compressed=compressed):
                source = self.root / f"{kind}-{mutate}-{compressed}.input"
                configuration = self.root / f"{kind}-{mutate}-{compressed}.json"
                output = self.root / f"{kind}-{mutate}-{compressed}.jsonl"
                source.write_bytes(original)
                configuration.write_text(json.dumps(config, indent=2), encoding="utf-8")
                before = source.stat()
                actual_reader = getattr(cli, reader)
                injected = []

                def read_after_replacement(*args, **kwargs):
                    if mutate:
                        self.assertEqual(len(original), len(replacement))
                        source.write_bytes(replacement)
                        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                        injected.append(source.read_bytes())
                    return actual_reader(*args, **kwargs)

                with patch.object(cli, reader, side_effect=read_after_replacement) as call:
                    code, _, errors = self.invoke(
                        "convert", kind, str(source), str(configuration), str(output)
                    )
                self.assertEqual(code, 0, errors)
                call.assert_called_once()
                self.assertEqual(injected, [replacement] if mutate else [])
                records = list(read_records(output))
                self.assertEqual(
                    records[0].notes["native_sha256"], hashlib.sha256(original).hexdigest()
                )
                self.assertEqual(records[0].config_digest, digest(config))
                self.assertEqual(records[1].capability, expected)
                self.assertEqual(source.read_bytes(), replacement if mutate else original)
                self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)
                self.assertEqual(self.invoke("validate", str(output))[0], 0)

    def test_dal_conversion_binds_digest_to_parsed_bytes(self):
        self.check_conversion_identity("dal-csv")

    def test_dal_conversion_accepts_crlf_and_formatted_configuration(self):
        self.check_conversion_identity("dal-csv", compressed=True)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy arrays dependency")
    def test_array_conversion_binds_digest_and_accepts_compressed_container(self):
        self.check_conversion_identity("array-npz")
        self.check_conversion_identity("array-npz", compressed=True)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy arrays dependency")
    def test_forager_conversion_binds_digest_and_accepts_compressed_container(self):
        self.check_conversion_identity("forager-npz")
        self.check_conversion_identity("forager-npz", compressed=True)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy arrays dependency")
    def test_invalid_npz_reports_cause_and_cleans_snapshot(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("rewards", b"not an array")
            archive.writestr("total_loss", b"not an array")
        cases = (
            (b"invalid-native-input", "valid.*archive"),
            (b"PK\x03\x04broken-archive", "valid.*archive"),
            (stream.getvalue(), "members.*NumPy arrays"),
        )
        temporary_directory = tempfile.TemporaryDirectory
        snapshots = []

        def capture_directory(*args, **kwargs):
            temporary = temporary_directory(*args, **kwargs)
            snapshots.append(Path(temporary.name))
            return temporary

        with patch.object(cli.tempfile, "TemporaryDirectory", side_effect=capture_directory):
            for kind in ("array-npz", "forager-npz"):
                _, _, config, _, _ = self.conversion_fixture(kind)
                source, configuration, output = (
                    self.root / f"{kind}.{extension}" for extension in ("npz", "json", "jsonl")
                )
                configuration.write_text(json.dumps(config), encoding="utf-8")
                for native, reason in cases:
                    with self.subTest(format=kind, reason=reason):
                        source.write_bytes(native)
                        code, _, errors = self.invoke(
                            "convert", kind, str(source), str(configuration), str(output)
                        )
                        self.assertEqual(code, 2)
                        self.assertRegex(errors, "ValueError: NPZ .*" + reason)
                        self.assertNotIn(str(self.root), errors)
                        self.assertFalse(output.exists())
                        self.assertEqual(source.read_bytes(), native)
                        self.assertFalse(snapshots[-1].exists())
        self.assertEqual(len(snapshots), 6)

    def test_conversion_failure_cleans_snapshot_and_preserves_input_and_output(self):
        original, _, config, _, _ = self.conversion_fixture("dal-csv")
        source, configuration, output = (
            self.root / name for name in ("native.csv", "config.json", "result.jsonl")
        )
        configuration.write_text(json.dumps(config), encoding="utf-8")
        snapshots = []
        temporary_directory = tempfile.TemporaryDirectory

        def capture_directory(*args, **kwargs):
            temporary = temporary_directory(*args, **kwargs)
            snapshots.append(Path(temporary.name))
            return temporary

        with patch.object(cli.tempfile, "TemporaryDirectory", side_effect=capture_directory):
            source.write_bytes(b"invalid,row\n")
            self.assertEqual(
                self.invoke("convert", "dal-csv", str(source), str(configuration), str(output))[0],
                2,
            )
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), b"invalid,row\n")
            source.write_bytes(original)
            output.write_bytes(b"existing output\n")
            for destination in (output, source):
                self.assertEqual(
                    self.invoke(
                        "convert", "dal-csv", str(source), str(configuration), str(destination)
                    )[0],
                    2,
                )
            self.assertEqual(output.read_bytes(), b"existing output\n")
            self.assertEqual(source.read_bytes(), original)
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(not path.exists() for path in snapshots))

    def test_batch_template_requires_runtime_image_identity(self):
        template_path = Path(__file__).resolve().parents[1] / "configs/batch-job.json"
        template = json.loads(template_path.read_text(encoding="utf-8"))
        self.assertEqual(template["image"], "example.invalid/research@sha256:<supply-at-runtime>")
        rejected = self.root / "rejected.json"
        self.assertEqual(self.invoke("batch-spec", str(template_path), str(rejected))[0], 2)
        self.assertFalse(rejected.exists())

        template["image"] = "example.invalid/research@sha256:" + digest(
            {"synthetic_image": "cli-batch-structure"}
        )
        private = self.root / "batch-private.json"
        private.write_text(json.dumps(template), encoding="utf-8")
        output = self.root / "batch.json"
        self.assertEqual(self.invoke("batch-spec", str(private), str(output))[0], 0)
        request = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            request["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]["imageUri"],
            template["image"],
        )


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        queue = Queue(Path(self.temp.name) / "queue.sqlite")
        self.controller = Mock(queue=queue)
        self.controller.tick.return_value = None
        self.token = "synthetic-" + "x" * 32
        self.app = Application(self.controller, bearer_token=self.token)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, path, payload=None, *, auth=True, method="POST"):
        body = b"" if payload is None else canonical_bytes(payload)
        env = {
            "PATH_INFO": path,
            "REQUEST_METHOD": method,
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.input": io.BytesIO(body),
        }
        if auth:
            env["HTTP_AUTHORIZATION"] = "Bearer " + self.token
        response = []
        content = b"".join(
            self.app(env, lambda status, headers: response.append((status, headers)))
        )
        return response[0][0], content

    def test_unauthorized_has_no_effect(self):
        self.assertEqual(self.request("/tick", auth=False)[0], "401 Unauthorized")
        self.assertFalse(self.controller.tick.called)

    def test_routes(self):
        self.assertEqual(self.request("/health", method="GET")[0], "200 OK")
        self.assertEqual(self.request("/tick")[0], "200 OK")
        self.assertEqual(
            self.request("/jobs", {"request_id": "r", "specification": {"task": 1}})[0], "200 OK"
        )
        self.assertIn(b"QUEUED", self.request("/describe", {"request_id": "r"})[1])
        self.assertEqual(self.request("/cancel", {"request_id": "r"})[0], "200 OK")
        self.assertEqual(self.request("/jobs", {"request_id": "r"})[0], "400 Bad Request")
        self.assertEqual(self.request("/describe", {"request_id": "absent"})[0], "404 Not Found")
        self.assertEqual(self.request("/bad")[0], "404 Not Found")

    def test_errors_hide_provider_details(self):
        self.controller.tick.side_effect = RuntimeError("synthetic-private-detail")
        status, body = self.request("/tick")
        self.assertEqual(status, "503 Service Unavailable")
        self.assertNotIn(b"synthetic-private-detail", body)
