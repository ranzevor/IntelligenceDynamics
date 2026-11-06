import tempfile
import unittest
from pathlib import Path

from intelligence_dynamics.runtime import Controller, ProviderError, Queue


class Provider:
    def __init__(self):
        self.remote = None
        self.pending = None
        self.creates = 0
        self.cancels = 0
        self.create_error = None
        self.cancel_errors = []

    def get(self, identity):
        return self.remote

    def create(self, identity, specification):
        self.creates += 1
        self.pending = {"labels": specification["labels"], "status": {"state": "RUNNING"}}
        if self.create_error:
            raise self.create_error
        self.remote = self.pending
        return self.remote

    def cancel(self, identity):
        self.cancels += 1
        if self.cancel_errors:
            error = self.cancel_errors.pop(0)
            if error:
                raise error
        self.remote["status"]["state"] = "CANCELLATION_IN_PROGRESS"


class CancellationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "jobs.sqlite"
        self.queue = Queue(self.path)
        self.provider = Provider()
        self.now = 100.0
        self.resources = False
        self.controller = Controller(
            self.queue,
            self.provider,
            resources_present=lambda *a: self.resources,
            artifacts_valid=lambda *a: True,
            clock=lambda: self.now,
        )
        self.queue.enqueue("request", {"taskGroups": [{}]})

    def tearDown(self):
        self.temp.cleanup()

    def begin_cancel(self):
        self.controller.tick()
        self.queue.cancel("request")

    def test_transient_failure_retries_then_accepted_once(self):
        self.begin_cancel()
        self.provider.cancel_errors = [ProviderError(503, "not accepted")]
        first = self.controller.tick()
        self.assertEqual(
            (first.state, first.cancel_attempts, first.cancel_accepted), ("CANCEL_RETRY", 1, False)
        )
        second = self.controller.tick()
        self.assertEqual(
            (second.state, second.cancel_attempts, second.cancel_accepted),
            ("CANCEL_REQUESTED", 2, True),
        )
        for _ in range(4):
            self.assertEqual(self.controller.tick().state, "CLEANUP_WAIT")
        self.assertEqual(self.provider.cancels, 2)
        self.resources = True
        self.provider.remote["status"]["state"] = "CANCELLED"
        self.assertEqual(self.controller.tick().state, "CLEANUP_WAIT")
        self.resources = False
        self.assertEqual(self.controller.tick().state, "CANCELLED")
        self.assertIsNone(self.controller.tick())

    def test_retry_classification_and_budget(self):
        self.begin_cancel()
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.provider.cancel_errors = [ProviderError(status, "temporary")] * 3
                self.assertEqual(
                    [self.controller.tick().state for _ in range(3)],
                    ["CANCEL_RETRY", "CANCEL_RETRY", "CANCEL_BLOCKED"],
                )
                count = self.provider.cancels
                self.assertEqual(self.controller.tick().state, "CANCEL_BLOCKED")
                self.assertEqual(self.provider.cancels, count)
                self.queue.retry_cancellation("request", now=self.now)
        self.provider.cancel_errors = []
        self.assertEqual(self.controller.tick().state, "CANCEL_REQUESTED")

    def test_denial_does_not_retry_without_explicit_request(self):
        self.begin_cancel()
        for status in (400, 401, 403):
            with self.subTest(status=status):
                self.provider.cancel_errors = [ProviderError(status, "denied")]
                self.assertEqual(self.controller.tick().state, "CANCEL_BLOCKED")
                count = self.provider.cancels
                self.assertEqual(self.controller.tick().state, "CANCEL_BLOCKED")
                self.assertEqual(self.provider.cancels, count)
                self.queue.retry_cancellation("request", now=self.now)
        self.assertEqual(self.controller.tick().state, "CANCEL_REQUESTED")
        with self.assertRaises(ValueError):
            self.queue.retry_cancellation("request", now=self.now)

    def test_retry_preserves_lease_fencing(self):
        self.begin_cancel()
        self.provider.cancel_errors = [ProviderError(403, "denied")]
        self.controller.tick()
        claim = self.queue.claim(now=self.now)
        with self.assertRaises(RuntimeError):
            self.queue.retry_cancellation("request", now=self.now)
        self.queue.transition(claim, "CANCEL_BLOCKED", now=self.now)
        self.assertEqual(
            self.queue.retry_cancellation("request", now=self.now).state, "CANCEL_RETRY"
        )

    def test_unknown_cancel_reconciles_before_retry(self):
        self.begin_cancel()

        def unknown(identity):
            self.provider.cancels += 1
            self.provider.remote["status"]["state"] = "CANCELLATION_IN_PROGRESS"
            raise TimeoutError("acknowledgement lost")

        self.provider.cancel = unknown
        self.assertEqual(self.controller.tick().state, "CANCEL_RETRY")
        self.assertEqual(self.controller.tick().state, "CLEANUP_WAIT")
        self.assertTrue(self.queue.get("request").cancel_accepted)
        self.assertEqual(self.provider.cancels, 1)

    def test_transport_wrapped_timeout_retries_but_certificate_failure_blocks(self):
        import ssl
        from urllib.error import URLError

        self.begin_cancel()
        self.provider.cancel_errors = [URLError(TimeoutError("network timeout"))]
        self.assertEqual(self.controller.tick().state, "CANCEL_RETRY")
        self.provider.cancel_errors = [
            URLError(ssl.SSLCertVerificationError("certificate mismatch"))
        ]
        self.assertEqual(self.controller.tick().state, "CANCEL_BLOCKED")
        count = self.provider.cancels
        self.controller.tick()
        self.assertEqual(self.provider.cancels, count)

    def test_unknown_submission_waits_for_late_acceptance(self):
        self.provider.create_error = TimeoutError("create still pending")
        self.assertEqual(self.controller.tick().state, "SUBMITTING")
        self.queue.cancel("request")
        for _ in range(5):
            self.assertEqual(self.controller.tick().state, "SUBMITTING")
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual(self.provider.cancels, 0)
        self.provider.remote = self.provider.pending
        self.assertEqual(self.controller.tick().state, "CANCEL_REQUESTED")
        self.assertEqual(self.provider.cancels, 1)
        self.provider.remote["status"]["state"] = "CANCELLED"
        self.assertEqual(self.controller.tick().state, "CANCELLED")

    def test_missing_job_after_accepted_cancel_is_not_terminal_proof(self):
        self.begin_cancel()
        self.controller.tick()
        remote = self.provider.remote
        self.provider.remote = None
        self.assertNotEqual(self.controller.tick().state, "CANCELLED")
        self.provider.remote = remote
        self.provider.remote["status"]["state"] = "CANCELLED"
        self.assertEqual(self.controller.tick().state, "CANCELLED")

    def test_terminal_remote_race_retains_actual_outcome(self):
        self.begin_cancel()
        self.provider.cancel_errors = [ProviderError(503, "temporary")]
        self.controller.tick()
        self.provider.remote["status"]["state"] = "SUCCEEDED"
        self.assertEqual(self.controller.tick().state, "SUCCEEDED")
        self.assertEqual(self.provider.cancels, 1)

    def test_intent_persists_across_worker_crash(self):
        self.begin_cancel()
        claim = self.queue.claim(now=self.now, lease_seconds=1)
        self.queue.transition(
            claim, "CANCEL_SUBMITTING", now=self.now, cancel_attempts=1, release=False
        )
        self.now += 2
        self.assertEqual(self.controller.tick().state, "CANCEL_REQUESTED")
        self.assertEqual(self.queue.get("request").cancel_attempts, 2)
        self.assertEqual(self.provider.cancels, 1)

    def test_reopen_preserves_acceptance_and_attempts(self):
        self.begin_cancel()
        self.controller.tick()
        reopened = Queue(self.path).get("request")
        self.assertTrue(reopened.cancel_accepted)
        self.assertEqual(reopened.cancel_attempts, 1)

    def test_cancel_before_submission_has_no_provider_effect(self):
        self.queue.cancel("request")
        self.assertEqual(self.controller.tick().state, "CANCELLED")
        self.assertEqual((self.provider.creates, self.provider.cancels), (0, 0))


class PollingOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "jobs.sqlite"
        self.queue = Queue(self.path)

    def add(self, name, **kwargs):
        return self.queue.enqueue(name, {"task": name}, **kwargs)

    def provider(self):
        class MultipleJobs:
            def __init__(self):
                self.remote = {}
                self.pending = {}
                self.unknown = set()
                self.denied = set()
                self.gets = []
                self.creates = []

            def get(self, identity):
                self.gets.append(identity)
                return self.remote.get(identity)

            def create(self, identity, specification):
                self.creates.append(identity)
                remote = {"labels": specification["labels"], "status": {"state": "RUNNING"}}
                self.pending[identity] = remote
                if identity in self.unknown:
                    raise TimeoutError("synthetic unknown submission")
                self.remote[identity] = remote
                return remote

            def cancel(self, identity):
                if identity in self.denied:
                    raise ProviderError(403, "synthetic cancellation denied")
                self.remote[identity]["status"]["state"] = "CANCELLED"

        return MultipleJobs()

    def controller(self, provider, resources=()):
        return Controller(
            self.queue,
            provider,
            resources_present=lambda job, remote: job.request_id in resources,
            artifacts_valid=lambda *a: True,
            clock=lambda: 100.0,
        )

    def test_blocked_cancel_is_polled_while_independent_jobs_advance(self):
        first = self.add("first")
        provider = self.provider()
        controller = self.controller(provider)
        self.assertEqual(controller.tick().request_id, "first")
        self.queue.cancel("first")
        provider.denied.add(first.provider_id)
        self.assertEqual(controller.tick().state, "CANCEL_BLOCKED")
        second, third = self.add("second"), self.add("third")
        self.add("dependent", requires=["first"])
        selected = [controller.tick().request_id for _ in range(9)]
        self.assertEqual(selected, ["first", "second", "third"] * 3)
        self.assertEqual(self.queue.get("first").state, "CANCEL_BLOCKED")
        self.assertEqual(self.queue.get("second").state, "RUNNING")
        self.assertEqual(self.queue.get("third").state, "RUNNING")
        self.assertEqual(self.queue.get("dependent").state, "QUEUED")
        self.assertEqual(
            set(provider.creates), {first.provider_id, second.provider_id, third.provider_id}
        )
        self.assertGreaterEqual(provider.gets.count(first.provider_id), 4)

    def test_unknown_submission_and_cleanup_wait_remain_in_polling(self):
        unknown = self.add("unknown")
        provider = self.provider()
        provider.unknown.add(unknown.provider_id)
        controller = self.controller(provider, resources={"cleanup"})
        self.assertEqual(controller.tick().state, "SUBMITTING")
        cleanup = self.add("cleanup")
        independent = self.add("independent")
        selected = [controller.tick().request_id for _ in range(3)]
        self.assertEqual(selected, ["unknown", "cleanup", "independent"])
        provider.remote[cleanup.provider_id]["status"]["state"] = "SUCCEEDED"
        results = [controller.tick() for _ in range(6)]
        self.assertEqual(
            [job.request_id for job in results], ["unknown", "cleanup", "independent"] * 2
        )
        self.assertEqual(self.queue.get("cleanup").state, "CLEANUP_WAIT")
        self.assertEqual(self.queue.get("unknown").state, "SUBMITTING")
        self.assertEqual(self.queue.get("independent").state, "RUNNING")
        self.assertIn(independent.provider_id, provider.creates)
        provider.remote[unknown.provider_id] = provider.pending[unknown.provider_id]
        self.assertEqual(controller.tick().state, "RUNNING")

    def test_cancel_and_explicit_retry_promote_once(self):
        for name in ("first", "second", "third"):
            self.add(name)
        self.queue.cancel("first")
        first = self.queue.claim(now=100)
        self.queue.transition(first, "CANCEL_BLOCKED", now=100, cancel_attempts=3)
        for name in ("second", "third"):
            self.queue.cancel("first")
            job = self.queue.claim(now=100)
            self.assertEqual(job.request_id, name)
            self.queue.transition(job, "RUNNING", now=100)
        self.queue.retry_cancellation("first", now=100)
        with self.assertRaises(ValueError):
            self.queue.retry_cancellation("first", now=100)
        first = self.queue.claim(now=100)
        self.assertEqual((first.request_id, first.cancel_attempts), ("first", 0))
        self.queue.transition(first, "CANCEL_RETRY", now=100)
        self.queue.cancel("first")
        self.assertEqual(self.queue.claim(now=100).request_id, "second")

    def test_first_cancel_promotes_a_job_at_the_tail(self):
        for name in ("first", "second", "third"):
            self.add(name)
        self.queue.cancel("third")
        job = self.queue.claim(now=100)
        self.assertEqual(job.request_id, "third")
        self.queue.transition(job, "CANCELLED", now=100)
        self.assertEqual(self.queue.claim(now=100).request_id, "first")

    def test_reopen_preserves_rotation_and_new_jobs_join_tail(self):
        self.add("first")
        self.add("second")
        job = self.queue.claim(now=100)
        self.queue.transition(job, "RUNNING", now=100)
        self.queue = Queue(self.path)
        self.add("third")
        selected = []
        for _ in range(6):
            job = self.queue.claim(now=100)
            selected.append(job.request_id)
            self.queue.transition(job, "RUNNING", now=100)
        self.assertEqual(selected, ["second", "first", "third"] * 2)

    def test_expired_lease_is_fenced_after_rotation_and_restart(self):
        self.add("first")
        self.add("second")
        old = self.queue.claim(now=100, lease_seconds=5)
        self.queue = Queue(self.path)
        second = self.queue.claim(now=106)
        self.assertEqual(second.request_id, "second")
        current = self.queue.claim(now=106)
        self.assertEqual(current.request_id, "first")
        with self.assertRaises(RuntimeError):
            self.queue.transition(old, "RUNNING", now=107)
        self.queue.transition(current, "RUNNING", now=107)

    def test_old_database_upgrade_preserves_jobs_history_and_live_lease(self):
        import sqlite3
        from contextlib import closing

        old_path = self.path.parent / "legacy.sqlite"
        with closing(sqlite3.connect(old_path)) as db, db:
            db.executescript("""
                CREATE TABLE jobs (
                    request_id TEXT PRIMARY KEY, provider_id TEXT UNIQUE NOT NULL,
                    specification TEXT NOT NULL, state TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT, lease_until REAL NOT NULL DEFAULT 0,
                    artifact_attempts INTEGER NOT NULL DEFAULT 0, detail TEXT,
                    sequence INTEGER UNIQUE NOT NULL, prerequisites TEXT NOT NULL DEFAULT '[]',
                    cancel_attempts INTEGER NOT NULL DEFAULT 0,
                    cancel_accepted INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE transitions (
                    id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, state TEXT NOT NULL,
                    detail TEXT, observed_at REAL NOT NULL
                );
            """)
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "first",
                    "id-old-first",
                    '{"task":1}',
                    "CANCEL_BLOCKED",
                    1,
                    "live-lease",
                    1000,
                    2,
                    "retained detail",
                    1,
                    "[]",
                    3,
                    0,
                ),
            )
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "second",
                    "id-old-second",
                    '{"task":2}',
                    "QUEUED",
                    0,
                    None,
                    0,
                    0,
                    None,
                    2,
                    "[]",
                    0,
                    0,
                ),
            )
            db.execute(
                "INSERT INTO transitions VALUES (1, 'first', 'CANCEL_BLOCKED', 'retained history', 90)"
            )
            before = db.execute("SELECT * FROM jobs ORDER BY sequence").fetchall()
        upgraded = Queue(old_path)
        with closing(sqlite3.connect(old_path)) as db:
            after = db.execute("SELECT * FROM jobs ORDER BY sequence").fetchall()
        self.assertEqual([row[:-1] for row in after], before)
        self.assertEqual([row[-1] for row in after], [1, 2])
        self.assertEqual(
            upgraded.history("first"),
            [{"state": "CANCEL_BLOCKED", "detail": "retained history", "observed_at": 90.0}],
        )
        old = upgraded.get("first")
        self.assertEqual((old.lease_token, old.lease_until), ("live-lease", 1000))
        self.assertEqual(upgraded.claim(now=100).request_id, "second")
        reopened = Queue(old_path)
        current = reopened.claim(now=1001)
        self.assertEqual(current.request_id, "first")
        with self.assertRaises(RuntimeError):
            reopened.transition(old, "RUNNING", now=1002)
        self.assertEqual(reopened.get("first").detail, "retained detail")
