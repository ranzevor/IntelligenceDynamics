"""Transactional job state with fenced leases and deterministic provider IDs."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

from ..storage import canonical_bytes, digest, loads
from ..validation import integer, is_linked_path, json_value, number, text
from .batch import ProviderError

TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
STATES = TERMINAL | {
    "QUEUED",
    "SUBMITTING",
    "SUBMITTED",
    "RUNNING",
    "CANCEL_SUBMITTING",
    "CANCEL_RETRY",
    "CANCEL_BLOCKED",
    "CANCEL_REQUESTED",
    "CLEANUP_WAIT",
    "ARTIFACT_WAIT",
}


@dataclass(frozen=True)
class Job:
    request_id: str
    provider_id: str
    specification: dict
    state: str
    cancel_requested: bool
    lease_token: str | None
    lease_until: float
    artifact_attempts: int
    detail: str | None
    prerequisites: tuple[str, ...] = ()
    cancel_attempts: int = 0
    cancel_accepted: bool = False


class Queue:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        for part in (self.path, *self.path.parents):
            if is_linked_path(part):
                raise ValueError("queue path cannot traverse symbolic links or reparse points")
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                request_id TEXT PRIMARY KEY, provider_id TEXT UNIQUE NOT NULL,
                specification TEXT NOT NULL, state TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                lease_token TEXT, lease_until REAL NOT NULL DEFAULT 0,
                artifact_attempts INTEGER NOT NULL DEFAULT 0, detail TEXT, sequence INTEGER UNIQUE NOT NULL, prerequisites TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY, request_id TEXT NOT NULL, state TEXT NOT NULL,
                detail TEXT, observed_at REAL NOT NULL
            );""")
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("cancel_attempts", "cancel_accepted"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
            if "claim_order" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN claim_order INTEGER NOT NULL DEFAULT 0")
                db.execute("UPDATE jobs SET claim_order=sequence")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        except BaseException as primary:
            try:
                db.close()
            except BaseException as secondary:
                if hasattr(primary, "add_note"):
                    primary.add_note(f"database close failed: {secondary}")
                else:
                    primary.cleanup_errors = (str(secondary),)
            raise
        else:
            db.close()

    @staticmethod
    def _job(row):
        if row is None:
            return None
        return Job(
            row["request_id"],
            row["provider_id"],
            loads(row["specification"]),
            row["state"],
            bool(row["cancel_requested"]),
            row["lease_token"],
            row["lease_until"],
            row["artifact_attempts"],
            row["detail"],
            tuple(loads(row["prerequisites"])),
            row["cancel_attempts"],
            bool(row["cancel_accepted"]),
        )

    def enqueue(self, request_id: str, specification: dict, *, requires=()) -> Job:
        text(request_id, "request_id")
        if not isinstance(requires, (tuple, list)):
            raise ValueError("prerequisites must be a sequence")
        for dependency in requires:
            text(dependency, "prerequisite")
        if request_id in requires or len(set(requires)) != len(requires):
            raise ValueError("prerequisites must be unique and cannot include the request itself")
        prerequisites = canonical_bytes(list(requires)).decode("utf-8")
        specification = json_value(specification)
        if not isinstance(specification, dict) or not specification:
            raise ValueError("job specification must be a nonempty object")
        encoded = canonical_bytes(specification).decode("utf-8")
        provider_id = "id-" + digest({"request_id": request_id})[:40]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if old["specification"] != encoded or old["prerequisites"] != prerequisites:
                    raise ValueError("request ID already identifies a different specification")
                return self._job(old)
            for dependency in requires:
                if (
                    db.execute("SELECT 1 FROM jobs WHERE request_id=?", (dependency,)).fetchone()
                    is None
                ):
                    raise ValueError("prerequisites must identify existing jobs")
            sequence = db.execute("SELECT COALESCE(MAX(sequence), 0)+1 FROM jobs").fetchone()[0]
            order = db.execute("SELECT COALESCE(MAX(claim_order), 0)+1 FROM jobs").fetchone()[0]
            db.execute(
                "INSERT INTO jobs(request_id,provider_id,specification,state,sequence,prerequisites,claim_order) VALUES(?,?,?,?,?,?,?)",
                (request_id, provider_id, encoded, "QUEUED", sequence, prerequisites, order),
            )
            db.execute(
                "INSERT INTO transitions(request_id,state,observed_at) VALUES(?,?,?)",
                (request_id, "QUEUED", time.time()),
            )
        return self.get(request_id)

    def get(self, request_id: str):
        with self._connect() as db:
            return self._job(
                db.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            )

    def cancel(self, request_id: str):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise KeyError(request_id)
            if row["state"] not in TERMINAL and not row["cancel_requested"]:
                order = db.execute(
                    "SELECT COALESCE(MIN(claim_order), 0)-1 FROM jobs WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')"
                ).fetchone()[0]
                db.execute(
                    "UPDATE jobs SET cancel_requested=1,claim_order=? WHERE request_id=?",
                    (order, request_id),
                )
        return self.get(request_id)

    def retry_cancellation(self, request_id: str, *, now: float):
        """Resume a blocked cancellation after its cause has been addressed."""
        number(now, "now", minimum=0)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise KeyError(request_id)
            if row["state"] != "CANCEL_BLOCKED" or row["cancel_accepted"]:
                raise ValueError("only an unaccepted blocked cancellation can be retried")
            if row["lease_until"] > now:
                raise RuntimeError("job has an active lease")
            order = db.execute(
                "SELECT COALESCE(MIN(claim_order), 0)-1 FROM jobs WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')"
            ).fetchone()[0]
            db.execute(
                "UPDATE jobs SET state='CANCEL_RETRY',cancel_attempts=0,claim_order=? WHERE request_id=?",
                (order, request_id),
            )
            db.execute(
                "INSERT INTO transitions(request_id,state,detail,observed_at) VALUES(?,?,?,?)",
                (request_id, "CANCEL_RETRY", "operator requested cancellation retry", now),
            )
        return self.get(request_id)

    def claim(self, *, now: float, lease_seconds: float = 60) -> Job | None:
        number(now, "now", minimum=0)
        number(lease_seconds, "lease_seconds", minimum=0)
        if lease_seconds == 0:
            raise ValueError("lease_seconds must be positive")
        end = now + lease_seconds
        number(end, "lease expiry")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            candidates = db.execute(
                """SELECT * FROM jobs WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')
                AND lease_until<=? ORDER BY claim_order, sequence""",
                (now,),
            ).fetchall()
            row = None
            for candidate in candidates:
                dependencies = loads(candidate["prerequisites"])
                if candidate["cancel_requested"] or all(
                    db.execute(
                        "SELECT state FROM jobs WHERE request_id=?", (dependency,)
                    ).fetchone()[0]
                    == "SUCCEEDED"
                    for dependency in dependencies
                ):
                    row = candidate
                    break
            if row is None:
                return None
            token = uuid.uuid4().hex
            order = db.execute("SELECT COALESCE(MAX(claim_order), 0)+1 FROM jobs").fetchone()[0]
            db.execute(
                "UPDATE jobs SET lease_token=?,lease_until=?,claim_order=? WHERE request_id=?",
                (token, end, order, row["request_id"]),
            )
            return self._job(
                db.execute("SELECT * FROM jobs WHERE request_id=?", (row["request_id"],)).fetchone()
            )

    def transition(
        self,
        job: Job,
        state: str,
        *,
        now: float,
        detail: str | None = None,
        artifact_attempts: int | None = None,
        cancel_attempts: int | None = None,
        cancel_accepted: bool | None = None,
        release: bool = True,
    ):
        if state not in STATES:
            raise ValueError("unknown job state")
        number(now, "now", minimum=0)
        if detail is not None:
            text(detail, "detail")
        attempts = (
            job.artifact_attempts
            if artifact_attempts is None
            else integer(artifact_attempts, "artifact_attempts")
        )
        cancellations = (
            job.cancel_attempts
            if cancel_attempts is None
            else integer(cancel_attempts, "cancel_attempts")
        )
        accepted = job.cancel_accepted if cancel_accepted is None else cancel_accepted
        if type(accepted) is not bool:
            raise ValueError("cancel_accepted must be a boolean")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE request_id=?", (job.request_id,)).fetchone()
            if (
                row is None
                or not job.lease_token
                or row["lease_token"] != job.lease_token
                or row["lease_until"] <= now
            ):
                raise RuntimeError("job lease is no longer valid")
            if row["state"] in TERMINAL:
                raise RuntimeError("terminal jobs cannot transition")
            db.execute(
                "UPDATE jobs SET state=?,detail=?,artifact_attempts=?,lease_token=?,lease_until=?,cancel_attempts=?,cancel_accepted=? WHERE request_id=?",
                (
                    state,
                    detail,
                    attempts,
                    None if release else job.lease_token,
                    0 if release else row["lease_until"],
                    cancellations,
                    int(accepted),
                    job.request_id,
                ),
            )
            db.execute(
                "INSERT INTO transitions(request_id,state,detail,observed_at) VALUES(?,?,?,?)",
                (job.request_id, state, detail, now),
            )
        return self.get(job.request_id)

    def history(self, request_id):
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT state,detail,observed_at FROM transitions WHERE request_id=? ORDER BY id",
                    (request_id,),
                )
            ]


class Controller:
    """Advance one job using explicit provider, cleanup and artifact checks."""

    def __init__(
        self,
        queue: Queue,
        provider,
        *,
        resources_present,
        artifacts_valid,
        clock=time.time,
        lease_seconds=120,
        artifact_attempts=3,
        cancel_attempts=3,
        execution_timeout=604800,
    ):
        integer(artifact_attempts, "artifact_attempts", minimum=1)
        self.max_cancel_attempts = integer(cancel_attempts, "cancel_attempts", minimum=1)
        number(execution_timeout, "execution_timeout", minimum=1)
        if execution_timeout > 604800:
            raise ValueError("execution_timeout exceeds seven days")
        self.execution_timeout = execution_timeout
        number(lease_seconds, "lease_seconds", minimum=1)
        if not all(callable(fn) for fn in (resources_present, artifacts_valid, clock)):
            raise ValueError("resource, artifact and clock callbacks are required")
        self.queue, self.provider = queue, provider
        self.resources_present, self.artifacts_valid, self.clock = (
            resources_present,
            artifacts_valid,
            clock,
        )
        self.lease_seconds, self.max_artifact_attempts = lease_seconds, artifact_attempts

    def tick(self):
        job = self.queue.claim(now=self.clock(), lease_seconds=self.lease_seconds)
        if job is None:
            return None

        def save(
            state, detail=None, attempts=None, release=True, *, cancellations=None, accepted=None
        ):
            nonlocal job
            job = self.queue.transition(
                job,
                state,
                now=self.clock(),
                detail=detail,
                artifact_attempts=attempts,
                cancel_attempts=cancellations,
                cancel_accepted=accepted,
                release=release,
            )
            return job

        try:
            if job.state == "QUEUED" and job.cancel_requested:
                return save("CANCELLED", "cancelled before submission")
            if job.state == "QUEUED":
                save("SUBMITTING", "submission intent recorded", release=False)
            history = self.queue.history(job.request_id)
            started = next(
                (row["observed_at"] for row in history if row["state"] == "SUBMITTING"), None
            )
            if started is not None and self.clock() - started >= self.execution_timeout:
                self.queue.cancel(job.request_id)
                job = self.queue.get(job.request_id)
            remote = self.provider.get(job.provider_id)
            if remote is None:
                if job.cancel_requested:
                    return save(
                        job.state, "provider job is not observable; cancellation remains unresolved"
                    )
                if job.state != "SUBMITTING":
                    return save(job.state, "provider job is not observable")
                specification = json_value(job.specification)
                labels = specification.setdefault("labels", {})
                if not isinstance(labels, dict):
                    raise ValueError("job labels must be an object")
                labels["id-spec"] = digest(job.specification)[:40]
                allocation = specification.setdefault("allocationPolicy", {})
                allocation_labels = allocation.setdefault("labels", {})
                allocation_labels["id-job"] = job.provider_id
                remote = self.provider.create(job.provider_id, specification)
            if not isinstance(remote, dict) or not isinstance(remote.get("status"), dict):
                raise ValueError("provider job response has no status object")
            if remote.get("labels", {}).get("id-spec") != digest(job.specification)[:40]:
                return save("FAILED", "provider job identity does not match specification")
            state = remote["status"].get("state")
            if state in {"CANCELLATION_IN_PROGRESS", "DELETION_IN_PROGRESS"}:
                return save("CLEANUP_WAIT", f"provider reports {state}", accepted=True)
            if state == "CANCELLED":
                return self._finish_cancel(job, save)
            if job.cancel_requested and state not in {"SUCCEEDED", "FAILED"}:
                if job.cancel_accepted:
                    return save("CANCEL_REQUESTED", "awaiting provider cancellation and cleanup")
                if job.state == "CANCEL_BLOCKED" or job.cancel_attempts >= self.max_cancel_attempts:
                    return save(
                        "CANCEL_BLOCKED",
                        "cancellation requires operator attention; monitoring continues",
                    )
                save(
                    "CANCEL_SUBMITTING",
                    "cancellation intent recorded",
                    release=False,
                    cancellations=job.cancel_attempts + 1,
                )
                try:
                    self.provider.cancel(job.provider_id)
                except Exception as exc:
                    reason = exc.reason if isinstance(exc, URLError) else exc
                    retryable = (
                        isinstance(exc, ProviderError)
                        and exc.status in {408, 429, 500, 502, 503, 504}
                    ) or isinstance(reason, (TimeoutError, ConnectionError))
                    state = (
                        "CANCEL_RETRY"
                        if retryable and job.cancel_attempts < self.max_cancel_attempts
                        else "CANCEL_BLOCKED"
                    )
                    return save(
                        state, f"cancellation not acknowledged: {type(exc).__name__}: {exc}"
                    )
                return save("CANCEL_REQUESTED", "provider accepted cancellation", accepted=True)
            state = remote["status"].get("state")
            if state in {"QUEUED", "SCHEDULED"}:
                return save("SUBMITTED")
            if state == "RUNNING":
                return save("RUNNING")
            if state in {"SUCCEEDED", "FAILED"}:
                save("CLEANUP_WAIT", "checking provider resource cleanup", release=False)
                resources = self.resources_present(job, remote)
                if type(resources) is not bool:
                    raise ValueError("resource check must return a boolean")
                if resources:
                    return save("CLEANUP_WAIT", "provider resources remain")
                if state == "FAILED":
                    return save("FAILED", "provider reported failure")
                attempts = job.artifact_attempts + 1
                detail = "required artifacts are not ready"
                try:
                    valid = self.artifacts_valid(job, remote)
                    if type(valid) is not bool:
                        raise ValueError("artifact check must return a boolean")
                except Exception as exc:
                    valid = False
                    detail = f"artifact validation failed: {type(exc).__name__}: {exc}"
                if valid:
                    return save("SUCCEEDED", attempts=attempts)
                if attempts >= self.max_artifact_attempts:
                    return save("FAILED", detail, attempts)
                return save("ARTIFACT_WAIT", detail, attempts)
            return save(job.state, "provider state is not recognized")
        except Exception as exc:
            if job.lease_token is None:
                raise
            return save(job.state, f"{type(exc).__name__}: {exc}")

    def _finish_cancel(self, job, save):
        resources = self.resources_present(job, None)
        if type(resources) is not bool:
            raise ValueError("resource check must return a boolean")
        return save(
            "CLEANUP_WAIT" if resources else "CANCELLED",
            "provider resources remain"
            if resources
            else "provider confirmed cancellation and resources are absent",
        )
