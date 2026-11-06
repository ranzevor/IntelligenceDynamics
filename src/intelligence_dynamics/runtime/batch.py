"""Google Batch request construction and explicit authenticated transport."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..storage import canonical_bytes, loads
from ..validation import integer, json_value, text

PROJECT = r"[a-z][a-z0-9-]{4,28}[a-z0-9]"


def bucket_name(value: str) -> str:
    text(value, "bucket")
    if not 3 <= len(value) <= (222 if "." in value else 63):
        raise ValueError("bucket name length is invalid")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*[a-z0-9]", value):
        raise ValueError("bucket name syntax is invalid")
    if any(not part or len(part) > 63 for part in value.split(".")):
        raise ValueError("bucket dot-separated components require 1–63 characters")
    if value.startswith("goog") or re.search(r"g[o0][o0]gle", value):
        raise ValueError("bucket uses a reserved name")
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value):
        raise ValueError("bucket cannot use dotted-decimal IP notation")
    return value


@dataclass(frozen=True)
class BatchTarget:
    project: str
    region: str
    service_account: str
    bucket: str

    def __post_init__(self):
        if not isinstance(self.project, str) or not re.fullmatch(PROJECT, self.project):
            raise ValueError("project must be a 6–30 character project ID")
        if not isinstance(self.region, str) or not re.fullmatch(
            r"[a-z]+-[a-z]+[0-9]+", self.region
        ):
            raise ValueError("region syntax is invalid")
        if not isinstance(self.service_account, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]{4,28}[a-z0-9]@"
            + re.escape(self.project)
            + r"\.iam\.gserviceaccount\.com",
            self.service_account,
        ):
            raise ValueError(
                "service account requires a 6–30 character account ID in the target project"
            )
        bucket_name(self.bucket)

    @property
    def parent(self):
        return f"projects/{self.project}/locations/{self.region}"


def job_spec(
    target: BatchTarget,
    *,
    image: str,
    arguments: list[str],
    timeout_seconds: int = 3600,
    cpu_milli: int = 1000,
    memory_mib: int = 1024,
    machine_type: str = "e2-standard-2",
    task_count: int = 1,
    parallelism: int = 1,
    task_environments: list[dict[str, str]] | None = None,
) -> dict:
    if not isinstance(target, BatchTarget):
        raise ValueError("target must be BatchTarget")
    if not isinstance(image, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._/:\-]*@sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("container image requires a SHA256 digest reference")
    if not isinstance(arguments, list) or not arguments:
        raise ValueError("arguments must be a nonempty string list")
    for argument in arguments:
        text(argument, "container argument")
    for name, value in (
        ("timeout_seconds", timeout_seconds),
        ("cpu_milli", cpu_milli),
        ("memory_mib", memory_mib),
    ):
        integer(value, name, minimum=1)
    integer(task_count, "task_count", minimum=1)
    integer(parallelism, "parallelism", minimum=1)
    if parallelism > task_count:
        raise ValueError("parallelism cannot exceed task_count")
    if task_environments is not None:
        if not isinstance(task_environments, list) or len(task_environments) != task_count:
            raise ValueError("task_environments must match task_count")
        for environment in task_environments:
            if not isinstance(environment, dict):
                raise ValueError("each task environment must be an object")
            for key, value in environment.items():
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise ValueError("environment variable name is invalid")
                if not isinstance(value, str) or "\x00" in value:
                    raise ValueError("environment variable value is invalid")
    if timeout_seconds > 7 * 86400:
        raise ValueError("timeout_seconds exceeds seven days")
    if not isinstance(machine_type, str) or not re.fullmatch("[a-z][a-z0-9-]+", machine_type):
        raise ValueError("machine_type syntax is invalid")
    result = {
        "taskGroups": [
            {
                "taskCount": str(task_count),
                "parallelism": str(parallelism),
                "taskSpec": {
                    "runnables": [{"container": {"imageUri": image, "commands": list(arguments)}}],
                    "computeResource": {"cpuMilli": cpu_milli, "memoryMib": memory_mib},
                    "maxRetryCount": 0,
                    "maxRunDuration": f"{timeout_seconds}s",
                    "volumes": [
                        {"gcs": {"remotePath": target.bucket}, "mountPath": "/mnt/results"}
                    ],
                },
            }
        ],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": machine_type}}],
            "serviceAccount": {"email": target.service_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    if task_environments is not None:
        result["taskGroups"][0]["taskEnvironments"] = [
            {"variables": dict(item)} for item in task_environments
        ]
    return result


class ProviderError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ProviderError(code, "provider redirect refused")


def http_transport(method, url, headers, body):
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with build_opener(_NoRedirect).open(request, timeout=30) as response:
            data = response.read(4 * 1024 * 1024 + 1)
            status = response.status
    except HTTPError as exc:
        raise ProviderError(exc.code, f"provider HTTP status {exc.code}") from exc
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("provider response exceeds four MiB")
    if not 200 <= status < 300:
        raise ProviderError(status, f"provider HTTP status {status}")
    return loads(data) if data else {}


class BatchClient:
    """Perform Batch requests using a caller-supplied short-lived token provider."""

    def __init__(self, target: BatchTarget, token_provider, *, transport=http_transport):
        if (
            not isinstance(target, BatchTarget)
            or not callable(token_provider)
            or not callable(transport)
        ):
            raise ValueError("target, token provider and transport are required")
        self.target, self.token_provider, self.transport = target, token_provider, transport

    def _request(self, method, suffix, body=None):
        token = self.token_provider()
        text(token, "access token")
        if any(ch.isspace() for ch in token):
            raise ValueError("access token cannot contain whitespace")
        return self.transport(
            method,
            "https://batch.googleapis.com/v1/" + self.target.parent + suffix,
            {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            None if body is None else canonical_bytes(body),
        )

    def _id(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(
            r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", job_id
        ):
            raise ValueError("job ID syntax is invalid")
        return job_id

    def get(self, job_id):
        try:
            return self._request("GET", "/jobs/" + self._id(job_id))
        except ProviderError as exc:
            if exc.status == 404:
                return None
            raise

    def create(self, job_id, specification):
        self._id(job_id)
        body = json_value(specification)
        if not isinstance(body, dict):
            raise ValueError("job specification must be an object")
        request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, self.target.parent + "/" + job_id))
        return self._request(
            "POST", "/jobs?" + urlencode({"jobId": job_id, "requestId": request_id}), body
        )

    def cancel(self, job_id):
        self._id(job_id)
        request_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, self.target.parent + "/" + job_id + ":cancel")
        )
        return self._request("POST", "/jobs/" + job_id + ":cancel", {"requestId": request_id})


def estimate_timeout(
    *, observed_seconds, observed_units, target_units, slack=2.25, grace_seconds=300, maximum=604800
):
    """Scale a measured workload duration and add an explicit bounded margin."""
    from ..validation import number

    seconds = number(observed_seconds, "observed_seconds", minimum=0)
    integer(observed_units, "observed_units", minimum=1)
    integer(target_units, "target_units", minimum=1)
    margin = number(slack, "slack", minimum=1)
    integer(grace_seconds, "grace_seconds")
    integer(maximum, "maximum", minimum=1)
    if maximum > 604800:
        raise ValueError("maximum exceeds seven days")
    estimate = seconds * (target_units / observed_units) * margin + grace_seconds
    number(estimate, "estimated duration", minimum=0)
    import math

    return min(maximum, max(1, math.ceil(estimate)))
