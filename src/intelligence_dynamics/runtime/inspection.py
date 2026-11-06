"""Paginated resource cleanup and JSON object checks for Google Cloud jobs."""

from __future__ import annotations

from urllib.parse import quote, urlencode

from ..storage import digest
from ..validation import text
from .batch import BatchTarget, ProviderError, http_transport


class GoogleInspector:
    def __init__(self, target: BatchTarget, token_provider, *, transport=http_transport):
        if (
            not isinstance(target, BatchTarget)
            or not callable(token_provider)
            or not callable(transport)
        ):
            raise ValueError("target, token provider and transport are required")
        self.target, self.token_provider, self.transport = target, token_provider, transport

    def _get(self, url):
        token = self.token_provider()
        text(token, "access token")
        if any(character.isspace() for character in token):
            raise ValueError("access token cannot contain whitespace")
        return self.transport("GET", url, {"Authorization": "Bearer " + token}, None)

    def resources_present(self, job, remote=None):
        """Find VM instances labelled with this exact controller job identity."""
        del remote
        token, seen = None, set()
        while True:
            parameters = {
                "filter": f'labels.id-job = "{job.provider_id}"',
                "returnPartialSuccess": "false",
            }
            if token is not None:
                parameters["pageToken"] = token
            result = self._get(
                "https://compute.googleapis.com/compute/v1/projects/"
                + self.target.project
                + "/aggregated/instances?"
                + urlencode(parameters)
            )
            if not isinstance(result, dict) or not isinstance(result.get("items", {}), dict):
                raise ValueError("instance response must contain a scope map")
            for scope in result.get("items", {}).values():
                if not isinstance(scope, dict) or "error" in scope:
                    raise ValueError("instance scope query failed")
                warning = scope.get("warning")
                if warning is not None and (
                    not isinstance(warning, dict) or warning.get("code") != "NO_RESULTS_ON_PAGE"
                ):
                    raise ValueError("instance scope has an unresolved warning")
                instances = scope.get("instances", [])
                if not isinstance(instances, list):
                    raise ValueError("instance collection must be a list")
                for instance in instances:
                    if not isinstance(instance, dict) or not isinstance(
                        instance.get("labels", {}), dict
                    ):
                        raise ValueError("instance identity is invalid")
                    if instance.get("labels", {}).get("id-job") == job.provider_id:
                        return True
            following = result.get("nextPageToken")
            if following is None or following == "":
                return False
            if not isinstance(following, str) or following in seen or not following.strip():
                raise ValueError("instance pagination token is invalid or repeated")
            seen.add(following)
            token = following

    def json_artifact(
        self, object_name: str, *, expected_digest: str | None = None, required_fields=()
    ):
        """Load a JSON object and check its canonical content digest and required keys."""
        text(object_name, "object name")
        for field in required_fields:
            text(field, "required field")
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + self.target.bucket
            + "/o/"
            + quote(object_name, safe="")
            + "?alt=media"
        )
        try:
            value = self._get(url)
        except ProviderError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(value, dict) or any(field not in value for field in required_fields):
            raise ValueError("JSON artifact is missing required fields")
        if expected_digest is not None and digest(value) != expected_digest:
            raise ValueError("JSON artifact canonical digest differs")
        return value
