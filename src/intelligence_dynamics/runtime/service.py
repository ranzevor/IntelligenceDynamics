"""Authenticated WSGI control surface for an explicitly configured controller."""

import hmac

from ..storage import canonical_bytes, loads
from ..validation import text


class Application:
    def __init__(self, controller, *, bearer_token: str):
        text(bearer_token, "bearer token")
        if len(bearer_token) < 32 or any(character.isspace() for character in bearer_token):
            raise ValueError("control token requires at least 32 non-whitespace characters")
        self.controller = controller
        self._authorization = ("Bearer " + bearer_token).encode("utf-8")

    def __call__(self, environ, start_response):
        def respond(status, payload):
            body = canonical_bytes(payload) + b"\n"
            start_response(
                status,
                [
                    ("Content-Type", "application/json; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                    ("Cache-Control", "no-store"),
                ],
            )
            return [body]

        authorization = environ.get("HTTP_AUTHORIZATION", "")
        if not isinstance(authorization, str) or not hmac.compare_digest(
            authorization.encode("utf-8"), self._authorization
        ):
            return respond("401 Unauthorized", {"error": "authentication required"})
        path, method = environ.get("PATH_INFO"), environ.get("REQUEST_METHOD")
        if path == "/health" and method == "GET":
            return respond("200 OK", {"status": "ok"})
        if method != "POST":
            return respond("405 Method Not Allowed", {"error": "POST required"})
        try:
            if path == "/tick":
                job = self.controller.tick()
                return respond(
                    "200 OK",
                    {
                        "job": None
                        if job is None
                        else {"request_id": job.request_id, "state": job.state}
                    },
                )
            if path not in {"/jobs", "/cancel", "/describe"}:
                return respond("404 Not Found", {"error": "unknown route"})
            length = int(environ.get("CONTENT_LENGTH") or 0)
            if not 0 < length <= 65536:
                return respond(
                    "413 Content Too Large", {"error": "request body requires 1–65536 bytes"}
                )
            raw = environ["wsgi.input"].read(length)
            if len(raw) != length:
                raise ValueError("request body is incomplete")
            payload = loads(raw)
            if not isinstance(payload, dict) or set(payload) != (
                {"request_id", "specification"} if path == "/jobs" else {"request_id"}
            ):
                raise ValueError("request fields are invalid")
            text(payload["request_id"], "request_id")
            queue = self.controller.queue
            if path == "/jobs":
                job = queue.enqueue(payload["request_id"], payload["specification"])
            elif path == "/cancel":
                job = queue.cancel(payload["request_id"])
            else:
                job = queue.get(payload["request_id"])
            if job is None:
                return respond("404 Not Found", {"error": "job not found"})
            return respond(
                "200 OK",
                {
                    "request_id": job.request_id,
                    "provider_id": job.provider_id,
                    "state": job.state,
                    "cancel_requested": job.cancel_requested,
                },
            )
        except (ValueError, TypeError, KeyError):
            return respond("400 Bad Request", {"error": "invalid request"})
        except Exception:
            return respond("503 Service Unavailable", {"error": "controller operation unavailable"})
