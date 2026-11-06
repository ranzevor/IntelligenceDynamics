"""Explicit-codec checkpoints with verified state and history identity."""

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ..storage import canonical_bytes, loads
from ..validation import integer, is_linked_path, text
from ._state import codec_state_identity, state_snapshot
from .drqn import ExecutionState


@dataclass(frozen=True)
class NativeCheckpoint:
    state: ExecutionState
    history: tuple = ()

    def __post_init__(self):
        if not isinstance(self.state, ExecutionState) or type(self.history) is not tuple:
            raise ValueError("checkpoint requires ExecutionState and tuple history")
        integer(self.state.total_steps, "total steps")
        if len(self.history) != self.state.total_steps:
            raise ValueError("history length must equal total environment interactions")
        state_snapshot(self)

    @property
    def sha256(self):
        return state_snapshot(self)["sha256"]


@dataclass(frozen=True)
class CheckpointCodec:
    name: str
    encode: object
    decode: object

    def __post_init__(self):
        text(self.name, "codec name")
        if not callable(self.encode) or not callable(self.decode):
            raise ValueError("codec encode and decode must be callable")


class CheckpointStore:
    """Store new named checkpoints; decode only with the caller's trusted codec."""

    def __init__(self, root, codec, *, max_bytes=67108864):
        if not isinstance(codec, CheckpointCodec):
            raise ValueError("codec must be CheckpointCodec")
        integer(max_bytes, "max_bytes", minimum=1)
        self.root, self.codec, self.max_bytes = Path(root), codec, max_bytes
        self._safe(self.root)

    @staticmethod
    def _safe(path):
        if any(is_linked_path(part) for part in (path, *path.parents)):
            raise ValueError("checkpoint paths cannot traverse symbolic links or reparse points")

    def _directory(self, name):
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) is None
        ):
            raise ValueError(
                "checkpoint names require 1-80 letters, digits, hyphens or underscores"
            )
        path = self.root / name
        self._safe(path)
        return path

    def _decode(self, payload):
        value = self.codec.decode(payload)
        if not isinstance(value, NativeCheckpoint):
            raise ValueError("codec must decode NativeCheckpoint")
        NativeCheckpoint(value.state, value.history)
        return value

    def save(self, name, checkpoint):
        if not isinstance(checkpoint, NativeCheckpoint):
            raise ValueError("expected NativeCheckpoint")
        expected = checkpoint.sha256
        expected_types = codec_state_identity(checkpoint)
        copied = deepcopy(checkpoint)
        if codec_state_identity(copied) != expected_types:
            raise ValueError("checkpoint copying changed state or array types")
        payload = self.codec.encode(copied)
        if not isinstance(payload, bytes) or not payload or len(payload) > self.max_bytes:
            raise ValueError("encoded checkpoint must be nonempty bounded bytes")
        decoded = self._decode(payload)
        if (
            decoded.sha256 != expected
            or checkpoint.sha256 != expected
            or codec_state_identity(decoded) != expected_types
            or codec_state_identity(checkpoint) != expected_types
        ):
            raise ValueError("checkpoint codec changed state or history")
        directory = self._directory(name)
        self.root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(exist_ok=False)
        manifest = {
            "format": "native-checkpoint-v1",
            "codec": self.codec.name,
            "bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint_sha256": expected,
        }
        with (directory / "payload.bin").open("xb") as stream:
            stream.write(payload)
        with (directory / "manifest.json").open("xb") as stream:
            stream.write(canonical_bytes(manifest))
        return expected

    def _read(self, name):
        directory = self._directory(name)
        for file in (directory / "manifest.json", directory / "payload.bin"):
            self._safe(file)
            if not file.is_file():
                raise ValueError("checkpoint files are missing")
        if (directory / "manifest.json").stat().st_size > 8192:
            raise ValueError("checkpoint manifest is too large")
        manifest = loads((directory / "manifest.json").read_bytes())
        keys = {"format", "codec", "bytes", "payload_sha256", "checkpoint_sha256"}
        if (
            not isinstance(manifest, dict)
            or set(manifest) != keys
            or manifest["format"] != "native-checkpoint-v1"
            or manifest["codec"] != self.codec.name
        ):
            raise ValueError("checkpoint manifest has an incompatible contract")
        integer(manifest["bytes"], "checkpoint bytes", minimum=1)
        if (
            manifest["bytes"] > self.max_bytes
            or (directory / "payload.bin").stat().st_size != manifest["bytes"]
        ):
            raise ValueError("checkpoint payload size is invalid")
        with (directory / "payload.bin").open("rb") as stream:
            payload = stream.read(self.max_bytes + 1)
        if (
            len(payload) != manifest["bytes"]
            or hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]
        ):
            raise ValueError("checkpoint payload digest mismatch")
        checkpoint = self._decode(payload)
        if checkpoint.sha256 != manifest["checkpoint_sha256"]:
            raise ValueError("decoded checkpoint identity differs from its manifest")
        return checkpoint, payload, manifest

    def load(self, name):
        """Verify bytes before decoding, then verify complete typed state identity."""
        return self._read(name)[0]

    def clone(self, source, destination):
        """Create a byte-identical copy and verify the complete reloaded state."""
        checkpoint, payload, manifest = self._read(source)
        expected = checkpoint.sha256
        expected_types = codec_state_identity(checkpoint)
        directory = self._directory(destination)
        directory.mkdir(exist_ok=False)
        with (directory / "payload.bin").open("xb") as stream:
            stream.write(payload)
        with (directory / "manifest.json").open("xb") as stream:
            stream.write(canonical_bytes(manifest))
        copied = self.load(destination)
        if copied.sha256 != expected or codec_state_identity(copied) != expected_types:
            raise ValueError("checkpoint clone changed state")
        return copied
