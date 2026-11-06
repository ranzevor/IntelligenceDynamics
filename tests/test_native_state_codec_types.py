"""CPU array codec qualification; requires the optional real-native stack."""

import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from intelligence_dynamics.native import (
    CheckpointCodec,
    CheckpointStore,
    ExecutionState,
    NativeCheckpoint,
    native_state_snapshot,
)
from intelligence_dynamics.native._state import codec_state_identity

try:
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    import optax
except ImportError:
    jax = jnp = ml_dtypes = np = optax = None


if np is not None:

    class TypeDroppingArray(np.ndarray):
        def __deepcopy__(self, memo):
            return np.asarray(self).copy()


def fixture(array=None, impl="threefry2x32"):
    if array is None:
        array = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    params = {"weight": array}
    state = {
        "params": params,
        "optim": optax.adam(0.001).init(params),
        "key": jax.random.key(17, impl=impl),
        "hypers": {"freeze_steps": float("inf")},
    }
    return NativeCheckpoint(ExecutionState(state, {}, 0))


def codec(decode=pickle.loads, encode=pickle.dumps):
    return CheckpointCodec("real-native-local-pickle", encode, decode)


def changed_array(value, convert):
    state = copy.deepcopy(value.state.agent_state)
    state["params"]["weight"] = convert(state["params"]["weight"])
    return replace(value, state=replace(value.state, agent_state=state))


@unittest.skipIf(jax is None, "requires real JAX, Optax, NumPy and ml_dtypes")
class NativeCodecArrayTypeTests(unittest.TestCase):
    def assert_rejected_codec(self, value, convert):
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(
                root, codec(lambda raw: changed_array(pickle.loads(raw), convert))
            )
            with self.assertRaisesRegex(ValueError, "codec changed"):
                store.save("lossy", value)
            self.assertFalse((Path(root) / "lossy").exists())

    def test_jax_to_numpy_and_numpy_to_jax_rejected(self):
        self.assert_rejected_codec(fixture(), lambda value: np.asarray(value).copy())
        self.assert_rejected_codec(fixture(np.asarray([1.0, 2.0], dtype=np.float32)), jnp.asarray)

    def test_numpy_scalar_to_array_rejected(self):
        self.assert_rejected_codec(fixture(np.float32(1)), np.asarray)

    def test_deepcopy_array_type_loss_rejected_before_encoder(self):
        value = fixture(np.asarray([1.0, 2.0], dtype=np.float32).view(TypeDroppingArray))
        encoded = []

        def encode(checkpoint):
            encoded.append(True)
            return pickle.dumps(checkpoint)

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, codec(encode=encode))
            with self.assertRaisesRegex(ValueError, "copying changed"):
                store.save("lossy", value)
            self.assertEqual(encoded, [])
            self.assertFalse((Path(root) / "lossy").exists())

    def test_jax_weak_type_change_rejected(self):
        original = fixture(jnp.asarray(1.0))
        self.assertTrue(original.state.agent_state["params"]["weight"].weak_type)
        self.assert_rejected_codec(original, lambda value: jnp.asarray(value, dtype=jnp.float32))

    def test_numpy_jax_and_bfloat16_preserving_codecs_pass(self):
        values = [
            np.asarray([1.0, 2.0], dtype=np.float32),
            jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16),
            jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16),
        ]
        for array in values:
            with self.subTest(type=type(array), dtype=str(array.dtype)):
                with tempfile.TemporaryDirectory() as root:
                    value = fixture(array)
                    store = CheckpointStore(root, codec())
                    store.save("source", value)
                    loaded = store.load("source")
                    cloned = store.clone("source", "copy")
                    self.assertEqual(codec_state_identity(loaded), codec_state_identity(value))
                    self.assertEqual(codec_state_identity(cloned), codec_state_identity(value))
                    self.assertIs(
                        type(loaded.state.agent_state["optim"][0]), optax.ScaleByAdamState
                    )
                    self.assertEqual(int(loaded.state.agent_state["optim"][0].count), 0)
                    self.assertEqual(
                        (Path(root) / "source/payload.bin").read_bytes(),
                        (Path(root) / "copy/payload.bin").read_bytes(),
                    )

    def test_three_typed_keys_keep_codec_semantics(self):
        for impl in ("threefry2x32", "rbg", "unsafe_rbg"):
            with self.subTest(impl=impl), tempfile.TemporaryDirectory() as root:
                value = fixture(impl=impl)
                store = CheckpointStore(root, codec())
                store.save("source", value)
                copied = store.clone("source", "copy")
                self.assertEqual(str(jax.random.key_impl(copied.state.agent_state["key"])), impl)
                self.assertEqual(codec_state_identity(value), codec_state_identity(copied))

    def test_clone_rejects_decoder_array_type_instability(self):
        calls = []

        def decode(raw):
            value = pickle.loads(raw)
            calls.append(True)
            return changed_array(value, np.asarray) if len(calls) == 3 else value

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, codec(decode))
            store.save("source", fixture())
            with self.assertRaisesRegex(ValueError, "clone changed state"):
                store.clone("source", "copy")
            self.assertEqual(len(calls), 3)

    def test_encoder_mutating_caller_array_type_rejected(self):
        original = fixture()

        def encode(value):
            original.state.agent_state["params"]["weight"] = np.asarray(
                original.state.agent_state["params"]["weight"]
            )
            return pickle.dumps(value)

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, codec(encode=encode))
            with self.assertRaisesRegex(ValueError, "codec changed"):
                store.save("lossy", original)
            self.assertFalse((Path(root) / "lossy").exists())

    def test_clone_captures_type_before_cached_decoder_mutates_it(self):
        cached = []
        calls = []

        def decode(raw):
            calls.append(True)
            if not cached:
                cached.append(pickle.loads(raw))
            if len(calls) == 3:
                cached[0].state.agent_state["params"]["weight"] = np.asarray(
                    cached[0].state.agent_state["params"]["weight"]
                )
            return cached[0]

        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore(root, codec(decode))
            store.save("source", fixture())
            with self.assertRaisesRegex(ValueError, "clone changed state"):
                store.clone("source", "copy")

    def test_array_dtype_change_rejected(self):
        self.assert_rejected_codec(fixture(), lambda value: value.astype(jnp.float16))

    def test_ordinary_content_digest_stays_backend_compatible(self):
        a = np.asarray([1.0, 2.0], dtype=np.float32)
        b = jnp.asarray(a)
        self.assertEqual(native_state_snapshot(a), native_state_snapshot(b))
        self.assertNotEqual(codec_state_identity(a), codec_state_identity(b))

    def test_bfloat16_backend_type_erasure_rejected(self):
        self.assert_rejected_codec(fixture(jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16)), np.asarray)


@unittest.skipIf(jax is None, "requires real JAX, Optax, NumPy and ml_dtypes")
class Bfloat16BoundaryTests(unittest.TestCase):
    def test_finite_jax_numpy_and_scalar_bfloat16_metadata(self):
        for value in (
            jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16),
            np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16),
            ml_dtypes.bfloat16(1.0),
        ):
            with self.subTest(type=type(value)):
                snap = native_state_snapshot(value)
                self.assertEqual(snap["tensors"][0]["dtype"], "bfloat16")
                self.assertEqual(snap["tensor_bytes"], np.asarray(value).size * 2)
                self.assertEqual(
                    snap["tensors"][0]["sha256"],
                    hashlib.sha256(np.asarray(value).tobytes()).hexdigest(),
                )

    def test_bfloat16_semantic_dtype_not_raw_void_or_float16(self):
        bfloat = np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16)
        float16 = bfloat.view(np.float16)
        self.assertEqual(bfloat.tobytes(), float16.tobytes())
        self.assertNotEqual(
            native_state_snapshot(bfloat)["sha256"], native_state_snapshot(float16)["sha256"]
        )
        for invalid in (
            bfloat.view("V2"),
            np.asarray([object()], dtype=object),
            np.asarray([(1,)], dtype=[("x", "i2")]),
        ):
            with self.subTest(dtype=str(invalid.dtype)), self.assertRaises(ValueError):
                native_state_snapshot(invalid)

    def test_bfloat16_nan_and_infinities_rejected_in_arrays_and_optax(self):
        for scalar in (float("nan"), float("inf"), float("-inf")):
            for factory in (
                lambda x: jnp.asarray(x, dtype=jnp.bfloat16),
                lambda x: np.asarray(x, dtype=ml_dtypes.bfloat16),
            ):
                with self.subTest(scalar=repr(scalar)):
                    with self.assertRaises(ValueError):
                        native_state_snapshot({"params": factory([scalar])})
                    value = fixture(jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16))
                    value.state.agent_state["optim"][0].mu["weight"] = factory([scalar, 0.0])
                    with self.assertRaises(ValueError):
                        native_state_snapshot(value)

    def test_bfloat16_positive_freeze_infinity_only_at_correct_path(self):
        for factory in (
            lambda x: jnp.asarray(x, dtype=jnp.bfloat16),
            lambda x: np.asarray(x, dtype=ml_dtypes.bfloat16),
        ):
            native_state_snapshot({"hypers": {"freeze_steps": factory(float("inf"))}})
            for bad in (float("nan"), float("-inf")):
                with self.assertRaises(ValueError):
                    native_state_snapshot({"hypers": {"freeze_steps": factory(bad)}})
            with self.assertRaises(ValueError):
                native_state_snapshot({"freeze_steps": factory(float("inf"))})


@unittest.skipIf(jax is None, "requires real JAX, Optax, NumPy and ml_dtypes")
class CheckpointV1CompatibilityTests(unittest.TestCase):
    def test_v1_manifest_contract_and_corrupt_formats(self):
        cases = [
            lambda value: {**value, "format": "native-checkpoint-v2"},
            lambda value: {**value, "unexpected": "x"},
            lambda value: {k: v for k, v in value.items() if k != "checkpoint_sha256"},
        ]
        for change in cases:
            with tempfile.TemporaryDirectory() as root:
                store = CheckpointStore(root, codec())
                store.save("source", fixture())
                path = Path(root) / "source/manifest.json"
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["format"], "native-checkpoint-v1")
                self.assertEqual(
                    set(manifest),
                    {"format", "codec", "bytes", "payload_sha256", "checkpoint_sha256"},
                )
                path.write_text(json.dumps(change(manifest)))
                with self.assertRaisesRegex(ValueError, "incompatible contract"):
                    store.load("source")


@unittest.skipIf(jax is None, "requires real JAX, Optax, NumPy and ml_dtypes")
class CodecScalarDepthCompatibilityTests(unittest.TestCase):
    frontiers = ((1, 11), (3, 8), (5, 5), (7, 2))

    @staticmethod
    def wrapped(leaf, mappings, lists):
        for _ in range(mappings):
            leaf = {"x": leaf}
        for _ in range(lists):
            leaf = [leaf]
        return leaf

    def test_scalar_frontiers_save_load_clone_resave(self):
        leaves = (
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(1.0),
            np.asarray(1.0, dtype=np.float32),
            np.float32(1.0),
            jnp.asarray(1.0, dtype=jnp.bfloat16),
            ml_dtypes.bfloat16(1.0),
        )
        for mappings, lists in self.frontiers:
            for leaf in leaves:
                with self.subTest(layers=(mappings, lists), type=type(leaf), dtype=str(leaf.dtype)):
                    value = NativeCheckpoint(
                        ExecutionState(self.wrapped(leaf, mappings, lists), {}, 0)
                    )
                    with tempfile.TemporaryDirectory() as root:
                        store = CheckpointStore(root, codec())
                        store.save("source", value)
                        loaded = store.load("source")
                        cloned = store.clone("source", "copy")
                        store.save("resaved", loaded)
                        self.assertEqual(value.sha256, loaded.sha256)
                        self.assertEqual(codec_state_identity(value), codec_state_identity(cloned))
                        for name in ("copy", "resaved"):
                            for file in ("payload.bin", "manifest.json"):
                                self.assertEqual(
                                    (Path(root) / "source" / file).read_bytes(),
                                    (Path(root) / name / file).read_bytes(),
                                )

    def test_extra_layer_still_exceeds_public_depth_limit(self):
        for mappings, lists in self.frontiers:
            for leaf in (jnp.asarray(1.0, dtype=jnp.float32), np.asarray(1.0, dtype=np.float32)):
                with self.subTest(layers=(mappings, lists), type=type(leaf)):
                    state = ExecutionState(self.wrapped(leaf, mappings, lists + 1), {}, 0)
                    with self.assertRaisesRegex(ValueError, "maximum JSON nesting depth"):
                        NativeCheckpoint(state)

    def test_frontier_backend_and_weak_type_changes_rejected(self):
        pairs = (
            (jnp.asarray(1.0, dtype=jnp.float32), np.asarray(1.0, dtype=np.float32)),
            (jnp.asarray(1.0), jnp.asarray(1.0, dtype=jnp.float32)),
            (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(1.0)),
        )
        for mappings, lists in self.frontiers:
            for before, after in pairs:
                with self.subTest(layers=(mappings, lists), type=type(after)):
                    original = NativeCheckpoint(
                        ExecutionState(self.wrapped(before, mappings, lists), {}, 0)
                    )
                    changed = NativeCheckpoint(
                        ExecutionState(self.wrapped(after, mappings, lists), {}, 0)
                    )
                    self.assertEqual(original.sha256, changed.sha256)
                    with tempfile.TemporaryDirectory() as root:
                        store = CheckpointStore(root, codec(lambda raw: changed))
                        with self.assertRaisesRegex(ValueError, "codec changed"):
                            store.save("lossy", original)
                        self.assertFalse((Path(root) / "lossy").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
