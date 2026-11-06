import unittest
from collections import namedtuple
from types import ModuleType
from unittest.mock import patch

from intelligence_dynamics.native import native_state_snapshot as state_snapshot

try:
    import numpy as np
except ImportError:
    np = None

AdamState = namedtuple("AdamState", "count mu nu")


class TypedKey:
    dtype = "key<fry>"

    def __array__(self, *args, **kwargs):
        raise TypeError("typed keys require key_data")


@unittest.skipIf(np is None, "requires the arrays dependency group")
class NativeArrayIdentityTests(unittest.TestCase):
    def test_optimizer_namedtuple_and_scalar_native_freeze_infinity(self):
        value = {
            "optim": AdamState(
                np.asarray(1), {"weight": np.asarray([0.1])}, {"weight": np.asarray([0.2])}
            ),
            "hypers": {"freeze_steps": np.asarray(float("inf"))},
        }
        observed = state_snapshot(value)
        self.assertEqual(observed["tensor_elements"], 4)
        changed = {"optim": value["optim"]._replace(count=np.asarray(2)), "hypers": value["hypers"]}
        self.assertNotEqual(state_snapshot(changed)["sha256"], observed["sha256"])
        with self.assertRaises(ValueError):
            state_snapshot({"params": np.asarray([float("inf")])})

    def test_typed_key_uses_key_data_and_includes_prng_implementation(self):
        module = ModuleType("jax.random")
        calls = []
        module.key_data = lambda key: (calls.append(key), np.asarray([1, 2], dtype=np.uint32))[1]
        key = TypedKey()
        with patch.dict("sys.modules", {"jax.random": module}):
            first = state_snapshot(key)
            key.dtype = "key<rbg>"
            second = state_snapshot(key)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["tensor_bytes"], 8)


class NativeStructuredReplacementTests(unittest.TestCase):
    def test_namedtuple_field_and_index_replacement_preserve_type(self):
        from intelligence_dynamics.native._values import replace_field

        state = AdamState(1, {"weight": 0.1}, {"weight": 0.2})
        first = replace_field(state, (0,), 2)
        second = replace_field(state, ("mu", "weight"), 0.9)
        self.assertIs(type(first), AdamState)
        self.assertIs(type(second), AdamState)
        self.assertEqual(first.count, 2)
        self.assertEqual(second.mu, {"weight": 0.9})
        self.assertEqual(state.mu, {"weight": 0.1})
        self.assertEqual(second.nu, state.nu)

    def test_immutable_mapping_has_explicit_replacement_rejection(self):
        from types import MappingProxyType

        from intelligence_dynamics.native._values import replace_field

        with self.assertRaisesRegex(ValueError, "mutable mapping"):
            replace_field(MappingProxyType({"weight": 0.1}), ("weight",), 0.2)
