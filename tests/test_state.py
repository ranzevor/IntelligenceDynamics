import unittest
from dataclasses import dataclass

from intelligence_dynamics.state import state_snapshot, verify_state_intervention


class StateTests(unittest.TestCase):
    def test_typed_keys_and_cycles(self):
        self.assertNotEqual(state_snapshot({1: 2})["sha256"], state_snapshot({"1": 2})["sha256"])
        self.assertNotEqual(state_snapshot([1])["sha256"], state_snapshot((1,))["sha256"])
        value = []
        value.append(value)
        with self.assertRaises(ValueError):
            state_snapshot(value)
        with self.assertRaises(ValueError):
            state_snapshot({True: 1})

    def test_dataclass_fields(self):
        @dataclass
        class State:
            carry: list
            parameter: float

        before = State([1, 2], 0.5)
        after = State([0, 0], 0.5)
        self.assertEqual(
            verify_state_intervention(before, after, allowed_paths=[("carry",)]),
            (("carry", 0), ("carry", 1)),
        )
        with self.assertRaises(ValueError):
            verify_state_intervention(before, State([0, 0], 0.6), allowed_paths=[("carry",)])

    def test_arrays_preserve_shape_dtype_bytes_and_footprint(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("arrays dependency is not installed")
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        before = {"carry": original, "parameters": np.array([5.0], dtype=np.float64)}
        snapshot = state_snapshot(before)
        self.assertEqual(snapshot["tensor_elements"], 5)
        self.assertEqual(snapshot["tensor_bytes"], 24)
        self.assertEqual(original.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        after = {"carry": np.zeros_like(original), "parameters": before["parameters"].copy()}
        self.assertEqual(
            verify_state_intervention(before, after, allowed_paths=[("carry",)]), (("carry",),)
        )
        self.assertNotEqual(
            state_snapshot(original)["sha256"],
            state_snapshot(original.astype(np.float64))["sha256"],
        )
        self.assertNotEqual(
            state_snapshot(original)["sha256"], state_snapshot(original.reshape(4))["sha256"]
        )
        with self.assertRaises(ValueError):
            state_snapshot(np.array([float("nan")]))
