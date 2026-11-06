import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intelligence_dynamics.records import (
    AdaptationEvent,
    ComputationCounters,
    InterventionRecord,
    MemoryAccounting,
    RoutingSummary,
    RunMetadata,
    StepRecord,
    from_object,
    to_object,
)
from intelligence_dynamics.storage import digest, loads, read_records, write_json, write_records
from tests.support import create_symlink


class RecordsTests(unittest.TestCase):
    def test_record_roundtrips(self):
        records = [
            RunMetadata("demo", "A", "tabular", "switching", 1, "0.1.0", digest({"steps": 8})),
            StepRecord(
                "demo",
                1,
                1,
                capability=0,
                computation=ComputationCounters(active_unit_updates=1, potential_unit_updates=4),
                memory=MemoryAccounting(parameter_bytes=64),
                routing=RoutingSummary(active_modules=1, available_modules=2),
            ),
            AdaptationEvent("change", 4, "mapping", True, "Mapping reverses"),
            InterventionRecord("demo", "branch", "memory", "reset", 4, 8, delta_capability=-0.2),
        ]
        for record in records:
            with self.subTest(kind=type(record).__name__):
                self.assertEqual(from_object(to_object(record)), record)

    def test_missing_measurements_are_null(self):
        item = to_object(StepRecord("x", 0, 0))
        self.assertIsNone(item["capability"])
        self.assertIsNone(item["computation"])
        self.assertEqual(StepRecord("x", 0, 0, capability=0).capability, 0)

    def test_invalid_scalars(self):
        for bad in [None, True, -1, 1.5, "0"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                StepRecord("x", bad, 0)
        for bad in [None, True, -1, math.inf, math.nan, "0", 10**1000]:
            with self.subTest(bad=repr(bad)[:30]), self.assertRaises(ValueError):
                StepRecord("x", 0, bad)
        for bad in [True, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                StepRecord("x", 0, 0, capability=bad)
        with self.assertRaises(ValueError):
            StepRecord("", 0, 0)
        with self.assertRaises(ValueError):
            StepRecord("x", 0, 0, protocol_version="2")

    def test_counter_and_routing_bounds(self):
        with self.assertRaises(ValueError):
            ComputationCounters(active_unit_updates=3, potential_unit_updates=2)
        with self.assertRaises(ValueError):
            ComputationCounters(learning_updates=True)
        with self.assertRaises(ValueError):
            MemoryAccounting(parameter_bytes=-1)
        with self.assertRaises(ValueError):
            RoutingSummary(active_modules=2, available_modules=1)
        self.assertEqual(
            ComputationCounters(
                active_unit_updates=0, potential_unit_updates=10
            ).unit_activation_ratio,
            0,
        )
        self.assertIsNone(
            ComputationCounters(
                active_unit_updates=0, potential_unit_updates=0
            ).unit_activation_ratio
        )

    def test_metadata_event_intervention(self):
        with self.assertRaises(ValueError):
            RunMetadata("r", "D", "s", "e", 1, "v", "c")
        with self.assertRaises(ValueError):
            RunMetadata("r", "A", "s", "e", True, "v", "c")
        with self.assertRaises(ValueError):
            AdaptationEvent("e", 1, "change", 1, "changed")
        with self.assertRaises(ValueError):
            AdaptationEvent("e", 1, "change", True, "changed", tags={"a": 1})
        with self.assertRaises(ValueError):
            InterventionRecord("r", "i", "m", "reset", 3, 2)
        with self.assertRaises(ValueError):
            InterventionRecord("r", "i", "m", "reset", 0, 2, delta_compute_ratio=math.nan)

    def test_nested_and_unknown_fields(self):
        with self.assertRaises(ValueError):
            StepRecord("r", 0, 0, computation={})
        with self.assertRaises(ValueError):
            from_object({"record_type": "mapping", "value": 1})
        value = to_object(StepRecord("r", 0, 0))
        value["unknown"] = 1
        with self.assertRaises(ValueError):
            from_object(value)
        value = to_object(StepRecord("r", 0, 0))
        value["computation"] = {"nonsense": 1}
        with self.assertRaises(ValueError):
            from_object(value)

    def test_json_extension_contract(self):
        for value in [{1: "x"}, {"x": math.nan}, {"x": object()}]:
            with self.assertRaises(ValueError):
                StepRecord("r", 0, 0, extras=value)
        source = {"nested": [1, 2]}
        record = StepRecord("r", 0, 0, extras=source)
        source["nested"].append(3)
        self.assertEqual(record.extras, {"nested": [1, 2]})
        record.extras["bad"] = math.nan
        with self.assertRaises(ValueError):
            to_object(record)


class StorageTests(unittest.TestCase):
    def test_strict_json(self):
        for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "[1e999]", b'"\xff"']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                loads(value)
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            digest({1: "a"})

    def test_atomic_records(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "measurements.jsonl"
            records = [StepRecord("run", 0, 0), StepRecord("run", 1, 1, capability=0.5)]
            self.assertEqual(write_records(path, records), 2)
            self.assertEqual(list(read_records(path)), records)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_records(path, [])
            self.assertEqual(path.read_bytes(), original)

            def failure():
                yield StepRecord("run", 2, 2)
                raise RuntimeError("synthetic iterator failure")

            with self.assertRaisesRegex(RuntimeError, "iterator"):
                write_records(path, failure(), overwrite=True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(sorted(p.name for p in Path(td).iterdir()), [path.name])
            write_records(path, [], overwrite=True)
            self.assertEqual(path.read_bytes(), b"")

    def test_invalid_data_leaves_target(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x"
            p.write_bytes(b"original")
            record = StepRecord("r", 0, 0)
            record.extras["bad"] = math.inf
            with self.assertRaises(ValueError):
                write_records(p, [record], overwrite=True)
            self.assertEqual(p.read_bytes(), b"original")
            with patch(
                "intelligence_dynamics.storage.os.replace", side_effect=PermissionError("denied")
            ):
                with self.assertRaises(PermissionError):
                    write_json(p, {"a": 1}, overwrite=True)
            self.assertEqual(p.read_bytes(), b"original")

    def test_directories_are_not_output_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x").write_bytes(b"old")
            with self.assertRaises(ValueError):
                write_json(root, {}, overwrite=True)
            self.assertEqual((root / "x").read_bytes(), b"old")

    def test_symbolic_links_are_not_output_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real = root / "real"
            real.mkdir()
            target = real / "x"
            target.write_bytes(b"old")
            link = root / "link"
            create_symlink(self, link, real, directory=True)
            with self.assertRaises(ValueError):
                write_json(link / "x", {}, overwrite=True)
            self.assertEqual(target.read_bytes(), b"old")

    def test_record_line_bounds_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x"
            p.write_text("\n")
            with self.assertRaisesRegex(ValueError, "line 1"):
                list(read_records(p))
            p.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "one MiB"):
                list(read_records(p))
            p.write_text('{"record_type":"StepRecord","run_id":"x"}\n')
            with self.assertRaisesRegex(ValueError, "line 1"):
                list(read_records(p))


if __name__ == "__main__":
    unittest.main()


class SchemaTests(unittest.TestCase):
    def test_record_schema(self):
        from intelligence_dynamics.schema import record_schema

        schema = record_schema()
        self.assertEqual(len(schema["oneOf"]), 4)
        for name in ("RunMetadata", "StepRecord", "AdaptationEvent", "InterventionRecord"):
            self.assertFalse(schema["$defs"][name]["additionalProperties"])
            self.assertIn("protocol_version", schema["$defs"][name]["required"])
        self.assertEqual(schema["$defs"]["StepRecord"]["properties"]["step"]["minimum"], 0)

    def test_json_utf8_only(self):
        from intelligence_dynamics.storage import loads

        with self.assertRaises(ValueError):
            loads('{"value":1}'.encode("utf-16"))


class SchemaFileTests(unittest.TestCase):
    def test_schema_file_matches_runtime_contract(self):
        import json
        from pathlib import Path

        from intelligence_dynamics.schema import record_schema

        path = Path(__file__).resolve().parents[1] / "configs" / "measurement-record.schema.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record_schema())
