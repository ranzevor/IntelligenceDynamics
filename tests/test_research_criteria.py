import json
import unittest
from dataclasses import asdict

from intelligence_dynamics.research import Criterion, assess


class CriteriaTests(unittest.TestCase):
    def test_each_direction_includes_its_exact_boundary(self):
        expected = {"ge": True, "gt": False, "le": True, "lt": False, "eq": True}
        for direction, outcome in expected.items():
            with self.subTest(direction=direction):
                rule = Criterion("accuracy", "score", direction, 0.75)
                report = assess({"score": 0.75}, (rule,))
                self.assertIs(report.passed, outcome)
                self.assertEqual(report.results[0].value, 0.75)
                self.assertEqual(report.results[0].threshold, 0.75)
                json.dumps(asdict(report), allow_nan=False)

    def test_missing_metric_is_never_zero(self):
        rule = Criterion("cost", "cost", "le", 0.1)
        for values in ({}, {"cost": None}):
            with self.subTest(values=values):
                report = assess(values, (rule,))
                self.assertIsNone(report.passed)
                self.assertIsNone(report.results[0].value)
                self.assertIsNone(report.results[0].passed)
                self.assertEqual((report.evaluated, report.missing), (0, 1))
        observed = assess({"cost": 0}, (rule,))
        self.assertIs(observed.passed, True)
        self.assertEqual(observed.missing, 0)

    def test_three_valued_joint_truth_tables(self):
        rules = (Criterion("first", "a", "ge", 1), Criterion("second", "b", "ge", 1))
        for left in (True, False, None):
            for right in (True, False, None):
                values = {
                    "a": None if left is None else int(left),
                    "b": None if right is None else int(right),
                }
                with self.subTest(left=left, right=right):
                    all_expected = (
                        False
                        if left is False or right is False
                        else None
                        if left is None or right is None
                        else True
                    )
                    any_expected = (
                        True
                        if left is True or right is True
                        else None
                        if left is None or right is None
                        else False
                    )
                    self.assertIs(assess(values, rules, combine="all").passed, all_expected)
                    self.assertIs(assess(values, rules, combine="any").passed, any_expected)

    def test_directions_and_thresholds_come_only_from_config(self):
        strict = Criterion("advantage", "difference", "ge", 0.2)
        loose = Criterion("advantage", "difference", "ge", 0.1)
        self.assertFalse(assess({"difference": 0.15}, (strict,)).passed)
        self.assertTrue(assess({"difference": 0.15}, (loose,)).passed)

    def test_invalid_criteria_and_values_fail_explicitly(self):
        for arguments in (
            ("", "x", "ge", 0),
            ("name", "", "ge", 0),
            ("name", "x", "bad", 0),
            ("name", "x", "ge", float("nan")),
            ("name", "x", "ge", True),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Criterion(*arguments)
        rule = Criterion("name", "x", "ge", 0)
        for values in ({"x": float("nan")}, {"x": True}, {1: 0}, {"x": "1"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                assess(values, (rule,))
        for rules in ((), (rule, rule), (object(),)):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                assess({}, rules)
        with self.assertRaises(ValueError):
            assess({}, (rule,), combine="majority")


if __name__ == "__main__":
    unittest.main()
