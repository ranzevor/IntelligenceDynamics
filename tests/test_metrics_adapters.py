import math
import unittest

from intelligence_dynamics.adapters import DalScale, dal_rows, forager_rows
from intelligence_dynamics.metrics import observations, pareto_front, plasticity, recovery


class MetricTests(unittest.TestCase):
    def test_recovery_floor_and_auc(self):
        r = recovery([0, 2, 4], [2, 4, 6], baseline=6, floor=2, fraction=0.5)
        self.assertEqual(r.first_recovery_step, 2)
        self.assertEqual(r.target, 4)
        self.assertAlmostEqual(r.normalized_auc, 0.5)
        r = recovery([4], [6], baseline=6, floor=2)
        self.assertEqual(r.normalized_auc, 1)
        r = recovery([0, 1], [1e308, 1e308], baseline=1e308)
        self.assertEqual(r.normalized_auc, 1)

    def test_bad_recovery_series(self):
        for steps, values in [
            ([], []),
            ([0, 0], [1, 2]),
            ([True], [1]),
            ([0], [math.nan]),
            ([0, 1], [1]),
        ]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                recovery(steps, values, baseline=1)
        for options in [
            {"baseline": 0},
            {"baseline": 1, "floor": 1},
            {"baseline": True},
            {"baseline": 1, "fraction": 1.1},
        ]:
            with self.assertRaises(ValueError):
                recovery([0], [1], **options)

    def test_plasticity_and_missing_observations(self):
        self.assertEqual(plasticity([(0, 2), (4, 1)]), [(0, 1), (4, 0.5)])
        with self.assertRaises(ValueError):
            plasticity([(0, 0)])
        with self.assertRaises(ValueError):
            plasticity([(0, 1), (0, 2)])
        self.assertEqual(
            observations([1, None, 3]), {"count": 3, "observed": 2, "missing": 1, "mean": None}
        )
        self.assertEqual(observations([1, 3])["mean"], 2)
        self.assertEqual(observations([])["mean"], None)
        with self.assertRaises(ValueError):
            observations([True])

    def test_pareto_front(self):
        pts = [
            {"score": 2, "cost": 1},
            {"score": 1, "cost": 2},
            {"score": 3, "cost": 3},
            {"score": 2, "cost": 1},
        ]
        self.assertEqual(pareto_front(pts, maximize=["score"], minimize=["cost"]), [0, 2, 3])
        with self.assertRaises(ValueError):
            pareto_front([{"score": None}], maximize=["score"], minimize=[])
        with self.assertRaises(ValueError):
            pareto_front([], maximize=["a"], minimize=["a"])


class AdapterTests(unittest.TestCase):
    def test_dal_positional_contract(self):
        line = "1,2,3,4,.1,.2,.3,.4,.5,None,0,0.7,0.8"
        r = list(dal_rows([line], run_id="r", scale=DalScale(2, 3, 4)))[0]
        self.assertEqual(r.step, 24)
        self.assertEqual(r.capability, 2)
        self.assertIsNone(r.plasticity_probe_score)
        self.assertEqual(r.extras["native"]["learning_progress"], [0.7, 0.8])
        self.assertIsNone(r.extras["native"]["nm_only_fitness"])

    def test_dal_bad_data(self):
        valid = "1,2,3,4,.1,.2,.3,.4,.5,None,0"
        for row in [
            "",
            valid.replace(".1", "nan"),
            valid.replace(".5", "2"),
            valid.replace("None", ""),
            "01" + valid[1:],
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                list(dal_rows([row], run_id="r", scale=DalScale(1, 1, 1)))
        with self.assertRaises(ValueError):
            list(dal_rows([valid, valid], run_id="r", scale=DalScale(1, 1, 1)))
        with self.assertRaises(ValueError):
            DalScale(True, 1, 1)

    def test_forager_explicit_fields(self):
        r = list(
            forager_rows(
                [{"steps": 4, "reward": 0.3, "grad_norm": 2, "novel": {"a": 1}}],
                run_id="r",
                capability_field="reward",
            )
        )[0]
        self.assertEqual(r.capability, 0.3)
        self.assertEqual(r.extras["native"]["novel"], {"a": 1})
        self.assertIsNone(r.plasticity_probe_score)
        for rows in [
            [{"steps": True, "reward": 1}],
            [{"steps": 1, "reward": math.inf}],
            [{"steps": 1, "reward": 1}] * 2,
        ]:
            with self.assertRaises(ValueError):
                list(forager_rows(rows, run_id="r", capability_field="reward"))


if __name__ == "__main__":
    unittest.main()
