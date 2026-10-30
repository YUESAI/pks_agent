"""Regression tests for sparse longitudinal Motor feature construction.

Run from repository root:
  python -m unittest tests.test_phase77_sparse_motor -v
"""
import unittest
import numpy as np
from scripts.phase36_masked_site_feasibility import VISIBLE
from scripts.phase76_reliability_gated_wm import engineered

class SparseMotorFeatureTests(unittest.TestCase):
    def record(self,visits):
        return {
            "history":visits,
            "anchor":0,
            "horizon":12,
            "optional":{},
        }

    def test_all_motor_items_missing(self):
        row=self.record([{"month":0,"scores":{}}])
        x=engineered([row],())
        self.assertEqual(x.shape,(1,6*len(VISIBLE)+4))
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.all(x[0,0:6]==0))

    def test_single_motor_item_measured_once(self):
        item=VISIBLE[0]
        row=self.record([{"month":0,"scores":{item:2}}])
        x=engineered([row],())
        np.testing.assert_allclose(x[0,:6],[.5,.5,0.,0.,1/6,1.])

    def test_item_only_observed_at_later_visit(self):
        item=VISIBLE[0]
        row=self.record([
            {"month":0,"scores":{}},
            {"month":6,"scores":{item:3}},
        ])
        x=engineered([row],())
        np.testing.assert_allclose(x[0,:6],[.75,.75,0.,0.,1/6,1.])

    def test_slope_from_two_measured_visits(self):
        item=VISIBLE[0]
        row=self.record([
            {"month":0,"scores":{item:1}},
            {"month":6,"scores":{}},
            {"month":12,"scores":{item:3}},
        ])
        x=engineered([row],())
        np.testing.assert_allclose(x[0,:6],[.25,.75,.5,.25,2/6,1.])

    def test_empty_visit_history_fails(self):
        with self.assertRaisesRegex(ValueError,"Missing chronological"):
            engineered([self.record([])],())

if __name__=="__main__":
    unittest.main()
