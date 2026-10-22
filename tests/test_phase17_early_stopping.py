"""Synthetic Phase17 early-stopping regression tests, no private records."""
import unittest
from scripts.phase17_early_stopping import replay, brier_by_class
from scripts.phase13_numeric_policy_cv import train_model

class Phase17Tests(unittest.TestCase):
    def setUp(self):
        examples=[
            ("PD",True,{"age":"66","motor_exam":"40","cognitive_test":"18"}),
            ("HC",True,{"age":"65","motor_exam":"3","cognitive_test":"29"})
        ]*16
        self.model,self.success=train_model(examples)
        self.values=examples[0][2]

    def test_budget_and_stop(self):
        for space in ("motor_allowed","motor_excluded"):
            for policy in ("clinical","motor_first","static_ig","adaptive_ig"):
                for threshold in (None,.7,.8,.9,.95,.99):
                    value=replay(self.model,self.success,self.values,policy,space,3,threshold)
                    self.assertLessEqual(value[1],3)
                    self.assertGreaterEqual(value[3],1)
                    self.assertIn(value[4],("budget_or_catalog_exhausted","score_threshold"))
                    self.assertEqual(value,replay(self.model,self.success,self.values,
                                                 policy,space,3,threshold))

    def test_no_stop_reference_and_brier(self):
        score,cost,n_returned,n_requested,reason=replay(
            self.model,self.success,self.values,"static_ig","motor_allowed",2,None)
        self.assertEqual(cost,2)
        self.assertEqual(reason,"budget_or_catalog_exhausted")
        samples=[(2.,True,2,2,2,"budget_or_catalog_exhausted"),
                 (-2.,False,2,2,2,"budget_or_catalog_exhausted")]
        self.assertGreaterEqual(brier_by_class(samples),0.)
        self.assertLessEqual(brier_by_class(samples),1.)

if __name__=="__main__":
    unittest.main()
