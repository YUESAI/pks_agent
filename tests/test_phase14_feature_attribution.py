"""Synthetic Phase14 tests: no private patient data."""
import unittest
from scripts.phase14_feature_attribution import (
 record_score, paired_bootstrap, motor_source_counts)
from scripts.phase13_numeric_policy_cv import train_model

class Phase14Tests(unittest.TestCase):
 def test_ablation_only_uses_selected_values(self):
  rows=[("PD",True,{"motor_exam":"40","cognitive_test":"20"}),
        ("HC",True,{"motor_exam":"3","cognitive_test":"29"})]*8
  model,success=train_model(rows)
  score,n=record_score(model,rows[0][2],("motor_exam",))
  self.assertEqual(n,1)
  no_motor,n2=record_score(model,rows[0][2],("age",))
  self.assertEqual(n2,0)
  self.assertEqual(no_motor,0.0)
 def test_paired_bootstrap_same_subjects(self):
  pairs=[(True,2.,3.) for _ in range(10)]+[(False,-2.,-3.) for _ in range(10)]
  rows=paired_bootstrap(pairs,100,12)
  self.assertEqual(len(rows),2)
  self.assertTrue(all(r["point_difference"]==0 for r in rows))
  self.assertTrue(all(r["bootstrap_lower_95"]==0 for r in rows))

if __name__=="__main__":
 unittest.main()
