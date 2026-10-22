"""Synthetic Phase 13: no clinical source material required."""
import unittest
from scripts.phase13_numeric_policy_cv import (
 bin_value,train_model,policy_episode,evaluate)
class Phase13Tests(unittest.TestCase):
 def test_numeric_and_action_costs(self):
  self.assertIsNone(bin_value("cognitive_test",None))
  self.assertIsNone(bin_value("cognitive_test",99))
  rows=[("PD",True,{"age":"70","motor_exam":"35","cognitive_test":"20"}),
        ("HC",True,{"age":"65","motor_exam":"3","cognitive_test":"28"})]*8
  model,success=train_model(rows)
  for p in ("clinical","smell_first","random","adaptive"):
   score,cost,n=policy_episode(rows[0][2],model,success,p,5,42,"SECRET")
   self.assertLessEqual(cost,5)
   self.assertGreaterEqual(n,1)
  self.assertEqual(policy_episode(rows[0][2],model,success,"adaptive",5,42,"SECRET"),
                   policy_episode(rows[0][2],model,success,"adaptive",5,42,"SECRET"))
 def test_aggregate_metrics(self):
  score=evaluate([(2,True,2,2),(-2,False,2,2)])
  self.assertEqual(score["balanced_accuracy"],1.0)
  self.assertEqual(score["auroc"],1.0)
if __name__=="__main__":unittest.main()
