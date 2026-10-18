"""Phase16 synthetic tests for adaptive-vs-static benchmark."""
import unittest
from scripts.phase16_static_adaptive import run_episode, compare
from scripts.phase13_numeric_policy_cv import train_model

class Phase16Tests(unittest.TestCase):
 def test_shared_model_and_budget(self):
  rows=[("PD",True,{"age":"65","motor_exam":"35","smell_test":"15"}),
        ("HC",True,{"age":"65","motor_exam":"4","smell_test":"32"})]*12
  model,success=train_model(rows)
  for budget in (1,2,3,4,5):
   for space in ("motor_allowed","motor_excluded"):
    for policy in ("clinical","motor_first","static_ig","adaptive_ig"):
     score,cost,returned,requested=run_episode(model,success,rows[0][2],
                                               policy,budget,space)
     self.assertLessEqual(cost,budget)
     self.assertEqual(len(set(requested)),len(requested))
     if space=="motor_excluded":
      self.assertNotIn("motor_exam",requested)
  a={"x":(2.,True,2,2,("age","motor_exam")),
     "y":(-2.,False,2,2,("age","motor_exam"))}
  b={"x":(2.,True,2,2,("age","motor_exam")),
     "y":(-2.,False,2,2,("age","motor_exam"))}
  differences=compare(a,b,2,100,42,10)
  self.assertTrue(all(d["point_difference"]==0 for d in differences))
  self.assertTrue(all(d["different_request_sequences_pct"]==0 for d in differences))

if __name__=="__main__":
 unittest.main()
