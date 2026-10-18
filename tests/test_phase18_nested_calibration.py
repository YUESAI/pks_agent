"""Synthetic Phase18 nested calibration and paired stopping tests."""
import unittest
from scripts.phase18_nested_calibration import (
    calibration_split, IsotonicCalibrator, trace_static,
    choose_snapshot, summarize, paired_delta)
from scripts.phase13_numeric_policy_cv import train_model

class Phase18Tests(unittest.TestCase):
 def test_patient_disjoint_calibration_and_monotone(self):
  records=[("PD_"+str(i),"PD",True,{"age":"68","motor_exam":"30"})
           for i in range(24)]+[
           ("HC_"+str(i),"HC",True,{"age":"68","motor_exam":"3"})
           for i in range(24)]
  proper,cal=calibration_split(records,11)
  self.assertFalse({r[0] for r in proper}&{r[0] for r in cal})
  self.assertEqual(len(proper)+len(cal),48)
  curve=IsotonicCalibrator([(-2,False,.5),(-1,False,.5),
                           (1,True,.5),(2,True,.5)])
  self.assertLessEqual(curve.predict(-2),curve.predict(2))
 def test_same_trajectories_and_zero_differences(self):
  rows=[("PD",True,{"age":"68","motor_exam":"35"}),
        ("HC",True,{"age":"68","motor_exam":"3"})]*15
  model,success=train_model(rows)
  trajectory=trace_static(model,success,rows[0][2],"motor_allowed",5)
  self.assertLessEqual(trajectory[-1][1],5)
  self.assertEqual(choose_snapshot(trajectory,"no_stop")[:2],
                   (trajectory[-1][0],trajectory[-1][1]))
  sample={"a":(True,2.,2,True,.8),"b":(False,-2.,2,False,.2)}
  metrics=summarize(list(sample.values()))
  self.assertEqual(metrics["balanced_accuracy"],1.)
  intervals=paired_delta(sample,sample,100,17)
  self.assertTrue(all(x["difference"]==0 for x in intervals))
  self.assertTrue(all(x["lower_95"]==0 for x in intervals))

if __name__=="__main__":
 unittest.main()
