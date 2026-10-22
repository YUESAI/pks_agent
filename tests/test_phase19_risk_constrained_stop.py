"""Synthetic Phase19 threshold-selection and direction-consistency tests."""
import unittest
from scripts.phase19_risk_constrained_stop import (
    select,choose_threshold,score_metrics,strata_bootstrap)
class MockCal:
    def predict(self,score):
        return .98 if score>=0 else .02
class ReverseCal:
    def predict(self,score):
        return .02 if score>=0 else .98
class Phase19Tests(unittest.TestCase):
    def test_direction_conflict_never_stops(self):
        trace=[(3.0,2,1),(4.0,4,2)]
        self.assertEqual(select(trace,"stop",ReverseCal(),.9)[3],False)
        self.assertEqual(select(trace,"stop",MockCal(),.9)[3],True)
        self.assertEqual(select(trace,"no_stop",MockCal(),.9)[1],4)
    def test_stratified_paired_bootstrap(self):
        rows=[(True,2.0,5,False,False)]*10+[(False,-2.0,5,False,False)]*10
        output=strata_bootstrap(rows,rows,100,17,
                                 ("balanced_accuracy","mean_cost"))
        self.assertEqual(output["balanced_accuracy"],(0.0,0.0,0.0))
        self.assertEqual(output["mean_cost"],(0.0,0.0,0.0))
    def test_threshold_selection_conservative_fallback(self):
        traces=[]
        for i in range(12):
            traces.append(("PD"+str(i),"PD",[(2.0,2,1),(3.0,5,2)],MockCal()))
            traces.append(("HC"+str(i),"HC",[(-2.0,2,1),(-3.0,5,2)],MockCal()))
        threshold,cost,loss=choose_threshold(traces,.02,100,3)
        self.assertIsNotNone(threshold)
        self.assertLess(cost,5)
        self.assertEqual(score_metrics([(True,2.0,2,True,False),
                                         (False,-2.0,2,True,False)])["balanced_accuracy"],1.)
if __name__=="__main__":
    unittest.main()
