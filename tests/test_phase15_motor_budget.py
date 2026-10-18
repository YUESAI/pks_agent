"""Synthetic Phase 15 tests, no patient-level clinical data."""
import unittest
from scripts.phase15_motor_budget import EvidenceOracle, play, expected_gain
from scripts.phase13_numeric_policy_cv import train_model

class Phase15Tests(unittest.TestCase):
    def test_request_budget_and_masked_motor(self):
        rows=[("PD",True,{"age":"65","motor_exam":"30","smell_test":"14"}),
              ("HC",True,{"age":"66","motor_exam":"2","smell_test":"33"})]*10
        model,success=train_model(rows)
        values=rows[0][2]
        for policy in ("clinical","smell_first","random","adaptive"):
            for budget in (1,2,3,4,5):
                a=play(model,success,values,policy,"SYNTHETIC",budget,17,"motor_allowed")
                b=play(model,success,values,policy,"SYNTHETIC",budget,17,"motor_excluded")
                self.assertLessEqual(a[1],budget)
                self.assertLessEqual(b[1],budget)
                self.assertEqual(b,play(model,success,values,policy,"SYNTHETIC",
                                        budget,17,"motor_excluded"))
    def test_oracle_does_not_report_unrequested_result(self):
        oracle=EvidenceOracle({"age":"60","motor_exam":"12"},2)
        self.assertEqual(oracle.request("age"),"60")
        self.assertEqual(oracle.request("cognitive_test"),None)
        self.assertEqual(oracle.returned,1)
        with self.assertRaises(ValueError):oracle.request("cognitive_test")
        with self.assertRaises(ValueError):oracle.request("smell_test")
if __name__=="__main__":unittest.main()
