"""Synthetic Phase45 evidence-access, WM and RSI nested-CV regression tests."""
import random,tempfile,unittest
from pathlib import Path
from scripts.phase45_world_model import (
    ObservationWorldModel,PatientEvidenceService,query_policy
)
from scripts.phase45_wm_rsi_agents import run

def mock_data(n=120):
    rng=random.Random(123)
    names=tuple("NP3FAKE"+str(i) for i in range(12))
    records=[]
    for i in range(n):
        features={name:rng.choice([0,0,0,1,2,3]) for name in names}
        label=int(sum(features[name] for name in names[:3])>=2)
        records.append({"pid":"SYNTHETIC_SECRET_"+str(i),
                        "features":features,"label":label,
                        "hidden":float(label*(1+i%3))})
    return records,names

class Phase45Tests(unittest.TestCase):
    def test_patient_cannot_access_hidden_or_repeat(self):
        p=PatientEvidenceService({"a":1,"b":2},("a","b"),1)
        self.assertEqual(p.request("a")["recorded_score"],1)
        with self.assertRaises(PermissionError):p.request("b")
        with self.assertRaises(PermissionError):p.request("a")
        with self.assertRaises(PermissionError):p.request("hidden_six")

    def test_world_model_distribution_and_dynamic_queries(self):
        rows,names=mock_data()
        wm=ObservationWorldModel(names).fit(rows)
        self.assertAlmostEqual(sum(wm.predict_item({},names[0])),1.)
        self.assertGreaterEqual(wm.information_gain({},names[0]),0.)
        for planner in ("random","static","dynamic","variance"):
            patient=PatientEvidenceService(rows[0]["features"],names,4)
            p,n=query_policy(wm,patient,planner,4,rows[0]["pid"],123)
            self.assertEqual(n,4)
            self.assertTrue(0<=p<=1)

    def test_train_only_synthetic_end_to_end(self):
        records,items=mock_data()
        with tempfile.TemporaryDirectory() as td:
            result=run(records,items,Path(td)/"report",folds=3,
                       questions=4,rounds=2,seed=123,bootstraps=100)
            self.assertEqual(result["n_train_episodes"],120)
            for path in (Path(td)/"report").iterdir():
                self.assertNotIn("SYNTHETIC_SECRET_",path.read_text())
            self.assertTrue((Path(td)/"report"/"evolution_trace.csv").is_file())

if __name__=="__main__":
    unittest.main()
