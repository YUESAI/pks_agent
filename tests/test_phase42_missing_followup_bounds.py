"""Synthetic Phase42 tests for sharp unknown-followup bounds and privacy."""
import csv,sqlite3,tempfile,unittest,sys
from unittest.mock import patch
from pathlib import Path
from scripts.phase42_missing_followup_bounds import (
    exact_bounds,build_registry,topk,audit
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE

class Phase42Tests(unittest.TestCase):
 def test_sharp_bounds(self):
  registry={
   "A":{"visible":8.,"future_positive":True},
   "B":{"visible":7.,"future_positive":None},
   "C":{"visible":3.,"future_positive":False},
   "D":{"visible":2.,"future_positive":None},
   "E":{"visible":1.,"future_positive":True},
  }
  got=exact_bounds(registry,{"A","B"})
  self.assertEqual(got["unknown_selected"],1)
  self.assertEqual(got["unknown_not_selected"],1)
  self.assertAlmostEqual(got["precision_lower"],.5)
  self.assertAlmostEqual(got["precision_upper"],1.)
  self.assertAlmostEqual(got["recall_lower"],1/3)
  self.assertAlmostEqual(got["recall_upper"],2/3)
  self.assertEqual(topk(registry,.4,42),{"A","B"})
 def test_train_only_aggregate(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);raw=root/"data"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"manifest.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(35):
     db.execute("INSERT INTO patients VALUES(?,?,?)",
                (f"PRIVATE_TRAIN_{i}","train",2025*12+12))
    db.execute("INSERT INTO patients VALUES(?,?,?)",
               ("PRIVATE_TEST","test",2025*12+12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(35):
     visible={name:(1 if j<i%27 else 0) for j,name in enumerate(VISIBLE)}
     remote=[101 if item in HIDDEN else visible[item] for item in ALL]
     w.writerow([f"PRIVATE_TRAIN_{i}","R01","06/2026","","",*remote])
     if i<25:
      onsite=[(1 if i%2==0 else 0) if item in HIDDEN else visible[item]
              for item in ALL]
      w.writerow([f"PRIVATE_TRAIN_{i}","V04","12/2026",
                  sum(onsite),"",*onsite])
    w.writerow(["PRIVATE_TEST","V04","12/2026","132","",*[4]*33])
   output=root/"reports"
   with patch.dict(sys.modules,{"scripts.phase41_future_v04_risk":None}):
    result=audit(raw.parent,manifest,output,42,20,10)
   self.assertEqual(result["all_eligible_R01_patients"],35)
   self.assertEqual(result["future_V04_observed"],25)
   self.assertEqual(result["future_V04_missing"],10)
   with (output/"missing_outcome_budget_bounds.csv").open() as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),5)
   for r in rows:
    self.assertLessEqual(float(r["future_positive_precision_lower_pct"]),
                         float(r["future_positive_precision_upper_pct"]))
    self.assertLessEqual(float(r["future_positive_recall_lower_pct"]),
                         float(r["future_positive_recall_upper_pct"]))
   for file in output.iterdir():
    value=file.read_text()
    self.assertNotIn("PRIVATE_TRAIN",value)
    self.assertNotIn("PRIVATE_TEST",value)

if __name__=="__main__":
 unittest.main()
