"""Phase40 synthetic exact-budget, tie and matched-rank checks."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase40_budget_calibration import (
    quota,exact_select,boundary_ties,rank_correlation,run
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

class Phase40Tests(unittest.TestCase):
 def test_exact_quota_and_tie_resolution(self):
  values={str(i):{"sum":5 if i<8 else 2} for i in range(20)}
  self.assertEqual(quota(20,.2),4)
  self.assertEqual(len(exact_select(values,.2,42)),4)
  self.assertEqual(exact_select(values,.2,42),exact_select(values,.2,42))
  self.assertEqual(boundary_ties(values,.2)[:2],(0,8))
  self.assertAlmostEqual(rank_correlation([1,2,2,3],[2,4,4,6]),1.)
 def test_train_only_aggregate_and_matched(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);raw=root/"raw"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"manifest.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(30):
     db.execute("INSERT INTO patients VALUES(?,?,?)",
                ("TRAIN_PRIVATE_"+str(i),"train",2025*12+12))
    db.execute("INSERT INTO patients VALUES(?,?,?)",
               ("TEST_PRIVATE","test",2025*12+12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(30):
     remote=[("101" if item in HIDDEN else str(i%3))
             for j,item in enumerate(ALL)]
     onsite=[str((i+1)%3) for j,item in enumerate(ALL)]
     w.writerow(["TRAIN_PRIVATE_"+str(i),"R01","06/2026","", "",*remote])
     w.writerow(["TRAIN_PRIVATE_"+str(i),"V04","12/2026",
                 str(sum(map(int,onsite))),"",*onsite])
    w.writerow(["TEST_PRIVATE","V04","12/2026","33","",*["1"]*33])
   output=root/"reports"
   result=run(raw.parent,manifest,output,42,100,10)
   self.assertEqual(result["eligible_R01"],30)
   self.assertEqual(result["eligible_V04"],30)
   self.assertEqual(result["paired_R01_to_later_V04"],30)
   with (output/"exact_budget_calibration.csv").open() as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),5)
   self.assertEqual(rows[1]["R01_exact_k"],"6")
   for p in output.iterdir():
    text=p.read_text()
    self.assertNotIn("TRAIN_PRIVATE_",text)
    self.assertNotIn("TEST_PRIVATE",text)

if __name__=="__main__":
 unittest.main()
