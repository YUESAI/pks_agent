"""Synthetic Phase38 ranking, budget, and privacy regression checks."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase38_budgeted_acquisition import (
    rank_indices,selected_at_budget,aggregate,run,
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

class Phase38Tests(unittest.TestCase):
 def test_budget_ranking_never_uses_hidden_labels(self):
  rows=[
   {"pid":"A","fold":0,"hidden":0.,"visible":1.,
    "risk_visible_sum":0.3,"risk_visible_27":3.},
   {"pid":"B","fold":0,"hidden":4.,"visible":7.,
    "risk_visible_sum":2.3,"risk_visible_27":1.},
   {"pid":"C","fold":0,"hidden":1.,"visible":3.,
    "risk_visible_sum":1.3,"risk_visible_27":2.},
  ]
  a=selected_at_budget(rows,"ridge_visible_27",1/3,42)
  mutated=[dict(r,hidden=100-r["hidden"]) for r in rows]
  b=selected_at_budget(mutated,"ridge_visible_27",1/3,42)
  self.assertEqual(a,b)
  self.assertEqual(a,{"A"})
  self.assertEqual(selected_at_budget(rows,"visible_sum",1/3,42),{"B"})
  self.assertEqual(aggregate(rows,{"B"})["captured"],1)
 def test_end_to_end_training_only(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);raw=root/"raw"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"private.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(55):
     db.execute("INSERT INTO patients VALUES (?,?,?)",
                ("TRAIN_SECRET_"+str(i),"train",2025*12))
    db.execute("INSERT INTO patients VALUES (?,?,?)",
               ("TEST_SECRET","test",2025*12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","PDSTATE","NP3TOT",*ALL])
    for i in range(55):
     scores=[str(i%5 if name in HIDDEN and j==2 else i%3)
             for j,name in enumerate(ALL)]
     w.writerow(["TRAIN_SECRET_"+str(i),"V04","12/2026","",
                 str(sum(map(int,scores))),*scores])
    w.writerow(["TEST_SECRET","V04","12/2026","","132",*["4"]*33])
   out=root/"aggregate"
   info=run(raw.parent,manifest,out,5,42,100,10)
   self.assertEqual(info["eligible_train_V04_patients"],55)
   with (out/"budgeted_acquisition_curve.csv").open() as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(len(rows),35)
   for p in out.iterdir():
    text=p.read_text()
    self.assertNotIn("TRAIN_SECRET_",text)
    self.assertNotIn("TEST_SECRET",text)

if __name__=="__main__":
 unittest.main()
