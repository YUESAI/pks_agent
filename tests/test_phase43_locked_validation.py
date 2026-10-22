"""Phase43 synthetic checks for precommitted validation-only policy."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase43_locked_validation import (
    bounds,select_ids,load_val,summarize,VERSION
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE

class Phase43Tests(unittest.TestCase):
 def test_sharp_bounds_and_future_not_used_in_ranking(self):
  registry={"A":(8,True),"B":(7,None),"C":(3,False),
            "D":(2,None),"E":(1,True)}
  result=bounds(registry,{"A","B"})
  self.assertAlmostEqual(result["recall_lower"],1/3)
  self.assertAlmostEqual(result["recall_upper"],2/3)
  self.assertAlmostEqual(result["precision_lower"],.5)
  selected=select_ids(registry,.4)
  changed={pid:(score,False) for pid,(score,label) in registry.items()}
  self.assertEqual(selected,select_ids(changed,.4))
  self.assertEqual(selected,{"A","B"})
 def test_strict_val_filter_and_aggregates(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);folder=root/"raw"/"PPMI";folder.mkdir(parents=True)
   manifest=root/"private.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(35):
     db.execute("INSERT INTO patients VALUES (?,?,?)",
                ("VAL_PRIVATE_"+str(i),"val",2025*12+12))
    for split in ("train","test"):
     db.execute("INSERT INTO patients VALUES (?,?,?)",
                ("OTHER_PRIVATE_"+split,split,2025*12+12))
   with (folder/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(35):
     values={name:(1 if j<i%27 else 0) for j,name in enumerate(VISIBLE)}
     remote=[101 if name in HIDDEN else values[name] for name in ALL]
     writer.writerow(["VAL_PRIVATE_"+str(i),"R01","06/2026","","",*remote])
     if i<25:
      site=[(1 if i%2==0 else 0) if name in HIDDEN else values[name]
            for name in ALL]
      writer.writerow(["VAL_PRIVATE_"+str(i),"V04","12/2026",
                       sum(site),"",*site])
    writer.writerow(["OTHER_PRIVATE_test","V04","12/2026","132","",*[4]*33])
   registry,qc,cohort=load_val(folder.parent,manifest)
   self.assertEqual(cohort,35)
   self.assertEqual(len(registry),35)
   self.assertEqual(sum(label is None for score,label in registry.values()),10)
   out=root/"report"
   summary=summarize(registry,qc,cohort,out)
   self.assertEqual(summary["protocol_version"],VERSION)
   self.assertEqual(summary["split"],"val")
   self.assertTrue(summary["TEST_unopened"])
   for file in out.iterdir():
    body=file.read_text()
    self.assertNotIn("VAL_PRIVATE_",body)
    self.assertNotIn("OTHER_PRIVATE_",body)
if __name__=="__main__":unittest.main()
