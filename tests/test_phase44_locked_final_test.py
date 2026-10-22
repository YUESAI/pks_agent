"""Synthetic Phase44 tests: locked TEST-only semantics; no real TEST access."""
import csv,sqlite3,tempfile,unittest,sys
from unittest.mock import patch
from pathlib import Path
from scripts.phase44_locked_final_test import (
    bounds,select_ids,load_test,summarize,VERSION,main
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE

class Phase44Tests(unittest.TestCase):
 def test_explicit_confirmation_required(self):
  with patch.object(sys,"argv",["phase44","--data-root","/missing",
              "--manifest-db","/missing"]):
   with self.assertRaises(SystemExit) as err:main()
   self.assertEqual(err.exception.code,2)
 def test_future_labels_do_not_change_selection(self):
  registry={"A":(8,True),"B":(7,None),"C":(3,False),
            "D":(2,None),"E":(1,True)}
  r=bounds(registry,{"A","B"})
  self.assertAlmostEqual(r["recall_lower"],1/3)
  self.assertAlmostEqual(r["recall_upper"],2/3)
  modified={pid:(score,False) for pid,(score,label) in registry.items()}
  self.assertEqual(select_ids(registry,.4),select_ids(modified,.4))
 def test_only_test_synthetic_and_aggregate(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/"raw"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"private.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(35):
     db.execute("INSERT INTO patients VALUES (?,?,?)",
                (f"SYNTHETIC_TEST_{i}","test",2025*12+12))
    for split in ("train","val"):
     db.execute("INSERT INTO patients VALUES (?,?,?)",
                ("SYNTHETIC_OTHER_"+split,split,2025*12+12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(35):
     visible={name:(1 if j<i%27 else 0) for j,name in enumerate(VISIBLE)}
     remote=[101 if name in HIDDEN else visible[name] for name in ALL]
     w.writerow([f"SYNTHETIC_TEST_{i}","R01","06/2026","","",*remote])
     if i<25:
      onsite=[(1 if i%2==0 else 0) if name in HIDDEN else visible[name]
              for name in ALL]
      w.writerow([f"SYNTHETIC_TEST_{i}","V04","12/2026",
                  sum(onsite),"",*onsite])
    w.writerow(["SYNTHETIC_OTHER_train","V04","12/2026","132","",*[4]*33])
   registry,qc,total=load_test(raw.parent,manifest)
   self.assertEqual(total,35)
   self.assertEqual(len(registry),35)
   self.assertEqual(sum(label is None for score,label in registry.values()),10)
   reports=root/"results"
   summary=summarize(registry,qc,total,reports)
   self.assertEqual(summary["protocol_version"],VERSION)
   self.assertEqual(summary["split"],"test")
   self.assertTrue(summary["TEST_consumed"])
   for f in reports.iterdir():
    s=f.read_text()
    self.assertNotIn("SYNTHETIC_TEST_",s)
    self.assertNotIn("SYNTHETIC_OTHER_",s)

if __name__=="__main__":unittest.main()
