"""Phase32: synthetic manifest reconciliation and missing-value taxonomy."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase32_exclusion_forensics import audit,score_kind,reason

class Phase32Tests(unittest.TestCase):
 def test_value_categories(self):
  self.assertEqual(score_kind("")[0],"missing")
  self.assertEqual(score_kind("-9")[0],"negative_or_sentinel")
  self.assertEqual(score_kind("140")[0],"above_max")
  self.assertEqual(score_kind("unknown")[0],"non_numeric")
  self.assertEqual(score_kind("5")[0],"valid")
  self.assertEqual(reason({"values":set(),"errors":{"missing":1},"events":{"V02"},"rows":1}),"missing_only")
 def test_exact_replay_matches_frozen_manifest(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); raw=root/"data"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"private.sqlite"; out=root/"report"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY, split TEXT,baseline_month INTEGER,baseline_motor REAL)")
    db.execute("CREATE TABLE readiness(pid TEXT,horizon INTEGER,status TEXT, PRIMARY KEY(pid,horizon))")
    db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
    db.execute("INSERT INTO patients VALUES(?,?,?,?)",("PRIVATE_TRAIN","train",2025*12,3.))
    db.execute("INSERT INTO patients VALUES(?,?,?,?)",("PRIVATE_TEST","test",2025*12,8.))
    for h,status in ((6,"no_in_window_target"),(12,"paired"),(24,"no_in_window_target")):
     db.execute("INSERT INTO readiness VALUES(?,?,?)",("PRIVATE_TRAIN",h,status))
    db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?,?)",
               ("PRIVATE_TRAIN",12,2025*12,2026*12,12,3.,7.,4.))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"])
    w.writerows([
      ["PRIVATE_TRAIN","BL","3","12/2025",""],
      ["PRIVATE_TRAIN","V04","7","12/2026",""],
      ["PRIVATE_TRAIN","V02","","06/2026",""],
      ["PRIVATE_TEST","V04","100","12/2026",""],
    ])
   result=audit(raw.parent,manifest,out,10)
   self.assertEqual(result["manifest_comparison_total_mismatches"],0)
   self.assertEqual(result["phase23_replayed_excluded_keys_train"],1)
   self.assertEqual(result["matched_manifest_targets_by_horizon"]["12"],1)
   for file in out.iterdir():
    text=file.read_text()
    self.assertNotIn("PRIVATE_TRAIN",text)
    self.assertNotIn("PRIVATE_TEST",text)

if __name__=="__main__":
 unittest.main()
