"""Phase33 source taxonomy checks with synthetic values."""
import unittest
from scripts.phase33_visit_audit import total_kind,item_bucket,numeric_present
class Phase33Tests(unittest.TestCase):
 def test_missing_and_valid(self):
  self.assertEqual(total_kind(""),"empty")
  self.assertEqual(total_kind("5"),"valid")
  self.assertEqual(total_kind("-9"),"invalid")
  self.assertFalse(numeric_present(""))
  self.assertFalse(numeric_present("-9"))
  self.assertTrue(numeric_present("0"))
  self.assertEqual(item_bucket(0),"none")
  self.assertEqual(item_bucket(3),"1_to_4")
  self.assertEqual(item_bucket(11),"10plus")
 def test_item_column_count_not_overwritten(self):
  import csv,sqlite3,tempfile
  from pathlib import Path
  from scripts.phase33_visit_audit import analyze
  with tempfile.TemporaryDirectory() as tmp:
   raw=Path(tmp)/"raw"/"PPMI"
   raw.mkdir(parents=True)
   manifest=Path(tmp)/"patients.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT, split TEXT, baseline_month INTEGER)")
    db.execute("INSERT INTO patients VALUES (?,?,?)",("FAKE1","train",2025*12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE","NP3SPCH","NP3FACXP"])
    w.writerow(["FAKE1","R01","","06/2026","","2","1"])
   result=analyze(raw.parent,manifest,Path(tmp)/"out",10)
   self.assertEqual(result["NP3_item_columns"],2)
if __name__=="__main__":unittest.main()
