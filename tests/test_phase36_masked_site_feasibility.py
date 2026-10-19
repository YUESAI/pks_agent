"""Synthetic Phase36 tests for remote masking with same-visit V04 truth."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase36_masked_site_feasibility import audit,ALL,HIDDEN,VISIBLE,score

class Phase36Tests(unittest.TestCase):
 def test_item_partition(self):
  self.assertEqual(len(ALL),33)
  self.assertEqual(len(HIDDEN),6)
  self.assertEqual(len(VISIBLE),27)
  self.assertEqual(score("101"),("unable",None))
  self.assertEqual(score("2"),("valid",2))
 def test_train_only_and_same_visit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/"raw"/"PPMI";folder.mkdir(parents=True)
   manifest=root/"patients.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT, split TEXT, baseline_month INTEGER)")
    db.execute("INSERT INTO patients VALUES(?,?,?)",("TRAIN_SECRET","train",2025*12))
    db.execute("INSERT INTO patients VALUES(?,?,?)",("TEST_SECRET","test",2025*12))
   with (folder/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","PDSTATE","NP3TOT",*ALL])
    r=[("101" if x in HIDDEN else "1") for x in ALL]
    onsite=["1" for _ in ALL]
    w.writerow(["TRAIN_SECRET","R01","06/2026","","",*r])
    w.writerow(["TRAIN_SECRET","V04","12/2026","","33",*onsite])
    w.writerow(["TEST_SECRET","V04","12/2026","","33",*onsite])
   result=audit(folder.parent,manifest,root/"report",10)
   self.assertEqual(result["eligible_V04_same_visit_labels"],"<10")
   self.assertEqual(result["real_R01_six_unable_27valid"],"<10")
   self.assertEqual(result["R01_to_later_V04_verified_pairs"],"<10")
   for path in (root/"report").iterdir():
    body=path.read_text()
    self.assertNotIn("TRAIN_SECRET",body)
    self.assertNotIn("TEST_SECRET",body)
if __name__=="__main__":unittest.main()
