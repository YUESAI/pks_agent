"""Synthetic Phase34 six-item remote Motor status and privacy tests."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase34_remote_motor import audit,item_status,SPECIAL

class Phase34Tests(unittest.TestCase):
 def test_item_status(self):
  self.assertEqual(item_status("UR"),"UR")
  self.assertEqual(item_status(""),"blank")
  self.assertEqual(item_status("3"),"numeric_0_to_4")
  self.assertEqual(item_status("-9"),"other_code")
 def test_train_only_source_and_blank_examdate_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   folder=root/"data"/"PPMI"
   folder.mkdir(parents=True)
   db=root/"manifest.sqlite"
   with sqlite3.connect(db) as con:
    con.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    con.execute("INSERT INTO patients VALUES (?,?,?)",("FAKE_TRAIN","train",2025*12))
    con.execute("INSERT INTO patients VALUES (?,?,?)",("FAKE_TEST","test",2025*12))
   src=folder/"MDS-UPDRS_Part_III_06Oct2026.csv"
   with src.open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","NP3TOT","EXAMDT","INFODT","PDSTATE",*SPECIAL])
    w.writerow(["FAKE_TRAIN","R01","","","06/2026","",*["UR"]*6])
    w.writerow(["FAKE_TEST","R01","","","06/2026","",*["4"]*6])
   out=root/"reports"
   result=audit(folder.parent,db,out,10)
   self.assertEqual(result["missing_six_item_headers"],[])
   with (out/"event_missing_total_counts.csv").open() as f:
    rows=list(csv.DictReader(f))
   self.assertEqual(rows,[{"window":"6m_window","event_code":"R01",
                           "missing_total_rows":"<10"}])
   for f in out.iterdir():
    s=f.read_text()
    self.assertNotIn("FAKE_TRAIN",s)
    self.assertNotIn("FAKE_TEST",s)
if __name__=="__main__":
 unittest.main()
