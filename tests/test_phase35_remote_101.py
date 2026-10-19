"""Phase35 synthetic TRAIN-only numeric 101 and onsite overlap tests."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase35_remote_101 import calculate,status,SPECIAL

class Phase35Tests(unittest.TestCase):
 def test_code_101(self):
  self.assertEqual(status("101"),"numeric_101")
  self.assertEqual(status("UR"),"literal_UR")
  self.assertEqual(status("4"),"valid_0to4")
  self.assertEqual(status(""),"blank")
 def test_train_only_and_clinic_overlap(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); raw=root/"raw"/"PPMI";raw.mkdir(parents=True)
   db=root/"manifest.sqlite"
   with sqlite3.connect(db) as con:
    con.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    con.execute("INSERT INTO patients VALUES(?,?,?)",("TRAIN_ID","train",2025*12))
    con.execute("INSERT INTO patients VALUES(?,?,?)",("TEST_ID","test",2025*12))
   other=["NP3FAKE"+str(i) for i in range(21)]
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE",*SPECIAL,*other])
    w.writerow(["TRAIN_ID","R01","","06/2026","",*["101"]*6,*["2"]*21])
    w.writerow(["TRAIN_ID","V04","3","12/2026","",*["1"]*6,*["2"]*21])
    w.writerow(["TEST_ID","R01","","06/2026","",*["101"]*6,*["2"]*21])
   out=root/"report";summary=calculate(raw.parent,db,out,10)
   self.assertEqual(summary["R01_six_month_blank_total_rows_processed"],1)
   with (out/"remote_clinic_overlap.csv").open() as f:
    lines=list(csv.DictReader(f))
   self.assertEqual(next(r["TRAIN_patients"] for r in lines if r["measure"]=="V04_valid_total_patient"),"<10")
   for file in out.iterdir():
    content=file.read_text()
    self.assertNotIn("TRAIN_ID",content)
    self.assertNotIn("TEST_ID",content)
if __name__=="__main__":unittest.main()
