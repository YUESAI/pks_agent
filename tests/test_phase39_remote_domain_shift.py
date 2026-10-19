"""Phase39 synthetic test: R01 versus later V04 feature-shift audit."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase39_remote_domain_shift import audit,ks_distance,threshold_benchmark
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

class Phase39Tests(unittest.TestCase):
 def test_ks_and_threshold(self):
  self.assertEqual(ks_distance([1,1],[2,2]),1.)
  remote={"a":{"sum":2},"b":{"sum":5}}
  site={"a":{"sum":3},"b":{"sum":7}}
  paired={"a":(remote["a"],site["a"]),"b":(remote["b"],site["b"])}
  result=threshold_benchmark(site,remote,paired,.5)
  self.assertEqual(result["site_selected_n"],1)
  self.assertEqual(result["remote_selected_n"],0)
 def test_train_only_aggregate(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);src=base/"data"/"PPMI";src.mkdir(parents=True)
   manifest=base/"private.sqlite"
   with sqlite3.connect(manifest) as conn:
    conn.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(15):
     conn.execute("INSERT INTO patients VALUES (?,?,?)",
                  ("PRIVATE_TRAIN_"+str(i),"train",2025*12))
    conn.execute("INSERT INTO patients VALUES (?,?,?)",
                 ("PRIVATE_TEST","test",2025*12))
   with (src/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as file:
    writer=csv.writer(file)
    writer.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(15):
     rem=["101" if name in HIDDEN else str(i%3) for name in ALL]
     site=[str(i%3) for name in ALL]
     writer.writerow(["PRIVATE_TRAIN_"+str(i),"R01","06/2026","","",*rem])
     writer.writerow(["PRIVATE_TRAIN_"+str(i),"V04","12/2026",
                      str(sum(map(int,site))),"",*site])
    writer.writerow(["PRIVATE_TEST","V04","12/2026","33","",*["1"]*33])
   out=base/"reports"
   result=audit(src.parent,manifest,out,100,10,42)
   self.assertEqual(result["remote_R01_eligible"],15)
   self.assertEqual(result["onsite_V04_eligible"],15)
   self.assertEqual(result["same_patient_R01_then_V04"],15)
   for file in out.iterdir():
    data=file.read_text()
    self.assertNotIn("PRIVATE_TRAIN_",data)
    self.assertNotIn("PRIVATE_TEST",data)
if __name__=="__main__":unittest.main()
