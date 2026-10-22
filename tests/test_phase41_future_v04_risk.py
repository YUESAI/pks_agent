"""Phase41 synthetic observational future-risk audit and leakage checks."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase41_future_v04_risk import run,roc_auc,rank_select,quota
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

class Phase41Tests(unittest.TestCase):
 def test_no_future_label_used_in_r01_ranking(self):
  rows=[{"pid":f"FAKE_{i}","remote_sum":float(i),
         "future_six":float(i%3),"future_visible_sum":float(30-i)}
        for i in range(25)]
  selected={r["pid"] for r in rank_select(rows,"remote_sum",.2,42)}
  changed=[dict(r,future_six=100-r["future_six"],
                future_visible_sum=1000+r["future_visible_sum"]) for r in rows]
  self.assertEqual(selected,
   {r["pid"] for r in rank_select(changed,"remote_sum",.2,42)})
  self.assertEqual(len(selected),quota(25,.2))
 def test_train_only_future_labels(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/"raw"/"PPMI";raw.mkdir(parents=True)
   manifest=root/"private.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(30):
     db.execute("INSERT INTO patients VALUES(?,?,?)",
                (f"PRIVATE_TRAIN_{i}","train",2025*12+12))
    db.execute("INSERT INTO patients VALUES(?,?,?)",
               ("PRIVATE_TEST","test",2025*12+12))
   with (raw/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["PATNO","EVENT_ID","EXAMDT","NP3TOT","PDSTATE",*ALL])
    for i in range(30):
     visible=["1" if j<i%27 else "0" for j,name in enumerate(
              [n for n in ALL if n not in HIDDEN])]
     visible_by_name=dict(zip([n for n in ALL if n not in HIDDEN],visible))
     remote=[("101" if name in HIDDEN else visible_by_name[name]) for name in ALL]
     onsite=[str((i%3)) if name in HIDDEN else visible_by_name[name] for name in ALL]
     w.writerow([f"PRIVATE_TRAIN_{i}","R01","06/2026","","",*remote])
     w.writerow([f"PRIVATE_TRAIN_{i}","V04","12/2026",str(sum(map(int,onsite))),"",*onsite])
    w.writerow(["PRIVATE_TEST","V04","12/2026","132","",*["4"]*33])
   out=root/"reports"
   result=run(raw.parent,manifest,out,42,100,100,20,10)
   self.assertEqual(result["paired_R01_then_V04_people"],30)
   self.assertEqual(result["R01_eligible_remote_people"],30)
   with (out/"prospective_rank_discrimination.csv").open() as f:
    self.assertEqual(len(list(csv.DictReader(f))),2)
   with (out/"future_outcome_budget_curves.csv").open() as f:
    self.assertEqual(len(list(csv.DictReader(f))),20)
   for f in out.iterdir():
    body=f.read_text()
    self.assertNotIn("PRIVATE_TRAIN_",body)
    self.assertNotIn("PRIVATE_TEST",body)
if __name__=="__main__":
 unittest.main()
