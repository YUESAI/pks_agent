"""Synthetic same-visit Phase37 checks without real PPMI records."""
import csv,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.phase37_masked_site_benchmark import (
    load_site_episodes,roc_auc,clip_hidden,report
)
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN

class Phase37Tests(unittest.TestCase):
 def test_clipping_and_rank_auc(self):
  self.assertEqual(clip_hidden(200,5),24.)
  self.assertEqual(clip_hidden(1,5),0.)
  self.assertEqual(roc_auc({"a":(0,0.),"b":(2,1.)},True),1.)
 def test_train_only_and_same_visit_labels(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);folder=root/"raw"/"PPMI";folder.mkdir(parents=True)
   manifest=root/"private.sqlite"
   with sqlite3.connect(manifest) as db:
    db.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
    for i in range(55):
     db.execute("INSERT INTO patients VALUES(?,?,?)",
                ("TRAIN_"+str(i),"train",2025*12))
    db.execute("INSERT INTO patients VALUES(?,?,?)",
               ("SECRET_TEST","test",2025*12))
   with (folder/"MDS-UPDRS_Part_III_06Oct2026.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["PATNO","EVENT_ID","EXAMDT","PDSTATE","NP3TOT",*ALL])
    for i in range(55):
     scores=[str((i%5) if item in HIDDEN and j==2 else (i%3))
             for j,item in enumerate(ALL)]
     w.writerow(["TRAIN_"+str(i),"V04","12/2026","",str(sum(map(int,scores))),*scores])
    w.writerow(["SECRET_TEST","V04","12/2026","","132",*["4"]*33])
   data,qc,source=load_site_episodes(folder.parent,manifest)
   self.assertEqual(len(data),55)
   out=root/"report"
   result=report(manifest,folder.parent,out,5,42,100,10)
   self.assertEqual(result["eligible_train_V04_patients"],55)
   with (out/"masked_site_oof_metrics.csv").open() as f:
    models=list(csv.DictReader(f))
   self.assertEqual(len(models),4)
   for file in out.iterdir():
    body=file.read_text()
    self.assertNotIn("TRAIN_",body)
    self.assertNotIn("SECRET_TEST",body)

if __name__=="__main__":unittest.main()
