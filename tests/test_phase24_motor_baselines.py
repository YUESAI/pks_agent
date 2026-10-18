"""Synthetic Phase24 tests; no patient-level clinical source data."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase24_motor_baselines import audit,fit,fixed_folds,load_train_only,predict

class Phase24Tests(unittest.TestCase):
 def test_regression_models_and_deterministic_patient_folds(self):
  rows=[(f"FAKE_{i}",float(i%12),float(i%12)+2) for i in range(60)]
  self.assertEqual(fixed_folds(rows,5,42),fixed_folds(rows[::-1],5,42))
  params=fit(rows)
  self.assertAlmostEqual(predict("persistence",params,10),10)
  self.assertAlmostEqual(predict("mean_delta",params,10),12)
  self.assertAlmostEqual(predict("median_delta",params,10),12)
  self.assertAlmostEqual(predict("ridge_delta",params,10),12)
 def test_train_only_targets_and_aggregate_only_reports(self):
  with tempfile.TemporaryDirectory() as tmp:
   dbpath=Path(tmp)/"private.sqlite"
   with sqlite3.connect(dbpath) as db:
    db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
    db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
    for i in range(60):
     pid=f"FAKE_TRAIN_{i}";x=float(i%15);y=x+2
     db.execute("INSERT INTO patients VALUES (?,?,?,?)",(pid,"train",2023*12,x))
     db.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)",(pid,12,2023*12,2024*12,12,x,y,2))
    for split in ("val","test"):
     for i in range(4):
      pid=f"FAKE_{split}_{i}"
      db.execute("INSERT INTO patients VALUES (?,?,?,?)",(pid,split,2023*12,2))
      db.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?)",(pid,12,2023*12,2024*12,12,2,132,130))
   rows,registry,count=load_train_only(dbpath)
   self.assertEqual(len(rows),60)
   self.assertEqual(registry["test"],4)
   output=Path(tmp)/"reports"
   self.assertEqual(audit(dbpath,output,5,42,100)["train_pair_count"],60)
   with (output/"train_oof_motor_metrics.csv").open() as f:
    metrics={row["model"]:row for row in csv.DictReader(f)}
   self.assertAlmostEqual(float(metrics["mean_delta"]["mae"]),0)
   self.assertAlmostEqual(float(metrics["persistence"]["mae"]),2)
   for path in output.iterdir():
    self.assertNotIn("FAKE_",path.read_text())
if __name__=="__main__":unittest.main()
