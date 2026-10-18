"""Phase27 synthetic tests: inner-only tuning, heldout isolation and privacy."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase27_fair_incremental import audit,tune_penalty,PENALTIES

class Phase27Tests(unittest.TestCase):
    def test_inner_penalty_grid(self):
        rows=[(f"FAKE_{i}",float(i%15),float(i%15)+1,
               {"moca_total":float(i%20),"rbd_vivid_dreams":float(i%2),
                "scopa_aut_item1":float(i%3),"upsit_total":None})
              for i in range(45)]
        choice,rmse=tune_penalty(rows,"motor_core",42,0)
        self.assertIn(choice,PENALTIES)
        self.assertGreaterEqual(rmse,0)
    def test_end_to_end_train_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/"private.sqlite";src=root/"clinical.sqlite"
            with sqlite3.connect(manifest) as db:
                db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
                db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
                for i in range(50):
                    pid=f"PRIVATE_{i}";x=float(i%16)
                    db.execute("INSERT INTO patients VALUES(?,?,?,?)",(pid,"train",2025*12+5,x))
                    db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?,?)",
                               (pid,12,2025*12+5,2026*12+5,12,x,x+1,1))
                for split in ("val","test"):
                    pid="SENSITIVE_"+split
                    db.execute("INSERT INTO patients VALUES(?,?,?,?)",(pid,split,2025*12+5,2))
                    db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?,?)",
                               (pid,12,2025*12+5,2026*12+5,12,2,132,130))
            with sqlite3.connect(src) as db:
                db.execute("""CREATE TABLE observations(cohort TEXT,patient_id TEXT,
                  event_id TEXT,concept TEXT,value_raw TEXT,exam_month TEXT,
                  quality_status TEXT,imputed_flag TEXT)""")
                for i in range(50):
                    pid=f"PRIVATE_{i}"
                    for concept,value in (("moca_total","24"),("rbd_vivid_dreams","1"),
                                          ("scopa_aut_item1","2")):
                        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
                                   ("PPMI",pid,"BL",concept,value,"2025-05",
                                    "observed_unverified",None))
            out=root/"reports"
            result=audit(manifest,src,"sqlite",out,5,42,100,10)
            self.assertEqual(result["train_pairs"],50)
            with (out/"nested_multimodal_oof.csv").open() as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len(records),5)
            for p in out.iterdir():
                body=p.read_text()
                self.assertNotIn("PRIVATE_",body)
                self.assertNotIn("SENSITIVE_",body)

if __name__=="__main__":
    unittest.main()
