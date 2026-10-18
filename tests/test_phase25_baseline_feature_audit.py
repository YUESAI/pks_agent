"""Synthetic Phase25 tests: BL/SC temporal gates and TRAIN-only feature audit."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase25_baseline_feature_audit import (
    choose_feature,evaluate_visit,audit
)

class Phase25Tests(unittest.TestCase):
    def test_baseline_time_and_screening_window(self):
        anchor=2025*12+5
        month="2025-05"
        self.assertEqual(evaluate_visit([("25",month,"observed_unverified",None)],
                                        "moca_total",anchor,"BL")[0],"ready_BL")
        self.assertEqual(evaluate_visit([("25","2025-06","observed_unverified",None)],
                                        "moca_total",anchor,"BL")[0],"date_rejected")
        data={"SC":[("21","2025-03","observed_unverified",None)]}
        self.assertEqual(choose_feature("upsit_total",data,anchor)[0],"ready_SC")
        data["SC"]=[("21","2025-05","observed_unverified",None)]
        self.assertEqual(choose_feature("upsit_total",data,anchor)[0],"date_rejected")
        data={"BL":[("20","2025-05","observed_unverified",None),
                    ("21","2025-05","observed_unverified",None)]}
        self.assertEqual(choose_feature("moca_total",data,anchor)[0],"ambiguous_repeat")
    def test_training_sql_scope_no_private_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=root/"manifest.sqlite";clinical=root/"clinical.sqlite"
            with sqlite3.connect(manifest) as db:
                db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
                db.execute("CREATE TABLE readiness(pid TEXT,horizon INTEGER,status TEXT,PRIMARY KEY(pid,horizon))")
                for split,pid,status in (("train","SECRET_TRAIN","paired"),
                                         ("val","SECRET_VAL","paired"),
                                         ("test","SECRET_TEST","paired")):
                    db.execute("INSERT INTO patients VALUES(?,?,?,?)",(pid,split,2025*12+5,3))
                    db.execute("INSERT INTO readiness VALUES (?,?,?)",(pid,12,status))
            with sqlite3.connect(clinical) as db:
                db.execute("""CREATE TABLE observations(cohort TEXT,patient_id TEXT,
                    event_id TEXT,concept TEXT,value_raw TEXT,exam_month TEXT,
                    quality_status TEXT,imputed_flag TEXT)""")
                db.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",[
                    ("PPMI","SECRET_TRAIN","BL","moca_total","24","2025-05","observed_unverified",None),
                    ("PPMI","SECRET_VAL","BL","moca_total","1","2025-05","observed_unverified",None),
                    ("PPMI","SECRET_TEST","BL","moca_total","30","2025-05","observed_unverified",None)])
            output=root/"reports"
            result=audit(clinical,"sqlite",manifest,output,10)
            self.assertEqual(result["train_patients"],1)
            with (output/"train_feature_status_by_followup.csv").open() as f:
                rows=list(csv.DictReader(f))
            got=[r for r in rows if r["feature"]=="moca_total" and r["status"]=="ready_BL"]
            self.assertEqual(sum(r["patients"]=="<10" for r in got),1)
            for file in output.iterdir():
                value=file.read_text()
                for forbidden in ("SECRET_TRAIN","SECRET_VAL","SECRET_TEST"):
                    self.assertNotIn(forbidden,value)

if __name__=="__main__":
    unittest.main()
