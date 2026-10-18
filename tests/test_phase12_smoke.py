"""Synthetic state-blind budget/replay regression checks."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.budgeted_env_v0 import BudgetedDoctorPatientEnv
from scripts.phase12_smoke_benchmark import run, action_order

class Phase12Tests(unittest.TestCase):
    def test_budgeted_actions_and_hidden_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            src=base/"source.sqlite"
            man=base/"manifest.sqlite"
            spl=base/"splits.sqlite"
            with sqlite3.connect(src) as db:
                db.execute("""CREATE TABLE observations(
                    cohort TEXT,patient_id TEXT,event_id TEXT,concept TEXT,
                    value_raw TEXT,exam_month TEXT,date_precision TEXT,
                    medication_state TEXT,quality_status TEXT,source_file TEXT,
                    source_row INTEGER)""")
                db.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?)",[
                    ("PPMI","SECRET_A","BL","updrs3_total","30",
                     "2025-03","month_candidate","ON",
                     "observed_unverified","test.csv",2),
                    ("PPMI","SECRET_A","BL","moca_total","25",
                     "2025-03","month_candidate",None,
                     "observed_unverified","test.csv",3),
                ])
            with sqlite3.connect(man) as db:
                db.execute("CREATE TABLE episode_candidates(cohort TEXT,patient_id TEXT,baseline_month TEXT)")
                db.execute("CREATE TABLE action_catalog(cohort TEXT,action_key TEXT,concept TEXT,required_medication_state TEXT)")
                db.execute("""CREATE TABLE episode_actions(cohort TEXT,patient_id TEXT,
                    action_key TEXT,status TEXT,selected_visit TEXT,source_file TEXT,
                    source_row INTEGER,record_month TEXT)""")
                db.execute("INSERT INTO episode_candidates VALUES(?,?,?)",("PPMI","SECRET_A","2025-03"))
                db.executemany("INSERT INTO action_catalog VALUES(?,?,?,?)",[
                    ("PPMI","updrs3_total@ON","updrs3_total","ON"),
                    ("PPMI","updrs3_total@OFF","updrs3_total","OFF"),
                    ("PPMI","moca_total","moca_total",None)])
                db.executemany("INSERT INTO episode_actions VALUES(?,?,?,?,?,?,?,?)",[
                    ("PPMI","SECRET_A","updrs3_total@ON","ready","BL","test.csv",2,"2025-03"),
                    ("PPMI","SECRET_A","updrs3_total@OFF","not_recorded","BL",None,None,None),
                    ("PPMI","SECRET_A","moca_total","ready","BL","test.csv",3,"2025-03")])
            with sqlite3.connect(spl) as db:
                db.execute("CREATE TABLE patient_splits(cohort TEXT,patient_id TEXT,split TEXT)")
                db.execute("INSERT INTO patient_splits VALUES (?,?,?)",
                           ("PPMI","SECRET_A","train"))
            with BudgetedDoctorPatientEnv(man,src,"SECRET_A",budget=3,source_engine="sqlite") as env:
                self.assertNotIn("updrs3_total@ON",env.catalog())
                response=env.request("motor_exam")
                self.assertEqual(response["value"],"30")
                for private in ("medication_state","recorded_month","source_file","patient_id"):
                    self.assertNotIn(private,response)
                self.assertEqual(env.request("motor_exam")["status"],"already_requested")
                self.assertEqual(env.request("smell_test")["status"],"insufficient_budget")
                self.assertEqual(env.request("cognitive_test")["value"],"25")
                self.assertEqual(env.spent,3)
                self.assertEqual(env.request("sleep_item")["status"],"insufficient_budget")
            self.assertEqual(action_order("random",42,"SECRET_A"),
                             action_order("random",42,"SECRET_A"))
            reports=base/"reports"
            res=run(src,"sqlite",man,spl,reports,budget=3)
            self.assertEqual(res["n_train_patients"],1)
            for report in reports.iterdir():
                self.assertNotIn("SECRET_A",report.read_text())
