"""Synthetic episode-readiness regression checks (no private patient data)."""
import csv
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from scripts.episode_readiness import resolve, qualify, audit

class EpisodeReadinessTests(unittest.TestCase):
    def test_reject_conflicts_and_imputation(self):
        def item(value, day="2025-02-01", med="", quality="observed_unverified", imputed=""):
            return dict(value=value, date=date.fromisoformat(day) if day else None,
                        med=med, quality=quality, imputed=imputed)
        self.assertEqual(resolve([item("24"), item("25")])[0], "conflicting_values")
        self.assertEqual(resolve([item("24",med="ON"),item("24",med="OFF")])[0],
                         "conflicting_medication")
        self.assertEqual(resolve([item("24"),item("24")])[0], "identical_repeats")
        self.assertEqual(resolve([item("24",imputed="true")])[0], "quality_blocked")
        self.assertEqual(resolve([item("999")])[0], "quality_blocked")
        v=qualify({"moca_total":{"SC":[item("24", "2025-01-01")],
                                 "BL":[item("25","2025-05-02")]}},
                  "moca_total", date(2025,5,1),180)
        self.assertTrue(v["sc_dated"])
        self.assertFalse(v["bl_dated"])
        self.assertTrue(v["union_dated"])

    def test_aggregated_report_never_exposes_patient_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);folder=base/"raw"/"PPMI";folder.mkdir(parents=True)
            with (folder/"Baseline_Visit_Start_06Oct2026.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["PATNO","EVENT_ID","INFODT"])
                w.writerows([["SECRET_A","BL","2025-03-01"],
                             ["SECRET_B","BL","2025-03-10"]])
            dbpath=base/"patient.sqlite"
            db=sqlite3.connect(dbpath)
            db.execute("CREATE TABLE restricted_labels (cohort TEXT,patient_id TEXT,candidate_label TEXT)")
            db.execute("CREATE TABLE observations (cohort TEXT,patient_id TEXT,event_id TEXT,concept TEXT,value_raw TEXT,exam_date TEXT,quality_status TEXT,medication_state TEXT,imputed_flag TEXT)")
            db.executemany("INSERT INTO restricted_labels VALUES(?,?,?)",
                           [("PPMI","SECRET_A","PD"),("PPMI","SECRET_B","HC")])
            data=[("BL","updrs3_total","20","2025-03-01"),
                  ("BL","moca_total","25","2025-03-01"),
                  ("BL","rbd_vivid_dreams","1","2025-03-01"),
                  ("SC","upsit_total","15","2025-02-01")]
            db.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?)",
                           [("PPMI","SECRET_A",event,concept,value,day,
                             "observed_unverified",None,None)
                            for event,concept,value,day in data])
            db.commit();db.close()
            reports=base/"reports"
            result=audit(dbpath,"sqlite",base/"raw",reports,180,10)
            self.assertEqual(result["n_candidate_PPMI"],2)
            with (reports/"joint_readiness.csv").open() as f:
                rows=list(csv.DictReader(f))
            match=next(r for r in rows if r["label"]=="PD" and
                       r["features"]=="updrs3_total+moca_total+rbd_vivid_dreams"
                       and r["view"]=="bl_dated")
            self.assertEqual(match["patients"],"<10")
            for file in reports.iterdir():
                contents=file.read_text()
                self.assertNotIn("SECRET_A",contents)
                self.assertNotIn("SECRET_B",contents)
