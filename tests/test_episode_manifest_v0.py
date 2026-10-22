"""Synthetic Phase 9 tests: never rely on clinical source files."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.episode_manifest_v0 import build
from scripts.doctor_patient_env_v0 import DoctorPatientEnv


class EpisodeManifestTests(unittest.TestCase):
    def test_state_actions_sc_prebaseline_and_label_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            raw=base/"raw"
            folder=raw/"PPMI"
            folder.mkdir(parents=True)
            with (folder/"Baseline_Visit_Start_06Oct2026.csv").open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["PATNO","EVENT_ID","INFODT"])
                w.writerow(["PRIVATE_PATIENT_A","BL","03/2025"])
            src=base/"source.sqlite"
            db=sqlite3.connect(src)
            db.execute("""CREATE TABLE restricted_labels
                (cohort TEXT,patient_id TEXT,candidate_label TEXT)""")
            db.execute("""CREATE TABLE observations
                (cohort TEXT,patient_id TEXT,event_id TEXT,concept TEXT,value_raw TEXT,
                exam_month TEXT,date_precision TEXT,quality_status TEXT,medication_state TEXT,
                imputed_flag TEXT,source_file TEXT,source_row INTEGER)""")
            db.execute("INSERT INTO restricted_labels VALUES (?,?,?)",
                       ("PPMI","PRIVATE_PATIENT_A","PD"))
            entries=[
                ("BL","updrs3_total","24","2025-03","OFF",2),
                ("BL","updrs3_total","18","2025-03","ON",3),
                ("BL","moca_total","25","2025-03","",4),
                ("BL","rbd_vivid_dreams","1","2025-03","",5),
                ("SC","upsit_total","16","2025-01","",6),
            ]
            db.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                ("PPMI","PRIVATE_PATIENT_A",event,concept,value,month,
                 "month_candidate","observed_unverified",med,None,"test.csv",line)
                for event,concept,value,month,med,line in entries])
            db.commit()
            db.close()
            manifest=base/"manifest.sqlite"
            reports=base/"reports"
            summary=build(raw,src,"sqlite",manifest,reports,months=6,minimum=10)
            self.assertEqual(summary["episodes"],1)
            with DoctorPatientEnv(manifest,src,"PPMI","PRIVATE_PATIENT_A","sqlite") as env:
                self.assertEqual(env.order_test("updrs3_total@OFF")["value"],"24")
                self.assertEqual(env.order_test("updrs3_total@ON")["value"],"18")
                self.assertEqual(env.order_test("upsit_total")["visit"],"SC")
                self.assertEqual(env.order_test("updrs3_total@UNSPECIFIED")["status"],
                                 "not_available")
                self.assertEqual(env.order_test("PRIMDIAG")["status"],"invalid_action")
                self.assertEqual(env.queries,["updrs3_total@OFF","updrs3_total@ON",
                                              "upsit_total","updrs3_total@UNSPECIFIED"])
            with sqlite3.connect(manifest) as check:
                columns=[x[1] for x in check.execute("PRAGMA table_info(episode_candidates)")]
                self.assertNotIn("candidate_label",columns)
            for report in reports.iterdir():
                self.assertNotIn("PRIVATE_PATIENT_A",report.read_text())


if __name__=="__main__":
    unittest.main()
