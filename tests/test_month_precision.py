import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.build_patient_db import parse_exam_time, build
from scripts.month_precision_audit import as_month, start_months, audit

class MonthPrecisionTests(unittest.TestCase):
    def test_month_has_no_invented_day(self):
        self.assertEqual(parse_exam_time("07/2022"),(None,"2022-07","month_candidate"))
        self.assertEqual(parse_exam_time("2022-07-19"),("2022-07-19","2022-07","day"))
        self.assertEqual(parse_exam_time("15/2022"),(None,None,"unknown"))
        self.assertEqual(as_month("07/2022"),"2022-07")
    def test_monthly_anchor_aggregation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"raw";f=root/"PPMI";f.mkdir(parents=True)
            with (f/"Baseline_Visit_Start_06Oct2026.csv").open("w",newline="") as h:
                w=csv.writer(h);w.writerow(["PATNO","EVENT_ID","INFODT"])
                w.writerows([["SECRET_A","BL","02/2025"],["SECRET_B","BL","03/2025"]])
            months,stats=start_months(root)
            self.assertEqual(months["SECRET_A"],"2025-02")
            self.assertEqual(stats["resolved_BL_month_anchor"],2)
            dbpath=Path(d)/"db.sqlite"
            db=sqlite3.connect(dbpath)
            db.execute("CREATE TABLE observations (cohort TEXT,patient_id TEXT,event_id TEXT,concept TEXT,value_raw TEXT,exam_month TEXT,date_precision TEXT,date_source TEXT,medication_state TEXT,quality_status TEXT)")
            db.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",[
                ("PPMI","SECRET_A","SC","moca_total","20","2025-01","month_candidate","INFODT",None,"observed_unverified"),
                ("PPMI","SECRET_A","BL","updrs3_total","22","2025-02","month_candidate","EXAMDT","OFF","observed_unverified"),
                ("PPMI","SECRET_A","BL","updrs3_total","18","2025-02","month_candidate","EXAMDT","ON","observed_unverified")])
            db.commit();db.close()
            out=Path(d)/"reports"
            summary=audit(root,dbpath,"sqlite",out,1)
            self.assertEqual(summary["n_unique_BL_month_anchors"],2)
            text=(out/"month_chronology.csv").read_text()
            self.assertIn("earlier_month",text)
            self.assertIn("same_month_order_unknown",text)
            for p in out.iterdir():
                self.assertNotIn("SECRET_A",p.read_text())

if __name__=="__main__":
    unittest.main()
