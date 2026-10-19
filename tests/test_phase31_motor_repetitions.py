"""Synthetic Phase31 repeat-mechanism tests, no private PPMI records."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase31_motor_repetitions import (
    categorize,context_for,score_spread_bin,summarize
)

class Phase31Tests(unittest.TestCase):
    def test_mechanisms_and_event_gate(self):
        from collections import defaultdict
        def item(events,rows=2,invalid=False):
            return {"invalid":invalid,"rows":rows,
                    "event_values":{k:set(v) for k,v in events.items()},
                    "scores":set(x for values in events.values() for x in values)}
        self.assertEqual(categorize(item({"BL":[10]})),"exact_repeat_same_event")
        self.assertEqual(categorize(item({"V02":[10],"V03":[10]})),
                         "different_events_same_score")
        self.assertEqual(categorize(item({"V02":[10],"V03":[14]})),
                         "different_events_different_scores")
        self.assertEqual(categorize(item({"V02":[10,14]})),
                         "same_event_different_scores")
        self.assertEqual(categorize(item({"V02":[10]},invalid=True)),
                         "invalid_value_in_key")
        self.assertEqual(context_for("BL",2025*12+12,2025*12),
                         "bad_BL_month")
        self.assertEqual(context_for("V04",2025*12+12,2025*12),
                         "12m_window")
        self.assertEqual(score_spread_bin(4),"3_to_5")

    def test_training_only_and_no_identifier_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            raw=root/"data"/"PPMI";raw.mkdir(parents=True)
            database=root/"private.sqlite"
            with sqlite3.connect(database) as conn:
                conn.execute("CREATE TABLE patients(pid TEXT,split TEXT,baseline_month INTEGER)")
                conn.executemany("INSERT INTO patients VALUES(?,?,?)",[
                    ("SECRET_TRAIN","train",2025*12),
                    ("SECRET_TEST","test",2025*12)])
            path=raw/"MDS-UPDRS_Part_III_06Oct2026.csv"
            with path.open("w",newline="") as f:
                w=csv.writer(f)
                w.writerow(["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"])
                w.writerows([
                    ["SECRET_TRAIN","BL","4","12/2025",""],
                    ["SECRET_TRAIN","BL","4","12/2025",""],
                    ["SECRET_TRAIN","V04","8","12/2026",""],
                    ["SECRET_TRAIN","V05","9","12/2026",""],
                    ["SECRET_TEST","V04","20","12/2026",""]])
            out=root/"reports"
            result=summarize(raw.parent,database,out,10)
            self.assertEqual(result["frozen_train_population"],1)
            with (out/"collision_mechanisms.csv").open() as f:
                rows=list(csv.DictReader(f))
            found=[r for r in rows if r["visit_context"]=="12m_window" and
                   r["mechanism"]=="different_events_different_scores"]
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]["patient_month_keys"],"<10")
            for file in out.iterdir():
                body=file.read_text()
                self.assertNotIn("SECRET_TRAIN",body)
                self.assertNotIn("SECRET_TEST",body)
if __name__=="__main__":
    unittest.main()
