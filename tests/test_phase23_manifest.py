import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase23_manifest import build, split_all

class ManifestTests(unittest.TestCase):
    def test_stable_split(self):
        ids = [f"ID{i}" for i in range(60)]
        self.assertEqual(split_all(ids, 42), split_all(ids[::-1], 42))
        self.assertEqual(len(split_all(ids, 42)), 60)

    def test_strict_followup_and_no_raw_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data" / "PPMI"
            folder.mkdir(parents=True)
            files = {
                "Participant_Status_06Oct2026.csv": (
                    ["PATNO","COHORT"], [["FAKE_A","4"],["FAKE_B","4"]]),
                "Primary_Research_Diagnosis_06Oct2026.csv": (
                    ["PATNO","EVENT_ID","PRIMDIAG"],
                    [["FAKE_A","BL","25"],["FAKE_B","BL","25"]]),
                "Baseline_Visit_Start_06Oct2026.csv": (
                    ["PATNO","EVENT_ID","INFODT"],
                    [["FAKE_A","BL","01/2023"],["FAKE_B","BL","01/2023"]]),
                "MDS-UPDRS_Part_III_06Oct2026.csv": (
                    ["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"],
                    [["FAKE_A","BL","10","01/2023",""],
                     ["FAKE_A","V04","15","01/2024",""],
                     ["FAKE_B","BL","11","01/2024",""],
                     ["FAKE_B","V04","12","01/2024",""]])
            }
            for name, (head, rows) in files.items():
                with (folder/name).open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(head)
                    writer.writerows(rows)
            dbpath, reports = Path(tmp)/"private.sqlite", Path(tmp)/"report"
            result = build(folder.parent, dbpath, reports, 42)
            self.assertEqual(result["baseline_candidates"], 2)
            self.assertEqual(result["ready_12m"], 1)
            with sqlite3.connect(dbpath) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM targets WHERE horizon=12").fetchone()[0], 1)
            for path in reports.iterdir():
                body = path.read_text()
                self.assertNotIn("FAKE_A", body)
                self.assertNotIn("01/2023", body)
