"""Synthetic tests for duplicated headers and aggregate clinical QC."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.clinical_qc import scan, choose
from scripts.episode_audit import read_records

class ClinicalQCTests(unittest.TestCase):
    def test_duplicate_header_conflicts_are_not_silently_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"motor.csv"
            with file.open("w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(["PATNO","EVENT_ID","NP3TOT","INFODT","PATNO","EVENT_ID","PDSTATE"])
                writer.writerow(["SECRET_P1","BL","24","2025-01-01","SECRET_P1","BL","ON"])
                writer.writerow(["SECRET_P1","BL","25","2025-01-02","OTHER","BL","OFF"])
                writer.writerow(["","SC","26","2025-01-03","SECRET_P2","SC",""])
            stats,_,_,_,duplicates=scan(file,"NP3TOT")
            self.assertEqual(stats["patients"],2)
            self.assertEqual(stats["key_conflicts"],1)
            self.assertIn(("PATNO",2),duplicates)
            records=read_records(file,"NP3TOT")
            self.assertEqual(set(records),{"SECRET_P1","SECRET_P2"})
            self.assertEqual(len(records["SECRET_P1"]["BL"]),1)

    def test_missing_duplicate_value_prefer_populated(self):
        row=["","PPMI42"]
        value,conflict=choose(row,{"PATNO":[0,1]},"PATNO")
        self.assertEqual(value,"PPMI42")
        self.assertFalse(conflict)

if __name__=="__main__":
    unittest.main()
