import csv
import tempfile
import unittest
from pathlib import Path
from scripts.profile_clinical import scan, find_table

class CoverageProfileTests(unittest.TestCase):
    def test_patient_keys_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PPMI" / "Montreal_Cognitive_Assessment__MoCA__06Oct2026.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["PATNO", "EVENT_ID", "MCATOT"])
                w.writerows([
                    ["SECRET_A", "BL", "24"],
                    ["SECRET_A", "BL", "25"],
                    ["SECRET_B", "SC", ""],
                    ["SECRET_C", "BL", "30"],
                ])
            self.assertEqual(find_table(path.parent, "PPMI",
                             "Montreal_Cognitive_Assessment__MoCA__"), path)
            s = scan(path, "PATNO", "EVENT_ID", {"moca_total": "MCATOT"}, 0)
            self.assertEqual(s["n_rows"], 4)
            self.assertEqual(len(s["patient_ids"]), 3)
            self.assertEqual(s["unique_keys"], 3)
            self.assertEqual(s["duplicates"], 1)
            self.assertEqual(len(s["feature_patients"]["moca_total"]), 2)
