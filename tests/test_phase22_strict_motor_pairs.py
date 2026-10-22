"""Synthetic Phase22 tests; no real patient data."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.phase22_strict_motor_pairs import audit, target_for_window,parse_month

class Phase22Tests(unittest.TestCase):
    def test_tie_refused_and_months(self):
        self.assertEqual(parse_month("01/2024"),2024*12+1)
        self.assertEqual(target_for_window({2024*12+1:15},2023*12+1,12,3)[0],"paired")
        self.assertEqual(target_for_window(
            {2023*12+11:13,2024*12+3:18},2023*12+1,12,3)[0],"equidistant_months")

    def test_baseline_not_reclassified_as_followup_and_reports_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw";folder=root/"PPMI";folder.mkdir(parents=True)
            files={
                "Participant_Status_06Oct2026.csv":(
                    ["PATNO","COHORT"],[["PRIVATE_A","4"],["PRIVATE_B","4"]]),
                "Primary_Research_Diagnosis_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","PRIMDIAG"],[
                        ["PRIVATE_A","BL","25"],["PRIVATE_B","BL","25"]]),
                "Baseline_Visit_Start_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","INFODT"],[
                        ["PRIVATE_A","BL","01/2023"],["PRIVATE_B","BL","01/2023"]]),
                "MDS-UPDRS_Part_III_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"],[
                        ["PRIVATE_A","BL","10","01/2023",""],
                        ["PRIVATE_A","V04","18","01/2024",""],
                        # This BL event is a year late and must not count as either baseline or follow-up.
                        ["PRIVATE_B","BL","11","01/2024",""],
                        ["PRIVATE_B","V04","15","01/2024",""]]),
            }
            for name,(headers,rows) in files.items():
                with (folder/name).open("w",newline="") as f:
                    w=csv.writer(f);w.writerow(headers);w.writerows(rows)
            out=Path(tmp)/"reports"
            summary=audit(root,out,10)
            self.assertEqual(summary["status"],
                             "strict_motor_month_pair_feasibility_not_training_ready")
            with (out/"strict_pair_window_counts.csv").open() as f:
                counts=list(csv.DictReader(f))
            r=next(row for row in counts
                   if row["baseline_group"]=="prodromal"
                   and row["medication_state"]=="UNSPECIFIED"
                   and row["target_month"]=="12")
            self.assertEqual(r["unique_strict_pairs"],"<10")
            self.assertEqual(r["baseline_motor_ready"],"<10")
            for file in out.iterdir():
                self.assertNotIn("PRIVATE_A",file.read_text())
                self.assertNotIn("01/2023",file.read_text())

if __name__=="__main__":
    unittest.main()
