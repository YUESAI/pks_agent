"""Synthetic Phase21 regression tests; no private patient material."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.phase21_longitudinal_target_readiness import (
    evaluate, month_number, motor_state
)

class Phase21Tests(unittest.TestCase):
    def test_month_state_parsing(self):
        self.assertEqual(month_number("02/2025"),2025*12+2)
        self.assertIsNone(month_number("13/2025"))
        self.assertEqual(motor_state("OFF"),"OFF")
        self.assertEqual(motor_state(""),"UNSPECIFIED")

    def test_horizons_and_motor_repeats_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw";folder=root/"PPMI";folder.mkdir(parents=True)
            fixture={
                "Participant_Status_06Oct2026.csv":(
                    ["PATNO","COHORT"],[["PRIVATE_PD","4"],["PRIVATE_STABLE","4"]]),
                "Baseline_Visit_Start_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","INFODT"],
                    [["PRIVATE_PD","BL","01/2023"],["PRIVATE_STABLE","BL","01/2023"]]),
                "Primary_Research_Diagnosis_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","PRIMDIAG","INFODT"],[
                        ["PRIVATE_PD","BL","25","01/2023"],
                        ["PRIVATE_PD","V02","25","07/2023"],
                        ["PRIVATE_PD","V04","1","03/2024"],
                        ["PRIVATE_PD","V06","1","10/2024"],
                        ["PRIVATE_STABLE","BL","25","01/2023"],
                        ["PRIVATE_STABLE","V04","25","03/2024"]]),
                "MDS-UPDRS_Part_III_06Oct2026.csv":(
                    ["PATNO","EVENT_ID","PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"],[
                        ["PRIVATE_PD","BL","PRIVATE_PD","BL","10","01/2023","OFF"],
                        ["PRIVATE_PD","V02","PRIVATE_PD","V02","12","07/2023","OFF"],
                        ["PRIVATE_PD","V04","PRIVATE_PD","V04","14","03/2024","OFF"],
                        ["PRIVATE_STABLE","BL","PRIVATE_STABLE","BL","2","01/2023",""]])
            }
            for name,(header,rows) in fixture.items():
                with (folder/name).open("w",newline="") as h:
                    w=csv.writer(h);w.writerow(header);w.writerows(rows)
            out=Path(tmp)/"aggregates"
            summary=evaluate(root,out,10)
            self.assertEqual(summary["candidate_first_later_PD_code_patients"],1)
            with (out/"diagnosis_observability_horizons.csv").open() as f:
                rows=list(csv.DictReader(f))
            h12={r["status"]:r["patients"] for r in rows if r["horizon_months"]=="12"}
            self.assertEqual(h12["PD_code_seen_by_horizon"],"0")
            self.assertEqual(h12["followup_through_horizon_no_PD_code"],"<10")
            h24={r["status"]:r["patients"] for r in rows if r["horizon_months"]=="24"}
            self.assertEqual(h24["PD_code_seen_by_horizon"],"<10")
            self.assertEqual(h24["not_observable_to_horizon"],"<10")
            for file in out.iterdir():
                body=file.read_text()
                self.assertNotIn("PRIVATE_PD",body)
                self.assertNotIn("03/2024",body)

if __name__=="__main__":
    unittest.main()
