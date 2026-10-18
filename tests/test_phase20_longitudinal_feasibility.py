"""Synthetic longitudinal feasibility checks; never read real patient data."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.phase20_longitudinal_feasibility import audit,parse_month,classify

class Phase20Tests(unittest.TestCase):
 def test_month_and_baseline_group(self):
  self.assertEqual(parse_month("02/2025"),2025*12+2)
  self.assertIsNone(parse_month("13/2025"))
  self.assertEqual(classify("4","25"),"prodromal_concordant")
 def test_no_patient_data_export(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/"raw";folder=root/"PPMI";folder.mkdir(parents=True)
   fixtures={
    "Participant_Status_06Oct2026.csv":(
      ["PATNO","COHORT"],[["SECRET_A","4"],["SECRET_B","1"]]),
    "Baseline_Visit_Start_06Oct2026.csv":(
      ["PATNO","EVENT_ID","INFODT"],
      [["SECRET_A","BL","01/2023"],["SECRET_B","BL","02/2023"]]),
    "Primary_Research_Diagnosis_06Oct2026.csv":(
      ["PATNO","EVENT_ID","PRIMDIAG","INFODT"],
      [["SECRET_A","BL","25","01/2023"],
       ["SECRET_A","V02","25","07/2023"],
       ["SECRET_A","V04","1","03/2024"],
       ["SECRET_B","BL","1","02/2023"]]),
    "MDS-UPDRS_Part_III_06Oct2026.csv":(
      ["PATNO","EVENT_ID","PATNO","EVENT_ID","NP3TOT","EXAMDT","PDSTATE"],
      [["SECRET_A","V02","SECRET_A","V02","16","07/2023","OFF"],
       ["SECRET_A","V04","SECRET_A","V04","22","03/2024","ON"]]),
   }
   for name,(headers,rows) in fixtures.items():
    with (folder/name).open("w",newline="") as file:
     w=csv.writer(file);w.writerow(headers);w.writerows(rows)
   out=Path(temp)/"reports"
   summary=audit(root,out,10)
   self.assertEqual(summary["status"],
      "aggregate_longitudinal_feasibility_not_conversion_labels")
   with (out/"longitudinal_feasibility.csv").open() as file:
    rows=list(csv.DictReader(file))
   prod=next(r for r in rows if r["baseline_group"]=="prodromal_concordant")
   self.assertEqual(prod["baseline_patients"],"<10")
   for f in out.iterdir():
    content=f.read_text()
    self.assertNotIn("SECRET_A",content)
    self.assertNotIn("01/2023",content)
if __name__=="__main__": unittest.main()
