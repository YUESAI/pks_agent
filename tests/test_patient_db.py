import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.build_patient_db import build, candidate_label
from scripts.patient_store import PatientStore
from scripts.report_patient_db import group_coverage

class PatientDatabaseTests(unittest.TestCase):
    def test_candidate_label_policy(self):
        self.assertEqual(candidate_label("PPMI","1","1"),"PD")
        self.assertEqual(candidate_label("PPMI","2","17"),"HC")
        self.assertIsNone(candidate_label("PPMI","4","1"))
        self.assertEqual(candidate_label("BioFIND","1","01"),"PD")

    def test_local_build_separated_labels_and_on_demand_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw"; ppmi=root/"PPMI";ppmi.mkdir(parents=True)
            fixtures={
             "Participant_Status_06Oct2026.csv":(["PATNO","COHORT"],[["SECRET_A","1"],["SECRET_B","2"]]),
             "Primary_Research_Diagnosis_06Oct2026.csv":
                 (["PATNO","EVENT_ID","PRIMDIAG"],[["SECRET_A","BL","1"],["SECRET_B","BL","17"]]),
             "Age_at_visit_06Oct2026.csv":
                 (["PATNO","EVENT_ID","AGE_AT_VISIT"],[["SECRET_A","BL","61"]]),
             "MDS-UPDRS_Part_III_06Oct2026.csv":
                 (["PATNO","EVENT_ID","PATNO","EVENT_ID","NP3TOT","NP3GAIT","NP3BRADY"],
                  [["SECRET_A","BL","SECRET_A","BL","23","2","1"],
                   ["SECRET_A","BL","SECRET_A","BL","24","3","2"]]),
             "Montreal_Cognitive_Assessment__MoCA__06Oct2026.csv":
                 (["PATNO","EVENT_ID","MCATOT"],[["SECRET_A","BL","26"]]),
             "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_06Oct2026.csv":
                 (["PATNO","EVENT_ID","TOTAL_CORRECT","IMPUTED_DATA"],
                  [["SECRET_A","SC","18","true"]]),
             "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_06Oct2026.csv":
                 (["PATNO","EVENT_ID","DRMVIVID"],[["SECRET_A","BL","1"]]),
             "SCOPA-AUT_06Oct2026.csv":
                 (["PATNO","EVENT_ID","SCAU1"],[["SECRET_A","BL","2"]])
            }
            for name,(header,rows) in fixtures.items():
                with (ppmi/name).open("w",newline="") as f:
                    w=csv.writer(f);w.writerow(header);w.writerows(rows)
            db=Path(tmp)/"work"/"patient.sqlite"
            summary=build(root,db,"sqlite",["PPMI"])
            self.assertEqual(summary["counts"]["PPMI_candidate_PD"],1)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM restricted_labels").fetchone()[0],2)
                self.assertEqual(group_coverage(conn,"PPMI",("moca_total",),"BL",True),(1,0))
            store=PatientStore(db,"sqlite")
            self.assertNotIn("PRIMDIAG",str(store.public_catalog("PPMI")))
            self.assertEqual(store.order_test("PPMI","SECRET_A","moca_total")["value_raw"],"26")
            self.assertEqual(store.order_test("PPMI","SECRET_A","updrs3_total")["status"],
                             "requires_duplicate_resolution")
            self.assertEqual(store.order_test("PPMI","SECRET_A","upsit_total","SC")["status"],
                             "quality_review_required")
            store.close()

if __name__=="__main__":
    unittest.main()
