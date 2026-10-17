import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path
from scripts.episode_audit import feature_views, label_and_date, analyze

class EpisodeTests(unittest.TestCase):
    def test_chronology_and_label_conflicts(self):
        rows={"secret1":{"SC":[{"value":"20","date":date(2025,1,1),"state":""}]},
              "secret2":{"SC":[{"value":"21","date":date(2025,5,1),"state":""}]},
              "secret3":{"BL":[{"value":"22","date":None,"state":""}]}}
        dates={p:date(2025,3,1) for p in rows}
        views,_=feature_views(rows,dates,180)
        self.assertEqual(views["strict_BL"],{"secret3"})
        self.assertEqual(views["SC_BL_candidate"],set(rows))
        self.assertEqual(views["SC_BL_date_checked"],{"secret1","secret3"})
        labels,_,conflicts,_=label_and_date({
            "hidden":{"BL":[{"value":"1","date":None},{"value":"17","date":None}]}})
        self.assertEqual(labels,{})
        self.assertEqual(conflicts,1)

    def test_no_patient_values_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"raw"
            folder=root/"PPMI"
            folder.mkdir(parents=True)
            tables={
                "Participant_Status_06Oct2026.csv":(["PATNO","COHORT"],[["SECRET_A","4"]]),
                "Primary_Research_Diagnosis_06Oct2026.csv":
                    (["PATNO","EVENT_ID","INFODT","PRIMDIAG"],[["SECRET_A","BL","2025-03-01","1"]]),
                "MDS-UPDRS_Part_III_06Oct2026.csv":
                    (["PATNO","EVENT_ID","NP3TOT"],[["SECRET_A","BL","21"]]),
                "Montreal_Cognitive_Assessment__MoCA__06Oct2026.csv":
                    (["PATNO","EVENT_ID","MCATOT"],[["SECRET_A","BL","24"]]),
                "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_06Oct2026.csv":
                    (["PATNO","EVENT_ID","INFODT","TOTAL_CORRECT"],[["SECRET_A","SC","2025-02-01","19"]]),
                "REM_Sleep_Behavior_Disorder_Screening_Questionnaire_06Oct2026.csv":
                    (["PATNO","EVENT_ID","DRMVIVID"],[["SECRET_A","BL","1"]]),
                "SCOPA-AUT_06Oct2026.csv":(["PATNO","EVENT_ID","SCAU1"],[["SECRET_A","BL","1"]]),
                "Age_at_visit_06Oct2026.csv":
                    (["PATNO","EVENT_ID","AGE_AT_VISIT"],[["SECRET_A","BL","60"]]),
            }
            for name,(header,rows) in tables.items():
                with (folder/name).open("w",newline="") as f:
                    w=csv.writer(f);w.writerow(header);w.writerows(rows)
            output=Path(tmp)/"reports"
            result=analyze(root,output,["PPMI"],180,10)
            self.assertEqual(result["overview"][0]["anchor_intersect_BL_diagnosis"],1)
            for file in output.iterdir():
                self.assertNotIn("SECRET_A",file.read_text())
