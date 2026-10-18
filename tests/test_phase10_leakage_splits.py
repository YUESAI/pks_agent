import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from scripts.phase10_leakage_splits import auc, audit, split_patients, classify_naive_bayes, evaluate

class Phase10Tests(unittest.TestCase):
    def test_auc_and_deterministic_stratification(self):
        self.assertAlmostEqual(auc([(2,True),(1,False)]),1.0)
        self.assertAlmostEqual(auc([(1,True),(1,False)]),0.5)
        groups={("PPMI","PD",True):["PRIVATE_PD_"+str(i) for i in range(20)],
                ("PPMI","HC",False):["PRIVATE_HC_"+str(i) for i in range(20)],
                ("BioFIND","HC",False):["PRIVATE_EXTERNAL"]}
        first=split_patients(groups,42)
        self.assertEqual(first,split_patients(groups,42))
        self.assertEqual(first[("BioFIND","PRIVATE_EXTERNAL")],"external_test")
        self.assertEqual(len(first),41)

    def test_reports_aggregate_and_split_stays_local(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/"source.sqlite";manifest=root/"manifest.sqlite"
            db=sqlite3.connect(source)
            db.execute("CREATE TABLE restricted_labels(cohort TEXT,patient_id TEXT,candidate_label TEXT)")
            db.executemany("INSERT INTO restricted_labels VALUES (?,?,?)",
               [("PPMI","SECRET_PD_"+str(i),"PD") for i in range(12)]+
               [("PPMI","SECRET_HC_"+str(i),"HC") for i in range(12)]+
               [("BioFIND","SECRET_EXTERNAL","HC")])
            db.commit();db.close()
            db=sqlite3.connect(manifest)
            db.execute("CREATE TABLE episode_candidates(cohort TEXT,patient_id TEXT,baseline_month TEXT)")
            db.execute("CREATE TABLE action_catalog(cohort TEXT,action_key TEXT)")
            db.execute("CREATE TABLE episode_actions(cohort TEXT,patient_id TEXT,action_key TEXT,status TEXT)")
            db.executemany("INSERT INTO action_catalog VALUES(?,?)",
                           [("PPMI","moca_total"),("BioFIND","moca_total")])
            patients=[("PPMI","SECRET_PD_"+str(i),"2025-01") for i in range(12)]+[("PPMI","SECRET_HC_"+str(i),None) for i in range(12)]+[("BioFIND","SECRET_EXTERNAL",None)]
            db.executemany("INSERT INTO episode_candidates VALUES(?,?,?)",patients)
            db.executemany("INSERT INTO episode_actions VALUES(?,?,?,?)",
                [(c,p,"moca_total","ready" if ("PD" in p or c=="BioFIND") else "not_recorded")
                 for c,p,_ in patients])
            db.commit();db.close()
            reports=root/"reports";splits=root/"local_splits.sqlite"
            result=audit(source,"sqlite",manifest,splits,reports,42,10)
            self.assertEqual(result["patients_total"],25)
            with sqlite3.connect(splits) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM patient_splits").fetchone()[0],25)
            for file in reports.iterdir():
                text=file.read_text()
                self.assertNotIn("SECRET_PD_",text)
                self.assertNotIn("SECRET_HC_",text)
                self.assertNotIn("SECRET_EXTERNAL",text)
            with (reports/"availability_only_baseline.csv").open() as f:
                rows=list(csv.DictReader(f))
                self.assertEqual({r["split"] for r in rows},{"train","val","test"})

if __name__=="__main__":
    unittest.main()
