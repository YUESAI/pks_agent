"""Phase26 synthetic tests: strict TRAIN-only access and fitted-fold imputation."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.phase26_multimodal_oof import (
    audit, fit_multi, predict_multi, bootstrap_differences,
)

class Phase26Tests(unittest.TestCase):
    def test_fold_only_median_imputation_and_model(self):
        rows=[
            (f"P{i}",float(i%10),float(i%10)+2,
             {"moca_total":float(20+i%5),
              "rbd_vivid_dreams":float(i%2),
              "scopa_aut_item1":float(i%3),
              "upsit_total":None if i%2 else float(15+i%9)})
            for i in range(40)]
        model=fit_multi(rows,("moca_total","upsit_total"))
        self.assertEqual(model["names"],("moca_total","upsit_total"))
        self.assertTrue(0<=predict_multi(model,5,{"moca_total":None,"upsit_total":None})<=132)
        ref={f"P{i}":(float(i),float(i)) for i in range(20)}
        other={f"P{i}":(float(i),float(i)+1) for i in range(20)}
        pairs=bootstrap_differences(ref,other,100,42)
        self.assertEqual(len(pairs),2)
        self.assertAlmostEqual(pairs[0]["difference"],1.)
    def test_no_val_test_source_data_used(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);manifest=root/"private.sqlite";source=root/"clinical.sqlite"
            with sqlite3.connect(manifest) as db:
                db.execute("CREATE TABLE patients(pid TEXT PRIMARY KEY,split TEXT,baseline_month INTEGER,baseline_motor REAL)")
                db.execute("CREATE TABLE targets(pid TEXT,horizon INTEGER,baseline_month INTEGER,target_month INTEGER,months_apart INTEGER,baseline_motor REAL,target_motor REAL,delta_motor REAL,PRIMARY KEY(pid,horizon))")
                for i in range(30):
                    pid=f"PRIVATE_TRAIN_{i}";x=float(i%10)
                    db.execute("INSERT INTO patients VALUES(?,?,?,?)",(pid,"train",2025*12+5,x))
                    db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?,?)",(pid,12,2025*12+5,2026*12+5,12,x,x+2,2))
                for split in ("val","test"):
                    pid="SECRET_"+split
                    db.execute("INSERT INTO patients VALUES(?,?,?,?)",(pid,split,2025*12+5,2))
                    db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?,?)",(pid,12,2025*12+5,2026*12+5,12,2,132,130))
            with sqlite3.connect(source) as db:
                db.execute("""CREATE TABLE observations(
                    cohort TEXT,patient_id TEXT,event_id TEXT,concept TEXT,
                    value_raw TEXT,exam_month TEXT,quality_status TEXT,imputed_flag TEXT)""")
                for i in range(30):
                    pid=f"PRIVATE_TRAIN_{i}"
                    for concept,value in (("moca_total","25"),
                                          ("rbd_vivid_dreams","1"),
                                          ("scopa_aut_item1","0")):
                        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
                            ("PPMI",pid,"BL",concept,value,"2025-05","observed_unverified",None))
                    db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
                        ("PPMI",pid,"BL","age_at_visit","67",None,"observed_unverified",None))
                for split in ("val","test"):
                    db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)",
                        ("PPMI","SECRET_"+split,"BL","moca_total","1","2025-05","observed_unverified",None))
            reports=root/"report"
            result=audit(manifest,source,"sqlite",reports,5,42,100,10)
            self.assertEqual(result["n_train_pairs"],30)
            with (reports/"multimodal_train_oof.csv").open() as h:
                models=list(csv.DictReader(h))
            self.assertEqual(len(models),5)
            for f in reports.iterdir():
                text=f.read_text()
                for private in ("PRIVATE_TRAIN","SECRET_val","SECRET_test"):
                    self.assertNotIn(private,text)
if __name__=="__main__":
    unittest.main()
