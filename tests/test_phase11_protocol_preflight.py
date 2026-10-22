"""Synthetic Phase 11 protocol tests; no real patient records."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.phase11_protocol_preflight import (
    assign_folds, audit, evaluate_scores, project_actions,
)

class Phase11Tests(unittest.TestCase):
    def test_motor_collapse_and_fold_determinism(self):
        catalog={"updrs3_total@ON","updrs3_total@OFF","updrs3_total@UNSPECIFIED",
                 "updrs3_gait@ON","moca_total","rbd_vivid_dreams"}
        ready={"updrs3_total@ON","updrs3_gait@ON","moca_total"}
        self.assertEqual(project_actions(ready,"collapsed_motor",catalog),
                         frozenset({"motor_any_state","moca_total"}))
        self.assertEqual(project_actions(ready,"nonmotor_only",catalog),
                         frozenset({"moca_total"}))
        examples=[("SUBJ_"+str(i),"PD" if i%2 else "HC",bool(i%3),set())
                  for i in range(24)]
        self.assertEqual(assign_folds(examples,3,11),assign_folds(examples,3,11))
        result=evaluate_scores([(1.,True),(-1.,False)])
        self.assertEqual(result["balanced_accuracy"],1.0)

    def test_train_only_and_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            src=base/"source.sqlite"
            man=base/"manifest.sqlite"
            splits=base/"splits.sqlite"
            people=[("PPMI","PRIVATE_"+str(i),"PD" if i%2 else "HC",bool(i%3))
                    for i in range(30)]
            with sqlite3.connect(src) as db:
                db.execute("CREATE TABLE restricted_labels (cohort TEXT,patient_id TEXT,candidate_label TEXT)")
                db.executemany("INSERT INTO restricted_labels VALUES (?,?,?)",
                               [(c,p,l) for c,p,l,a in people])
            with sqlite3.connect(splits) as db:
                db.execute("CREATE TABLE patient_splits (cohort TEXT,patient_id TEXT,split TEXT)")
                db.executemany("INSERT INTO patient_splits VALUES (?,?,?)",
                               [(c,p,"train" if int(p.split("_")[-1])<24 else "test")
                                for c,p,l,a in people])
            actions=["updrs3_total@ON","updrs3_total@OFF","updrs3_total@UNSPECIFIED",
                     "age_at_visit","moca_total","rbd_vivid_dreams",
                     "scopa_aut_item1","upsit_total"]
            with sqlite3.connect(man) as db:
                db.execute("CREATE TABLE episode_candidates (cohort TEXT,patient_id TEXT,baseline_month TEXT)")
                db.execute("CREATE TABLE action_catalog (cohort TEXT,action_key TEXT)")
                db.execute("CREATE TABLE episode_actions (cohort TEXT,patient_id TEXT,action_key TEXT,status TEXT)")
                db.executemany("INSERT INTO episode_candidates VALUES (?,?,?)",
                               [(c,p,"2025-01" if a else None) for c,p,l,a in people])
                db.executemany("INSERT INTO action_catalog VALUES (?,?)",[("PPMI",a) for a in actions])
                db.executemany("INSERT INTO episode_actions VALUES (?,?,?,?)",
                               [(c,p,a,("ready" if ((l=="PD" and a=="updrs3_total@ON") or
                                     a=="moca_total") else "not_recorded"))
                                for c,p,l,_ in people for a in actions])
            out=base/"report"
            result=audit(src,"sqlite",man,splits,out,folds=3,seed=11,minimum=10)
            self.assertEqual(result["train_patient_count"],24)
            with (out/"train_only_shortcut_cv.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertEqual({int(r["n"]) for r in rows},{24})
            for path in out.iterdir():
                self.assertNotIn("PRIVATE_",path.read_text())

if __name__=="__main__":
    unittest.main()
