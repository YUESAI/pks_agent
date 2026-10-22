"""Phase58 regression tests for fixed 10-strategy matching and privacy."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN
from scripts.phase58_ten_strategy_sweep import METHODS,fit_features,fitted_scores,run
from scripts.phase58_paired_metrics import compute_matrix

def synthetic(n=130):
    patients=[]
    hidden={ALL.index(k) for k in HIDDEN}
    for i in range(n):
        visits=[]
        for t,event in ((0,"BL"),(6,"R01"),(12,"V04")):
            score=[int((i+2*k+t)%17==0) for k in range(33)]
            status=[1]*33
            if event=="R01":
                for k in hidden:
                    score[k]=0;status[k]=2
            visits.append({"event":event,"month":24241+t,
                "score":tuple(score),"status":tuple(status),
                "documented_total":sum(score) if event!="R01" else None})
        patients.append({"pid":"SIMPRIVATE_"+str(i),"visits":visits})
    return patients

class Phase58Tests(unittest.TestCase):
    def test_paired_resampling_same_patients(self):
        labels={str(i):int(i%3==0) for i in range(90)}
        s={str(i):float(i%12) for i in range(90)}
        scores={"visible27_sum":s,"duplicated":dict(s)}
        table,paired,overlap=compute_matrix(scores,labels,sorted(labels),
                                            42,nboot=100)
        self.assertEqual(len(table),2)
        self.assertEqual(paired[0]["delta_recall_at_k"],0)
        self.assertEqual(paired[0]["delta_auroc"],0)
        self.assertAlmostEqual(overlap[0]["selected_jaccard"],1.)
    def test_feature_fitting_has_no_target_feature(self):
        from scripts.phase56_remote_to_site_wm import find_pairs
        pairs,_=find_pairs(synthetic())
        self.assertGreater(len(pairs),100)
        fitted=fit_features(pairs)
        estimates=fitted_scores(fitted,pairs)
        for v in estimates.values():
            self.assertEqual(len(v),len(pairs))
            self.assertTrue(np.isfinite(list(v.values())).all())
    def test_smoke_ten_methods_no_patient_records_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary=run(synthetic(),Path(tmp)/"aggregate",
                        epochs=2,folds=5,bootstrap=100,device_name="cpu")
            self.assertEqual(summary["n_strategies"],10)
            self.assertEqual(len(METHODS),10)
            report=Path(tmp)/"aggregate"
            for name in ("ten_strategy_metrics.csv","paired_bootstrap.csv",
                         "topk_overlap.csv","fold_training_compute.csv",
                         "summary.json"):
                p=report/name
                self.assertTrue(p.is_file(),name)
                self.assertNotIn("SIMPRIVATE_",p.read_text())
            self.assertEqual(
                len((report/"ten_strategy_metrics.csv").read_text().splitlines()),11)

if __name__=="__main__":unittest.main()
