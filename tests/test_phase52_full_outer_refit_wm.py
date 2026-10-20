"""Phase52: patient-disjoint nested folds and full outer-refit regression tests."""
import random,tempfile,unittest
from pathlib import Path
import torch
from scripts.phase52_full_outer_refit_wm import (
    assign_folds,choose_inner,refit_on_full_outer_train,evaluate
)

def demo(n=120):
    rng=random.Random(15)
    rows=[]
    for i in range(n):
        bl=float(rng.randint(0,10))
        recent=float(max(0,bl+rng.randint(-2,4)))
        later=float(max(0,min(132,(bl+recent)/2+rng.randint(-2,3))))
        rows.append(dict(pid="SYNTHETIC_PRIVATE_"+str(i),bl=bl,last=recent,
                         target=later,gap12=12,future_gap=12))
    return rows

class Phase52Tests(unittest.TestCase):
    def test_5_and_10_folds_are_patient_disjoint(self):
        records=demo()
        for n in (5,10):
            folds=assign_folds(records,n,42)
            self.assertEqual(len(folds),len(records))
            self.assertEqual(set(folds.values()),set(range(n)))
            for k in range(n):
                fit=[r for r in records if folds[r["pid"]]!=k]
                held=[r for r in records if folds[r["pid"]]==k]
                self.assertFalse({r["pid"] for r in fit}&
                                 {r["pid"] for r in held})
                self.assertEqual(len(fit)+len(held),len(records))
    def test_fresh_full_refit_uses_all_outer_people(self):
        torch.set_num_threads(2)
        records=demo()[:80]
        fit,dev=choose_inner(records,12)
        self.assertEqual(len(fit)+len(dev),len(records))
        model=refit_on_full_outer_train(records,2,123,8,torch.device("cpu"))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
    def test_end_to_end_5fold_aggregate_only(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"metrics"
            r=evaluate(demo(),output,folds=5,members=1,epochs=2,
                       seed=12,width=8,device_name="cpu",bootstraps=100)
            self.assertEqual(r["outer_folds"],5)
            self.assertEqual(r["n_independent_train_trajectories"],120)
            self.assertEqual(r["full_outer_training_patients_per_fold"],[96])
            for file in output.iterdir():
                self.assertNotIn("SYNTHETIC_PRIVATE_",file.read_text())
            self.assertTrue((output/"oof_metrics.csv").exists())

if __name__=="__main__":unittest.main()
