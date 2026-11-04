"""Phase89: independent expert OOF gate boundary and leakage tests."""
import unittest
import numpy as np
import torch
from scripts.phase89_oof_pretrained_expert_gate import (
    OOFDecisionGate,gate_features,patient_weights,inner_partition)

class Phase89Tests(unittest.TestCase):
    def sample_rows(self):
        history=[{"month":0,"scores":{"NP3SPCH":1}}]
        rows=[]
        for i in range(24):
            rows.append({"pid":str(i//2),"history":history,
                         "anchor":0,"horizon":12 if i%2==0 else 24,
                         "site":[i%3,0,0,0,0,0]})
        return rows
    def probability(self,n,seed=0):
        rng=np.random.default_rng(seed)
        p=rng.dirichlet(np.ones(25),size=n)
        return p.astype(np.float32)
    def test_patient_partition_shared_across_horizons(self):
        self.assertEqual(inner_partition("31"),inner_partition("31"))
        self.assertIn(inner_partition("31"),(0,1))
    def test_feature_shapes(self):
        rows=self.sample_rows()
        p=self.probability(len(rows))
        q=self.probability(len(rows),1)
        self.assertEqual(tuple(gate_features(rows,p,q,"horizon").shape),(24,2))
        self.assertEqual(tuple(gate_features(rows,p,q,"patient").shape),(24,10))
    def test_patient_weights_sum_once(self):
        w=patient_weights(self.sample_rows())
        self.assertAlmostEqual(float(w.sum()),1.0,places=6)
        self.assertTrue(torch.allclose(w,torch.full((24,),1/24)))
    def test_gate_training_and_predict(self):
        rows=self.sample_rows()
        s=self.probability(24)
        d=self.probability(24,1)
        for kind in ("horizon","patient"):
            with self.subTest(kind=kind):
                gate=OOFDecisionGate(kind).fit(rows,s,d)
                p,w=gate.predict(rows,s,d)
                self.assertEqual(p.shape,(24,25))
                self.assertEqual(w.shape,(24,))
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((w>=0)&(w<=1)).all())
                np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-6)
    def test_reject_post_target_history(self):
        rows=self.sample_rows()
        rows[0]["history"]=[{"month":13,"scores":{}}]
        with self.assertRaises(AssertionError):
            gate_features(rows,self.probability(24),
                          self.probability(24,1),"horizon")

if __name__=="__main__":
    unittest.main()
