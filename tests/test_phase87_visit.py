"""Phase87 transition supervision and FIT-only auxiliary accounting."""
import unittest
import numpy as np
import torch
from scripts.phase87_visit_anchored_progression_wm import (
    VisitAnchoredFlowNet,merge_fit_auxiliary)
from scripts.phase85_correlated_site_wm import correlated_total_distribution

class Phase87Test(unittest.TestCase):
    def test_visit_logits_distribution_and_gradients(self):
        net=VisitAnchoredFlowNet()
        x=torch.randn(4,6,54)
        dt=torch.zeros(4,6)
        valid=torch.ones(4,6,dtype=torch.bool)
        horizon=torch.tensor([[1/3,1/3],[2/3,2/3],[1/3,1/6],[1/2,1/3]])
        mix,sites,next_motor=net(x,dt,valid,torch.empty(4,0),horizon)
        self.assertEqual(next_motor.shape,(4,27,5))
        p=correlated_total_distribution(mix,sites)
        self.assertEqual(p.shape,(4,25))
        np.testing.assert_allclose(p.detach().sum(-1),np.ones(4),atol=1e-6)
        loss=-p[:,2].clamp_min(1e-12).log().mean()
        loss+=torch.nn.functional.cross_entropy(
            next_motor.reshape(-1,5),torch.zeros(4*27,dtype=torch.long))
        loss.backward()
        self.assertTrue(torch.isfinite(net.drift[0].weight.grad).all())
        self.assertTrue(torch.isfinite(net.next_motor.weight.grad).all())

    def test_no_future_or_zero_time(self):
        net=VisitAnchoredFlowNet()
        x=torch.randn(2,6,54)
        dt=torch.zeros(2,6)
        valid=torch.ones(2,6,dtype=torch.bool)
        with self.assertRaises(ValueError):
            net(x,dt,valid,torch.empty(2,0),
                torch.tensor([[1/3,0.],[1/3,1/3]]))

    def test_auxiliary_union_deduplicated(self):
        common={"pid":"A","anchor":0,"horizon":6,
                "aux_motor":[0]*27}
        unique={"pid":"B","anchor":6,"horizon":6,
                "aux_motor":[1]*27}
        datasets={
            ("prodromal",12):{"fit":[{"pid":"A"},{"pid":"B"}],
                              "aux":[common,unique]},
            ("prodromal",24):{"fit":[{"pid":"A"}],
                              "aux":[dict(common)]}}
        union=merge_fit_auxiliary(datasets)
        self.assertEqual(len(union),2)
        self.assertEqual([r["pid"] for r in union],["A","B"])

    def test_reject_auxiliary_from_dev_person(self):
        datasets={
            ("prodromal",12):{"fit":[{"pid":"A"}],
                "aux":[{"pid":"DEV","anchor":0,"horizon":12,
                        "aux_motor":[0]*27}]},
            ("prodromal",24):{"fit":[{"pid":"A"}],"aux":[]}}
        with self.assertRaises(AssertionError):
            merge_fit_auxiliary(datasets)

if __name__=="__main__":
    unittest.main()
