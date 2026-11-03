"""Phase88 unit tests: coherent normalized mixture and gradients."""
import unittest
import numpy as np
import torch
from scripts.phase88_selective_progression_wm import (
    SelectiveProgressionNet,SelectiveProgressionWM,blend_posteriors)
from scripts.phase85_correlated_site_wm import correlated_total_distribution

class Phase88Tests(unittest.TestCase):
    def test_blend_normalized_and_extreme_gates(self):
        p=torch.softmax(torch.randn(4,25),dim=-1)
        q=torch.softmax(torch.randn(4,25),dim=-1)
        torch.testing.assert_close(blend_posteriors(p,q,torch.zeros(4,1)),p)
        torch.testing.assert_close(blend_posteriors(p,q,torch.ones(4,1)),q)
        z=blend_posteriors(p,q,torch.full((4,1),.5))
        torch.testing.assert_close(z.sum(-1),torch.ones(4))

    def test_reject_incorrect_gate(self):
        p=torch.softmax(torch.randn(2,25),dim=-1)
        with self.assertRaises(ValueError):
            blend_posteriors(p,p,torch.ones(2,2))
        with self.assertRaises(ValueError):
            blend_posteriors(p,p,torch.full((2,1),1.3))

    def test_gate_and_flow_have_gradient(self):
        for mode in ("patient","horizon"):
            with self.subTest(mode=mode):
                net=SelectiveProgressionNet(mode)
                x=torch.randn(4,6,54)
                dt=torch.zeros(4,6)
                valid=torch.ones(4,6,dtype=torch.bool)
                horizon=torch.tensor([[1/3,1/3],[2/3,2/3],
                                      [1/3,1/6],[2/3,1/2]])
                sm,ss,dm,ds,gate,next_motor=net(
                    x,dt,valid,torch.empty(4,0),horizon)
                self.assertEqual(tuple(gate.shape),(4,1))
                self.assertEqual(tuple(next_motor.shape),(4,27,5))
                p=blend_posteriors(
                    correlated_total_distribution(sm,ss),
                    correlated_total_distribution(dm,ds),gate)
                self.assertEqual(tuple(p.shape),(4,25))
                np.testing.assert_allclose(p.detach().sum(-1),1.,atol=1e-6)
                (-p[:,2].clamp_min(1e-12).log().mean()).backward()
                self.assertTrue(torch.isfinite(net.gate_layer.weight.grad).all())
                self.assertTrue(torch.isfinite(net.drift[0].weight.grad).all())

    def test_conservative_dynamic_prior(self):
        for mode in ("patient","horizon"):
            p=torch.sigmoid(SelectiveProgressionNet(mode).gate_layer.bias)
            self.assertAlmostEqual(float(p.item()),.2,places=5)

    def test_reject_foreign_auxiliary_patient(self):
        wm=SelectiveProgressionWM("patient",epochs=4)
        fit=[{"pid":"FIT"} for _ in range(32)]
        with self.assertRaises(AssertionError):
            wm.fit(fit,[{"pid":"OTHER"}])

if __name__=="__main__":
    unittest.main()
