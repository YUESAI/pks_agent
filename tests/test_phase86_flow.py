"""Phase86 continuous-time latent flow regression."""
import unittest
import torch
import numpy as np
from scripts.phase86_continuous_time_progression_wm import ContinuousTimeMixtureNet
from scripts.phase85_correlated_site_wm import correlated_total_distribution

class TestPhase86(unittest.TestCase):
    def test_zero_duration_is_identity(self):
        net=ContinuousTimeMixtureNet()
        h=torch.randn(3,80)
        same=net.evolve(h,torch.zeros(3,1))
        self.assertTrue(torch.allclose(h,same,atol=1e-7))
    def test_time_split_numerical_consistency_and_gradients(self):
        net=ContinuousTimeMixtureNet()
        x=torch.randn(3,6,54)
        dt=torch.zeros(3,6)
        valid=torch.ones(3,6,dtype=torch.bool)
        horizon=torch.tensor([[1/3,1/3],[2/3,2/3],[1/3,1/6]])
        clinical=torch.empty(3,0)
        mix,sites,penalty=net(x,dt,valid,clinical,horizon,True)
        p=correlated_total_distribution(mix,sites)
        self.assertEqual(p.shape,(3,25))
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(3),atol=1e-6))
        loss=-torch.log(p[:,2]).mean()+.25*penalty
        loss.backward()
        self.assertTrue(torch.isfinite(net.drift[0].weight.grad).all())
        self.assertTrue(np.isfinite(float(penalty.detach())))
    def test_cannot_predict_past_target(self):
        net=ContinuousTimeMixtureNet()
        with self.assertRaises(AssertionError):
            net(torch.randn(1,6,54),torch.zeros(1,6),
                torch.ones(1,6,dtype=torch.bool),torch.empty(1,0),
                torch.tensor([[.333,0.]]))
if __name__=="__main__":unittest.main()
