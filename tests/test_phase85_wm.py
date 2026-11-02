"""Phase85 mixture normalization, gradients, joint-site score and split tests."""
import unittest
import numpy as np
import torch
from scripts.phase85_correlated_site_wm import (
    correlated_total_distribution,joint_site_log_probability,
    dev_split)

class Phase85Test(unittest.TestCase):
    def test_support_and_gradient(self):
        mix=torch.randn(3,4,requires_grad=True)
        site=torch.randn(3,4,6,5,requires_grad=True)
        probability=correlated_total_distribution(mix,site)
        self.assertEqual(probability.shape,(3,25))
        np.testing.assert_allclose(probability.detach().sum(-1),
                                   np.ones(3),atol=1e-6)
        (-torch.log(probability[:,2])).mean().backward()
        self.assertTrue(torch.isfinite(site.grad).all())
        self.assertTrue(torch.isfinite(mix.grad).all())

    def test_mixture_reduces_to_identical_components(self):
        sites=torch.randn(2,1,6,5).repeat(1,4,1,1)
        a=correlated_total_distribution(torch.randn(2,4),sites)
        b=correlated_total_distribution(torch.zeros(2,4),sites)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_deterministic_joint_log_score(self):
        mix=torch.zeros(2,4)
        site=torch.zeros(2,4,6,5)
        obs=torch.zeros(2,6,dtype=torch.long)
        logp=joint_site_log_probability(mix,site,obs)
        self.assertTrue(torch.allclose(logp,torch.full((2,),-6*np.log(5)),
                                        atol=1e-6))

    def test_global_patient_split_stable(self):
        rows=[{"pid":str(i)} for i in range(100)]
        a,b=dev_split(rows)
        c,d=dev_split(rows[::-1])
        self.assertEqual([x["pid"] for x in a],[x["pid"] for x in c])
        self.assertEqual([x["pid"] for x in b],[x["pid"] for x in d])

if __name__=="__main__":unittest.main()
