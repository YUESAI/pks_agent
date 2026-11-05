"""Phase90 exact hidden6 conditional inference and mask handling.

python -m unittest discover -s tests -p test_phase90_joint33.py -v
"""
import unittest
import numpy as np
import torch
from scripts.phase62_ppmi_multicohort_audit import HIDX,VIDX
from scripts.phase90_masked_joint33_wm import (
    masked_joint_loglik,hidden_total_posterior,
    auxiliary_partial_targets,Full33Net)


class Phase90Tests(unittest.TestCase):
    def test_masks_exclude_unmeasured_items(self):
        mix=torch.randn(2,4,requires_grad=True)
        item=torch.randn(2,4,33,5,requires_grad=True)
        y=torch.full((2,33),-100,dtype=torch.long)
        y[:,list(VIDX)[:3]]=1
        lp=masked_joint_loglik(mix,item,y)
        self.assertEqual(tuple(lp.shape),(2,))
        (-lp.mean()).backward()
        self.assertTrue(torch.isfinite(item.grad).all())

    def test_exact_full33_uniform_loglik(self):
        mix=torch.zeros(2,4)
        scores=torch.zeros(2,4,33,5)
        full=torch.zeros(2,33,dtype=torch.long)
        loss=masked_joint_loglik(mix,scores,full)
        np.testing.assert_allclose(loss.numpy(),
            np.full(2,-33*np.log(5)),atol=1e-5)

    def test_hidden6_total_normalization_and_conditioning(self):
        mix=torch.zeros(2,4)
        items=torch.randn(2,4,33,5)
        prior=hidden_total_posterior(mix,items)
        visible=torch.randint(0,5,(2,27))
        conditioned=hidden_total_posterior(mix,items,visible)
        self.assertEqual(tuple(prior.shape),(2,25))
        np.testing.assert_allclose(prior.sum(-1),np.ones(2),atol=1e-6)
        np.testing.assert_allclose(conditioned.sum(-1),np.ones(2),atol=1e-6)

    def test_conditioning_on_all_missing_recovers_prior(self):
        mix=torch.randn(2,4)
        sites=torch.randn(2,4,33,5)
        none=torch.full((2,27),-100,dtype=torch.long)
        torch.testing.assert_close(
            hidden_total_posterior(mix,sites),
            hidden_total_posterior(mix,sites,none),atol=1e-6,rtol=1e-6)

    def test_auxiliary_only_real_visible_ratings(self):
        aux=[{"aux_motor":[2]*27},{"aux_motor":[-100]*26+[4]}]
        t=auxiliary_partial_targets(aux)
        self.assertEqual(tuple(t.shape),(2,33))
        self.assertTrue(torch.all(t[:,list(HIDX)]==-100))
        self.assertEqual(int((t!=-100).sum()),28)

    def test_static_dynamic_network_shapes_gradients(self):
        for dynamic in (False,True):
            with self.subTest(dynamic=dynamic):
                net=Full33Net(dynamic=dynamic)
                x=torch.randn(2,6,54)
                gaps=torch.zeros(2,6)
                observed=torch.ones(2,6,dtype=torch.bool)
                horizon=torch.tensor([[1/3,1/3],[2/3,1/2]])
                mix,sites=net(x,gaps,observed,
                              torch.empty(2,0),horizon)
                self.assertEqual(tuple(sites.shape),(2,4,33,5))
                total=hidden_total_posterior(mix,sites)
                (-torch.log(total[:,2].clamp_min(1e-12)).mean()).backward()
                self.assertTrue(torch.isfinite(net.full33_head.weight.grad).all())

if __name__=="__main__":
    unittest.main()
