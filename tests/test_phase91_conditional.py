"""Phase91 proper conditional inference, masking, and gradient tests."""
import unittest
import numpy as np
import torch
from scripts.phase62_ppmi_multicohort_audit import VIDX
from scripts.phase91_proper_conditional_wm import (
    training_masks,deterministic_eval_mask,proper_ranked_score,
    ConditionallyScoredWM,real_visible)
from scripts.phase90_masked_joint33_wm import hidden_total_posterior

class Phase91Tests(unittest.TestCase):
    def test_training_full_and_subset_masks_have_real_values_only(self):
        real=torch.randint(0,5,(20,27))
        random=np.random.default_rng(20261018)
        full=training_masks(real,random,"full")
        torch.testing.assert_close(full,real)
        subsets=training_masks(real,random,"subsets")
        self.assertEqual(tuple(subsets.shape),(20,27))
        for orig,masked in zip(real,subsets):
            observed=(masked!=-100)
            self.assertIn(int(observed.sum()),(7,14,21,27))
            torch.testing.assert_close(masked[observed],orig[observed])
    def test_visible_index_is_real_27(self):
        values=torch.arange(33)[None,:].repeat(3,1)
        torch.testing.assert_close(real_visible(values),
                                   values[:,list(VIDX)])
    def test_eval_mask_reproducible(self):
        rows=[{"pid":"a","horizon":12},
              {"pid":"b","horizon":24}]
        values=torch.randint(0,5,(2,27))
        first=deterministic_eval_mask(rows,values,7)
        second=deterministic_eval_mask(list(rows),values,7)
        torch.testing.assert_close(first,second)
        self.assertTrue(torch.all((first==-100)|(first==values)))
        self.assertTrue(torch.all((first!=-100).sum(-1)==7))
        third=deterministic_eval_mask(rows,values,14)
        self.assertTrue(torch.all((third!=-100).sum(-1)==14))
    def test_bayesian_posterior_reduces_to_prior_if_mask_empty(self):
        mix=torch.randn(4,4)
        logits=torch.randn(4,4,33,5)
        empty=torch.full((4,27),-100,dtype=torch.long)
        prior=hidden_total_posterior(mix,logits)
        conditioned=hidden_total_posterior(mix,logits,empty)
        torch.testing.assert_close(prior,conditioned,rtol=1e-6,atol=1e-6)
    def test_proper_score_gradient(self):
        p=torch.softmax(torch.randn(4,25,requires_grad=True),dim=-1)
        labels=torch.tensor([0,1,8,24],dtype=torch.long)
        loss=proper_ranked_score(p,labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(p.grad_fn)
    def test_mask_selection_never_reads_hidden_six(self):
        full=torch.randint(0,5,(2,33))
        view=real_visible(full)
        rows=[{"pid":"a","horizon":12},{"pid":"b","horizon":24}]
        mask=deterministic_eval_mask(rows,view,14)
        full2=full.clone()
        for j in set(range(33))-set(VIDX):
            full2[:,j]=(full2[:,j]+1)%5
        torch.testing.assert_close(
            mask,deterministic_eval_mask(rows,real_visible(full2),14))
    def test_model_modes(self):
        self.assertFalse(ConditionallyScoredWM(mask_mode="full").dynamic)
        self.assertTrue(ConditionallyScoredWM(dynamic=True,
                          mask_mode="subsets").dynamic)
        with self.assertRaises(ValueError):
            ConditionallyScoredWM(mask_mode="unknown")

if __name__=="__main__":
    unittest.main()
