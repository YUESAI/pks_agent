"""Phase84 convolution and global split regression tests.

python -m unittest discover -s tests -p test_phase84_wm.py -v
"""
import unittest
import numpy as np
import torch
from scripts.phase84_single_split_site_wm import convolve_sites,dev_split

class Phase84Test(unittest.TestCase):
    def test_zero_all_sites(self):
        logits=torch.full((2,6,5),-100.)
        logits[:,:,0]=100.
        p=convolve_sites(logits).detach().numpy()
        self.assertEqual(p.shape,(2,25))
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-6)
        self.assertGreater(p[0,0],.999)
    def test_all_sites_four(self):
        logits=torch.full((2,6,5),-100.)
        logits[:,:,4]=100.
        p=convolve_sites(logits).detach().numpy()
        self.assertGreater(p[0,24],.999)
    def test_gradient(self):
        logits=torch.randn(4,6,5,requires_grad=True)
        p=convolve_sites(logits)
        loss=-(p[:,3].log()).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
    def test_split_independent_of_input_order(self):
        rows=[{"pid":str(i)} for i in range(200)]
        train1,dev1=dev_split(rows)
        train2,dev2=dev_split(list(reversed(rows)))
        self.assertEqual([r["pid"] for r in train1],
                         [r["pid"] for r in train2])
        self.assertEqual([r["pid"] for r in dev1],
                         [r["pid"] for r in dev2])
        self.assertEqual(len(train1)+len(dev1),200)

if __name__=="__main__":unittest.main()
