"""Synthetic smoke checks for Phase56."""
import unittest
from scripts.phase56_remote_to_site_wm import masked_input
class Phase56Tests(unittest.TestCase):
    def test_dummy_future_contains_no_measurements(self):
        x={"month":10,"score":(1,)*33,"status":(1,)*33}
        r={"month":16,"score":(0,)*33,"status":(2,)*33}
        future={"month":22,"score":(4,)*33,"status":(1,)*33}
        dummy=masked_input((x,r,future))["visits"][-1]
        self.assertEqual(sum(dummy["status"]),0)
        self.assertEqual(sum(dummy["score"]),0)
if __name__=="__main__":unittest.main()
