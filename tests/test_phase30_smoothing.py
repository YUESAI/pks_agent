"""Phase30 synthetic checks."""
import unittest
from scripts.phase30_smoothing import choose_weight
class Phase30Tests(unittest.TestCase):
 def test_fixed_fit_blend(self):
  rows=[{"bl":2.,"last":6.,"target":4.} for _ in range(20)]
  self.assertEqual(choose_weight(rows),.5)
 def test_last_value_is_optimal(self):
  rows=[{"bl":2.,"last":6.,"target":6.} for _ in range(20)]
  self.assertEqual(choose_weight(rows),0.)
if __name__=="__main__":unittest.main()
