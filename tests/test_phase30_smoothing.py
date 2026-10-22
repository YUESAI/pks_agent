"""Phase30 synthetic checks."""
import unittest
from scripts.phase30_smoothing import choose_weight
class Phase30Tests(unittest.TestCase):
 def test_phase29_dependency_contract(self):
  from scripts import phase29_sequential as p29
  from scripts import phase30_smoothing as p30
  import inspect
  self.assertIs(p30.matrix,p29.matrix)
  self.assertEqual(list(inspect.signature(p30.tune).parameters),
                   ["rows","model","seed","outer"])
 def test_fixed_fit_blend(self):
  rows=[{"bl":2.,"last":6.,"target":4.} for _ in range(20)]
  self.assertEqual(choose_weight(rows),.5)
 def test_last_value_is_optimal(self):
  rows=[{"bl":2.,"last":6.,"target":6.} for _ in range(20)]
  self.assertEqual(choose_weight(rows),0.)
if __name__=="__main__":unittest.main()
