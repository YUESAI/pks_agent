"""Phase33 source taxonomy checks with synthetic values."""
import unittest
from scripts.phase33_visit_audit import total_kind,item_bucket,numeric_present
class Phase33Tests(unittest.TestCase):
 def test_missing_and_valid(self):
  self.assertEqual(total_kind(""),"empty")
  self.assertEqual(total_kind("5"),"valid")
  self.assertEqual(total_kind("-9"),"invalid")
  self.assertFalse(numeric_present(""))
  self.assertFalse(numeric_present("-9"))
  self.assertTrue(numeric_present("0"))
  self.assertEqual(item_bucket(0),"none")
  self.assertEqual(item_bucket(3),"1_to_4")
  self.assertEqual(item_bucket(11),"10plus")
if __name__=="__main__":unittest.main()
