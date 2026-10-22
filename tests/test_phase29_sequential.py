"""Phase29 synthetic regression checks."""
import unittest
from scripts.phase29_sequential import predict_simple,pair_diff

class Phase29Tests(unittest.TestCase):
 def test_future_target_and_date_never_enter_predictor(self):
  x=dict(bl=4.,last=8.,gap12=12,target=9.,future_gap=10)
  y=dict(x,target=120.,future_gap=20)
  self.assertEqual(predict_simple("linear_extrapolation",x),
                   predict_simple("linear_extrapolation",y))
  self.assertEqual(predict_simple("last_value",x),8.)
 def test_pairwise_bootstrap(self):
  a={str(i):(float(i),float(i)) for i in range(20)}
  b={str(i):(float(i),float(i)+1) for i in range(20)}
  values=pair_diff(a,b,100,42)
  self.assertEqual(len(values),2)
  self.assertEqual(values[0]["difference"],1.)
if __name__=="__main__":unittest.main()
