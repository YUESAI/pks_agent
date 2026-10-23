"""Phase63 public synthetic regressions, no job-level smoke test."""
import unittest
import numpy as np
from scripts.phase63_twenty_arm_cpu import (
 arm_from_id,train_fold_preprocess,fixed_folds,build_inputs
)

class Phase63Tests(unittest.TestCase):
 def test_twenty_distinct_configs(self):
  self.assertEqual(len({arm_from_id(j) for j in range(20)}),20)
 def test_fixed_folds_independent_of_order(self):
  rows=[{"pid":f"subject_{i}"} for i in range(50)]
  a=fixed_folds(rows,("prodromal",12))
  b=fixed_folds(list(reversed(rows)),("prodromal",12))
  self.assertEqual(a.tolist(),list(reversed(b.tolist())))
 def test_missing_imputed_on_training_only(self):
  a=np.array([[0.,1.],[10.,1.],[np.nan,1.]])
  b=np.array([[999.,1.],[np.nan,1.]])
  fit,held=train_fold_preprocess(a,b,"visible27_scores")
  self.assertTrue(np.isfinite(fit).all())
  self.assertTrue(np.isfinite(held).all())
