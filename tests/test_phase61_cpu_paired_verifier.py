"""Deterministic Phase61 interface regressions, not a required smoke job."""
import unittest
from scripts.phase61_cpu_paired_verifier import (
 audit_split,episode,shuffled_priors,search_and_audit)
from scripts.phase60_cpu_multistep_wm import split_train
from scripts.phase45_world_model import ObservationWorldModel

ITEMS=tuple(f"item_{i}" for i in range(8))
def people(n=240):
 return [{"pid":f"S_{i:04d}","label":int(i%3==0),
          "features":{k:int((i+j)%9==0) for j,k in enumerate(ITEMS)}}
         for i in range(n)]

class Phase61Tests(unittest.TestCase):
 def test_patient_audit_disjoint(self):
  fit,dev,outer=split_train(people())
  tune,audit=audit_split(dev)
  self.assertFalse({r["pid"] for r in tune}&{r["pid"] for r in audit})
 def test_measured_patient_interface(self):
  fit,dev,outer=split_train(people())
  wm=ObservationWorldModel(ITEMS).fit(fit)
  x=episode(outer[0],wm,None,(0.,1.,"information_gain"),3)
  self.assertTrue(0<=x<=1)
 def test_prior_placebo_preserves_values(self):
  _,_,outer=split_train(people())
  priors={r["pid"]:0.1+i/1000 for i,r in enumerate(outer)}
  shuffled=shuffled_priors(outer,priors)
  self.assertEqual(sorted(shuffled.values()),sorted(priors.values()))
