"""Synthetic Phase50 CPU regression tests for clinical latent dynamics."""
import random,tempfile,unittest
from pathlib import Path
import torch
from scripts.phase50_clinical_latent_wm import (
 ClinicalLatentDynamics,batch,loss_fn,folds_for,dev_split,evaluate)

def fixture(n=108):
 rng=random.Random(213);rows=[]
 for i in range(n):
  bl=rng.randint(0,13)
  recent=max(0.,min(132.,bl+rng.randint(-2,4)))
  target=max(0.,min(132.,.5*bl+.5*recent+rng.randint(-2,3)))
  rows.append(dict(pid=f"PRIVATE_SYNTH_{i}",bl=float(bl),last=float(recent),
                   target=float(target),gap12=12,future_gap=12))
 return rows

class Phase50Tests(unittest.TestCase):
 def test_forward_gradient_and_latent_rollout(self):
  torch.set_num_threads(2)
  data=batch(fixture()[:10],torch.device("cpu"))
  model=ClinicalLatentDynamics(width=16)
  z12,z24,openloop=model(*[data[0],data[1],data[3]])
  self.assertEqual(tuple(z24.shape),(10,16))
  self.assertEqual(tuple(openloop.shape),(10,16))
  loss=loss_fn(model,*data)
  self.assertTrue(torch.isfinite(loss))
  loss.backward()
  self.assertTrue(any(p.grad is not None for p in model.parameters()))
 def test_future_target_not_a_prediction_input(self):
  rows=fixture()
  model=ClinicalLatentDynamics(width=8).eval()
  x=batch(rows[:3],torch.device("cpu"))
  with torch.no_grad():
   orig=model.forecast(x[0],x[1],x[3])[0]
  rows[0]["target"]=120.
  rows[0]["future_gap"]=18
  y=batch(rows[:3],torch.device("cpu"))
  with torch.no_grad():
   changed=model.forecast(y[0],y[1],y[3])[0]
  self.assertTrue(torch.equal(orig,changed))
 def test_patient_disjoint_and_no_private_report_rows(self):
  rows=fixture()
  mapping=folds_for(rows,3,44)
  self.assertEqual(set(mapping.values()),{0,1,2})
  train,dev=dev_split(rows[:70],44)
  self.assertFalse({r["pid"] for r in train}&{r["pid"] for r in dev})
  with tempfile.TemporaryDirectory() as tmp:
   result=evaluate(rows,Path(tmp)/"out",3,1,2,44,8,"cpu",100)
   self.assertEqual(result["train_trajectories"],108)
   self.assertTrue((Path(tmp)/"out"/"oof_metrics.csv").is_file())
   for f in (Path(tmp)/"out").iterdir():
    self.assertNotIn("PRIVATE_SYNTH_",f.read_text())

if __name__=="__main__":unittest.main()
