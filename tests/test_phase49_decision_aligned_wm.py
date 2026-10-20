"""Synthetic Phase49 ranking and nested TRAIN split checks."""
import random
import unittest
import torch
from scripts.phase49_decision_aligned_wm import (
    ranking_loss,inner_partition,train_model,inner_score
)
from scripts.phase48_masked_transformer_wm import NeuralObservationWM

def synthetic(n=180):
    rng=random.Random(55)
    items=tuple("SCORE"+str(j) for j in range(9))
    rows=[]
    for i in range(n):
        values={k:rng.choice((0,0,0,1,2,3,4)) for k in items}
        y=int(values[items[0]]+values[items[1]]>=3)
        rows.append({"pid":"SYNTH_PATIENT_"+str(i),"features":values,
                     "label":y,"hidden":float(y)})
    return rows,items

class Phase49Tests(unittest.TestCase):
    def test_ranking_gradient(self):
        logits=torch.tensor([.1,.4,.2,-.1],requires_grad=True)
        label=torch.tensor([1.,1.,0.,0.])
        loss=ranking_loss(logits,label)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        better=ranking_loss(logits.detach()+torch.tensor([2.,2.,0.,0.]),label)
        self.assertLess(float(better),float(loss))
    def test_nested_fit_dev_and_train(self):
        torch.set_num_threads(2)
        rows,items=synthetic()
        fit,dev=inner_partition(rows,444)
        self.assertFalse(set(r["pid"] for r in fit)&set(r["pid"] for r in dev))
        trained,epoch=train_model(fit,dev,items,444,
                                  torch.device("cpu"),.3,
                                  epochs=2,width=16,depth=1,patience=2)
        self.assertGreaterEqual(epoch,1)
        wm=NeuralObservationWM(items,[trained],torch.device("cpu"))
        val=wm.posterior({items[0]:1})
        self.assertGreater(val,0)
        self.assertLess(val,1)

if __name__=="__main__":unittest.main()
