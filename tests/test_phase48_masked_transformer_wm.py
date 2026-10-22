"""Phase48 synthetic WM training, masked score likelihood and privacy tests."""
import random
import unittest
import torch
from scripts.phase48_masked_transformer_wm import (
    MaskedItemTransformer, NeuralObservationWM, masked_batch,
    multitask_loss, train_one, dataset, split_inner
)

def mock_patients(n=90):
    rng=random.Random(41)
    items=tuple("NP3SIM"+str(i) for i in range(12))
    result=[]
    for i in range(n):
        features={k:rng.choice((0,0,0,1,2,3,4)) for k in items}
        y=int(features[items[0]]+features[items[1]]>2)
        result.append({"pid":"PRIVATE_SIM_"+str(i),"features":features,
                       "label":y,"hidden":float(y)})
    return result,items

class Phase48Tests(unittest.TestCase):
    def test_masked_zero_is_not_missing(self):
        x=torch.tensor([[0,1,2,0,4],[4,0,2,0,0]])
        mask=torch.tensor([[True,False,True,False,True],
                           [False,True,False,True,False]])
        m=MaskedItemTransformer(5,width=16,depth=1)
        logits,cond=m(x,mask)
        self.assertEqual(tuple(logits.shape),(2,))
        self.assertEqual(tuple(cond.shape),(2,5,2,5))
        self.assertEqual(int(m.value_embeddings.num_embeddings),6)
        y=torch.tensor([1,0])
        loss=multitask_loss((logits,cond),x,y,mask)
        self.assertTrue(torch.isfinite(loss))
    def test_training_and_probabilistic_observation(self):
        torch.set_num_threads(2)
        rows,items=mock_patients()
        fit,dev=split_inner(rows,123)
        self.assertFalse(set(x["pid"] for x in fit)&set(x["pid"] for x in dev))
        model,meta=train_one(rows,items,123,torch.device("cpu"),
                             epochs=2,width=16,depth=1)
        self.assertGreaterEqual(meta["best_epoch"],1)
        wm=NeuralObservationWM(items,[model],torch.device("cpu"))
        known={items[0]:0,items[1]:3}
        prediction=wm.predict_item(known,items[2])
        self.assertEqual(len(prediction),5)
        self.assertAlmostEqual(sum(prediction),1.,places=4)
        self.assertGreaterEqual(wm.information_gain(known,items[2]),0.)
        self.assertTrue(0<wm.posterior(known)<1)
        with self.assertRaises(ValueError):
            wm.predict_item(known,items[0])
        with self.assertRaises(ValueError):
            wm.posterior({items[0]:101})
if __name__=="__main__":
    unittest.main()
