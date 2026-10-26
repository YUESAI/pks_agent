"""Phase69 time-conditioned, masked clinical World Model.

Trains the Phase68 masked clinical observation/outcome heads on FIT people,
then adds an actual future-Motor transition head supervised by consecutive,
time-ordered Motor visits from the SAME TRAIN-SIDE people only.

A transition is (observed motor at t, delta months) -> scored Motor at t+delta.
It is a retrospective observational transition likelihood, NOT a causal
state transition, treatment intervention, counterfactual outcome or claim
that model-generated scores are real patient measurements.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scripts.phase36_masked_site_feasibility import ALL
from scripts.phase68_multimodal_neural_wm import MultimodalMaskedWM


def build_motor_transitions(people, approved_pids, min_gap=1, max_gap=36):
    """Only actually observed adjacent visits for explicitly FIT-approved IDs.

    Do not mix medication states, do not impute UR/101 missing measurements,
    and do not select pairs by target future burden.
    """
    approved=set(approved_pids)
    transitions=[]
    for person in people:
        if person["pid"] not in approved:
            continue
        visits=sorted(person["visits"],key=lambda v:(v["month"],v["event"]))
        for earlier,later in zip(visits,visits[1:]):
            gap=int(later["month"]-earlier["month"])
            if not min_gap<=gap<=max_gap:
                continue
            if earlier["state"]!=later["state"]:
                continue
            actual_before={
                "BL_"+name:float(earlier["score"][i])
                for i,name in enumerate(ALL) if earlier["status"][i]==1
            }
            actual_after={
                name:int(later["score"][i])
                for i,name in enumerate(ALL) if later["status"][i]==1
            }
            if len(actual_before)<5 or len(actual_after)<5:
                continue
            transitions.append({
                "pid":person["pid"],"observed":actual_before,
                "target":actual_after,"delta_months":gap,
            })
    return transitions


class TransitionHead(nn.Module):
    def __init__(self,width,n_motor):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(width+1,width),nn.GELU(),
            nn.Linear(width,n_motor*5))
        self.n_motor=n_motor

    def forward(self,latent,delta):
        z=torch.cat([latent,delta],dim=-1)
        return self.layers(z).reshape(-1,self.n_motor,5)


class TemporalMultimodalWM(MultimodalMaskedWM):
    def __init__(self,items,q=2,epochs=10,temporal_epochs=6,
                 seed=20261018,width=96,transition_weight=.35):
        super().__init__(items,q=q,epochs=epochs,seed=seed,width=width)
        self.temporal_epochs=temporal_epochs
        self.transition_weight=transition_weight
        self.motor_index={k:i for i,k in enumerate(ALL)}

    def fit(self,labelled,unlabelled=(),transitions=()):
        labelled=list(labelled);unlabelled=list(unlabelled)
        transitions=list(transitions)
        label_pids={r["pid"] for r in labelled}
        extra_pids={r["pid"] for r in unlabelled}
        if len(label_pids)!=len(labelled):
            raise ValueError("Duplicate labelled FIT patient")
        if label_pids & extra_pids:
            raise ValueError("Unlabelled profile overlaps labelled FIT")
        if any(t["pid"] not in label_pids|extra_pids for t in transitions):
            raise AssertionError("Non-FIT person in temporal WM supervision")

        # First learn actual measured multimodal evidence likelihoods and
        # future burden from FIT-only labelled people and other FIT profiles.
        super().fit(labelled,unlabelled)

        self.n_transitions=len(transitions)
        self.transition=TransitionHead(self.width,len(ALL))
        if not transitions or self.temporal_epochs==0:
            self.transition.eval()
            return self

        rng=np.random.default_rng(self.seed+351)
        torch.manual_seed(self.seed+351)
        # Jointly optimize shared encoder, transition likelihood and genuine
        # FIT outcome replay to mitigate task-forgetting during fine-tuning.
        opt=torch.optim.AdamW(
            list(self.net.parameters())+list(self.transition.parameters()),
            lr=.00035,weight_decay=.02)
        self.net.train();self.transition.train()
        for epoch in range(self.temporal_epochs):
            order=rng.permutation(len(transitions))
            for start in range(0,len(order),32):
                batch=[transitions[int(i)] for i in order[start:start+32]]
                x=[];gaps=[];target=[];present=[]
                for row in batch:
                    available=sorted(k for k in row["observed"] if k in self.index)
                    n=int(rng.integers(0,min(self.q,len(available))+1))
                    retained=(rng.choice(available,size=n,replace=False).tolist()
                              if n else [])
                    partial={k:row["observed"][k] for k in retained}
                    x.append(self._encode(partial))
                    gaps.append([min(1.,row["delta_months"]/36.)])
                    truth=[0]*len(ALL);mask=[0.]*len(ALL)
                    for name,score in row["target"].items():
                        j=self.motor_index[name]
                        if not 0<=score<=4:raise AssertionError("Bad actual Motor")
                        truth[j]=score;mask[j]=1.
                    target.append(truth);present.append(mask)

                inputs=torch.from_numpy(np.asarray(x,dtype=np.float32))
                dt=torch.from_numpy(np.asarray(gaps,dtype=np.float32))
                y=torch.tensor(target,dtype=torch.long)
                m=torch.tensor(present,dtype=torch.float32)
                latent=self.net.trunk(inputs)
                future_logits=self.transition(latent,dt)
                rec_loss=nn.functional.cross_entropy(
                    future_logits.reshape(-1,5),y.reshape(-1),
                    reduction="none").reshape(y.shape)
                loss=self.transition_weight*(rec_loss*m).sum()/m.sum().clamp_min(1.)

                # FIT outcome replay, genuine labels only. Not from
                # dev/audit/evaluation patients, and never synthetic labels.
                replay=[labelled[int(i)] for i in
                        rng.integers(0,len(labelled),size=len(batch))]
                bx=[];by=[]
                for record in replay:
                    observed=record["measured"]
                    keys=sorted(k for k in observed if k in self.index)
                    n=int(rng.integers(0,min(self.q,len(keys))+1))
                    sample=(rng.choice(keys,size=n,replace=False).tolist()
                            if n else [])
                    bx.append(self._encode({k:observed[k] for k in sample}))
                    burden=float(record["burden"])
                    by.append(0 if burden==0 else 1 if burden<=1 else
                              2 if burden<=3 else 3)
                lx=torch.from_numpy(np.asarray(bx,dtype=np.float32))
                ly=torch.tensor(by,dtype=torch.long)
                outcome,_=self.net(lx)
                loss=loss+nn.functional.cross_entropy(outcome,ly)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.parameters())+list(self.transition.parameters()),1.)
                opt.step()

        self.net.eval();self.transition.eval()
        return self

    def predicted_future_motor(self,observed,delta_months):
        """Model probabilities only: NEVER feed them to the patient tool."""
        if not self.fitted:raise RuntimeError("Unfitted")
        if delta_months<1 or delta_months>36:
            raise ValueError("Unsupported future interval")
        with torch.inference_mode():
            x=torch.tensor(self._encode(observed))[None,:]
            dt=torch.tensor([[delta_months/36.]],dtype=torch.float32)
            latent=self.net.trunk(x)
            return torch.softmax(self.transition(latent,dt)[0],-1).numpy()
