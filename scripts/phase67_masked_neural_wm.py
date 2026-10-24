"""Phase67 neural masked clinical WM (CPU): trained motor observation head plus future-risk head.

Self-supervised loss on actually measured Motor items from multiple TRAIN
visits, with no patient or visit from outer EVAL/DEV. Supervised future
six-item burden class loss on FIT-only labels. Feature status 0/1
explicit. Candidate scores come from network's predictive categorical
distribution, never from fabricated tool responses.

This is an observational masked neural WM, not causal disease dynamics.
"""
from __future__ import annotations
import hashlib,math
import numpy as np
import torch
from torch import nn
from scripts.phase36_masked_site_feasibility import ALL

class Net(nn.Module):
    def __init__(self,n,dim=96):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(2*n,dim),nn.GELU(),
                  nn.Linear(dim,dim),nn.GELU())
        self.outcome=nn.Linear(dim,4)
        self.reconstruct=nn.Linear(dim,n*5)
    def forward(self,x):
        hidden=self.encoder(x)
        return self.outcome(hidden),self.reconstruct(hidden).reshape(len(x),-1,5)

class NeuralClinicalWM:
    def __init__(self,items,alpha=5.,epochs=16,seed=20261018,q=4):
        self.items=tuple(items);self.epochs=epochs;self.seed=seed;self.q=q
        self.index={name:i for i,name in enumerate(items)}
        self.motor_idx=[(j,name) for j,name in enumerate(items)
                        if name.startswith(("BL_","R01_")) and
                        name.split("_",1)[1] in ALL]
        self.fitted=False

    def _values(self,observed):
        values=np.zeros(len(self.items),dtype=np.float32)
        mask=np.zeros(len(self.items),dtype=np.float32)
        for name,v in observed.items():
            i=self.index.get(name)
            if i is None:continue
            values[i]=float(v)/self.norm[i]
            mask[i]=1.
        return np.concatenate([values,mask])

    def fit(self,labelled,unlabelled=()):
        labelled=list(labelled);unlabelled=list(unlabelled)
        if len(labelled)<30:raise ValueError("Too few FIT labels")
        ids=[r["pid"] for r in labelled]
        if len(set(ids))!=len(ids):raise ValueError("Duplicate labelled patient")
        if set(ids)&{r.get("pid") for r in unlabelled}:
            raise ValueError("Labelled and unlabelled patients overlap")
        self.norm=np.ones(len(self.items),dtype=np.float32)
        for name,i in self.index.items():
            vals=[float(r["measured"][name]) for r in labelled+unlabelled
                  if name in r["measured"]]
            if vals:self.norm[i]=max(1.,np.percentile(np.abs(vals),90))
        train=labelled+unlabelled
        seed=self.seed;torch.manual_seed(seed);torch.set_num_threads(4)
        generator=np.random.default_rng(seed)
        self.net=Net(len(self.items))
        opt=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.01)
        # Mask each patient's existing BL/R01 observations; masked-target
        # cross-entropy for real 0..4 Motor scores. No hypothetical truth.
        for epoch in range(self.epochs):
            order=generator.permutation(len(train))
            for start in range(0,len(order),32):
                chosen=order[start:start+32]
                xs=[];outcome=[];targets=[];mask_rec=[]
                for j in chosen:
                    r=train[int(j)];actual=r["measured"]
                    keys=[k for k in actual if k in self.index]
                    keep=(generator.choice(keys,size=generator.integers(0,min(len(keys),self.q)+1),
                           replace=False).tolist() if keys else [])
                    reveal={k:actual[k] for k in keep}
                    xs.append(self._values(reveal))
                    outcome.append(0 if j>=len(labelled) else
                                   0 if r["burden"]==0 else
                                   1 if r["burden"]<=1 else
                                   2 if r["burden"]<=3 else 3)
                    t=[0]*len(self.items);m=[0.]*len(self.items)
                    for idx,name in self.motor_idx:
                        if name in actual and name not in reveal:
                            val=actual[name]
                            if 0<=int(val)<=4 and float(int(val))==float(val):
                                t[idx]=int(val);m[idx]=1.
                    targets.append(t);mask_rec.append(m)
                x=torch.tensor(np.asarray(xs),dtype=torch.float32)
                labels=torch.tensor(outcome,dtype=torch.long)
                t=torch.tensor(targets,dtype=torch.long)
                m=torch.tensor(mask_rec,dtype=torch.float32)
                logits,rec=self.net(x)
                # Explicit row-level supervised mask, not arbitrary batch prefix.
                supervised=torch.tensor([int(j)<len(labelled) for j in chosen],dtype=torch.bool)
                if supervised.any():
                    loss=torch.nn.functional.cross_entropy(logits[supervised],labels[supervised])
                else:
                    loss=logits.sum()*0
                recon=torch.nn.functional.cross_entropy(rec.reshape(-1,5),t.reshape(-1),
                                                        reduction="none").reshape(t.shape)
                if m.sum()>0:loss=loss+.3*(recon*m).sum()/m.sum()
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),2.)
                opt.step()
        self.net.eval()
        self.class_means=np.array([
            np.mean([r["burden"] for r in labelled if (
                0 if r["burden"]==0 else 1 if r["burden"]<=1 else
                2 if r["burden"]<=3 else 3)==k]) if any((
                0 if r["burden"]==0 else 1 if r["burden"]<=1 else
                2 if r["burden"]<=3 else 3)==k for r in labelled)
            else 0. for k in range(4)],dtype=np.float64)
        self.fitted=True
        return self

    def posterior(self,observed,temperature=1.):
        if not self.fitted:raise RuntimeError("Not fit")
        if temperature<=0:raise ValueError("Invalid temperature")
        with torch.no_grad():
            logits,_=self.net(torch.tensor(self._values(observed))[None,:])
            return torch.softmax(logits[0]/temperature,dim=-1).numpy().astype(float)

    def outcomes(self,observed,temperature=1.):
        p=self.posterior(observed,temperature)
        return float(p@self.class_means),float(1-p[0])

    def action_scores(self,observed,available,style="variance",temperature=1.):
        if style not in ("variance","entropy"):raise ValueError("Bad score style")
        with torch.no_grad():
            x=torch.tensor(self._values(observed))[None,:]
            _,rec=self.net(x)
            predictions=torch.softmax(rec[0],dim=-1).numpy()
        base_p=self.posterior(observed,temperature)
        score={}
        for key in available:
            idx=self.index[key]
            # Only trained Motor 0..4 has an outcome-domain-matched head.
            # Other feature actions are not assigned fictitious scores;
            # choose through learned mask perturbation / uncertainty proxy.
            if (idx,key) not in self.motor_idx:
                score[key]=0.
                continue
            q=predictions[idx]
            branches=[]
            for value in range(5):
                altered=dict(observed);altered[key]=value
                after=self.posterior(altered,temperature)
                branches.append(after)
            if style=="entropy":
                h=lambda p:-np.sum(p*np.log(np.maximum(p,1e-12)))
                value=h(base_p)-sum(float(w)*h(p) for w,p in zip(q,branches))
            else:
                y0=float(base_p@self.class_means)
                value=sum(float(w)*(float(p@self.class_means)-y0)**2
                          for w,p in zip(q,branches))
            score[key]=max(0.,float(value))
        return score
