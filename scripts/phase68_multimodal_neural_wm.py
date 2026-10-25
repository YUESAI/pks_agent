"""Phase68: fully mask-aware multimodal neural observational WM.

Uses real BL and pre-target R01 feature values. Learns:
 - probability of future six-item motor burden class (labelled FIT people)
 - five-bin conditional distribution of every actually recorded clinical
   feature, including MoCA, UPSIT, MDS-UPDRS I/II, RBD and SCOPA-AUT
   (labelled and unlabelled FIT people).
No target/evaluation-person observation enters training.
This models conditional observations, not causal clinical trajectories.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn

MOTOR_PREFIX=("BL_NP3","R01_NP3")
CLASS_COUNT=4

def outcome_class(v):
    v=float(v)
    return 0 if v==0 else 1 if v<=1 else 2 if v<=3 else 3

class MultimodalNet(nn.Module):
    def __init__(self,n,width=96):
        super().__init__()
        self.trunk=nn.Sequential(
            nn.Linear(n*2,width),nn.LayerNorm(width),nn.GELU(),
            nn.Linear(width,width),nn.GELU())
        self.outcome=nn.Linear(width,CLASS_COUNT)
        self.observation=nn.Linear(width,n*5)
    def forward(self,x):
        h=self.trunk(x)
        return self.outcome(h),self.observation(h).reshape(len(x),-1,5)

class MultimodalMaskedWM:
    def __init__(self,items,q=4,epochs=14,seed=20261018,width=96):
        self.items=tuple(items)
        if not self.items or len(self.items)!=len(set(self.items)):
            raise ValueError("Nonempty distinct item names required")
        self.index={item:i for i,item in enumerate(self.items)}
        self.q=q;self.epochs=epochs;self.seed=seed;self.width=width

    def _bin(self,key,v):
        return int(np.searchsorted(self.edges[key],float(v),side="right"))

    def _encode(self,observed):
        n=len(self.items)
        values=np.zeros(n,dtype=np.float32)
        masks=np.zeros(n,dtype=np.float32)
        for key,raw in observed.items():
            i=self.index.get(key)
            if i is not None:
                values[i]=float(raw)/self.scales[i]
                masks[i]=1.
        return np.r_[values,masks]

    def fit(self,labelled,unlabelled=()):
        labelled=list(labelled);unlabelled=list(unlabelled)
        if len(labelled)<30:raise ValueError("At least 30 labelled FIT persons")
        label_ids=[r["pid"] for r in labelled]
        if len(set(label_ids))!=len(label_ids):raise ValueError("Duplicate FIT")
        if set(label_ids)&{r.get("pid") for r in unlabelled}:
            raise ValueError("FIT overlap")
        all_rows=labelled+unlabelled
        self.scales=np.ones(len(self.items),dtype=np.float32)
        self.edges={};self.reps={};self.training_support={}
        for key,i in self.index.items():
            observed=np.array([float(r["measured"][key]) for r in all_rows
                               if key in r["measured"]],dtype=float)
            self.training_support[key]=len(observed)
            if observed.size==0:
                self.edges[key]=np.array([],dtype=float)
                self.reps[key]=np.array([0.],dtype=float)
                continue
            if not np.isfinite(observed).all():raise ValueError("Bad FIT evidence")
            self.scales[i]=max(1.,float(np.quantile(np.abs(observed),.90)))
            # Motor items are real 0..4 scores. Other features use up to
            # five quantile bins and original-scale training representatives.
            if key.startswith(MOTOR_PREFIX):
                edges=np.array([.5,1.5,2.5,3.5],dtype=float)
            else:
                candidate=np.unique(np.quantile(observed,[.2,.4,.6,.8]))
                edges=candidate[(candidate>observed.min())&
                                (candidate<observed.max())]
            self.edges[key]=edges
            indices=np.searchsorted(edges,observed,side="right")
            reps=np.array([np.median(observed[indices==b])
                           if np.any(indices==b) else float(np.median(observed))
                           for b in range(len(edges)+1)],dtype=float)
            self.reps[key]=reps
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        generator=np.random.default_rng(self.seed)
        self.net=MultimodalNet(len(self.items),self.width)
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.0008,weight_decay=.015)
        n=len(self.items)
        for epoch in range(self.epochs):
            order=generator.permutation(len(all_rows))
            for start in range(0,len(all_rows),32):
                indices=order[start:start+32]
                X=[];targets=[];mask=[];classes=[];supervision=[]
                for ridx in indices:
                    record=all_rows[int(ridx)];measured=record["measured"]
                    valid=sorted(k for k in measured if k in self.index)
                    keep_count=int(generator.integers(0,min(len(valid),self.q)+1))
                    retained=(generator.choice(valid,size=keep_count,
                               replace=False).tolist() if keep_count else [])
                    revealed={k:measured[k] for k in retained}
                    X.append(self._encode(revealed))
                    labels=[0]*n;observed_mask=[0.]*n
                    for key in valid:
                        if key not in revealed:
                            j=self.index[key]
                            labels[j]=self._bin(key,measured[key])
                            observed_mask[j]=1.
                    targets.append(labels);mask.append(observed_mask)
                    islabel=int(ridx)<len(labelled)
                    supervision.append(islabel)
                    classes.append(outcome_class(record["burden"]) if islabel else 0)
                if not X:continue
                inputs=torch.from_numpy(np.asarray(X,dtype=np.float32))
                classes_t=torch.tensor(classes,dtype=torch.long)
                supervised=torch.tensor(supervision,dtype=torch.bool)
                truth=torch.tensor(targets,dtype=torch.long)
                use=torch.tensor(mask,dtype=torch.float32)
                outcome_logits,obs_logits=self.net(inputs)
                loss=outcome_logits.sum()*0.
                if supervised.any():
                    loss=loss+nn.functional.cross_entropy(
                        outcome_logits[supervised],classes_t[supervised])
                if use.sum().item()>0:
                    pred_loss=nn.functional.cross_entropy(
                        obs_logits.reshape(-1,5),truth.reshape(-1),
                        reduction="none").reshape(truth.shape)
                    loss=loss+.5*(pred_loss*use).sum()/use.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()
        self.net.eval()
        class_burdens=defaultdict(list)
        for record in labelled:
            class_burdens[outcome_class(record["burden"])].append(
                float(record["burden"]))
        global_mean=float(np.mean([r["burden"] for r in labelled]))
        self.class_means=np.array([
            float(np.mean(class_burdens[k])) if class_burdens[k] else global_mean
            for k in range(CLASS_COUNT)])
        self.fitted=True
        return self

    def _outputs(self,observed,temperature=1.):
        if not getattr(self,"fitted",False):raise RuntimeError("Unfitted WM")
        if temperature<=0:raise ValueError("Bad temperature")
        with torch.inference_mode():
            inp=torch.tensor(self._encode(observed))[None,:]
            outcome,item=self.net(inp)
            p=torch.softmax(outcome[0]/temperature,-1).numpy()
            categorical=torch.softmax(item[0],-1).numpy()
        return p,categorical

    def outcomes(self,observed,temperature=1.):
        p,_=self._outputs(observed,temperature)
        return float(p@self.class_means),float(1.-p[0])

    def action_scores(self,observed,remaining,style="variance",temperature=1.):
        if style not in ("variance","entropy"):
            raise ValueError("Unknown acquisition objective")
        prior,distributions=self._outputs(observed,temperature)
        h=lambda p:-np.sum(p*np.log(np.clip(p,1e-12,1)))
        estimates={}
        for key in remaining:
            if key in observed:raise ValueError("Repeated observed key")
            idx=self.index[key]
            options=self.reps[key]
            weights=distributions[idx,:len(options)]
            weights=weights/weights.sum()
            branches=[]
            for value in options:
                simulated=dict(observed);simulated[key]=float(value)
                next_p,_=self._outputs(simulated,temperature)
                branches.append(next_p)
            if style=="entropy":
                before=h(prior)
                after=sum(float(w)*h(p) for w,p in zip(weights,branches))
                gain=float(before-after)
            else:
                center=sum(float(w)*float(p@self.class_means)
                           for w,p in zip(weights,branches))
                gain=sum(float(w)*(float(p@self.class_means)-center)**2
                         for w,p in zip(weights,branches))
            estimates[key]=max(0.,float(gain))
        return estimates
