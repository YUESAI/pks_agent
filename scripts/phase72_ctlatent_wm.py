"""Phase72: CPU continuous-time partially observed clinical world model.

Inspired by irregular-time recurrent state models and verifiable active
feature acquisition (NOT a port of Dreamer 4, L2M or AIDE2).
Inputs: real preceding Motor visits, exact month gaps, actually observed
baseline clinical modalities. Heads: six future onsite 0..4 categorical
scores, future nonzero risk, and self-supervised next-visit Motor items.
No simulated score may be returned as a real clinical observation.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from scripts.phase36_masked_site_feasibility import ALL,VISIBLE,HIDDEN

VINDEX=tuple(ALL.index(x) for x in VISIBLE)
SINDEX=tuple(ALL.index(x) for x in HIDDEN)
MAX_VISITS=6

def sequence_features(rows,keys):
    """Build reproducible variable-length tensors; no patient IDs in tensors."""
    b=len(rows);n=len(keys)
    x=np.zeros((b,MAX_VISITS,54),dtype=np.float32)
    dt=np.zeros((b,MAX_VISITS),dtype=np.float32)
    visitmask=np.zeros((b,MAX_VISITS),dtype=np.bool_)
    clinical=np.zeros((b,2*n),dtype=np.float32)
    horizons=np.zeros((b,2),dtype=np.float32)
    for i,r in enumerate(rows):
        visits=r["history"][-MAX_VISITS:]
        prev=None
        for j,v in enumerate(visits):
            visitmask[i,j]=True
            dt[i,j]=0 if prev is None else max(0.,float(v["month"]-prev))/36.
            prev=v["month"]
            for k,name in enumerate(VISIBLE):
                if name in v["scores"]:
                    value=v["scores"][name]
                    if not 0<=value<=4:raise ValueError("Invalid real Motor rating")
                    x[i,j,k]=float(value)/4.
                    x[i,j,k+27]=1.
        for j,key in enumerate(keys):
            if key in r.get("released",{}):
                clinical[i,j]=float(r["released"][key])/(
                    40. if key.startswith(("MoCA","UPSIT")) else 4.)
                clinical[i,j+n]=1.
        gap=float(r["horizon"])-(float(visits[-1]["month"])-float(r["anchor"]))
        if not 0<gap<=36:raise ValueError("Predictor saw a post-target visit")
        horizons[i]=[float(r["horizon"])/36.,gap/36.]
    return (torch.from_numpy(x),torch.from_numpy(dt),
            torch.from_numpy(visitmask),torch.from_numpy(clinical),
            torch.from_numpy(horizons))

def class_targets(records):
    labels=[]
    for r in records:
        if "site" not in r:raise ValueError("Missing actual target")
        ratings=r["site"]
        if len(ratings)!=6 or any(x is None or not 0<=int(x)<=4 for x in ratings):
            raise ValueError("True future six-site ratings required")
        labels.append([int(x) for x in ratings])
    return torch.tensor(labels,dtype=torch.long)

class CTLatentModel(nn.Module):
    def __init__(self,n_clinical,latent=80):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(54,64),nn.SiLU())
        self.decay_log=nn.Parameter(torch.full((latent,),-1.))
        self.cell=nn.GRUCell(64,latent)
        self.context=nn.Sequential(
            nn.Linear(latent+2*n_clinical+2,112),nn.SiLU(),
            nn.Linear(112,112),nn.SiLU())
        self.future_site=nn.Linear(112,6*5)
        self.future_any=nn.Linear(112,1)
        self.next_motor=nn.Linear(112,27*5)

    def forward(self,x,dt,valid,clinical,horizon):
        batch=x.shape[0]
        h=x.new_zeros((batch,self.cell.hidden_size))
        rate=nn.functional.softplus(self.decay_log)
        for t in range(x.shape[1]):
            next_h=self.cell(self.encoder(x[:,t]),h*torch.exp(-rate*dt[:,t,None]))
            h=torch.where(valid[:,t,None],next_h,h)
        z=self.context(torch.cat([h,clinical,horizon],dim=-1))
        return (self.future_site(z).reshape(-1,6,5),
                self.future_any(z).squeeze(-1),
                self.next_motor(z).reshape(-1,27,5))

def build_auxiliary(people):
    """True adjacent motor visits: no target labels needed.

    Caller MUST supply FIT-only patient objects; this never accesses
    heldout individuals.
    """
    rows=[]
    for p in people:
        visits=p["visits"]
        for j in range(len(visits)-1):
            first,second=visits[j],visits[j+1]
            gap=second["month"]-first["month"]
            if not 1<=gap<=36 or first["state"]!=second["state"]:continue
            target=[int(second["score"][idx]) if second["status"][idx]==1
                    else -100 for idx in VINDEX]
            if sum(x>=0 for x in target)<5:continue
            history=[]
            for v in visits[:j+1]:
                history.append({"month":int(v["month"]),
                    "scores":{name:int(v["score"][idx])
                         for name,idx in zip(VISIBLE,VINDEX)
                         if v["status"][idx]==1}})
            rows.append({"pid":p["pid"],"anchor":first["month"],
                "horizon":gap,"history":history,"released":{},
                "aux_motor":target})
    return rows

class EnsembleWM:
    def __init__(self,keys,members=3,epochs=12,seed=20261018):
        self.keys=tuple(keys)
        self.members=members;self.epochs=epochs;self.seed=seed
        self.networks=[]

    def fit(self,supervised,auxiliary):
        if len(supervised)<25:raise ValueError("Not enough FIT supervision")
        torch.set_num_threads(4)
        xy=class_targets(supervised)
        yy=torch.tensor([float(sum(r["site"])>0) for r in supervised])
        sup_features=sequence_features(supervised,self.keys)
        if auxiliary:
            aux_features=sequence_features(auxiliary,self.keys)
            aux_truth=torch.tensor([r["aux_motor"] for r in auxiliary])
        for member in range(self.members):
            seed=self.seed+member*1009
            torch.manual_seed(seed);gen=np.random.default_rng(seed)
            net=CTLatentModel(len(self.keys))
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for _ in range(self.epochs):
                order=gen.permutation(len(supervised))
                for off in range(0,len(order),48):
                    ids=order[off:off+48]
                    xs=[v[ids].clone() for v in sup_features]
                    # Retrospective missingness simulation for optional tests;
                    # never mask/unmask an actually absent observation.
                    # q=1 at inference; fit uses 0 or 1 released optional test.
                    for local_idx,j in enumerate(ids):
                        available=[key for key in self.keys if key in supervised[int(j)]["optional"]]
                        if len(available) and gen.random()>.3:
                            key=available[int(gen.integers(0,len(available)))]
                            k=self.keys.index(key)
                            val=float(supervised[int(j)]["optional"][key])
                            xs[3][local_idx,k]=val/(
                                40. if key.startswith(("MoCA","UPSIT")) else 4.)
                            xs[3][local_idx,k+len(self.keys)]=1.
                    logits,risk,_=net(*xs)
                    site=nn.functional.cross_entropy(
                        logits.reshape(-1,5),xy[ids].reshape(-1))
                    bce=nn.functional.binary_cross_entropy_with_logits(risk,yy[ids])
                    loss=site+.2*bce
                    if auxiliary:
                        n=min(48,len(auxiliary))
                        jj=gen.integers(0,len(auxiliary),size=n)
                        ax=[v[jj] for v in aux_features]
                        _,_,nextlogits=net(*ax)
                        target=aux_truth[jj]
                        mask=target.ne(-100)
                        if mask.any():
                            rec=nn.functional.cross_entropy(
                                nextlogits.reshape(-1,5),target.reshape(-1),
                                ignore_index=-100)
                            loss=loss+.15*rec
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(),1.)
                    opt.step()
            net.eval()
            self.networks.append(net)
        self.fit_people=len({r["pid"] for r in supervised})
        self.aux_pairs=len(auxiliary)
        return self

    def infer(self,rows,member="ensemble"):
        if not rows:return np.empty(0),np.empty(0),np.empty(0)
        xs=sequence_features(rows,self.keys)
        indices=(range(self.members) if member=="ensemble" else (int(member),))
        estimates=[];risks=[]
        with torch.inference_mode():
            for m in indices:
                if not 0<=m<len(self.networks):raise ValueError("Bad checkpoint")
                sites,binary,_=self.networks[m](*xs)
                probabilities=torch.softmax(sites,dim=-1).numpy()
                estimates.append(np.sum(probabilities*np.arange(5)[None,None,:],
                                        axis=(1,2)))
                risks.append(torch.sigmoid(binary).numpy())
        a=np.asarray(estimates);p=np.asarray(risks)
        return a.mean(axis=0),p.mean(axis=0),a.var(axis=0)

def fixed_ridge_features(rows,keys):
    """Same available core motor and q=1 optional evidence as neural arms."""
    x,_,mask,clinical,horizon=sequence_features(rows,keys)
    n=len(rows);last=np.zeros((n,54),dtype=float)
    for i in range(n):
        j=np.flatnonzero(mask[i].numpy())[-1]
        last[i]=x[i,j].numpy()
    return np.column_stack((last,clinical.numpy(),horizon.numpy()))
