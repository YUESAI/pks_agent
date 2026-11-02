#!/usr/bin/env python3
"""Phase84 single-split development: item-consistent probabilistic clinical WM.

No fivefold execution. Deterministic 80% FIT / 20% DEVELOPMENT split,
globally by hashed patient ID, shared across 12m/24m/PD tasks.
Previous consumed final TEST is NEVER accessed. The 20% is a repeatedly
usable DEV set, NOT confirmatory heldout evidence.

Three independent model arms: 0 phase78 categorical baseline,
1 six-site joint-consistent WM, 2 same WM + auxiliary true
FIT-person next Motor visit supervision. 25-class total score
distribution is differentiably derived by convolving six 5-way
site probability vectors; there is no independent inconsistent total
head. Evaluation: median MAE, mean RMSE, RPS, Brier, NLL, coverage80.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase73_matched_longitudinal_agent import data_rows
from scripts.phase72_ctlatent_wm import sequence_features,build_auxiliary
from scripts.phase78_distributional_wm import DistributionWM,DistributionNet
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase76_reliability_gated_wm import labels,write_csv

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))

def dev_split(rows):
    """Patient-global hash (not task- or horizon-dependent)."""
    fit=[];dev=[]
    for r in rows:
        digest=hashlib.sha256(
            ("phase84-global-v1|"+str(r["pid"])).encode()).hexdigest()
        (fit if int(digest[:12],16)%10<8 else dev).append(r)
    fit.sort(key=lambda r:str(r["pid"]))
    dev.sort(key=lambda r:str(r["pid"]))
    if set(r["pid"] for r in fit)&set(r["pid"] for r in dev):
        raise AssertionError("Patient leakage")
    return fit,dev

def convolve_sites(site_logits):
    """Independent-site approximation with exact total-score consistency.

    Six 0..4 site distributions imply a categorical burden 0..24.
    Independence is a modeling assumption, NOT a proven patient biology.
    """
    p=torch.softmax(site_logits,dim=-1)
    total=p[:,0,:]
    for k in range(1,6):
        previous=total
        candidates=[]
        for score in range(previous.shape[1]+4):
            contribution=[]
            for rating in range(5):
                prior=score-rating
                if 0<=prior<previous.shape[1]:
                    contribution.append(previous[:,prior]*p[:,k,rating])
            candidates.append(torch.stack(contribution,dim=1).sum(dim=1))
        total=torch.stack(candidates,dim=1)
    return total.clamp_min(1e-9)/total.sum(dim=1,keepdim=True)

class SiteDistributionNet(DistributionNet):
    def __init__(self):
        super().__init__()
        self.future_sites=nn.Linear(96,30)
    def forward(self,x,dt,valid,clinical,horizon):
        h=x.new_zeros((len(x),self.cell.hidden_size))
        rate=nn.functional.softplus(self.rate)
        for t in range(x.shape[1]):
            candidate=self.cell(self.observation(x[:,t]),
                 h*torch.exp(-rate*dt[:,t,None]))
            h=torch.where(valid[:,t,None],candidate,h)
        z=self.fuse(torch.cat((h,horizon),dim=-1))
        return self.future_sites(z).reshape(-1,6,5),self.next_motor(z).reshape(-1,27,5)

class SiteConsistentWM:
    def __init__(self,epochs=18,aux_weight=0.,seed=SEED):
        self.epochs=epochs;self.aux_weight=aux_weight;self.seed=seed
    def fit(self,fit,aux):
        torch.manual_seed(self.seed)
        gen=np.random.default_rng(self.seed)
        self.net=SiteDistributionNet()
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.0008,weight_decay=.025)
        X=sequence_features(fit,())
        sites,y,_=labels(fit)
        sites=torch.tensor(sites,dtype=torch.long)
        y=torch.tensor(y,dtype=torch.long)
        if self.aux_weight and aux:
            AX=sequence_features(aux,())
            A_target=torch.tensor([a["aux_motor"] for a in aux],dtype=torch.long)
        else:AX=None
        for epoch in range(self.epochs):
            self.net.train()
            perm=gen.permutation(len(fit))
            for start in range(0,len(perm),32):
                ii=perm[start:start+32]
                logits,aux_logits=self.net(*[x[ii] for x in X])
                prob=convolve_sites(logits)
                site_loss=nn.functional.cross_entropy(
                    logits.reshape(-1,5),sites[ii].reshape(-1))
                total_loss=nn.functional.nll_loss(torch.log(prob),y[ii])
                loss=site_loss+total_loss
                if AX is not None:
                    jj=gen.integers(0,len(aux),size=min(32,len(ii)))
                    _,nextlogits=self.net(*[x[jj] for x in AX])
                    loss=loss+self.aux_weight*nn.functional.cross_entropy(
                        nextlogits.reshape(-1,5),A_target[jj].reshape(-1),
                        ignore_index=-100)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()
        self.net.eval()
        return self
    def predict_distribution(self,rows):
        with torch.inference_mode():
            logits,_=self.net(*sequence_features(rows,()))
            return convolve_sites(logits).numpy()

def run(args):
    if args.arm not in (0,1,2) or not 4<=args.epochs<=100:
        raise ValueError("Allowed model arms 0,1,2; epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out=args.output.resolve();root=args.data_root.resolve()
    if out==root or root in out.parents:raise ValueError("Unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    people,_=load_cohorts(root,args.manifest_db)
    evidence,_,_=evidence_tables(root,people)
    start=time.monotonic()
    result=[];coverage=[]
    for task in TASKS:
        subset=[p for p in people if p["cohort"]==task[0]]
        truth,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,truth,evidence,task)
        fit,dev=dev_split(rows)
        if len(fit)<30 or len(dev)<10:raise ValueError("Split too small")
        train_ids={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in train_ids])
        seed=SEED+task[1]+(0 if task[0]=="prodromal" else 300)
        if args.arm==0:
            model=DistributionWM("categorical",epochs=args.epochs,seed=seed)
        else:
            model=SiteConsistentWM(epochs=args.epochs,seed=seed,
                                   aux_weight=.12 if args.arm==2 else 0.)
        model.fit(fit,aux)
        score=probabilities_scores(model.predict_distribution(dev),dev)
        result.append({"task":task[0],"horizon":task[1],"arm":args.arm,
                       "fit":len(fit),"dev":len(dev),
                       "fit_aux_pairs":len(aux),**score})
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(fit),"dev":len(dev),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",result)
        write_csv(out/"coverage.csv",coverage)
        print("PHASE84_TASK_COMPLETE",task,"arm",args.arm,flush=True)
    (out/"summary.json").write_text(json.dumps({
       "phase":84,"arm":args.arm,"completed_tasks":len(coverage),
       "seconds":round(time.monotonic()-start,2),
       "split":"global patient hash, 80% FIT / 20% DEVELOPMENT",
       "warning":"Development 20% is NOT independent TEST; no confirmation claims",
       "model":"categorical" if args.arm==0 else "six-item convolution WM",
       "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    print("PHASE84_SUCCESS",args.arm,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",type=int,required=True)
    p.add_argument("--epochs",type=int,default=18)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
