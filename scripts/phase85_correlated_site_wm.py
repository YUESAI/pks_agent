#!/usr/bin/env python3
"""Phase85: correlated future-Motor site mixture + shared-horizon development.

Four model jobs on ONE fixed patient-global 80/20 development split:
0: matched 25-class categorical (Phase84 arm0)
1: independent six-site convolution (Phase84 arm1)
2: 4-component correlated mixture of six-site distributions
3: same mixture, shared Prodromal 12m+24m FIT supervision

Not an independent holdout, prospective treatment model, or clinical RSI.
No patient-level data are saved. Already-consumed final TEST is excluded.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase73_matched_longitudinal_agent import data_rows
from scripts.phase72_ctlatent_wm import sequence_features,build_auxiliary
from scripts.phase76_reliability_gated_wm import labels,write_csv
from scripts.phase78_distributional_wm import DistributionWM,DistributionNet
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase84_single_split_site_wm import dev_split,SiteConsistentWM

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
N_COMPONENTS=4


def correlated_total_distribution(mixture_logits,site_logits):
    """Exactly marginalize a joint mixture over all six 0..4 ratings.

    P(S1..S6|H) = sum_k pi_k(H) * prod_j P(Sj|H,k).
    Unlike multiplying already-marginalized site probabilities,
    component mixing preserves conditional dependence across sites.
    """
    if (site_logits.ndim!=4 or site_logits.shape[-2:]!=(6,5)
            or mixture_logits.shape!=site_logits.shape[:2]):
        raise ValueError("Expected mixing logits [B,K], site logits [B,K,6,5]")
    site_p=torch.softmax(site_logits,dim=-1)
    weights=torch.softmax(mixture_logits,dim=-1)
    total=site_p[:,:,0,:]
    for site in range(1,6):
        # Each class in a new site contributes a shifted distribution
        # over total Motor burden. Pad creates differentiable tensors.
        shifted=[F.pad(total*site_p[:,:,site,rating:rating+1],
                       (rating,4-rating)) for rating in range(5)]
        total=sum(shifted)
    distribution=(weights.unsqueeze(-1)*total).sum(dim=1)
    return distribution.clamp_min(1e-12)/distribution.clamp_min(1e-12).sum(
        dim=-1,keepdim=True)


def joint_site_log_probability(mixture_logits,site_logits,sites):
    """Actual joint six-site log score, not pseudolikelihood."""
    if sites.ndim!=2 or sites.shape[1]!=6:
        raise ValueError("Expected six future site labels")
    k=site_logits.shape[1]
    index=sites[:,None,:,None].expand(-1,k,-1,-1)
    log_site=torch.log_softmax(site_logits,dim=-1).gather(-1,index)
    log_joint=(torch.log_softmax(mixture_logits,dim=-1)
               +log_site.squeeze(-1).sum(-1))
    return torch.logsumexp(log_joint,dim=-1)


class MixtureSiteNet(DistributionNet):
    def __init__(self,components=N_COMPONENTS):
        super().__init__()
        self.components=components
        self.mixture=nn.Linear(96,components)
        self.future_sites=nn.Linear(96,components*6*5)
    def forward(self,x,dt,valid,clinical,horizon):
        hidden=x.new_zeros((len(x),self.cell.hidden_size))
        decay=F.softplus(self.rate)
        for t in range(x.shape[1]):
            proposed=self.cell(self.observation(x[:,t]),
                hidden*torch.exp(-decay*dt[:,t,None]))
            hidden=torch.where(valid[:,t,None],proposed,hidden)
        context=self.fuse(torch.cat((hidden,horizon),dim=-1))
        return (self.mixture(context),
                self.future_sites(context).reshape(-1,self.components,6,5))


class CorrelatedSiteWM:
    def __init__(self,epochs=18,seed=SEED,components=N_COMPONENTS):
        self.epochs=epochs;self.seed=seed;self.components=components
    def fit(self,rows):
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=MixtureSiteNet(self.components)
        optimizer=torch.optim.AdamW(self.net.parameters(),
                                    lr=.0008,weight_decay=.025)
        X=sequence_features(rows,())
        site_truth,y,_=labels(rows)
        site_truth=torch.as_tensor(site_truth,dtype=torch.long)
        y=torch.as_tensor(y,dtype=torch.long)
        support=torch.arange(25)[None,:]
        for epoch in range(self.epochs):
            self.net.train()
            order=rng.permutation(len(rows))
            for start in range(0,len(order),32):
                ix=order[start:start+32]
                mix,sites=self.net(*[v[ix] for v in X])
                distribution=correlated_total_distribution(mix,sites)
                # Both losses are likelihoods of REAL future clinical
                # ratings, while RPS rewards ordered and calibrated tails.
                log_prob=joint_site_log_probability(
                    mix,sites,site_truth[ix])
                site_joint_nll=-log_prob.mean()
                total_nll=F.nll_loss(torch.log(distribution),y[ix])
                cdf=torch.cumsum(distribution,dim=-1)
                truth=(support>=y[ix,None]).to(cdf.dtype)
                rps=((cdf-truth)**2).sum(dim=1).mean()/24.
                loss=total_nll+.2*site_joint_nll+1.5*rps
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()
        self.net.eval()
        return self
    def predict_distribution(self,rows):
        with torch.inference_mode():
            mix,sites=self.net(*sequence_features(rows,()))
            return correlated_total_distribution(mix,sites).numpy()
    def effective_components(self,rows):
        with torch.inference_mode():
            mix,_=self.net(*sequence_features(rows,()))
            p=torch.softmax(mix,dim=-1)
            entropy=-(p*torch.log(p.clamp_min(1e-12))).sum(-1)
            return float(torch.exp(entropy).mean())


def fingerprint(rows):
    """Aggregate split fingerprint; never write individual identifiers."""
    unique=sorted({str(r["pid"]) for r in rows})
    digest=hashlib.sha256()
    digest.update(("phase85-split-v1|"+"|".join(unique)).encode())
    return digest.hexdigest()


def read_tasks(root,manifest):
    people,_=load_cohorts(root,manifest)
    evidence,_,_=evidence_tables(root,people)
    datasets={}
    for task in TASKS:
        population=[p for p in people if p["cohort"]==task[0]]
        events,_=future_site_pairs(population,task[1])
        records=data_rows(population,events,evidence,task)
        train,dev=dev_split(records)
        if min(len(train),len(dev))<15:
            raise ValueError("Unusable development split")
        train_ids={r["pid"] for r in train}
        auxiliary=build_auxiliary(
            [p for p in population if p["pid"] in train_ids])
        datasets[task]={"fit":train,"dev":dev,"aux":auxiliary}
    # Patient-disjoint even when paired horizons share the same person.
    fit_people={r["pid"] for d in datasets.values() for r in d["fit"]}
    dev_people={r["pid"] for d in datasets.values() for r in d["dev"]}
    if fit_people&dev_people:
        raise AssertionError("Global patient split leaked between horizons")
    return datasets


def run(args):
    if args.arm not in range(4) or not 4<=args.epochs<=100:
        raise ValueError("Arm must be 0..3; epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve();out=args.output.resolve()
    if root==out or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output or missing manifest")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tasks=read_tasks(root,args.manifest_db)
    results=[];coverage=[]
    pooled=None
    if args.arm==3:
        # Shared-horizon WM sees only TRAIN members of both horizons.
        # All DEV members remain excluded from every FIT task.
        combined=(tasks[("prodromal",12)]["fit"]+
                  tasks[("prodromal",24)]["fit"])
        combined.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
        pooled=CorrelatedSiteWM(args.epochs,seed=SEED+3000).fit(combined)
        print("PHASE85_SHARED_TRAIN_COMPLETE",len(combined),flush=True)
    for task in TASKS:
        d=tasks[task]
        seed=SEED+task[1]+(0 if task[0]=="prodromal" else 300)
        if args.arm==0:
            wm=DistributionWM("categorical",epochs=args.epochs,seed=seed)
            wm.fit(d["fit"],d["aux"])
        elif args.arm==1:
            wm=SiteConsistentWM(epochs=args.epochs,seed=seed,aux_weight=0.)
            wm.fit(d["fit"],d["aux"])
        elif args.arm==2 or task[0]!="prodromal":
            wm=CorrelatedSiteWM(args.epochs,seed=seed).fit(d["fit"])
        else:wm=pooled
        p=wm.predict_distribution(d["dev"])
        metrics=probabilities_scores(p,d["dev"])
        result={"cohort":task[0],"horizon":task[1],"arm":args.arm,
                "fit":len(d["fit"]),"dev":len(d["dev"]),
                "fit_aux_pairs":len(d["aux"]),
                "fit_fingerprint":fingerprint(d["fit"]),
                "dev_fingerprint":fingerprint(d["dev"]),
                "effective_components":(
                    wm.effective_components(d["dev"])
                    if isinstance(wm,CorrelatedSiteWM) else None),
                **metrics}
        results.append(result)
        coverage.append({"cohort":task[0],"horizon":task[1],
            "fit":len(d["fit"]),"dev":len(d["dev"]),"status":"complete"})
        write_csv(out/"dev_metrics.csv",results)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE85_TASK_COMPLETE",task,"arm",args.arm,flush=True)
    (out/"summary.json").write_text(json.dumps({
      "phase":85,"arm":args.arm,"completed_tasks":len(coverage),
      "seconds":round(time.monotonic()-started,2),
      "training_scope":"Global patient 80% FIT, 20% repeatedly-used DEV",
      "mixture_components":N_COMPONENTS if args.arm>=2 else None,
      "shared_horizon_fit":args.arm==3,
      "limitations":["Development only: not independent confirmation",
        "No patient-level outputs or original consumed final TEST",
        "Mixture components are latent statistical modes, not diagnosed subtypes",
        "Simple conditional mixtures are not a proven causal world model"],
      "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
      "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE85_SUCCESS",args.arm,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",required=True,type=int)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args())
