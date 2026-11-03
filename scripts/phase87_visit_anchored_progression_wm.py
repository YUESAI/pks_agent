#!/usr/bin/env python3
"""Phase87: visit-anchored continuous-time clinical World Model development.

Single fixed global 80/20 patient development split, no five folds.
Four independent CPU arms:
  0 Phase85 best shared correlated mixture (incumbent)
  1 Phase86 continuous-time flow WITHOUT observed-transition supervision
  2 Same flow + 0.05*true-next-visit 27-Motor-item cross-entropy
  3 Same flow + 0.15*true-next-visit 27-Motor-item cross-entropy

Unlike Phase86's almost-zero numerical semigroup penalty, auxiliary
supervision here is an *observed longitudinal endpoint*: the next
real visit's measured visible27 ratings, after a strictly pre-visit
input history. Only FIT patient histories are used for this loss.
This is exploratory; the repeatedly used DEV is not final TEST.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.phase72_ctlatent_wm import sequence_features
from scripts.phase76_reliability_gated_wm import labels,write_csv
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase85_correlated_site_wm import (
    CorrelatedSiteWM,correlated_total_distribution,
    joint_site_log_probability,read_tasks,fingerprint,SEED,TASKS)
from scripts.phase86_continuous_time_progression_wm import (
    ContinuousTimeMixtureNet,ProgressionWM)

AUX_WEIGHTS={0:0.0,1:0.0,2:0.05,3:0.15}


class VisitAnchoredFlowNet(ContinuousTimeMixtureNet):
    """Flow the last observed state to a real future visit/time horizon.

    The inherited next_motor head is trained on a real next-visit
    visible27 rating vector; it never receives the target scores as
    input. Its sites and mixture heads predict the 6 future site
    ratings and the correlated 0..24 total distribution.
    """
    def forward(self,x,dt,valid,clinical,horizon):
        if horizon.shape!=(len(x),2):
            raise ValueError("Invalid horizon shape")
        years=horizon[:,1:2]*3.
        if not torch.all((years>0)&(years<=3.)):
            raise ValueError("Future time must be after observed cutoff")
        current=self._observed_state(x,dt,valid)
        future=self.evolve(current,years)
        context=self.fuse(torch.cat((future,horizon),dim=-1))
        mixture=self.mixture(context)
        site_logits=self.future_sites(context).reshape(
            -1,self.components,6,5)
        next_motor_logits=self.next_motor(context).reshape(-1,27,5)
        return mixture,site_logits,next_motor_logits


class VisitAnchoredWM:
    def __init__(self,epochs=18,seed=SEED,aux_weight=0.05):
        if aux_weight<0:raise ValueError("Negative loss weight")
        self.epochs=epochs
        self.seed=seed
        self.aux_weight=float(aux_weight)

    def fit(self,rows,aux):
        if len(rows)<30:raise ValueError("FIT set too small")
        if self.aux_weight and not aux:
            raise ValueError("Real FIT next-visit evidence is required")
        ids={str(r["pid"]) for r in rows}
        if any(str(r["pid"]) not in ids for r in aux):
            raise AssertionError("Auxiliary evidence contains a non-FIT patient")
        torch.manual_seed(self.seed)
        gen=np.random.default_rng(self.seed)
        self.net=VisitAnchoredFlowNet()
        optimizer=torch.optim.AdamW(
            self.net.parameters(),lr=.0008,weight_decay=.025)
        X=sequence_features(rows,())
        site,y,_=labels(rows)
        site=torch.tensor(site,dtype=torch.long)
        y=torch.tensor(y,dtype=torch.long)
        if self.aux_weight:
            AX=sequence_features(aux,())
            true_next=torch.tensor(
                [r["aux_motor"] for r in aux],dtype=torch.long)
            if true_next.shape!=(len(aux),27):
                raise AssertionError("Auxiliary true visible27 ratings missing")
            if not torch.any(true_next!=-100):
                raise AssertionError("No actually measured next-visit items")
        else:
            AX=None
        support=torch.arange(25)[None,:]
        for epoch in range(self.epochs):
            self.net.train()
            order=gen.permutation(len(rows))
            for start in range(0,len(order),32):
                ii=order[start:start+32]
                mixture,sites,_=self.net(*[v[ii] for v in X])
                p=correlated_total_distribution(mixture,sites)
                nll=F.nll_loss(torch.log(p.clamp_min(1e-12)),y[ii])
                site_nll=-joint_site_log_probability(
                    mixture,sites,site[ii]).mean()
                truth=(support>=y[ii,None]).to(p.dtype)
                rps=(p.cumsum(-1)-truth).square().sum(-1).mean()/24.
                loss=nll+.2*site_nll+1.5*rps
                if AX is not None:
                    jj=gen.integers(0,len(aux),size=min(32,len(ii)))
                    _,_,next_logits=self.net(*[v[jj] for v in AX])
                    # Ignore genuinely unmeasured items, never use an
                    # invented rating to fill the missing value.
                    aux_loss=F.cross_entropy(
                        next_logits.reshape(-1,5),
                        true_next[jj].reshape(-1),ignore_index=-100)
                    loss=loss+self.aux_weight*aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()
        self.net.eval()
        self.aux_pairs_used=len(aux) if self.aux_weight else 0
        return self

    def predict_distribution(self,rows):
        with torch.inference_mode():
            mixture,sites,_=self.net(*sequence_features(rows,()))
            return correlated_total_distribution(mixture,sites).numpy()

    def diagnostics(self,rows):
        with torch.inference_mode():
            xs=sequence_features(rows,())
            mix,_,_=self.net(*xs)
            weights=torch.softmax(mix,dim=-1)
            entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
            return {"effective_components":float(entropy.exp().mean()),
                    "latent_flow_strength":float(
                        F.softplus(self.net.flow_strength).item())}


def merge_fit_auxiliary(datasets):
    """Union real adjacent visits of FIT people, do not double count
    overlapping 12m/24m patient trajectories when sharing training.
    """
    train=(datasets[("prodromal",12)]["fit"]+
           datasets[("prodromal",24)]["fit"])
    ids={str(r["pid"]) for r in train}
    dedup={}
    for task in (("prodromal",12),("prodromal",24)):
        for row in datasets[task]["aux"]:
            key=(str(row["pid"]),int(row["anchor"]),
                 int(row["horizon"]))
            if str(row["pid"]) not in ids:
                raise AssertionError("Auxiliary contains DEV person")
            if key in dedup:
                if dedup[key]["aux_motor"]!=row["aux_motor"]:
                    raise AssertionError("Conflicting observed next visit")
            else:
                dedup[key]=row
    return [dedup[key] for key in sorted(dedup)]


def run(args):
    if args.arm not in range(4) or not 4<=args.epochs<=100:
        raise ValueError("Expected arm 0..3 and 4..100 epochs")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve()
    out=args.output.resolve()
    if out==root or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output or missing frozen manifest")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tasks=read_tasks(root,args.manifest_db)
    combined=(tasks[("prodromal",12)]["fit"]+
              tasks[("prodromal",24)]["fit"])
    combined.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    aux=merge_fit_auxiliary(tasks)
    if args.arm==0:
        shared=CorrelatedSiteWM(args.epochs,seed=SEED+3000).fit(combined)
    elif args.arm==1:
        shared=ProgressionWM(args.epochs,seed=SEED+3000).fit(combined)
    else:
        shared=VisitAnchoredWM(args.epochs,seed=SEED+3000,
                 aux_weight=AUX_WEIGHTS[args.arm]).fit(combined,aux)
    print("PHASE87_SHARED_FIT_COMPLETE",len(combined),
          "true_adjacent_visit_pairs",len(aux),flush=True)
    metrics=[];coverage=[]
    for task in TASKS:
        d=tasks[task]
        if task[0]=="prodromal":
            wm=shared
            aux_count=len(aux)
        else:
            if args.arm==0:
                wm=CorrelatedSiteWM(args.epochs,seed=SEED+312).fit(
                    d["fit"])
            elif args.arm==1:
                wm=ProgressionWM(args.epochs,seed=SEED+312).fit(
                    d["fit"])
            else:
                wm=VisitAnchoredWM(args.epochs,seed=SEED+312,
                    aux_weight=AUX_WEIGHTS[args.arm]).fit(
                        d["fit"],d["aux"])
            aux_count=len(d["aux"])
        prediction=wm.predict_distribution(d["dev"])
        score=probabilities_scores(prediction,d["dev"])
        diagnostics=(wm.diagnostics(d["dev"]) if
             isinstance(wm,VisitAnchoredWM) else {})
        metrics.append({
            "cohort":task[0],"horizon":task[1],
            "arm":args.arm,"fit":len(d["fit"]),"dev":len(d["dev"]),
            "fit_fingerprint":fingerprint(d["fit"]),
            "dev_fingerprint":fingerprint(d["dev"]),
            "fit_adjacent_visit_pairs":aux_count,
            "aux_weight":AUX_WEIGHTS[args.arm],**score,
            "effective_components":diagnostics.get("effective_components"),
            "latent_flow_strength":diagnostics.get("latent_flow_strength")})
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",metrics)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE87_TASK_COMPLETE",task,"arm",args.arm,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":87,"arm":args.arm,"completed_tasks":len(coverage),
        "runtime_seconds":round(time.monotonic()-started,2),
        "model":("Phase85 shared correlated mixture" if args.arm==0
                 else "Phase86 flow" if args.arm==1
                 else "visit-anchored time-flow WM"),
        "aux_weight":AUX_WEIGHTS[args.arm],
        "shared_train_adjacent_pairs":len(aux),
        "split":"patient-global hash 80% FIT / 20% development",
        "limitations":["Repeatedly developed DEV; not independent TEST",
            "Actual FIT-only future 27-item motor visits used for training",
            "No DEV person's visits can enter auxiliary training",
            "The current WM predicts observed time gaps, not causal treatments",
            "Original consumed final TEST remains excluded",
            "No patient-level clinical records or predictions exported"],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE87_SUCCESS",args.arm,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",type=int,required=True)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())

if __name__=="__main__":
    main()
