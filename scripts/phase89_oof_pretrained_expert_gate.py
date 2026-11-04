#!/usr/bin/env python3
"""Phase89: truly separate pretrained static/dynamic experts + OOF gate.

Four ordinary standalone CPU jobs, ONE global patient 80% FIT / 20% DEV:
 arm0 Phase85 static correlated six-site WM
 arm1 Phase87 real-visit-supervised flow WM (aux weight=0.15)
 arm2 separate experts, horizon+remaining-gap linear gate
 arm3 separate experts, patient Motor-state+time+expert-disagreement gate

A gate MUST be trained on two inner patient-disjoint OOF expert
predictions from the 80% FIT pool, NOT on predictions from experts
that trained on those patients. After fitting gate, final two experts
are refitted on all FIT persons; neither gate nor experts touch DEV
labels. Inner OOF is not an outer fivefold benchmark.

This is exploratory model development on repeatedly used DEV.
Neither labels nor patient-level predictions/IDs are exported.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase76_reliability_gated_wm import write_csv
from scripts.phase85_correlated_site_wm import (
    CorrelatedSiteWM,read_tasks,fingerprint,SEED,TASKS)
from scripts.phase87_visit_anchored_progression_wm import (
    VisitAnchoredWM,merge_fit_auxiliary)

AUX_WEIGHT=.15
OOF_PARTITIONS=2
GATE_EPOCHS=180
GATE_PRIOR=.35
# A weak fixed prior stabilizes small OOF folds without forcing the
# gate to a constant. No selection of this value on outer DEV.
GATE_PRIOR_STRENGTH=.005
GATE_WEIGHT_L2=.001


def inner_partition(pid):
    """Patient-global assignment shared across both prediction horizons."""
    key="phase89-inner-v1|"+str(pid)
    digest=hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:12],16)%OOF_PARTITIONS


def train_expert_pair(fit,aux,seed):
    """Independent networks, identical FIT IDs but different objectives."""
    if len({str(r["pid"]) for r in fit})<25:
        raise AssertionError("Not enough distinct FIT participants")
    people={str(r["pid"]) for r in fit}
    if any(str(r["pid"]) not in people for r in aux):
        raise AssertionError("Auxiliary visit from a held-out patient")
    static=CorrelatedSiteWM(epochs=train_expert_pair.epochs,seed=seed)
    dynamic=VisitAnchoredWM(epochs=train_expert_pair.epochs,
                            seed=seed,aux_weight=AUX_WEIGHT)
    static.fit(fit)
    dynamic.fit(fit,aux)
    return static,dynamic


train_expert_pair.epochs=18


def _last_visit_features(row):
    """Only historical *observed* Motor ratings; never future labels."""
    history=row["history"]
    last=history[-1]
    first=history[0]
    values=list(last["scores"].values())
    initial=list(first["scores"].values())
    mean_last=(float(np.mean(values))/4. if values else 0.)
    mean_first=(float(np.mean(initial))/4. if initial else 0.)
    return [
        min(len(history),6)/6.,
        len(values)/27.,
        mean_last,
        mean_last-mean_first,
    ]


def gate_features(rows,p_static,p_dynamic,mode):
    if mode not in ("horizon","patient"):
        raise ValueError("Invalid mode")
    s=np.asarray(p_static,dtype=np.float64)
    d=np.asarray(p_dynamic,dtype=np.float64)
    if s.shape!=d.shape or s.shape!=(len(rows),25):
        raise AssertionError("Both experts must produce [n,25] posteriors")
    if not np.isfinite(s).all() or not np.isfinite(d).all():
        raise AssertionError("Invalid expert posterior")
    support=np.arange(25,dtype=np.float64)
    features=[]
    for i,r in enumerate(rows):
        h=float(r["horizon"])
        elapsed=float(r["history"][-1]["month"]-r["anchor"])
        remaining=h-elapsed
        if not (0<remaining<=36):
            raise AssertionError("History overlaps true future target")
        base=[h/36.,remaining/36.]
        if mode=="patient":
            uncertainty_s=-np.sum(s[i]*np.log(s[i]+1e-10))
            uncertainty_d=-np.sum(d[i]*np.log(d[i]+1e-10))
            base+=_last_visit_features(r)
            base+=[
                float((d[i]-s[i])@support)/24.,
                float(s[i,0]-d[i,0]),
                float(uncertainty_s-uncertainty_d)/np.log(25.),
                float(np.abs(s[i].cumsum()-d[i].cumsum()).sum()/24.),
            ]
        features.append(base)
    return torch.tensor(np.asarray(features),dtype=torch.float32)


class CalibratedGate(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.linear=nn.Linear(n_features,1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias,
                          float(np.log(GATE_PRIOR/(1-GATE_PRIOR))))
    def forward(self,x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def patient_weights(rows):
    """Patients appearing at both horizons contribute total weight 1."""
    cnt=Counter(str(r["pid"]) for r in rows)
    w=torch.tensor([1./cnt[str(r["pid"])] for r in rows],
                   dtype=torch.float32)
    return w/w.sum()


class OOFDecisionGate:
    def __init__(self,mode):
        self.mode=mode
        self.net=CalibratedGate(2 if mode=="horizon" else 10)
        self.fit_size=0
        self.oof_loss_before=None
        self.oof_loss_after=None
    def fit(self,rows,s_prob,d_prob):
        if len(rows)!=len(s_prob):
            raise AssertionError("Misaligned OOF rows")
        p_s=torch.tensor(np.asarray(s_prob),dtype=torch.float32)
        p_d=torch.tensor(np.asarray(d_prob),dtype=torch.float32)
        X=gate_features(rows,s_prob,d_prob,self.mode)
        y=torch.tensor([int(sum(r["site"])) for r in rows],
                       dtype=torch.long)
        truth=(torch.arange(25)[None,:]>=y[:,None]).float()
        weight=patient_weights(rows)
        opt=torch.optim.Adam(self.net.parameters(),lr=.02)
        torch.manual_seed(SEED)
        def losses():
            gate=self.net(X).unsqueeze(-1)
            mixed=(1-gate)*p_s+gate*p_d
            rps=((mixed.cumsum(dim=1)-truth)**2).sum(dim=1)/24.
            weighted=(rps*weight).sum()
            return weighted,gate
        with torch.no_grad():
            before,_=losses()
            self.oof_loss_before=float(before)
        for _ in range(GATE_EPOCHS):
            opt.zero_grad(set_to_none=True)
            ranked,gate=losses()
            regularizer=(GATE_PRIOR_STRENGTH*
                         ((gate-GATE_PRIOR).square().flatten()*weight).sum())
            l2=GATE_WEIGHT_L2*self.net.linear.weight.square().sum()
            (100.*ranked+regularizer+l2).backward()
            opt.step()
        with torch.no_grad():
            after,gate=losses()
            self.oof_loss_after=float(after)
            self.oof_mean_gate=float((gate.flatten()*weight).sum())
        self.net.eval()
        self.fit_size=len(rows)
        return self
    def predict(self,rows,s_prob,d_prob):
        with torch.inference_mode():
            X=gate_features(rows,s_prob,d_prob,self.mode)
            gate=self.net(X).numpy()
        mixed=np.asarray(s_prob)*(1-gate[:,None])+np.asarray(
            d_prob)*gate[:,None]
        mixed=np.maximum(mixed,1e-12)
        mixed/=mixed.sum(axis=1,keepdims=True)
        return mixed,gate


def make_oof(fit,aux,seed):
    """Inner-fold expert predictions. Same patient held out for BOTH horizons.

    All auxiliary adjacent visits belong to the current inner TRAIN
    people. In particular no Motor model sees a person's future
    visits before that person's OOF forecast is made.
    """
    people={str(r["pid"]) for r in fit}
    if any(str(a["pid"]) not in people for a in aux):
        raise AssertionError("Auxiliary union contains a non-FIT person")
    s=np.full((len(fit),25),np.nan)
    d=np.full_like(s,np.nan)
    fold_summary=[]
    for k in range(OOF_PARTITIONS):
        idx_train=[i for i,r in enumerate(fit) if
                   inner_partition(r["pid"])!=k]
        idx_eval=[i for i,r in enumerate(fit) if
                  inner_partition(r["pid"])==k]
        train=[fit[i] for i in idx_train]
        held=[fit[i] for i in idx_eval]
        train_ids={str(r["pid"]) for r in train}
        eval_ids={str(r["pid"]) for r in held}
        if train_ids&eval_ids:
            raise AssertionError("Same patient leaks across OOF fit and held")
        if min(len(train),len(held))<20:
            raise ValueError("Unbalanced two-fold gate training")
        current_aux=[a for a in aux if str(a["pid"]) in train_ids]
        if not current_aux:
            raise AssertionError("No genuine inner-train next visits")
        static,dynamic=train_expert_pair(train,current_aux,
                                         seed+1000+k*137)
        s[idx_eval]=static.predict_distribution(held)
        d[idx_eval]=dynamic.predict_distribution(held)
        fold_summary.append({"fold":k,
                             "fit_records":len(train),
                             "held_records":len(held),
                             "fit_patients":len(train_ids),
                             "held_patients":len(eval_ids),
                             "fit_aux_pairs":len(current_aux)})
    if not np.isfinite(s).all() or not np.isfinite(d).all():
        raise AssertionError("OOF expert predictions incomplete")
    return s,d,fold_summary


def evaluate_cohort(task,d,mode,seed,shared=None):
    if shared is None:
        if mode=="static":
            static=CorrelatedSiteWM(epochs=train_expert_pair.epochs,
                                     seed=seed).fit(d["fit"])
            p=static.predict_distribution(d["dev"])
            return p,{},{}
        if mode=="dynamic":
            dynamic=VisitAnchoredWM(
                epochs=train_expert_pair.epochs,
                seed=seed,aux_weight=AUX_WEIGHT).fit(d["fit"],d["aux"])
            return dynamic.predict_distribution(d["dev"]),{},{}
        s_oof,d_oof,inner=make_oof(d["fit"],d["aux"],seed)
        gate=OOFDecisionGate(mode).fit(d["fit"],s_oof,d_oof)
        static,dynamic=train_expert_pair(d["fit"],d["aux"],seed)
    else:
        gate,static,dynamic,inner=shared
    s=static.predict_distribution(d["dev"])
    dy=dynamic.predict_distribution(d["dev"])
    p,w=gate.predict(d["dev"],s,dy)
    diagnostics={"dynamic_gate_mean":float(w.mean()),
                 "dynamic_gate_std":float(w.std()),
                 "static_rps":probabilities_scores(s,d["dev"])["rps"],
                 "dynamic_rps":probabilities_scores(dy,d["dev"])["rps"],
                 "oof_gate_initial_rps":gate.oof_loss_before,
                 "oof_gate_trained_rps":gate.oof_loss_after}
    return p,diagnostics,{"inner_folds":inner,
                          "fit_gate_examples":gate.fit_size}


def run(args):
    if args.arm not in range(4) or not 4<=args.epochs<=100:
        raise ValueError("Expected arms 0..3 and epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train_expert_pair.epochs=args.epochs
    root=args.data_root.resolve();out=args.output.resolve()
    if out==root or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Invalid result path or frozen manifest")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    datasets=read_tasks(root,args.manifest_db)
    prod_rows=(datasets[("prodromal",12)]["fit"]+
               datasets[("prodromal",24)]["fit"])
    prod_rows.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    prod_aux=merge_fit_auxiliary(datasets)
    seed=SEED+3000
    if args.arm==0:
        static=CorrelatedSiteWM(args.epochs,seed=seed).fit(prod_rows)
        shared=static
        kind="static"
    elif args.arm==1:
        dynamic=VisitAnchoredWM(args.epochs,seed=seed,
                                aux_weight=AUX_WEIGHT).fit(
                                    prod_rows,prod_aux)
        shared=dynamic
        kind="dynamic"
    else:
        kind="horizon" if args.arm==2 else "patient"
        print("PHASE89_PROD_INNER_OOF_BEGIN",args.arm,flush=True)
        sp,dp,inner=make_oof(prod_rows,prod_aux,seed)
        gate=OOFDecisionGate(kind).fit(prod_rows,sp,dp)
        static,dynamic=train_expert_pair(prod_rows,prod_aux,seed)
        shared=(gate,static,dynamic,inner)
        print("PHASE89_PROD_INNER_OOF_DONE",args.arm,
              "rows",len(prod_rows),flush=True)
    results=[];coverage=[];details=[]
    for task in TASKS:
        d=datasets[task]
        if task[0]=="prodromal":
            if args.arm<2:
                predictions=shared.predict_distribution(d["dev"])
                diagnostics={};audit={}
            else:
                predictions,diagnostics,audit=evaluate_cohort(
                    task,d,kind,seed,shared=shared)
        else:
            predictions,diagnostics,audit=evaluate_cohort(
                task,d,kind,SEED+312)
        row={"cohort":task[0],"horizon":task[1],"arm":args.arm,
             "fit":len(d["fit"]),"dev":len(d["dev"]),
             "fit_fingerprint":fingerprint(d["fit"]),
             "dev_fingerprint":fingerprint(d["dev"]),
             **probabilities_scores(predictions,d["dev"]),**diagnostics}
        results.append(row)
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        if audit:
            details.append({"task":list(task),
                            "oof_internal_split":audit})
        write_csv(out/"dev_metrics.csv",results)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE89_TASK_COMPLETE",task,"arm",args.arm,flush=True)
    (out/"oof_audit.json").write_text(json.dumps(details,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
        "phase":89,"arm":args.arm,"kind":kind,
        "completed_tasks":len(coverage),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "outer_protocol":"one global 80% FIT, 20% repeatedly-used DEV",
        "inner_protocol":"2fold patient-global FIT-only OOF predictions for gate",
        "experts_are_fully_independent":True,
        "static_seed":SEED+3000,"dynamic_aux_weight":AUX_WEIGHT,
        "limitations":[
            "Exploratory repeatedly re-used DEV, not independent TEST",
            "2fold FIT-only gate crossfit is not fivefold outer evaluation",
            "OOF experts train on 50% FIT, refit experts on full FIT at DEV time",
            "Covariate/distribution shift between OOF experts and full FIT experts possible",
            "Site predictions are observational not causal clinical state transitions",
            "No original consumed final TEST or patient-level output"],
        "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE89_SUCCESS",args.arm,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",required=True,type=int)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
