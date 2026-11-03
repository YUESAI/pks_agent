#!/usr/bin/env python3
"""Phase88: selectively deployed longitudinal dynamics for future Parkinson motor burden.

Four model-development arms on the SAME patient-global 80% FIT/20% DEV:
0 Phase85 static shared-horizon correlated-site mixture WM
1 Phase87 observed-visit-supervised continuous flow (weight 0.15)
2 NEW horizon-only gate between static and evolving latent experts
3 NEW patient-state+horizon gate between both experts

The model mixes two coherent joint six-site distributions and trains its
dynamic branch with real FIT-only adjacent-visit targets. Development
results are NOT independent scientific confirmation.
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
from scripts.phase86_continuous_time_progression_wm import ContinuousTimeMixtureNet
from scripts.phase87_visit_anchored_progression_wm import (
    VisitAnchoredWM,merge_fit_auxiliary)

AUX_WEIGHT=.15
COMPANION_STATIC_WEIGHT=.10
COMPANION_DYNAMIC_WEIGHT=.04
GATE_PENALTY=.005

def blend_posteriors(static,dynamic,gate):
    """Convex combination of two valid score-0..24 distributions."""
    if static.shape!=dynamic.shape or static.ndim!=2 or static.shape[1]!=25:
        raise ValueError("Expected two [batch,25] probability matrices")
    if gate.shape!=(len(static),1):
        raise ValueError("Expected one gate per patient")
    if not bool(((gate>=0)&(gate<=1)).all()):
        raise ValueError("Gate outside [0,1]")
    result=static*(1-gate)+dynamic*gate
    return result/result.sum(dim=1,keepdim=True).clamp_min(1e-12)

class SelectiveProgressionNet(ContinuousTimeMixtureNet):
    def __init__(self,mode="patient"):
        super().__init__()
        if mode not in ("horizon","patient"):
            raise ValueError("Gate mode must be horizon or patient")
        self.mode=mode
        n_gate=2 if mode=="horizon" else self.cell.hidden_size+2
        self.gate_layer=nn.Linear(n_gate,1)
        # Conservative predeclared prior: initial dynamic gate is 20%.
        nn.init.zeros_(self.gate_layer.weight)
        nn.init.constant_(self.gate_layer.bias,-1.38629436112)
    def _heads(self,h,horizon):
        context=self.fuse(torch.cat((h,horizon),dim=-1))
        return (self.mixture(context),
                self.future_sites(context).reshape(
                    -1,self.components,6,5),
                self.next_motor(context).reshape(-1,27,5))
    def forward(self,x,dt,valid,clinical,horizon):
        if horizon.ndim!=2 or horizon.shape!=(len(x),2):
            raise ValueError("Expected forecast horizon and remaining gap")
        remaining_years=horizon[:,1:2]*3.
        if not bool(((remaining_years>0)&(remaining_years<=3)).all()):
            raise ValueError("Remaining time must be strictly future")
        observed=self._observed_state(x,dt,valid)
        progressed=self.evolve(observed,remaining_years)
        sm,ss,_=self._heads(observed,horizon)
        dm,ds,next_motor=self._heads(progressed,horizon)
        gate_input=(horizon if self.mode=="horizon" else
                    torch.cat((observed,horizon),dim=-1))
        gate=torch.sigmoid(self.gate_layer(gate_input))
        return sm,ss,dm,ds,gate,next_motor

class SelectiveProgressionWM:
    def __init__(self,mode="patient",epochs=18,seed=SEED):
        self.mode=mode;self.epochs=int(epochs);self.seed=int(seed)
    def fit(self,fit,aux):
        if len(fit)<30 or not aux:
            raise ValueError("Need >=30 FIT examples and real visit pairs")
        fit_ids={str(r["pid"]) for r in fit}
        if any(str(r["pid"]) not in fit_ids for r in aux):
            raise AssertionError("Non-FIT person in auxiliary transitions")
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=SelectiveProgressionNet(self.mode)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.0008,
                              weight_decay=.025)
        X=sequence_features(fit,())
        sites,y,_=labels(fit)
        sites=torch.as_tensor(sites,dtype=torch.long)
        y=torch.as_tensor(y,dtype=torch.long)
        AX=sequence_features(aux,())
        next_items=torch.as_tensor([r["aux_motor"] for r in aux],
                                   dtype=torch.long)
        if next_items.shape!=(len(aux),27):
            raise AssertionError("Invalid actual next-visit rating targets")
        support=torch.arange(25)[None,:]
        for epoch in range(self.epochs):
            self.net.train()
            order=rng.permutation(len(fit))
            for off in range(0,len(order),32):
                ix=order[off:off+32]
                sm,ss,dm,ds,gate,_=self.net(*[v[ix] for v in X])
                static_p=correlated_total_distribution(sm,ss)
                dynamic_p=correlated_total_distribution(dm,ds)
                blend=blend_posteriors(static_p,dynamic_p,gate)
                main_nll=F.nll_loss(blend.clamp_min(1e-12).log(),y[ix])
                s_log=joint_site_log_probability(sm,ss,sites[ix])
                d_log=joint_site_log_probability(dm,ds,sites[ix])
                g=gate.squeeze(-1).clamp(1e-6,1-1e-6)
                joint_log=torch.logaddexp(
                    torch.log1p(-g)+s_log,torch.log(g)+d_log)
                joint_nll=-joint_log.mean()
                truth=(support>=y[ix,None]).to(blend.dtype)
                ranked=(blend.cumsum(-1)-truth).square().sum(-1).mean()/24.
                loss=(main_nll+.2*joint_nll+1.5*ranked
                    +COMPANION_STATIC_WEIGHT*F.nll_loss(
                        static_p.clamp_min(1e-12).log(),y[ix])
                    +COMPANION_DYNAMIC_WEIGHT*F.nll_loss(
                        dynamic_p.clamp_min(1e-12).log(),y[ix])
                    +GATE_PENALTY*gate.mean())
                jj=rng.integers(0,len(aux),size=min(32,len(ix)))
                _,_,_,_,_,next_pred=self.net(*[v[jj] for v in AX])
                aux_loss=F.cross_entropy(next_pred.reshape(-1,5),
                    next_items[jj].reshape(-1),ignore_index=-100)
                loss=loss+AUX_WEIGHT*aux_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                opt.step()
        self.net.eval()
        return self
    def predict_distribution(self,rows):
        return self.predict_parts(rows)["blend"]
    def predict_parts(self,rows):
        with torch.inference_mode():
            sm,ss,dm,ds,gate,_=self.net(*sequence_features(rows,()))
            s=correlated_total_distribution(sm,ss)
            d=correlated_total_distribution(dm,ds)
            p=blend_posteriors(s,d,gate)
            return {"blend":p.numpy(),"static":s.numpy(),
                    "dynamic":d.numpy(),"gate":gate.numpy()}

def run(args):
    if args.arm not in range(4) or not 4<=args.epochs<=100:
        raise ValueError("Arm must be 0..3; epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve()
    out=args.output.resolve()
    if root==out or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output path or frozen manifest absent")
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    data=read_tasks(root,args.manifest_db)
    prod_fit=data[("prodromal",12)]["fit"]+data[("prodromal",24)]["fit"]
    prod_fit.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    prod_aux=merge_fit_auxiliary(data)
    def train_model(fit,aux,seed):
        if args.arm==0:
            return CorrelatedSiteWM(args.epochs,seed=seed).fit(fit)
        if args.arm==1:
            return VisitAnchoredWM(args.epochs,seed=seed,
                                   aux_weight=AUX_WEIGHT).fit(fit,aux)
        mode="horizon" if args.arm==2 else "patient"
        return SelectiveProgressionWM(mode,args.epochs,seed).fit(fit,aux)
    shared=train_model(prod_fit,prod_aux,SEED+3000)
    metrics=[];coverage=[]
    print("PHASE88_PROD_FIT_DONE",args.arm,len(prod_fit),
          len(prod_aux),flush=True)
    for task in TASKS:
        d=data[task]
        model=(shared if task[0]=="prodromal" else
               train_model(d["fit"],d["aux"],SEED+312))
        if isinstance(model,SelectiveProgressionWM):
            parts=model.predict_parts(d["dev"])
            p=parts["blend"]
            gate=parts["gate"].reshape(-1)
            diagnostics={
                "mean_dynamic_gate":float(gate.mean()),
                "gate_std":float(gate.std()),
                "static_rps":probabilities_scores(
                    parts["static"],d["dev"])["rps"],
                "dynamic_rps":probabilities_scores(
                    parts["dynamic"],d["dev"])["rps"]}
        else:
            p=model.predict_distribution(d["dev"])
            diagnostics={"mean_dynamic_gate":None,"gate_std":None,
                         "static_rps":None,"dynamic_rps":None}
        metrics.append({
            "cohort":task[0],"horizon":task[1],"arm":args.arm,
            "fit":len(d["fit"]),"dev":len(d["dev"]),
            "fit_fingerprint":fingerprint(d["fit"]),
            "dev_fingerprint":fingerprint(d["dev"]),
            "fit_transition_pairs":(len(prod_aux) if task[0]=="prodromal"
                                    else len(d["aux"])),
            **probabilities_scores(p,d["dev"]),**diagnostics})
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",metrics)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE88_TASK_COMPLETE",args.arm,task,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":88,"arm":args.arm,"completed_tasks":len(coverage),
        "seconds":round(time.monotonic()-start,2),
        "model":("phase85_static" if args.arm==0 else
                 "phase87_visit_flow" if args.arm==1 else
                 "adaptive_static_dynamic_gate"),
        "gate_type":("horizon" if args.arm==2 else
                     "patient" if args.arm==3 else None),
        "aux_weight":AUX_WEIGHT if args.arm else 0.,
        "fit_protocol":"patient-global 80% FIT / repeatedly explored 20% DEV",
        "limitations":[
            "Repeatedly used DEV, not independent confirmation",
            "Only real FIT-person next visits supervise dynamics",
            "Forecasts are observational, not causal intervention effects",
            "Mixture components are not verified clinical disease subtypes",
            "No patient-level exports; consumed final TEST remains excluded"],
        "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE88_SUCCESS",args.arm,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",type=int,required=True)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
