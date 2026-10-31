#!/usr/bin/env python3
"""Phase79: proper-scored ordinal clinical WM and audited distribution calibration.

Five prespecified person folds x four independent ordinary CPU arms:
0. Direct CT Motor / Motor HGB, reference.
1. Previous Phase78 categorical + hurdle distributions, dual decisions.
2. Same categorical baseline plus ordinal proper-scored categorical WM.
3. Ordinal WM plus DEV temperature selection and independent AUDIT veto.

For every fitted distribution, evaluate mean for RMSE, median for MAE,
p(Y>0) for Brier, and RPS/log score for the full 0..24 posterior.
Repeated original Prodromal TRAIN; NOT an independent confirmation set.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase73_matched_longitudinal_agent import data_rows,split_data
from scripts.phase72_ctlatent_wm import sequence_features,build_auxiliary
from scripts.phase76_reliability_gated_wm import (
    DirectWM,TimeTabular,labels,write_csv)
from scripts.phase78_distributional_wm import DistributionWM,DistributionNet
from scripts.phase60_cpu_multistep_wm import auc

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
SUPPORT=np.arange(25,dtype=np.float64)
TEMPERATURES=(.75,1.0,1.25,1.5,2.0)

def _targets(rows):
    _,total,positive=labels(rows)
    return total.astype(np.int64),positive.astype(np.float64)

def validate_probability(p,rows):
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=(len(rows),25):
        raise AssertionError("Expected probability for all 0..24 outcomes")
    if not np.isfinite(p).all() or (p<0).any() or np.max(
            np.abs(p.sum(axis=1)-1.))>1e-5:
        raise AssertionError("Distribution must be normalized and finite")
    return p

def quantile(prob,alpha):
    return (np.cumsum(prob,axis=1)<alpha).sum(axis=1).astype(int)

def probabilities_scores(prob,rows):
    p=validate_probability(prob,rows)
    y,z=_targets(rows)
    mean=p@SUPPORT
    med=quantile(p,.5)
    brier=(1-p[:,0]-z)**2
    cdf=np.cumsum(p,axis=1)
    truth=(SUPPORT[None,:]>=y[:,None]).astype(float)
    rps=((cdf-truth)**2).sum(axis=1)/24.
    nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-10))
    lower=quantile(p,.1);upper=quantile(p,.9)
    # Single calibrated posterior yields both point decisions.
    return {
        "mae_median":float(np.mean(np.abs(med-y))),
        "mae_mean":float(np.mean(np.abs(mean-y))),
        "rmse_mean":float(np.sqrt(np.mean((mean-y)**2))),
        "rmse_median":float(np.sqrt(np.mean((med-y)**2))),
        "brier":float(brier.mean()),
        "rps":float(rps.mean()),"nll":float(nll.mean()),
        "coverage80":float(np.mean((y>=lower)&(y<=upper))),
        "width80":float(np.mean(upper-lower)),
        "auroc":auc((1-p[:,0]).tolist(),z.astype(int).tolist()),
    }

def point_scores(pred,rows):
    burden,risk=pred
    y,z=_targets(rows)
    burden=np.asarray(burden);risk=np.asarray(risk)
    if len(burden)!=len(y) or not np.isfinite(burden).all():
        raise AssertionError("Bad direct burden forecasts")
    return {"mae_median":None,"mae_mean":float(np.mean(np.abs(burden-y))),
            "rmse_mean":float(np.sqrt(np.mean((burden-y)**2))),
            "rmse_median":None,"brier":float(np.mean((risk-z)**2)),
            "rps":None,"nll":None,"coverage80":None,"width80":None,
            "auroc":auc(risk.tolist(),z.astype(int).tolist())}

class ProperOrdinalWM(DistributionWM):
    """Categorical 0..24 distribution trained with CE + proper RPS and MSE.

    CE and RPS are proper scores; mean-specific MSE encourages useful
    conditional means. We do not falsely label these separate heads
    as a causal or autonomous world model.
    """
    def __init__(self,epochs=18,seed=SEED):
        super().__init__("categorical",epochs=epochs,seed=seed)
    def fit(self,fit,aux):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=DistributionNet()
        opt=torch.optim.AdamW(self.net.parameters(),lr=.0008,weight_decay=.025)
        X=sequence_features(fit,())
        site,y,z=labels(fit)
        y=torch.as_tensor(y,dtype=torch.long)
        classes=torch.arange(25)[None,:]
        if aux:
            AX=sequence_features(aux,())
            aux_targets=torch.tensor([r["aux_motor"] for r in aux],
                                     dtype=torch.long)
        for epoch in range(self.epochs):
            self.net.train()
            order=rng.permutation(len(fit))
            for start in range(0,len(order),32):
                ix=order[start:start+32]
                logits,_,_,_=self.net(*[v[ix] for v in X])
                p=torch.softmax(logits,dim=-1)
                cdf=torch.cumsum(p,dim=-1)
                truth=(classes>=y[ix,None]).to(cdf.dtype)
                ranked=((cdf-truth)**2).sum(dim=-1).mean()/24.
                expected=(p*classes.to(p.dtype)).sum(dim=-1)
                mse=((expected-y[ix].float())**2).mean()/25.
                ce=nn.functional.cross_entropy(logits,y[ix])
                loss=ce+1.5*ranked+.35*mse
                if aux:
                    jj=rng.integers(0,len(aux),min(32,len(ix)))
                    _,_,_,next_items=self.net(*[v[jj] for v in AX])
                    loss=loss+.12*nn.functional.cross_entropy(
                        next_items.reshape(-1,5),aux_targets[jj].reshape(-1),
                        ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                opt.step()
        self.net.eval()
        return self

def temperature_scale(p,temperature):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-10,1.)
    logit=np.log(p)/float(temperature)
    logit-=logit.max(axis=1,keepdims=True)
    value=np.exp(logit)
    return value/value.sum(axis=1,keepdims=True)

def select_temperature(model,dev,audit):
    dev_p=model.predict_distribution(dev)
    baseline_dev=probabilities_scores(dev_p,dev)
    # A proper score determines the single candidate temperature on DEV.
    ranked=[]
    for t in TEMPERATURES:
        test=probabilities_scores(temperature_scale(dev_p,t),dev)
        ranked.append((test["rps"],test["brier"],t))
    _,_,t=min(ranked)
    if t==1.:
        return 1.,{"candidate_temperature":1.,"accepted":False,
                   "reason":"dev_selected_identity",
                   "dev_before_rps":baseline_dev["rps"],
                   "dev_candidate_rps":baseline_dev["rps"],
                   "audit_proposals":0}
    selected_dev=probabilities_scores(temperature_scale(dev_p,t),dev)
    audit_p=model.predict_distribution(audit)
    before=probabilities_scores(audit_p,audit)
    after=probabilities_scores(temperature_scale(audit_p,t),audit)
    # Explicit once-only veto: not optimize test and not claim a free win.
    accepted=(selected_dev["rps"]+.0005<baseline_dev["rps"]
              and after["rps"]+.0005<before["rps"]
              and after["brier"]<=before["brier"]+.005
              and after["rmse_mean"]<=before["rmse_mean"]+.05
              and after["mae_median"]<=before["mae_median"]+.03)
    return (t if accepted else 1.),{
        "candidate_temperature":t,"accepted":bool(accepted),
        "reason":"audit_accept" if accepted else "audit_veto",
        "audit_proposals":1,"dev_before_rps":baseline_dev["rps"],
        "dev_candidate_rps":selected_dev["rps"],
        "audit_before_rps":before["rps"],"audit_candidate_rps":after["rps"],
        "audit_before_brier":before["brier"],
        "audit_candidate_brier":after["brier"],
        "audit_before_rmse_mean":before["rmse_mean"],
        "audit_candidate_rmse_mean":after["rmse_mean"],
        "audit_before_mae_median":before["mae_median"],
        "audit_candidate_mae_median":after["mae_median"]}

def paired_estimate(ref,cand,rows,measure,seed,rep=500):
    y,z=_targets(rows)
    def values(x):
        if isinstance(x,tuple):
            total,risk=x
            if measure=="mae":return np.abs(np.asarray(total)-y)
            if measure=="mse":return (np.asarray(total)-y)**2
            if measure=="brier":return (np.asarray(risk)-z)**2
            raise ValueError("Point predictions have no distributional RPS")
        p=validate_probability(x,rows)
        if measure=="mae":return np.abs(quantile(p,.5)-y)
        if measure=="mse":return (p@SUPPORT-y)**2
        if measure=="brier":return (1-p[:,0]-z)**2
        if measure=="rps":
            truth=(SUPPORT[None,:]>=y[:,None]).astype(float)
            return ((p.cumsum(axis=1)-truth)**2).sum(axis=1)/24.
        raise ValueError(measure)
    delta=values(cand)-values(ref)
    rng=np.random.default_rng(seed)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()
                   for _ in range(rep)])
    return {"metric":measure,"delta":float(delta.mean()),
            "ci_low":float(np.quantile(boot,.025)),
            "ci_high":float(np.quantile(boot,.975))}

def execute(args):
    if not 0<=args.arm_id<20 or not 4<=args.epochs<=80:
        raise ValueError("Bad arm or epochs")
    # Proven workaround for this cluster's pathological GRU backward.
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fold,variant=divmod(args.arm_id,4)
    root=args.data_root.resolve()
    out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing PPMI or original frozen TRAIN")
    if out==root or root in out.parents:raise ValueError("Unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,_,_=evidence_tables(root,people)
    metrics=[];paired=[];audits=[];coverage=[]
    for task in TASKS:
        subset=[p for p in people if p["cohort"]==task[0]]
        eligible,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,eligible,evidence,task)
        fit,dev,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,dev,audit,held)))<10:
            raise AssertionError("Expected fully populated heldout splits")
        fit_ids={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in fit_ids])
        seed=SEED+100*fold+task[1]
        ct=DirectWM((),style="balanced",epochs=args.epochs,seed=seed).fit(fit,aux)
        hgb=TimeTabular("hgb").fit(fit,())
        predictions={"ct_direct":ct.predict(held),
                     "motor_hgb":hgb.predict(held)}
        if variant>=1:
            categorical=DistributionWM("categorical",epochs=args.epochs,
                                       seed=seed).fit(fit,aux)
            hurdle=DistributionWM("hurdle",epochs=args.epochs,
                                  seed=seed).fit(fit,aux)
            predictions["categorical"]=categorical.predict_distribution(held)
            predictions["hurdle"]=hurdle.predict_distribution(held)
        if variant>=2:
            ordinal=ProperOrdinalWM(epochs=args.epochs,seed=seed).fit(fit,aux)
            predictions["ordinal_rps"]=ordinal.predict_distribution(held)
        if variant==3:
            T,verification=select_temperature(ordinal,dev,audit)
            predictions["ordinal_verified"]=temperature_scale(
                predictions["ordinal_rps"],T)
            audits.append({"cohort":task[0],"horizon":task[1],"fold":fold,
                           "selected_temperature":T,**verification})
        primary={0:"ct_direct",1:"categorical",2:"ordinal_rps",
                 3:"ordinal_verified"}[variant]
        for name,pred in predictions.items():
            evaluated=(point_scores(pred,held) if isinstance(pred,tuple)
                       else probabilities_scores(pred,held))
            metrics.append({"cohort":task[0],"horizon":task[1],
                "fold":fold,"arm":args.arm_id,"primary":int(name==primary),
                "method":name,"patients":len(held),"positive":int(_targets(held)[1].sum()),
                **evaluated})
        pairs=[("motor_hgb","ct_direct",("mae","mse","brier"))]
        for candidate in ("categorical","hurdle","ordinal_rps","ordinal_verified"):
            if candidate in predictions:
                pairs.append(("ct_direct",candidate,("mse","brier")))
                for reference in ("categorical","hurdle","ordinal_rps"):
                    if reference in predictions and reference!=candidate:
                        pairs.append((reference,candidate,("mae","mse","brier","rps")))
        for reference,candidate,measures in pairs:
            for measure in measures:
                paired.append({"cohort":task[0],"horizon":task[1],"fold":fold,
                    "arm":args.arm_id,"reference":reference,"candidate":candidate,
                    **paired_estimate(predictions[reference],predictions[candidate],
                                     held,measure,SEED+fold*100+task[1])})
        coverage.append({"cohort":task[0],"horizon":task[1],"fold":fold,
            "fit":len(fit),"dev":len(dev),"audit":len(audit),
            "heldout":len(held),"fit_aux_pairs":len(aux),"status":"complete"})
        write_csv(out/"method_metrics.csv",metrics)
        write_csv(out/"paired_bootstrap.csv",paired)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","last_complete_task":list(task),
            "completed_tasks":len(coverage)},indent=2)+"\n")
        print("PHASE79_TASK_COMPLETE",task,"arm",args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":79,"arm":args.arm_id,"fold":fold,"variant":variant,
        "completed_tasks":len(coverage),
        "duration_s":round(time.monotonic()-started,2),
        "limitations":[
          "Five folds of repeatedly used frozen Prodromal TRAIN plus PD exploratory",
          "No new independent patient confirmation; previous TEST is untouched",
          "Mean severity is used for RMSE, median for MAE",
          "Ordinal posterior evaluated by proper RPS and log score",
          "Brier is tied coherently to probability mass at score zero",
          "Tuning DEV and separate AUDIT are distinct from outer EVAL",
          "Calibration is not recursive self-improvement or Qwen clinical action",
          "No patient-level information written to output"],
        "finished_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE79_SUCCESS",args.arm_id,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=18)
    execute(p.parse_args())

if __name__=="__main__":main()
