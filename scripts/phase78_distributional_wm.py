#!/usr/bin/env python3
"""Phase78: true total-burden distributional WM with mean/median decisions.

5 original patient folds x 4 independent CPU arms.
0: matched CT-direct + HGB
1: two-part hurdle p(Y>0) and categorical p(Y|Y>0)
2: direct 25-class categorical p(Y)
3: identical candidate models, DEV selection + once-only AUDIT gate.

Only measured pre-cutoff 27 Motor items, identical folds; auxiliary
actual FIT-patient next-visit Motor. No fake scores, no patient exports.
"""
from __future__ import annotations
import argparse,csv,json,time
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
    DirectWM,TimeTabular,labels,score,paired_bootstrap,write_csv)

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
SUPPORT=torch.arange(25,dtype=torch.float32)

class DistributionNet(nn.Module):
    def __init__(self,h=80):
        super().__init__()
        self.observation=nn.Sequential(nn.Linear(54,64),nn.GELU())
        self.rate=nn.Parameter(torch.full((h,),-1.))
        self.cell=nn.GRUCell(64,h)
        self.fuse=nn.Sequential(nn.Linear(h+2,96),nn.LayerNorm(96),nn.GELU())
        self.categorical=nn.Linear(96,25)
        self.hurdle=nn.Linear(96,1)
        self.positive=nn.Linear(96,24)
        self.next_motor=nn.Linear(96,27*5)
    def forward(self,x,dt,valid,clinical,horizon):
        hidden=x.new_zeros((len(x),self.cell.hidden_size))
        decay=torch.nn.functional.softplus(self.rate)
        for t in range(x.shape[1]):
            next_h=self.cell(self.observation(x[:,t]),
                 hidden*torch.exp(-decay*dt[:,t,None]))
            hidden=torch.where(valid[:,t,None],next_h,hidden)
        z=self.fuse(torch.cat((hidden,horizon),dim=-1))
        return (self.categorical(z),self.hurdle(z).squeeze(-1),
                self.positive(z),
                self.next_motor(z).reshape(-1,27,5))

class DistributionWM:
    def __init__(self,kind,epochs=18,seed=SEED):
        if kind not in ("hurdle","categorical"):raise ValueError("Bad kind")
        self.kind=kind;self.epochs=epochs;self.seed=seed
    def fit(self,fit,aux):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        generator=np.random.default_rng(self.seed)
        self.net=DistributionNet()
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.0008,weight_decay=.025)
        inputs=sequence_features(fit,())
        _,burden,binary=labels(fit)
        y=torch.as_tensor(burden,dtype=torch.long)
        b=torch.as_tensor(binary,dtype=torch.float32)
        if aux:
            extra=sequence_features(aux,())
            extra_target=torch.tensor([r["aux_motor"] for r in aux],dtype=torch.long)
        for epoch in range(self.epochs):
            self.net.train()
            for start in range(0,len(fit),32):
                ix=generator.permutation(len(fit))[0:0] if False else None
                # Deterministic seeded shuffle once for each epoch.
                if start==0:order=generator.permutation(len(fit))
                ix=order[start:start+32]
                out,hit,pos,_=self.net(*[a[ix] for a in inputs])
                if self.kind=="categorical":
                    loss=nn.functional.cross_entropy(out,y[ix])
                else:
                    loss=nn.functional.binary_cross_entropy_with_logits(hit,b[ix])
                    positive=(y[ix]>0)
                    if positive.any():
                        loss=loss+nn.functional.cross_entropy(
                            pos[positive],y[ix][positive]-1)
                if aux:
                    jj=generator.integers(0,len(aux),min(32,len(ix)))
                    _,_,_,motor=self.net(*[a[jj] for a in extra])
                    loss=loss+.12*nn.functional.cross_entropy(
                        motor.reshape(-1,5),extra_target[jj].reshape(-1),
                        ignore_index=-100)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()
        self.net.eval()
        return self
    def predict_distribution(self,rows):
        with torch.inference_mode():
            logits,hit,positive,_=self.net(*sequence_features(rows,()))
            if self.kind=="categorical":
                p=torch.softmax(logits,-1)
            else:
                event=torch.sigmoid(hit).unsqueeze(-1)
                conditional=torch.softmax(positive,-1)
                p=torch.cat((1-event,event*conditional),dim=-1)
        a=p.numpy()
        if a.shape!=(len(rows),25):raise AssertionError("Distribution shape")
        if not np.isfinite(a).all():raise AssertionError("Invalid distribution")
        return a
    def predict(self,rows,decision="mean"):
        prob=self.predict_distribution(rows)
        if decision=="mean":
            severity=prob@np.arange(25)
        elif decision=="median":
            severity=(np.cumsum(prob,axis=1)<.5).sum(axis=1).astype(float)
        else:raise ValueError("Must use mean or median")
        return np.asarray(severity),np.asarray(1-prob[:,0])

def select_verification(candidates,dev,audit):
    """Select one of four allowed distribution/functional pairs on DEV.

    AUDIT sees ONLY selected best candidate and incumbent CT mean.
    Compare on identical people. A strict RMSE & Brier guardrail rejects
    improved MAE obtained solely by sacrificing tail prediction.
    """
    incumbent=candidates["ct_direct"]
    dev_scores={name:score(model.predict(dev) if isinstance(model,DistributionWM)
                          else model.predict(dev),dev)
                for name,model in candidates.items() if name!="ct_direct"}
    # Other decision functions are wrapped explicitly in DecisionModel.
    proposed=min(dev_scores,key=lambda k:(dev_scores[k]["mae"],k))
    baseline_dev=score(incumbent.predict(dev),dev)
    before=score(incumbent.predict(audit),audit)
    after=score(candidates[proposed].predict(audit),audit)
    accepted=(dev_scores[proposed]["mae"]+.01<baseline_dev["mae"] and
              after["mae"]+.01<before["mae"] and
              after["rmse"]<=before["rmse"]+.05 and
              after["brier"]<=before["brier"]+.01)
    return (proposed if accepted else "ct_direct"),{
        "candidate":proposed,"accepted":bool(accepted),"audit_proposals":1,
        "dev_before_mae":baseline_dev["mae"],
        "dev_candidate_mae":dev_scores[proposed]["mae"],
        "audit_before_mae":before["mae"],"audit_candidate_mae":after["mae"],
        "audit_before_rmse":before["rmse"],"audit_candidate_rmse":after["rmse"],
        "audit_before_brier":before["brier"],"audit_candidate_brier":after["brier"]}

class DecisionModel:
    def __init__(self,wm,decision):
        self.wm=wm;self.decision=decision
    def predict(self,rows):
        return self.wm.predict(rows,self.decision)

def execute(a):
    held,arm=divmod(a.arm_id,4)
    out=a.report_dir.resolve()
    root=a.data_root.resolve()
    if root==out or root in out.parents or not a.manifest_db.is_file():
        raise ValueError("Unsafe output or missing input")
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    people,_=load_cohorts(root,a.manifest_db)
    evidence,_,_=evidence_tables(root,people)
    metrics=[];contrasts=[];audits=[];coverage=[]
    for task in TASKS:
        subset=[p for p in people if p["cohort"]==task[0]]
        target,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,target,evidence,task)
        fit,dev,audit,eval_rows=split_data(rows,task,held)
        if min(map(len,(fit,dev,audit,eval_rows)))<10:continue
        seed=SEED+held*100+task[1]
        aux=build_auxiliary([p for p in subset
                             if p["pid"] in {r["pid"] for r in fit}])
        base=DirectWM((),style="balanced",epochs=a.epochs,seed=seed).fit(fit,aux)
        hgb=TimeTabular("hgb").fit(fit,())
        candidates={"ct_direct":base}
        if arm>=1:
            hurdle=DistributionWM("hurdle",epochs=a.epochs,seed=seed).fit(fit,aux)
            candidates["hurdle_mean"]=DecisionModel(hurdle,"mean")
            candidates["hurdle_median"]=DecisionModel(hurdle,"median")
        if arm>=2:
            categorical=DistributionWM("categorical",epochs=a.epochs,
                                       seed=seed).fit(fit,aux)
            candidates["categorical_mean"]=DecisionModel(categorical,"mean")
            candidates["categorical_median"]=DecisionModel(categorical,"median")
        if arm==3:
            chosen,ver=select_verification(candidates,dev,audit)
            audits.append({"cohort":task[0],"horizon":task[1],
                           "fold":held,**ver})
            candidates["verified_decision"]=candidates[chosen]
        predictions={"motor_hgb":hgb.predict(eval_rows)}
        predictions.update({k:model.predict(eval_rows) for k,model
                           in candidates.items()})
        main={0:"ct_direct",1:"hurdle_median",2:"categorical_median",
              3:"verified_decision"}[arm]
        for name,pred in predictions.items():
            metrics.append({"cohort":task[0],"horizon":task[1],
                "outer_fold":held,"arm_id":a.arm_id,
                "primary":int(name==main),"method":name,
                "patients":len(eval_rows),"positive":int(labels(eval_rows)[2].sum()),
                **score(pred,eval_rows)})
        for name,pred in predictions.items():
            if name=="motor_hgb":continue
            for reference in ("motor_hgb","ct_direct"):
                if reference==name:continue
                contrasts.append({"cohort":task[0],"horizon":task[1],
                    "fold":held,"reference":reference,"candidate":name,
                    **paired_bootstrap(predictions[reference],pred,eval_rows,
                                       SEED+held+task[1])})
        coverage.append({"cohort":task[0],"horizon":task[1],"fold":held,
            "fit":len(fit),"dev":len(dev),"audit":len(audit),
            "eval":len(eval_rows),"aux_pairs":len(aux),"status":"complete"})
        write_csv(out/"method_metrics.csv",metrics)
        write_csv(out/"paired_bootstrap.csv",contrasts)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE78_TASK_COMPLETE",task,a.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":78,"arm_id":a.arm_id,"outer_fold":held,"variant":arm,
        "completed_tasks":len(coverage),"elapsed_sec":time.monotonic()-start,
        "protocol":"True FIT-only Motor history and future six-site burden; distributional mean/median with once-only DEV/AUDIT gate",
        "limitations":["Frozen repeatedly developed Prodromal TRAIN only",
         "No LLM or autonomous RSI","No additional nonmotor modalities",
         "Hurdle model approximates conditional positive burden by 24-category bins",
         "Nonzero risk and severity share coherent distribution",
         "No independent final TEST or clinical generalization claim"],
        "finished_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({"status":"complete",
        "completed_tasks":len(coverage)},indent=2)+"\n")
    print(json.dumps({"status":"completed","arm":a.arm_id,
                      "completed_tasks":len(coverage),"out":str(out)}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=18)
    a=p.parse_args()
    if not 0<=a.arm_id<20 or not 4<=a.epochs<=80:
        raise ValueError("Bad Phase78 arm or epochs")
    execute(a)
if __name__=="__main__":
    main()
