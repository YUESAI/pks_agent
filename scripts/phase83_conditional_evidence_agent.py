#!/usr/bin/env python3
"""Phase83: risk-conditional clinical evidence and cross-fitted test utility.

Five frozen outer patient folds x four standalone CPU policies:
0 Motor q0 + old marginal evidence fixed baseline.
1 Risk-conditional clinical emission + fixed choice (information update only).
2 Risk-conditional emission + FIT OOF value-of-query abstention.
3 Same update and value model plus real Qwen constrained to supported
  tests; a once-only independent DEV/AUDIT gate chooses policy revision.

All real optional clinical values are BL measured values released only
after choice. Qwen receives names/estimated gains, never values or labels.
The original consumed final TEST remains untouched. Exploratory TRAIN
re-use, not causal WM or autonomous RSI.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase72_ctlatent_wm import build_auxiliary
from scripts.phase73_matched_longitudinal_agent import (
    data_rows,split_data,chosen_optional_keys)
from scripts.phase76_reliability_gated_wm import write_csv
from scripts.phase78_distributional_wm import DistributionWM
from scripts.phase79_proper_scored_ordinal_wm import (
    probabilities_scores,paired_estimate)
from scripts.phase80_active_evidence_agent import (
    ClinicalEvidenceChannel,choose_fixed,severity_bin,measured_bucket)
from scripts.phase82_value_guard_oracle_agent import oracle_headroom

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
ALPHA=.5  # fixed shrinkage of clinical likelihood ratio, never tuned on EVAL
COST=.00025
SHRINK=16.
MIN_SUPPORT=8
GAIN_MARGIN=.00005
GROUPS=np.array([severity_bin(y) for y in range(25)],dtype=int)

def risk_band(p):
    risk=1.-float(p[0])
    return 0 if risk<.33 else 1 if risk<.66 else 2

def oof_fold(pid,task):
    salted=f"pks-phase83|{SEED}|{task[0]}|{task[1]}|{pid}"
    return int(hashlib.sha256(salted.encode()).hexdigest()[:12],16)%3

def individual_rps(prob,target):
    truth=(np.arange(25)>=int(target)).astype(float)
    return float(((np.cumsum(prob)-truth)**2).sum()/24.)

class ConditionalChannel:
    """P(test bucket | future severity stratum, existing Motor risk band).

    Input risk band is an approximation to conditioning on MOTOR HISTORY;
    explicitly not a calibrated full P(test|Y,complete_history).
    Ratios are smoothed toward P(test|risk band), then clipped.
    """
    def __init__(self,rows,prior,keys):
        self.keys=tuple(keys)
        self.num=Counter()
        self.table={}
        if len(rows)!=len(prior):
            raise AssertionError("FIT prior alignment")
        count={key:np.zeros((3,4,5),dtype=np.float64)
               for key in self.keys}
        for r,p in zip(rows,prior):
            y=sum(r["site"])
            if not 0<=y<=24:raise AssertionError("Invalid training burden")
            band=risk_band(p)
            category=severity_bin(y)
            for key in self.keys:
                if key not in r["optional"]:continue
                v=measured_bucket(key,r["optional"][key])
                count[key][band,category,v]+=1.
                self.num[key]+=1
        for key in self.keys:
            arr=count[key]
            conditional=np.ones((3,4,5),dtype=np.float64)
            for band in range(3):
                # Strong class smoothing for small FIT; marginal uses
                # only actually observed values, plus symmetric prior.
                panel=arr[band]
                marginal=(panel.sum(axis=0)+2.)/(panel.sum()+10.)
                for s in range(4):
                    n=panel[s].sum()
                    conditional[band,s]=(panel[s]+SHRINK*marginal)/(n+SHRINK)
            self.table[key]=conditional

    def allowed(self,row):
        return tuple(key for key in self.keys
                     if key in row["optional"] and self.num[key]>=MIN_SUPPORT)

    def posterior(self,prior,key,value,alpha=ALPHA):
        if key not in self.table:raise ValueError("Not in FIT alphabet")
        p=np.asarray(prior,dtype=float)
        if p.shape!=(25,) or not np.isfinite(p).all():
            raise AssertionError("Invalid Motor prior")
        b=risk_band(p)
        v=measured_bucket(key,value)
        distribution=self.table[key][b,:,v]
        marginal=float((p*np.take(distribution,GROUPS)).sum())
        if marginal<=0:raise AssertionError("Invalid evidence likelihood")
        # Relative evidence conditional on Motor posterior. The
        # constant marginal cancels under final normalization.
        ratio=np.clip(distribution[GROUPS]/marginal,.25,4.)
        q=p*ratio**float(alpha)
        q/=q.sum()
        if not np.isfinite(q).all():raise AssertionError("Invalid update")
        return q

class OOFConditionalValue:
    def __init__(self,keys):
        self.keys=tuple(keys)
        self.sample=defaultdict(list)
        self.splits=[]
        self.fits=Counter()
    def fit(self,fit,subset,task,fold,epochs):
        if len({r["pid"] for r in fit})!=len(fit):
            raise AssertionError("Duplicate training subject")
        partitions=np.array([oof_fold(r["pid"],task) for r in fit])
        # First produce genuine OOF Motor predictions.
        motor_oof=np.full((len(fit),25),np.nan)
        for k in range(3):
            train_idx=np.flatnonzero(partitions!=k)
            valid_idx=np.flatnonzero(partitions==k)
            if not len(valid_idx) or len(train_idx)<20:
                raise AssertionError("Invalid inner split")
            train=[fit[i] for i in train_idx]
            validation=[fit[i] for i in valid_idx]
            ids={r["pid"] for r in train}
            aux=build_auxiliary([p for p in subset if p["pid"] in ids])
            wm=DistributionWM("categorical",epochs=epochs,
                   seed=SEED+2000+100*fold+task[1]+k).fit(train,aux)
            motor_oof[valid_idx]=wm.predict_distribution(validation)
            self.splits.append([len(train_idx),len(valid_idx)])
        if not np.isfinite(motor_oof).all():
            raise AssertionError("OOF Motor predictions incomplete")
        self.fit_priors=motor_oof.copy()
        # Second cross-fitting layer for the conditional test channel:
        # participant y is excluded both from the Motor WM used for
        # their OOF p0 and from their conditional emission.
        for k in range(3):
            fit_idx=np.flatnonzero(partitions!=k)
            val_idx=np.flatnonzero(partitions==k)
            # Critical: the all-FIT OOF priors on fit_idx may have been
            # generated by models trained on val_idx outcomes. Those
            # priors cannot be used to fit a channel evaluated on val_idx.
            # Fit a SEPARATE WM using only fit_idx people and predict
            # fit_idx to construct a leak-isolated, in-sample channel.
            # These are in-sample emission priors, NOT nested OOF priors;
            # the held person's outcome cannot influence this channel.
            inner_train=[fit[i] for i in fit_idx]
            ids={r["pid"] for r in inner_train}
            inner_aux=build_auxiliary(
                [person for person in subset if person["pid"] in ids])
            emitter_wm=DistributionWM("categorical",epochs=epochs,
                seed=SEED+5000+100*fold+task[1]+k).fit(
                    inner_train,inner_aux)
            channel=ConditionalChannel(inner_train,
                emitter_wm.predict_distribution(inner_train),self.keys)
            for i in val_idx:
                record=fit[i]
                p=motor_oof[i]
                y=sum(record["site"])
                base=individual_rps(p,y)
                for key in channel.allowed(record):
                    q=channel.posterior(p,key,record["optional"][key])
                    gain=base-individual_rps(q,y)
                    b=risk_band(p)
                    self.sample[(key,b)].append(gain)
                    self.sample[(key,-1)].append(gain)
                    self.fits[key]+=1
        return self
    def estimated_gain(self,key,band):
        values=self.sample.get((key,band),())
        if len(values)<10:values=self.sample.get((key,-1),())
        if len(values)<10:return -float("inf")
        x=np.asarray(values,dtype=float)
        shrunk=float(x.sum()/(len(x)+20.))
        penalty=float(.5*np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else 1.
        return shrunk-penalty
    def suggestions(self,record,prior,channel,cost=COST):
        b=risk_band(prior)
        return {key:self.estimated_gain(key,b) for key in channel.allowed(record)}
    def diagnostics(self):
        return {"oof_fit_splits":self.splits,
                "value_example_count":sum(self.fits.values()),
                "tested_keys":len(self.fits)}

def fixed(rows,priors,channel,kind="fixed"):
    out=[];stats=Counter()
    for row,p in zip(rows,priors):
        options=channel.allowed(row)
        if not options:
            out.append(p.copy());stats["no_test"]+=1
            continue
        if kind=="fixed":
            key=choose_fixed(options,channel,p)
        else:raise ValueError(kind)
        out.append(channel.posterior(p,key,row["optional"][key]))
        stats["released"]+=1
    return np.stack(out),stats

def value_query(rows,priors,channel,value,cost=COST):
    out=[];stats=Counter()
    for row,p in zip(rows,priors):
        choices=value.suggestions(row,p,channel,cost)
        good=sorted(((g,k) for k,g in choices.items() if g>cost),
                    key=lambda x:(-x[0],x[1]))
        if not good:
            out.append(p.copy());stats["abstain"]+=1
            continue
        key=good[0][1]
        out.append(channel.posterior(p,key,row["optional"][key]))
        stats["released"]+=1
    return np.stack(out),stats

def qwen_query(rows,priors,channel,value,doctor,horizon,cost=COST):
    out=[];stats=Counter()
    for row,p in zip(rows,priors):
        choices=value.suggestions(row,p,channel,cost)
        supported=sorted(((g,k) for k,g in choices.items()
                          if g>cost),key=lambda x:(-x[0],x[1]))
        if not supported:
            out.append(p.copy());stats["hard_abstain"]+=1
            continue
        allowed=[k for _,k in supported]
        gains={k:round(float(g),6) for g,k in supported}
        messages=[
         {"role":"system","content":
          "Research-only retrospective test choice. You may select one "
          "allowed baseline test name or ABSTAIN. No test values or future "
          "outcomes are known. NEVER fabricate a clinical observation. "
          'Return exactly JSON {"ask":"NAME_OR_ABSTAIN"}.'},
         {"role":"user","content":
          "Forecast future Motor score at "+str(horizon)+
          " months. Valid already-recorded baseline test NAMES: "+
          json.dumps(allowed)+
          ". FIT-crossvalidated incremental RPS improvement estimates: "+
          json.dumps(gains,sort_keys=True)+
          ". Motor-model probability of future nonzero score (estimate): "+
          str(round(1.-float(p[0]),5))+
          '. You may choose ABSTAIN. Output JSON {"ask":"NAME_OR_ABSTAIN"}.'}]
        reply=doctor.generate(messages)
        selected=extract_action(reply,set(allowed)|{"ABSTAIN"})
        stats["qwen_calls"]+=1
        if selected is None:
            stats["invalid"]+=1
            selected="ABSTAIN"
        if selected=="ABSTAIN":
            stats["llm_abstain"]+=1
            out.append(p.copy());continue
        out.append(channel.posterior(p,selected,row["optional"][selected]))
        stats["released"]+=1
    return np.stack(out),stats

def verify(o_dev,q_dev,o_audit,q_audit,dev,audit,
           o_d_count,q_d_count,o_a_count,q_a_count):
    def metrics(p,rows,count):
        s=probabilities_scores(p,rows)
        return s,s["rps"]+COST*count/len(rows)
    od,odn=metrics(o_dev,dev,o_d_count)
    qd,qdn=metrics(q_dev,dev,q_d_count)
    oa,oan=metrics(o_audit,audit,o_a_count)
    qa,qan=metrics(q_audit,audit,q_a_count)
    accepted=(qdn+GAIN_MARGIN<odn and
              qan+GAIN_MARGIN<oan and
              qa["mae_median"]<=oa["mae_median"]+.03 and
              qa["rmse_mean"]<=oa["rmse_mean"]+.05 and
              qa["brier"]<=oa["brier"]+.005)
    return bool(accepted),{
        "accepted":bool(accepted),"candidate":"qwen_conditional",
        "incumbent":"oof_conditional","audit_proposals":1,
        "dev_net_rps_before":odn,"dev_net_rps_candidate":qdn,
        "audit_net_rps_before":oan,"audit_net_rps_candidate":qan,
        "audit_mae_before":oa["mae_median"],
        "audit_mae_candidate":qa["mae_median"]}

def combine_qwen_stats(*parts):
    """Merge Counter-like Qwen statistics even when no calls were made.

    A Qwen policy can have zero supported tests for a task, so the
    Counter has no `qwen_calls` key. Converting to dict first and then
    incrementing an absent key caused all five Qwen jobs to crash.
    """
    wanted=("qwen_calls","released","hard_abstain",
            "llm_abstain","invalid")
    return {key:sum(int(part.get(key,0)) for part in parts)
            for key in wanted}


def run(a):
    if not 0<=a.arm_id<20 or not 4<=a.epochs<=80:
        raise ValueError("Bad arm or epoch")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fold,variant=divmod(a.arm_id,4)
    root=a.data_root.resolve();out=a.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not a.manifest_db.is_file():
        raise FileNotFoundError("Missing source or frozen manifest")
    if out==root or root in out.parents:raise ValueError("Unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    (out/"progress.json").write_text(
        json.dumps({"status":"started","arm":a.arm_id})+"\n")
    started=time.monotonic()
    people,_=load_cohorts(root,a.manifest_db)
    evidence,qc,_=evidence_tables(root,people)
    doctor=(TransformersDoctor(a.model_dir,max_new_tokens=32,device="cpu")
            if variant==3 else None)
    methods=[];contrasts=[];audit_log=[];coverage=[]
    for task in TASKS:
        t0=time.monotonic()
        subset=[p for p in people if p["cohort"]==task[0]]
        true,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,true,evidence,task)
        fit,dev,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,dev,audit,held)))<10:
            raise AssertionError("Empty person split")
        keys=chosen_optional_keys(fit)
        fid={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in fid])
        wm=DistributionWM("categorical",epochs=a.epochs,
                          seed=SEED+100*fold+task[1]).fit(fit,aux)
        priors=wm.predict_distribution(held)
        marginal=ClinicalEvidenceChannel(fit,keys)
        # Fixed names and true values, same prediction model, but two
        # different conditional-assumption likelihood updates.
        old=[]
        for r,p in zip(held,priors):
            allowed=marginal.allowed(r)
            key=choose_fixed(allowed,marginal,p)
            old.append(marginal.posterior(p,key,r["optional"][key],1.)
                       if key else p.copy())
        outcomes={"motor_q0":priors,"marginal_fixed":np.stack(old)}
        counts={"motor_q0":0,
                "marginal_fixed":sum(bool(marginal.allowed(r)) for r in held)}
        controls={}
        if variant>=1:
            # This channel requires FIT-only OOF priors for its
            # training risk bands; outcome label never from DEV/AUDIT.
            value=OOFConditionalValue(keys).fit(
                fit,subset,task,fold,a.epochs)
            # Important: build channel on the identical OOF FIT priors
            # rather than outer-EVAL or in-sample Motor FIT prediction.
            channel=ConditionalChannel(
                fit,value.fit_priors,keys)
            fixed_p,fs=fixed(held,priors,channel)
            outcomes["conditional_fixed"]=fixed_p
            counts["conditional_fixed"]=fs["released"]
            controls["fixed"]=fs
        if variant>=2:
            val_p,vs=value_query(held,priors,channel,value,COST)
            outcomes["conditional_oof"]=val_p
            counts["conditional_oof"]=vs["released"]
            controls["oof"]=vs
        if variant==3:
            q_p,qs=qwen_query(held,priors,channel,value,doctor,task[1])
            outcomes["conditional_qwen"]=q_p
            counts["conditional_qwen"]=qs["released"]
            d0=wm.predict_distribution(dev)
            a0=wm.predict_distribution(audit)
            od,os=value_query(dev,d0,channel,value)
            oa,ass=value_query(audit,a0,channel,value)
            qd,qds=qwen_query(dev,d0,channel,value,doctor,task[1])
            qa,qas=qwen_query(audit,a0,channel,value,doctor,task[1])
            accepted,decision=verify(
                od,qd,oa,qa,dev,audit,
                os["released"],qds["released"],
                ass["released"],qas["released"])
            outcomes["verified_policy"]=(
                q_p if accepted else outcomes["conditional_oof"])
            counts["verified_policy"]=(
                qs["released"] if accepted else vs["released"])
            # Some folds have zero eligible Qwen questions; Counter()
            # omits those keys. Aggregate explicitly with zero defaults.
            controls["qwen"]=combine_qwen_stats(qs,qds,qas)
            audit_log.append({"cohort":task[0],"horizon":task[1],
                              "fold":fold,**decision})
        primary={0:"motor_q0",1:"conditional_fixed",
                 2:"conditional_oof",3:"verified_policy"}[variant]
        for name,p in outcomes.items():
            s=probabilities_scores(p,held)
            methods.append({"cohort":task[0],"horizon":task[1],
                "fold":fold,"arm":a.arm_id,"model":name,
                "primary":int(name==primary),"patients":len(held),
                "queries":counts[name],"query_rate":counts[name]/len(held),
                "net_rps_cost250":s["rps"]+COST*counts[name]/len(held),**s})
        pairs=[("motor_q0","marginal_fixed"),
               ("motor_q0","conditional_fixed"),
               ("marginal_fixed","conditional_fixed"),
               ("conditional_fixed","conditional_oof"),
               ("conditional_oof","conditional_qwen"),
               ("conditional_oof","verified_policy")]
        for reference,candidate in pairs:
            if reference not in outcomes or candidate not in outcomes:continue
            for metric in ("rps","mae","mse","brier"):
                contrasts.append({"cohort":task[0],"horizon":task[1],
                  "fold":fold,"arm":a.arm_id,"reference":reference,
                  "candidate":candidate,
                  **paired_estimate(outcomes[reference],
                      outcomes[candidate],held,metric,SEED+fold*100+task[1])})
        row={"cohort":task[0],"horizon":task[1],"fold":fold,
             "fit":len(fit),"dev":len(dev),"audit":len(audit),
             "eval":len(held),"time_sec":round(time.monotonic()-t0,2),
             "status":"complete"}
        if variant>=1:row.update({
            "oof_splits":str(value.splits),
            "oof_test_instances":sum(value.fits.values())})
        for name,s in controls.items():
            for key in ("released","abstain","qwen_calls","llm_abstain",
                        "invalid"):
                row[name+"_"+key]=s.get(key,0)
        coverage.append(row)
        write_csv(out/"method_metrics.csv",methods)
        write_csv(out/"paired_bootstrap.csv",contrasts)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audit_log,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE83_TASK_COMPLETE",task,a.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":83,"arm_id":a.arm_id,"fold":fold,"variant":variant,
        "completed_tasks":len(coverage),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "qwen_prompts":doctor.total_prompts if doctor else 0,
        "scope":[
          "Exploratory re-used frozen Prodromal TRAIN and strict PD",
          "Previously consumed final TEST untouched",
          "True pre-cutoff Motor and BL optional evidence only",
          "FIT-only emission refitted within inner train to exclude held inner labels; emission training priors are in-sample",
          "Risk-band grouping is a coarse approximation to full conditional dependency",
          "One clinical test, value released only after selecting its name",
          "DEV/AUDIT gate is checkpoint-policy proxy, not autonomous RSI",
          "No patient-level identifiers, scores, prompts or forecasts exported",
          "No prospective clinical ordering or treatment effect claimed"],
        "generated_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps(
        {"status":"complete","completed_tasks":len(coverage)})+"\n")
    print("PHASE83_SUCCESS",a.arm_id,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",required=True,type=int)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--epochs",type=int,default=18)
    a=p.parse_args()
    if a.arm_id%4==3 and a.model_dir is None:
        raise ValueError("Qwen arm requires offline --model-dir")
    run(a)
if __name__=="__main__":
    main()
