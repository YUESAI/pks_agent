#!/usr/bin/env python3
"""Phase80: matched, real-Qwen active clinical evidence on a frozen Motor WM.

Five outer patient-disjoint folds x four independent CPU arms:
 0: no optional clinical test, Motor categorical WM q=0
 1: one actually recorded baseline clinical test, fixed FIT-only priority
 2: one actually recorded baseline test, WM expected-information selection
 3: real offline Qwen selects one test, then independent DEV/AUDIT verifier
    may revise the allowlisted test likelihood-update strength.

Each job evaluates its own same-fold q0, fixed and WM-guided references.
No item values, outcomes, patient identifiers or model predictions labelled
as real clinical observations enter LLM prompts. The 'tool' releases ONLY
the chosen pre-existing measured BL nonmotor value.

Scope: retrospective active-feature acquisition on frozen original
Prodromal TRAIN and exploratory PD. Not causal treatment, new data
collection, autonomous code-rewriting RSI or clinical deployment.
"""
from __future__ import annotations

import argparse,hashlib,json,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables,bucket
from scripts.phase72_ctlatent_wm import build_auxiliary
from scripts.phase73_matched_longitudinal_agent import (
    data_rows,split_data,chosen_optional_keys)
from scripts.phase76_reliability_gated_wm import write_csv
from scripts.phase78_distributional_wm import DistributionWM
from scripts.phase79_proper_scored_ordinal_wm import (
    probabilities_scores,paired_estimate)

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
BINS=np.asarray([0,1,2,3],dtype=np.int64)
SUPPORT=np.arange(25)
ALPHAS=(0.,.25,.5,.75,1.,1.25)
SMOOTH=12.

def severity_bin(total):
    return 0 if total==0 else 1 if total==1 else 2 if total<=3 else 3

def measured_bucket(key,value):
    """Only actual measured BL clinical values, never synthetic tools."""
    try:
        v=bucket(key,value)
    except (TypeError,ValueError) as exc:
        raise ValueError("Invalid measured optional clinical value") from exc
    if not isinstance(v,(int,np.integer)) or not 0<=v<=4:
        raise AssertionError("Clinical value bucket out of range")
    return int(v)

class ClinicalEvidenceChannel:
    """Smoothed FIT-only P(verified BL test value | future burden stratum).

    A fixed pretrained Motor posterior supplies prior p(Y). Only the
    recorded chosen BL test value updates posterior. This approximate
    Bayes update is NOT a causal intervention or calibrated joint model.
    """
    def __init__(self,fit,keys):
        self.keys=tuple(keys)
        self.counts={}
        self.support={}
        self.likelihood={}
        total_y=np.asarray([sum(r["site"]) for r in fit],dtype=int)
        if ((total_y<0)|(total_y>24)).any():
            raise AssertionError("Invalid FIT burden")
        for key in self.keys:
            cells=np.zeros((4,5),dtype=float)
            for row,y in zip(fit,total_y):
                if key not in row["optional"]:continue
                v=measured_bucket(key,row["optional"][key])
                cells[severity_bin(int(y)),v]+=1.
            n=int(cells.sum())
            self.support[key]=n
            # Preserve a positive marginal for rare and absent values.
            marginal=(cells.sum(axis=0)+1.)/(n+5.)
            # Each future-severity class is shrunk towards the global
            # empirically measured distribution, never a fabricated test.
            conditional=(cells+SMOOTH*marginal[None,:])/(
                cells.sum(axis=1)[:,None]+SMOOTH)
            if not np.allclose(conditional.sum(axis=1),1.):
                raise AssertionError("Emission not normalized")
            self.likelihood[key]=conditional
        self.classes=np.array([severity_bin(int(y)) for y in SUPPORT])
    def allowed(self,row):
        return tuple(k for k in self.keys if k in row["optional"]
                     and self.support.get(k,0)>=8)
    def posterior(self,prior,key,value,alpha=1.):
        p=np.asarray(prior,dtype=float)
        if p.shape!=(25,) or not np.isfinite(p).all():
            raise AssertionError("Expected finite 0..24 prior")
        if key not in self.likelihood:raise ValueError("Unallowlisted test")
        observed=measured_bucket(key,value)
        probs=self.likelihood[key][self.classes,observed]
        weight=p*np.power(np.maximum(probs,1e-12),float(alpha))
        total=weight.sum()
        if total<=0 or not np.isfinite(total):
            raise AssertionError("Invalid Bayesian evidence update")
        return weight/total
    def expected_information(self,prior,key):
        """FIT-only approximate expected KL, no current actual test value."""
        emission=self.likelihood[key][self.classes,:]
        prior=np.asarray(prior,dtype=float)
        likelihood=prior@emission
        gain=0.
        for v in range(5):
            if likelihood[v]<=0:continue
            posterior=prior*emission[:,v]/likelihood[v]
            good=(posterior>0)&(prior>0)
            gain+=float(likelihood[v]*np.sum(
                posterior[good]*np.log(posterior[good]/prior[good])))
        return max(0.,float(gain))

def choose_fixed(allowed,channel,prior):
    # Prespecified FIT-only priority, not patient test outcome.
    return next((key for key in channel.keys if key in allowed),None)

def choose_greedy(allowed,channel,prior):
    if not allowed:return None
    return min(allowed,key=lambda k:(-channel.expected_information(prior,k),k))

def choose_qwen(allowed,channel,prior,doctor,horizon):
    """LLM gets only permitted *names*, not their measured values or y."""
    if not allowed:return None,False,False
    utility={k:round(channel.expected_information(prior,k),6)
             for k in allowed}
    risk=round(float(1-prior[0]),5)
    messages=[{"role":"system","content":
        "Research-only clinical test-selection agent. Choose exactly one "
        "allowed previously measured baseline test name. Never infer a "
        "patient's test value or invent a clinical observation. Respond "
        "with one JSON object and no prose: {\"ask\":\"NAME\"}."},
        {"role":"user","content":
        "Predicting future motor burden at "+str(horizon)+
        " months. Allowed baseline test names: "+json.dumps(allowed)+
        ". Current MOTOR-only model nonzero-risk estimate (NOT a "
        "clinical observation): "+str(risk)+
        ". FIT-estimated expected information gains (NOT measured "
        "test scores): "+json.dumps(utility,sort_keys=True)+
        '. Select exactly one allowed test: {"ask":"NAME"}.'}]
    answer=doctor.generate(messages)
    item=extract_action(answer,set(allowed))
    invalid=item is None
    fallback=choose_greedy(allowed,channel,prior)
    return (item if item is not None else fallback),invalid,item==fallback

def acquire(rows,priors,channel,policy,alpha=1.,doctor=None,horizon=12):
    if policy not in ("none","fixed","wm","qwen"):
        raise ValueError("Unrecognized evidence-acquisition policy")
    if len(rows)!=len(priors):raise AssertionError("Row/prior misalignment")
    outputs=[];keys=[];selected=Counter();stats=Counter()
    for row,p in zip(rows,priors):
        allowed=channel.allowed(row)
        if not allowed or policy=="none":
            outputs.append(np.asarray(p,dtype=float).copy())
            keys.append(None)
            if not allowed:stats["no_available_test"]+=1
            continue
        stats["eligible_queries"]+=1
        if policy=="fixed":choice=choose_fixed(allowed,channel,p)
        elif policy=="wm":choice=choose_greedy(allowed,channel,p)
        else:
            if doctor is None:raise AssertionError("Qwen backend missing")
            choice,invalid,agree=choose_qwen(
                allowed,channel,p,doctor,horizon)
            stats["qwen_calls"]+=1
            stats["invalid_action"]+=int(invalid)
            stats["agreement_with_wm"]+=int(agree)
        if choice not in allowed:raise AssertionError("Invalid tool name")
        # Only here is the genuinely recorded value released. The LLM
        # and selection policies never receive it beforehand.
        observed=row["optional"][choice]
        outputs.append(channel.posterior(p,choice,observed,alpha))
        selected[choice]+=1
        keys.append(choice)
        stats["real_test_releases"]+=1
    return np.stack(outputs),keys,{
        "eligible_queries":stats["eligible_queries"],
        "real_test_releases":stats["real_test_releases"],
        "no_available_test":stats["no_available_test"],
        "qwen_calls":stats["qwen_calls"],
        "invalid_action":stats["invalid_action"],
        "agreement_with_wm":stats["agreement_with_wm"],
        "top_test_counts":dict(selected.most_common(5)),
    }

def replay(rows,priors,keys,channel,alpha):
    """Replay EXACT same chosen tests; no second LLM call or outcome use."""
    if not len(rows)==len(priors)==len(keys):
        raise AssertionError("Replay alignment")
    return np.stack([
        channel.posterior(p,k,r["optional"][k],alpha)
        if k is not None else np.asarray(p,dtype=float).copy()
        for r,p,k in zip(rows,priors,keys)])

def verify_update(dev,audit,dev_priors,audit_priors,dev_keys,audit_keys,channel):
    """Compare Qwen q1(alpha=1) against one allowlisted scalar revision.

    DEV chooses alpha. AUDIT is queried exactly once after candidate
    selection, with no EVAL access. Identity is a legitimate decision.
    """
    incumbent_dev=replay(dev,dev_priors,dev_keys,channel,1.)
    base_dev=probabilities_scores(incumbent_dev,dev)
    ranked=[]
    for alpha in ALPHAS:
        proposed=replay(dev,dev_priors,dev_keys,channel,alpha)
        stat=probabilities_scores(proposed,dev)
        ranked.append((stat["rps"],stat["brier"],abs(alpha-1.),alpha))
    _,_,_,candidate=min(ranked)
    if candidate==1.:
        return 1.,{"accepted":False,"proposal":1.,
            "reason":"dev_prefers_incumbent","audit_proposals":0,
            "dev_rps_before":base_dev["rps"],
            "dev_rps_proposal":base_dev["rps"]}
    cand_dev=probabilities_scores(
        replay(dev,dev_priors,dev_keys,channel,candidate),dev)
    before=probabilities_scores(
        replay(audit,audit_priors,audit_keys,channel,1.),audit)
    after=probabilities_scores(
        replay(audit,audit_priors,audit_keys,channel,candidate),audit)
    accepted=(cand_dev["rps"]+.0003<base_dev["rps"] and
        after["rps"]+.0003<before["rps"] and
        after["mae_median"]<=before["mae_median"]+.02 and
        after["rmse_mean"]<=before["rmse_mean"]+.05 and
        after["brier"]<=before["brier"]+.005)
    return (candidate if accepted else 1.),{
        "accepted":bool(accepted),"proposal":float(candidate),
        "reason":"audit_accept" if accepted else "audit_reject",
        "audit_proposals":1,"dev_rps_before":base_dev["rps"],
        "dev_rps_proposal":cand_dev["rps"],
        "audit_rps_before":before["rps"],
        "audit_rps_proposal":after["rps"],
        "audit_mae_before":before["mae_median"],
        "audit_mae_proposal":after["mae_median"],
        "audit_rmse_before":before["rmse_mean"],
        "audit_rmse_proposal":after["rmse_mean"],
        "audit_brier_before":before["brier"],
        "audit_brier_proposal":after["brier"]}

def execute(args):
    if not 0<=args.arm_id<20 or not 4<=args.epochs<=80:
        raise ValueError("20 arms and 4..80 epochs only")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fold,arm=divmod(args.arm_id,4)
    root=args.data_root.resolve()
    out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing PPMI or frozen manifest")
    if out==root or root in out.parents:
        raise ValueError("Unsafe output location")
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,people)
    doctor=None
    if arm==3:
        if args.model_dir is None:
            raise ValueError("Qwen arm requires --model-dir")
        # Qwen inference can exploit several CPU threads; all gradient
        # training already finished before this loop in each task.
        doctor=TransformersDoctor(args.model_dir,max_new_tokens=32,device="cpu")
    metrics=[];paired=[];audits=[];coverage=[]
    for task in TASKS:
        subset=[p for p in people if p["cohort"]==task[0]]
        truth,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,truth,evidence,task)
        fit,dev,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,dev,audit,held)))<10:
            raise AssertionError("Unexpected empty clinical split")
        ids={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in ids])
        categorical=DistributionWM("categorical",epochs=args.epochs,
                                   seed=SEED+100*fold+task[1]).fit(fit,aux)
        channel=ClinicalEvidenceChannel(fit,chosen_optional_keys(fit))
        base_dev=categorical.predict_distribution(dev)
        base_audit=categorical.predict_distribution(audit)
        base_held=categorical.predict_distribution(held)
        forecasts={"motor_q0":base_held}
        fixed,_,fixed_stats=acquire(
            held,base_held,channel,"fixed",horizon=task[1])
        forecasts["fixed_q1"]=fixed
        greedy,_,greedy_stats=acquire(
            held,base_held,channel,"wm",horizon=task[1])
        forecasts["wm_q1"]=greedy
        qstats={"qwen_calls":0,"invalid_action":0,
                "agreement_with_wm":0,"real_test_releases":0}
        decision={"accepted":False,"reason":"no_llm_arm",
                  "audit_proposals":0}
        if arm==3:
            dev_qwen,dev_keys,dev_stats=acquire(
                dev,base_dev,channel,"qwen",doctor=doctor,
                horizon=task[1])
            audit_qwen,audit_keys,audit_stats=acquire(
                audit,base_audit,channel,"qwen",doctor=doctor,
                horizon=task[1])
            held_qwen,held_keys,held_stats=acquire(
                held,base_held,channel,"qwen",doctor=doctor,
                horizon=task[1])
            forecasts["qwen_q1"]=held_qwen
            alpha,decision=verify_update(
                dev,audit,base_dev,base_audit,dev_keys,audit_keys,channel)
            forecasts["qwen_verified"]=replay(
                held,base_held,held_keys,channel,alpha)
            audits.append({"cohort":task[0],"horizon":task[1],
                           "fold":fold,"accepted_alpha":alpha,**decision})
            qstats={k:sum(s[k] for s in
                (dev_stats,audit_stats,held_stats)) for k in
                ("qwen_calls","invalid_action","agreement_with_wm",
                 "real_test_releases")}
        primary={0:"motor_q0",1:"fixed_q1",
                 2:"wm_q1",3:"qwen_verified"}[arm]
        for name,pred in forecasts.items():
            s=probabilities_scores(pred,held)
            metrics.append({"cohort":task[0],"horizon":task[1],
                "fold":fold,"arm_id":args.arm_id,
                "primary":int(primary==name),"method":name,
                "patients":len(held),
                "positive":sum(sum(row["site"])>0 for row in held),
                **s})
        for reference,candidate in [
            ("motor_q0","fixed_q1"),
            ("motor_q0","wm_q1"),
            ("fixed_q1","wm_q1"),
            ("fixed_q1","qwen_q1"),
            ("wm_q1","qwen_q1"),
            ("qwen_q1","qwen_verified")
        ]:
            if reference not in forecasts or candidate not in forecasts:
                continue
            for measure in ("mae","mse","brier","rps"):
                paired.append({"cohort":task[0],"horizon":task[1],
                    "fold":fold,"arm":args.arm_id,
                    "reference":reference,"candidate":candidate,
                    **paired_estimate(forecasts[reference],
                                     forecasts[candidate],
                                     held,measure,SEED+fold*100+task[1])})
        coverage.append({"cohort":task[0],"horizon":task[1],
            "fold":fold,"fit":len(fit),"dev":len(dev),
            "audit":len(audit),"heldout":len(held),
            "optional_keys":len(channel.keys),"aux_pairs":len(aux),
            "fixed_test_release":fixed_stats["real_test_releases"],
            "wm_test_release":greedy_stats["real_test_releases"],
            **qstats,"status":"complete"})
        write_csv(out/"method_metrics.csv",metrics)
        write_csv(out/"paired_bootstrap.csv",paired)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task),
            "elapsed_sec":round(time.monotonic()-start,2)},indent=2)+"\n")
        print("PHASE80_TASK_COMPLETE",task,"arm",args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":80,"arm":args.arm_id,"fold":fold,"strategy":arm,
        "elapsed_seconds":round(time.monotonic()-start,2),
        "completed_tasks":len(coverage),
        "qwen_total_prompts":doctor.total_prompts if doctor else 0,
        "scope":[
            "Repeated exploratory frozen Prodromal TRAIN, strict PD only",
            "Original already-consumed final TEST is NEVER accessed",
            "Same learned motor categorical WM for q0 and q1",
            "One actual previously recorded BL nonmotor value per eligible patient",
            "Selection receives only available names and FIT WM expected gains",
            "LLM receives no actual optional test values until AFTER its action",
            "Retrospective availability-aware query; not prospective ordering",
            "FIT-only likelihood update; probability outputs not independently calibrated",
            "DEV selects allowlisted alpha; separate AUDIT veto before outer EVAL",
            "Agent cannot retrain its own architecture: revision is scalar-only",
            "No individual patient records, prompts, predictions or IDs exported",
            "No causal treatment effect or clinical deployment claim"
        ],"qc":quality,
        "finished_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({"status":"complete",
        "completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE80_SUCCESS",args.arm_id,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--epochs",type=int,default=18)
    execute(p.parse_args())

if __name__=="__main__":
    main()
