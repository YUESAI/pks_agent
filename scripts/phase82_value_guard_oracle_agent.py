#!/usr/bin/env python3
"""Phase82: falsifiable, cost-aware clinical test acquisition with Qwen safety gate.

Five frozen subject folds x four standalone CPU strategies:
0 q0 / FIT-priority fixed / WM information gain / deterministic random
1 add FIT-only 3fold OOF benefits and prespecified query-cost frontiers
2 add hard OOF-value-filtered REAL Qwen Doctor (may ABSTAIN)
3 add one DEV/AUDIT-gated selection between safe Qwen and OOF policy

For each outer evaluation fold also estimate an *unattainable* retrospective
best single-test RPS using that subject's true outcome. It is explicitly
an oracle ceiling, NEVER a deployment policy or training signal.

No patient-level output, outcome in LLM prompts, unmeasured releases,
post-cutoff observations, or access to the previously consumed final TEST.
"""
from __future__ import annotations

import argparse,json,time
from collections import Counter
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
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores,paired_estimate
from scripts.phase80_active_evidence_agent import ClinicalEvidenceChannel
from scripts.phase81_oof_value_agent import (
    OOFValueModel,execute_policy,rps_one,SEED,TASKS)

MAIN_COST=.00025
COSTS=(.0001,.00025,.0005)
DEV_MARGIN=.0001

def eligible_value(row,prior,channel,benefits,cost):
    """FIT-only expectation chooses admissible positive-net-gain tests.

    Each candidate's benefit and evidence strength were determined before
    any value for this patient was released.
    """
    options=[]
    for key in channel.allowed(row):
        benefit,alpha=benefits.get(key,(-float("inf"),0.))
        if np.isfinite(benefit) and benefit>cost:
            options.append((float(benefit),key,float(alpha)))
    return sorted(options,key=lambda x:(-x[0],x[1]))

def safe_qwen(rows,priors,channel,value_model,doctor,horizon,cost):
    """Call Qwen only when at least one supported positive-value tool exists.

    Other queries, including available but negative-value tests, are
    *not* in the LLM's tool vocabulary. Invalid actions fall back to
    ABSTAIN, not to an unsupported clinical test.
    """
    stats=Counter()
    results=[]
    for row,prior in zip(rows,priors):
        allowed=channel.allowed(row)
        if not allowed:
            stats["no_recorded_test"]+=1
            results.append(np.asarray(prior,dtype=float).copy())
            continue
        stats["eligible_people"]+=1
        gains=value_model.suggestions(row,prior,channel)
        vetted=eligible_value(row,prior,channel,gains,cost)
        if not vetted:
            stats["guard_blocked"]+=1
            results.append(np.asarray(prior,dtype=float).copy())
            continue
        options=tuple(key for _,key,_ in vetted)
        utilities={key:round(gain,6) for gain,key,_ in vetted}
        messages=[
            {"role":"system","content":
                "Research-only retrospective clinical test selection. "
                "Choose ONE test from the exactly listed allowed names, "
                "or choose ABSTAIN. A test is chosen before its true measured "
                "value is available. Never guess a patient measurement, "
                "diagnosis or outcome. Output one JSON object only: "
                '{"ask":"ALLOWED_NAME_OR_ABSTAIN"}.'},
            {"role":"user","content":
                "Future Motor horizon months: "+str(horizon)+
                ". Available VALUE-VETTED test names: "+json.dumps(options)+
                ". FIT-only out-of-fold expected improvements in RPS "
                "(estimates, NOT patient measurements): "+
                json.dumps(utilities,sort_keys=True)+
                ". Motor WM estimated nonzero-risk (NOT a measured test): "+
                str(round(1-float(prior[0]),5))+
                ". You may ABSTAIN. Select one allowed name or ABSTAIN."}]
        response=doctor.generate(messages)
        stats["qwen_calls"]+=1
        choice=extract_action(response,set(options)|{"ABSTAIN"})
        if choice is None:
            stats["invalid_action"]+=1
            choice="ABSTAIN"
        if choice=="ABSTAIN":
            stats["llm_abstained"]+=1
            results.append(np.asarray(prior,dtype=float).copy())
            continue
        alpha=gains[choice][1]
        # The actual recorded value is revealed only *after* choice.
        results.append(channel.posterior(
            prior,choice,row["optional"][choice],alpha))
        stats["real_test_releases"]+=1
    return np.stack(results),{
        "eligible":stats["eligible_people"],
        "guard_blocked":stats["guard_blocked"],
        "qwen_calls":stats["qwen_calls"],
        "invalid_action":stats["invalid_action"],
        "llm_abstained":stats["llm_abstained"],
        "released":stats["real_test_releases"],
        "no_recorded_test":stats["no_recorded_test"]}

def oof_with_cost(rows,priors,channel,model,cost):
    answer=[];stats=Counter()
    for row,prior in zip(rows,priors):
        allowed=channel.allowed(row)
        if not allowed:
            stats["no_recorded_test"]+=1
            answer.append(np.asarray(prior,dtype=float).copy())
            continue
        stats["eligible_people"]+=1
        vetted=eligible_value(row,prior,channel,
                              model.suggestions(row,prior,channel),cost)
        if not vetted:
            stats["abstained"]+=1
            answer.append(np.asarray(prior,dtype=float).copy())
        else:
            _,key,alpha=vetted[0]
            answer.append(channel.posterior(
                prior,key,row["optional"][key],alpha))
            stats["real_test_releases"]+=1
    return np.stack(answer),{
        "eligible":stats["eligible_people"],
        "abstained":stats["abstained"],
        "released":stats["real_test_releases"],
        "no_recorded_test":stats["no_recorded_test"]}

def oracle_headroom(rows,priors,channel):
    """Outcome-aware hindsight oracle; only an upper bound, never an Agent.

    This evaluation uses true target burden AFTER all policy outputs
    are fixed, and MUST NEVER be used to select or fit a real policy.
    """
    total=0.;oracle=0.;available=0.;improvable=0.
    mean_potential=0.
    for row,prior in zip(rows,priors):
        y=sum(row["site"])
        baseline=rps_one(prior,y)
        candidates=[baseline]
        for key in channel.allowed(row):
            candidates.append(rps_one(channel.posterior(
                prior,key,row["optional"][key],alpha=1.),y))
        best=min(candidates)
        total+=baseline
        oracle+=best
        if len(candidates)>1:
            available+=1
            mean_potential+=baseline-best
            if best+1e-10<baseline:improvable+=1
    return {
        "n_patients":len(rows),
        "eligible_patients":available,
        "oracle_any_improvable":improvable,
        "q0_rps":total/len(rows),
        "clairvoyant_best_rps":oracle/len(rows),
        "clairvoyant_potential_rps":(total-oracle)/len(rows),
        "mean_potential_eligible":(
            mean_potential/available if available else 0.),
        "warning":"HINDSIGHT ONLY: chooses by TRUE EVAL OUTCOME, NOT a usable policy"
    }

def guard_policy(oo_dev,q_dev,oo_audit,q_audit,dev,audit,
                 oo_dev_rate,q_dev_rate,oo_audit_rate,q_audit_rate):
    def net_rps(p,people,queries):
        s=probabilities_scores(p,people)
        return s["rps"]+MAIN_COST*float(queries)/len(people)
    before_dev=net_rps(oo_dev,dev,oo_dev_rate)
    after_dev=net_rps(q_dev,dev,q_dev_rate)
    before_audit=net_rps(oo_audit,audit,oo_audit_rate)
    after_audit=net_rps(q_audit,audit,q_audit_rate)
    guard=probabilities_scores(oo_audit,audit)
    proposed=probabilities_scores(q_audit,audit)
    accepted=(after_dev+DEV_MARGIN<before_dev and
              after_audit+DEV_MARGIN<before_audit and
              proposed["brier"]<=guard["brier"]+.005 and
              proposed["rmse_mean"]<=guard["rmse_mean"]+.05 and
              proposed["mae_median"]<=guard["mae_median"]+.03)
    return accepted,{
        "candidate":"value_vetted_qwen","incumbent":"oof_cost250",
        "accepted":bool(accepted),"audit_proposals":1,
        "dev_net_rps_before":before_dev,"dev_net_rps_candidate":after_dev,
        "audit_net_rps_before":before_audit,
        "audit_net_rps_candidate":after_audit,
        "audit_before_mae":guard["mae_median"],
        "audit_candidate_mae":proposed["mae_median"],
        "audit_before_brier":guard["brier"],
        "audit_candidate_brier":proposed["brier"]}

def job(args):
    if not 0<=args.arm_id<20 or not 4<=args.epochs<=80:
        raise ValueError("Invalid arm or epochs")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fold,variant=divmod(args.arm_id,4)
    root=args.data_root.resolve()
    out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing PPMI data or frozen TRAIN manifest")
    if out==root or root in out.parents:
        raise ValueError("Unsafe result directory")
    out.mkdir(parents=True,exist_ok=False)
    (out/"progress.json").write_text(json.dumps({
        "status":"started","arm_id":args.arm_id},indent=2)+"\n")
    start=time.monotonic()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,people)
    doctor=None
    if variant>=2:
        if args.model_dir is None:
            raise ValueError("Qwen arms require --model-dir")
        doctor=TransformersDoctor(args.model_dir,max_new_tokens=32,
                                  device="cpu")
    metrics=[];paired=[];coverage=[];audits=[];oracle_info=[]
    for task in TASKS:
        began=time.monotonic()
        subset=[p for p in people if p["cohort"]==task[0]]
        targets,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,targets,evidence,task)
        fit,dev,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,dev,audit,held)))<10:
            raise ValueError("Insufficient patient split")
        ids={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in ids])
        wm=DistributionWM("categorical",epochs=args.epochs,
                          seed=SEED+100*fold+task[1]).fit(fit,aux)
        channel=ClinicalEvidenceChannel(fit,chosen_optional_keys(fit))
        eval_prior=wm.predict_distribution(held)
        policies={"motor_q0":eval_prior}
        query_rate={"motor_q0":0}
        stats={}
        for method,policy in (("fixed_q1","fixed"),("wm_q1","wm"),
                              ("random_q1","random")):
            policies[method],s=execute_policy(held,eval_prior,channel,
                                             policy,horizon=task[1])
            stats[method]=s
            query_rate[method]=s["released"]
        oracle_info.append({"cohort":task[0],"horizon":task[1],
                            "fold":fold,**oracle_headroom(
                                held,eval_prior,channel)})
        if variant>=1:
            value=OOFValueModel(channel.keys).fit(
                fit,subset,task,fold,args.epochs)
            for cost in COSTS:
                method="oof_cost"+str(int(round(cost*1_000_000)))
                policies[method],s=oof_with_cost(
                    held,eval_prior,channel,value,cost)
                stats[method]=s
                query_rate[method]=s["released"]
        if variant>=2:
            policies["qwen_safe"],s=safe_qwen(
                held,eval_prior,channel,value,doctor,task[1],MAIN_COST)
            stats["qwen_safe"]=s
            query_rate["qwen_safe"]=s["released"]
        if variant==3:
            dprior=wm.predict_distribution(dev)
            aprior=wm.predict_distribution(audit)
            od,os=oof_with_cost(dev,dprior,channel,value,MAIN_COST)
            oa,ats=oof_with_cost(audit,aprior,channel,value,MAIN_COST)
            qd,qs=safe_qwen(dev,dprior,channel,value,doctor,task[1],MAIN_COST)
            qa,qas=safe_qwen(audit,aprior,channel,value,doctor,task[1],MAIN_COST)
            accepted,decision=guard_policy(
                od,qd,oa,qa,dev,audit,
                os["released"],qs["released"],
                ats["released"],qas["released"])
            policies["verified_policy"]=(
                policies["qwen_safe"] if accepted else policies["oof_cost250"])
            query_rate["verified_policy"]=(
                query_rate["qwen_safe"] if accepted else
                query_rate["oof_cost250"])
            audits.append({"cohort":task[0],"horizon":task[1],"fold":fold,
                           **decision})
            # Report Qwen calls across dev/audit/eval for compute accounting.
            stats["qwen_safe"]["qwen_calls"]+=qs["qwen_calls"]+qas["qwen_calls"]
            stats["qwen_safe"]["invalid_action"]+=(
                qs["invalid_action"]+qas["invalid_action"])
        primary={0:"motor_q0",1:"oof_cost250",
                 2:"qwen_safe",3:"verified_policy"}[variant]
        for name,pred in policies.items():
            s=probabilities_scores(pred,held)
            cost=MAIN_COST*query_rate[name]/len(held)
            metrics.append({
                "cohort":task[0],"horizon":task[1],
                "fold":fold,"arm":args.arm_id,
                "primary":int(name==primary),"method":name,
                "patients":len(held),
                "queries":query_rate[name],
                "query_rate":query_rate[name]/len(held),
                "net_rps_cost250":s["rps"]+cost,
                **s})
        for ref,cand in (
            ("motor_q0","fixed_q1"),
            ("fixed_q1","wm_q1"),
            ("motor_q0","random_q1"),
            ("fixed_q1","oof_cost250"),
            ("oof_cost250","qwen_safe"),
            ("oof_cost250","verified_policy")
        ):
            if ref not in policies or cand not in policies:continue
            for measure in ("mae","mse","brier","rps"):
                paired.append({
                    "cohort":task[0],"horizon":task[1],
                    "fold":fold,"arm":args.arm_id,
                    "reference":ref,"candidate":cand,
                    **paired_estimate(policies[ref],policies[cand],
                                     held,measure,SEED+100*fold+task[1])})
        cov={"cohort":task[0],"horizon":task[1],"fold":fold,
             "fit":len(fit),"dev":len(dev),"audit":len(audit),
             "heldout":len(held),"fit_aux_pairs":len(aux),
             "seconds":round(time.monotonic()-began,2),
             "status":"complete"}
        for name,s in stats.items():
            cov[name+"_query"]=s.get("released",0)
            cov[name+"_calls"]=s.get("qwen_calls",0)
            cov[name+"_guard_abstain"]=s.get("guard_blocked",0)
            cov[name+"_llm_abstain"]=s.get("llm_abstained",0)
        coverage.append(cov)
        write_csv(out/"method_metrics.csv",metrics)
        write_csv(out/"paired_bootstrap.csv",paired)
        write_csv(out/"oracle_headroom.csv",oracle_info)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE82_TASK_COMPLETE",task,args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":82,"arm_id":args.arm_id,"fold":fold,"variant":variant,
        "tasks_completed":len(coverage),
        "elapsed_seconds":round(time.monotonic()-start,2),
        "qwen_total_calls":doctor.total_prompts if doctor else 0,
        "evaluation_scope":[
            "Only repeatedly explored original Prodromal TRAIN and strict PD",
            "Already-consumed final TEST never accessed",
            "Oracle headroom uses outcomes for HINDSIGHT upper bound ONLY",
            "No patient-level values, identifiers or LLM prompts exported",
            "No outcome or measured optional clinical value provided to LLM",
            "FIT-only 3fold OOF evidence benefit with predeclared query cost",
            "Only actually recorded BL tests can be queried",
            "Known availability retrospective masking, not real clinical ordering",
            "Hard value-of-information guard prevents unsupported Qwen tools",
            "One DEV/AUDIT policy-switch veto; not autonomous RSI or causal WM"],
        "quality":quality,
        "generated_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","tasks":len(coverage)},indent=2)+"\n")
    print("PHASE82_SUCCESS",args.arm_id,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--epochs",type=int,default=18)
    job(p.parse_args())

if __name__=="__main__":
    main()
