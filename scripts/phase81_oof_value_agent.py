#!/usr/bin/env python3
"""Phase81: cross-fitted expected-value-of-information acquisition with abstain.

20 standalone CPU jobs: five patient folds by four policies.
  0: Motor categorical distribution q=0 reference.
  1: q=1 FIT-priority, WM-expected-KL, and deterministic random controls.
  2: FIT-only inner-OOF empirical reduction in proper RPS; abstain if none
     appears reliably useful; alpha selected from a fixed small grid.
  3: real offline Qwen Doctor sees allowed names, expected RPS benefits
     and an ABSTAIN tool; once-only DEV/AUDIT chooses Qwen vs OOF policy.

The real optional clinical value is released ONLY AFTER the policy
chooses a name. FIT OOF evaluation excludes the evaluated person's
outcome from both training the CT-WM and fitting the evidence channel.
No causal patient simulator; no clinical deployment or autonomous RSI.
"""
from __future__ import annotations

import argparse,hashlib,json,time
from collections import defaultdict,Counter
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
from scripts.phase80_active_evidence_agent import ClinicalEvidenceChannel

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
ALPHAS=(.25,.5,1.)
MIN_EVIDENCE=10
STRENGTH_PRIOR=15.
REQUIRE_GAIN=0.00010

def oof_fold(pid,task):
    value=f"pks-phase81-OOF|{SEED}|{task[0]}|{task[1]}|{pid}"
    return int(hashlib.sha256(value.encode()).hexdigest()[:12],16)%3

def risk_band(prior):
    return int(1.-float(prior[0])>=.5)

def rps_one(prob,y):
    p=np.asarray(prob,dtype=np.float64)
    if p.shape!=(25,) or not np.isfinite(p).all():
        raise AssertionError("Invalid per-patient posterior")
    truth=(np.arange(25)>=int(y)).astype(float)
    return float(np.sum((np.cumsum(p)-truth)**2)/24.)

class OOFValueModel:
    """Shrunk conservative expected proper-score improvement of real tests.

    Each training target comes from an inner held-out subject evaluated
    by an inner WM and ClinicalEvidenceChannel both FIT on other people.
    No patient-level training rows/predictions are exported.
    """
    def __init__(self,keys):
        self.keys=tuple(keys)
        self.cells=defaultdict(list)
        self.fit_counts=Counter()
        self.inner_partitions=[]
    def fit(self,fit,people,task,fold,epochs):
        assignments=[oof_fold(r["pid"],task) for r in fit]
        if len(set(r["pid"] for r in fit))!=len(fit):
            raise AssertionError("Duplicate patient in OOF FIT")
        for inner in range(3):
            train=[r for r,b in zip(fit,assignments) if b!=inner]
            held=[r for r,b in zip(fit,assignments) if b==inner]
            if len(train)<20 or not held:
                raise AssertionError("Insufficient OOF partition")
            train_ids={r["pid"] for r in train}
            hold_ids={r["pid"] for r in held}
            if train_ids & hold_ids:
                raise AssertionError("OOF person leakage")
            aux=build_auxiliary([p for p in people if p["pid"] in train_ids])
            wm=DistributionWM("categorical",epochs=epochs,
                 seed=SEED+1000+100*fold+task[1]+inner).fit(train,aux)
            channel=ClinicalEvidenceChannel(train,self.keys)
            priors=wm.predict_distribution(held)
            for row,prior in zip(held,priors):
                y=sum(row["site"])
                baseline=rps_one(prior,y)
                band=risk_band(prior)
                for key in channel.allowed(row):
                    observed=row["optional"][key]
                    self.fit_counts[key]+=1
                    for alpha in ALPHAS:
                        posterior=channel.posterior(prior,key,observed,alpha)
                        improvement=baseline-rps_one(posterior,y)
                        # A global estimate (band -1) backs off when
                        # the patient-specific risk band is too sparse.
                        self.cells[(key,-1,alpha)].append(improvement)
                        self.cells[(key,band,alpha)].append(improvement)
            self.inner_partitions.append([len(train),len(held)])
        if sum(n for _,n in self.inner_partitions)!=len(fit):
            raise AssertionError("Incomplete FIT OOF coverage")
        return self
    def test_gain(self,key,band,alpha):
        values=self.cells.get((key,band,alpha),[])
        if len(values)<MIN_EVIDENCE:return -float("inf")
        arr=np.asarray(values,dtype=float)
        # Explicit shrinkage to zero and an uncertainty penalty.
        mean=float(arr.sum()/(len(arr)+STRENGTH_PRIOR))
        se=float(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr)>1 else 1.
        return mean-.5*se
    def suggestions(self,row,prior,channel):
        band=risk_band(prior)
        choices={}
        for key in channel.allowed(row):
            possibilities=[]
            for alpha in ALPHAS:
                # Use conditional risk-band gain if enough independent
                # OOF examples, else the more stable global score.
                gain=self.test_gain(key,band,alpha)
                if not np.isfinite(gain):
                    gain=self.test_gain(key,-1,alpha)
                possibilities.append((gain,alpha))
            gain,alpha=max(possibilities,key=lambda item:(item[0],-item[1]))
            choices[key]=(float(gain),float(alpha))
        return choices
    def choose(self,row,prior,channel):
        choices=self.suggestions(row,prior,channel)
        viable=[(gain,key,alpha) for key,(gain,alpha) in choices.items()
                if gain>REQUIRE_GAIN]
        if not viable:return None,0.
        _,key,alpha=max(viable,key=lambda a:(a[0],a[1]))
        return key,alpha
    def diagnostics(self):
        n=sum(self.fit_counts.values())
        useful=sum(1 for k in self.keys if any(
            self.test_gain(k,-1,a)>REQUIRE_GAIN for a in ALPHAS))
        return {"fit_oof_test_examples":n,
                "fit_oof_keys":len(self.fit_counts),
                "fit_oof_globally_positive_keys":useful,
                "inner_splits":self.inner_partitions}

def fixed_choice(allowed,channel):
    return next((key for key in channel.keys if key in allowed),None)

def wm_choice(allowed,channel,prior):
    return (min(allowed,key=lambda k:(-channel.expected_information(prior,k),k))
            if allowed else None)

def random_choice(allowed,row,horizon):
    if not allowed:return None
    seed=f"phase81-random|{SEED}|{horizon}|{row['pid']}"
    x=int(hashlib.sha256(seed.encode()).hexdigest()[:12],16)
    return allowed[x%len(allowed)]

def qwen_choice(row,prior,channel,values,doctor,horizon):
    """Qwen sees only test NAMES, benefits estimated from FIT-OOF,
    motor risk and ABSTAIN. Actual patient test values remain hidden."""
    allowed=channel.allowed(row)
    if not allowed:return None,False
    gains={k:round(float(values.get(k,(-1.,1.))[0]),6) for k in allowed}
    messages=[{"role":"system","content":
        "Research-only test acquisition. You may choose one already "
        "measured baseline test name from the allowed list OR ABSTAIN "
        "when expected improvement does not justify a query. "
        "The scores are estimates from training people, not measurements "
        "for this patient. Do not infer, guess or fabricate any test value. "
        'Respond only as JSON: {"ask":"EXACT_ALLOWED_NAME_OR_ABSTAIN"}.'},
        {"role":"user","content":
        "Future Motor burden horizon: "+str(horizon)+
        " months. Available test names: "+json.dumps(allowed)+
        ". Current Motor WM predicted nonzero risk (estimate only): "+
        str(round(float(1.-prior[0]),5))+
        ". Estimated improvement of ranked probability score from "
        "FIT cross-validation (larger more favorable): "+
        json.dumps(gains,sort_keys=True)+
        ". ABSTAIN is allowed. Reply JSON only."}]
    reply=doctor.generate(messages)
    action=extract_action(reply,set(allowed)|{"ABSTAIN"})
    fallback,_=values_policy(row,prior,channel,values)
    invalid=action is None
    if invalid:action=fallback
    return (None if action=="ABSTAIN" else action),invalid

def values_policy(row,prior,channel,values):
    viable=[(g,key,alpha) for key,(g,alpha) in values.items()
            if key in channel.allowed(row) and g>REQUIRE_GAIN]
    if not viable:return None,0.
    _,key,alpha=max(viable,key=lambda a:(a[0],a[1]))
    return key,alpha

def execute_policy(rows,priors,channel,policy,value_model=None,
                   doctor=None,horizon=12):
    if len(rows)!=len(priors):raise AssertionError("Prior misalignment")
    out=[];counts=Counter();choices=Counter()
    for row,p in zip(rows,priors):
        allowed=channel.allowed(row)
        if not allowed or policy=="none":
            out.append(np.asarray(p,dtype=float).copy())
            if not allowed:counts["without_allowed_test"]+=1
            continue
        counts["eligible"]+=1
        choices_with_gains=(value_model.suggestions(row,p,channel)
                            if value_model is not None else {})
        if policy=="fixed":key=fixed_choice(allowed,channel);alpha=1.
        elif policy=="wm":key=wm_choice(allowed,channel,p);alpha=1.
        elif policy=="random":key=random_choice(allowed,row,horizon);alpha=1.
        elif policy=="oof":key,alpha=values_policy(
            row,p,channel,choices_with_gains)
        elif policy=="qwen":
            if doctor is None or value_model is None:
                raise AssertionError("Qwen requires model and FIT OOF utility")
            key,invalid=qwen_choice(row,p,channel,choices_with_gains,
                                    doctor,horizon)
            counts["qwen_calls"]+=1
            counts["invalid_qwen_actions"]+=int(invalid)
            # Same post-selection evidence strength as the OOF baseline:
            # isolate test-selection and ABSTAIN decision, not scale.
            alpha=(choices_with_gains[key][1] if key is not None else 0.)
        else:raise ValueError("Unsupported policy")
        if key is None:
            counts["abstained"]+=1
            out.append(np.asarray(p,dtype=float).copy())
        else:
            if key not in allowed:raise AssertionError("Unmeasured chosen test")
            observed=row["optional"][key] # TRUE value released only here
            out.append(channel.posterior(p,key,observed,alpha))
            counts["observed_test_releases"]+=1
            choices[key]+=1
    if not rows:raise AssertionError("Expected evaluation rows")
    return np.stack(out),{
        "eligible":counts["eligible"],"abstained":counts["abstained"],
        "released":counts["observed_test_releases"],
        "qwen_calls":counts["qwen_calls"],
        "invalid_actions":counts["invalid_qwen_actions"],
        "top_selected_tests":dict(choices.most_common(3))}

def verify_policy(oo_dev,oo_audit,q_dev,q_audit,dev,audit):
    """One allowlisted Qwen-vs-data-driven policy revision decision.

    The same FIT-only WM and OOF benefit estimator are used. On DEV,
    Qwen must strictly improve RPS. AUDIT is a once-only independent
    veto with MAE, RMSE and Brier noninferiority guardrails.
    """
    dv0=probabilities_scores(oo_dev,dev)
    dv1=probabilities_scores(q_dev,dev)
    a0=probabilities_scores(oo_audit,audit)
    a1=probabilities_scores(q_audit,audit)
    accepted=(dv1["rps"]+.0002<dv0["rps"] and
              a1["rps"]+.0002<a0["rps"] and
              a1["brier"]<=a0["brier"]+.005 and
              a1["rmse_mean"]<=a0["rmse_mean"]+.05 and
              a1["mae_median"]<=a0["mae_median"]+.03)
    return bool(accepted),{
        "candidate":"qwen_oof_value",
        "incumbent":"oof_value",
        "accepted":bool(accepted),
        "reason":"audit_accept" if accepted else "dev_or_audit_reject",
        "audited_proposals":1,
        "dev_incumbent_rps":dv0["rps"],"dev_qwen_rps":dv1["rps"],
        "audit_incumbent_rps":a0["rps"],"audit_qwen_rps":a1["rps"],
        "audit_incumbent_mae":a0["mae_median"],
        "audit_qwen_mae":a1["mae_median"],
        "audit_incumbent_brier":a0["brier"],
        "audit_qwen_brier":a1["brier"]}

def run(args):
    if not 0<=args.arm_id<20 or not 4<=args.epochs<=80:
        raise ValueError("Invalid arm or epoch count")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fold,arm=divmod(args.arm_id,4)
    root=args.data_root.resolve()
    out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing PPMI or frozen original TRAIN")
    if out==root or root in out.parents:
        raise ValueError("Unsafe output path")
    out.mkdir(parents=True,exist_ok=False)
    (out/"progress.json").write_text(json.dumps({
        "status":"started","arm":args.arm_id},indent=2)+"\n")
    started=time.monotonic()
    print("PHASE81_BEGIN",args.arm_id,flush=True)
    people,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,people)
    doctor=None
    if arm==3:
        if args.model_dir is None:
            raise ValueError("Qwen arm requires --model-dir")
        doctor=TransformersDoctor(args.model_dir,max_new_tokens=32,device="cpu")
    metrics=[];paired=[];audits=[];coverage=[]
    for task in TASKS:
        task_started=time.monotonic()
        subset=[p for p in people if p["cohort"]==task[0]]
        targets,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,targets,evidence,task)
        fit,dev,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,dev,audit,held)))<10:
            raise AssertionError("Empty patient split")
        keys=chosen_optional_keys(fit)
        train_ids={r["pid"] for r in fit}
        aux=build_auxiliary([p for p in subset if p["pid"] in train_ids])
        wm=DistributionWM("categorical",epochs=args.epochs,
             seed=SEED+100*fold+task[1]).fit(fit,aux)
        channel=ClinicalEvidenceChannel(fit,keys)
        base_eval=wm.predict_distribution(held)
        outputs={"motor_q0":base_eval}
        stats={}
        for name,policy in (("fixed_q1","fixed"),("wm_q1","wm"),
                            ("random_q1","random")):
            outputs[name],stats[name]=execute_policy(
                held,base_eval,channel,policy,horizon=task[1])
        value_model=None
        if arm>=2:
            print("PHASE81_OOF_BEGIN",task,"arm",args.arm_id,flush=True)
            value_model=OOFValueModel(keys).fit(
                fit,subset,task,fold,args.epochs)
            outputs["oof_abstain"],stats["oof_abstain"]=execute_policy(
                held,base_eval,channel,"oof",value_model=value_model,
                horizon=task[1])
            print("PHASE81_OOF_DONE",task,value_model.diagnostics(),
                  flush=True)
        if arm==3:
            base_dev=wm.predict_distribution(dev)
            base_audit=wm.predict_distribution(audit)
            o_dev,_=execute_policy(dev,base_dev,channel,"oof",
                value_model=value_model,horizon=task[1])
            o_audit,_=execute_policy(audit,base_audit,channel,"oof",
                value_model=value_model,horizon=task[1])
            q_dev,sd=execute_policy(dev,base_dev,channel,"qwen",
                value_model=value_model,doctor=doctor,horizon=task[1])
            q_audit,sa=execute_policy(audit,base_audit,channel,"qwen",
                value_model=value_model,doctor=doctor,horizon=task[1])
            q_eval,se=execute_policy(held,base_eval,channel,"qwen",
                value_model=value_model,doctor=doctor,horizon=task[1])
            outputs["qwen_abstain"]=q_eval
            accepted,ver=verify_policy(
                o_dev,o_audit,q_dev,q_audit,dev,audit)
            outputs["verified_policy"]=(
                q_eval if accepted else outputs["oof_abstain"])
            stats["qwen_abstain"]={k:sum(s[k] for s in (sd,sa,se))
                                  for k in ("eligible","abstained","released",
                                            "qwen_calls","invalid_actions")}
            audits.append({"cohort":task[0],"horizon":task[1],
                           "fold":fold,**ver})
            print("PHASE81_VERIFIED",task,"accepted",accepted,flush=True)
        primary={0:"motor_q0",1:"wm_q1",2:"oof_abstain",
                 3:"verified_policy"}[arm]
        for name,pred in outputs.items():
            metrics.append({
                "cohort":task[0],"horizon":task[1],"fold":fold,
                "arm":args.arm_id,"method":name,
                "is_primary":int(name==primary),"patients":len(held),
                "positive":sum(sum(r["site"])>0 for r in held),
                **probabilities_scores(pred,held)})
        comparisons=[("motor_q0","fixed_q1"),
                     ("fixed_q1","wm_q1"),
                     ("fixed_q1","random_q1"),
                     ("motor_q0","random_q1")]
        if "oof_abstain" in outputs:
            comparisons += [("motor_q0","oof_abstain"),
                            ("fixed_q1","oof_abstain"),
                            ("wm_q1","oof_abstain")]
        if "qwen_abstain" in outputs:
            comparisons += [("oof_abstain","qwen_abstain"),
                            ("oof_abstain","verified_policy"),
                            ("qwen_abstain","verified_policy")]
        for reference,candidate in comparisons:
            for measure in ("mae","mse","brier","rps"):
                paired.append({"cohort":task[0],"horizon":task[1],
                    "fold":fold,"arm":args.arm_id,
                    "reference":reference,"candidate":candidate,
                    **paired_estimate(outputs[reference],
                                     outputs[candidate],held,measure,
                                     SEED+100*fold+task[1])})
        flat={"cohort":task[0],"horizon":task[1],"fold":fold,
              "fit":len(fit),"dev":len(dev),"audit":len(audit),
              "heldout":len(held),"available_keys":len(keys),
              "fit_aux_pairs":len(aux),
              "elapsed_sec":round(time.monotonic()-task_started,2),
              "status":"complete"}
        for name,fields in stats.items():
            for key in ("eligible","abstained","released",
                        "qwen_calls","invalid_actions"):
                flat[name+"_"+key]=fields.get(key,0)
        coverage.append(flat)
        write_csv(out/"method_metrics.csv",metrics)
        write_csv(out/"paired_bootstrap.csv",paired)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "task":list(task),
            "elapsed_sec":round(time.monotonic()-started,2)},indent=2)+"\n")
        print("PHASE81_TASK_COMPLETE",task,args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":81,"arm":args.arm_id,"outer_fold":fold,"strategy":arm,
        "completed_tasks":len(coverage),"elapsed_sec":round(
            time.monotonic()-started,2),
        "qwen_prompts":doctor.total_prompts if doctor else 0,
        "study_limits":[
         "Exploratory retrospective original frozen Prodromal TRAIN and strict PD",
         "Prior consumed TEST never accessed",
         "Cross-fitted FIT OOF utility but no independently confirmed policy gain",
         "Optional test values acquired from real already-recorded baseline",
         "Risk-band and per-test value estimates are shrinkage heuristics",
         "Available test names known before choice: retrospective oracle availability",
         "DEV/AUDIT audit one Qwen-vs-OOF policy switch, not autonomous RSI",
         "No individual participant records, scores, prompts or IDs exported",
         "No causal treatment effects or deployed clinical recommendations"],
        "quality":quality,
        "generated_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE81_SUCCESS",args.arm_id,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--epochs",type=int,default=18)
    run(p.parse_args())
if __name__=="__main__":
    main()
