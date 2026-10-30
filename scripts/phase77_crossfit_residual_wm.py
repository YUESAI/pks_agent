#!/usr/bin/env python3
"""Phase77: cross-fitted non-motor residual correction of an established Motor WM.

Five original patient folds x four arms (20 independent Slurm tasks):
0 Motor CT-WM, 1 OOF intercept/bias control, 2 cross-fitted nonmotor
residual correction, 3 DEV-selected and independently AUDIT-gated
bounded residual revision.

Every arm fits the same motor CT-WM and same-patient Ridge/HGB and
direct early-fusion references. No user-level output, no fictional
clinical values, no LLM inference, no external test claims.

Crucial: residual training targets use FIT-only 3-way OOF motor
predictions from models never trained on those residual subjects.
No DEV/AUDIT/outer patient is used for residual or WM training.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,faulthandler,sys,os
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase73_matched_longitudinal_agent import data_rows,split_data
from scripts.phase72_ctlatent_wm import build_auxiliary
from scripts.phase76_reliability_gated_wm import (
    DirectWM,TimeTabular,labels,optional_keys,score,paired_bootstrap,write_csv)

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
DEV_GAMMAS=(0.25,0.5,1.0)
T0=time.monotonic()

def stage(name,**data):
    """Safe cluster diagnostics: aggregate counts only, never patient data."""
    msg={"phase":77,"stage":name,
         "elapsed_seconds":round(time.monotonic()-T0,2),"pid":os.getpid()}
    msg.update(data)
    print("PHASE77_STAGE "+json.dumps(msg,sort_keys=True),flush=True)


def hash_fold(pid,task):
    s=f"phase77|inner|{SEED}|{task[0]}|{task[1]}|{pid}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%3

def get_aux(patients,ids):
    ids=set(ids)
    return build_auxiliary([p for p in patients if p["pid"] in ids])

def motor_fit(fit,people,fold,horizon,epochs):
    seed=SEED+100*fold+horizon
    return DirectWM((),style="balanced",epochs=epochs,seed=seed).fit(
        fit,get_aux(people,{r["pid"] for r in fit}))

def optional_matrix(rows,keys):
    x=np.zeros((len(rows),2*len(keys)),dtype=np.float32)
    for i,r in enumerate(rows):
        for j,k in enumerate(keys):
            if k not in r["optional"]:continue
            val=float(r["optional"][k])
            if not np.isfinite(val):raise ValueError("Nonfinite measured clinical value")
            scale=40. if k.startswith(("MoCA","UPSIT")) else 4.
            x[i,j]=val/scale
            x[i,j+len(keys)]=1.
    return x

class CrossFitClinicalResidual:
    """FIT-only nonmotor conditional correction on top of unbiased OOF Motor.

    Missing optional features do not receive a fictitious measured score.
    Conditional output is normalized to zero for an all-missing panel.
    """
    def __init__(self,keys):
        self.keys=tuple(keys)
    def fit(self,fit,people,task,outer_fold,epochs):
        rows=list(fit)
        person_ids=[r["pid"] for r in rows]
        if len(set(person_ids))!=len(rows):
            raise AssertionError("Duplicate cross-fit train patient")
        assignments=np.array([hash_fold(p,task) for p in person_ids])
        oof=np.full(len(rows),np.nan)
        splits=[]
        for k in range(3):
            tr=[rows[i] for i in range(len(rows)) if assignments[i]!=k]
            te=[i for i in range(len(rows)) if assignments[i]==k]
            if not te:continue
            if len(tr)<20:
                raise ValueError("Too few FIT persons for inner crossfit")
            valid_ids={r["pid"] for r in tr}
            # Inner validation person's FUTURE and ALL self-supervised
            # history stay excluded from this inner-fold model.
            model=DirectWM((),style="balanced",epochs=epochs,
                           seed=SEED+100*outer_fold+task[1]+1000+k)
            model.fit(tr,get_aux(people,valid_ids))
            oof[te]=model.predict([rows[i] for i in te])[0]
            splits.append((len(tr),len(te)))
        if not np.isfinite(oof).all():
            raise AssertionError("OOF residual predictions incomplete")
        _,truth,_=labels(rows)
        residual=truth-oof
        self.bias=float(np.median(residual))  # L1-optimal train-only correction
        self.has_nonmotor=(len(self.keys)>0 and
             sum(any(k in r["optional"] for k in self.keys) for r in rows)>=20)
        self.oof_fit_mae=float(np.mean(np.abs(oof-truth)))
        self.oof_bias_fit_mae=float(np.mean(np.abs(oof+self.bias-truth)))
        self.fit_subjects=len(rows)
        self.splits=splits
        if self.has_nonmotor:
            X=optional_matrix(rows,self.keys)
            # Regularized and shallow due to low effective sample count.
            self.model=HistGradientBoostingRegressor(
                max_iter=110,learning_rate=.035,max_leaf_nodes=5,
                min_samples_leaf=16,l2_regularization=60.,
                random_state=SEED+outer_fold+task[1])
            self.model.fit(X,residual-self.bias)
            self.empty_baseline=float(self.model.predict(
                np.zeros((1,X.shape[1]),dtype=np.float32))[0])
        return self
    def correction(self,rows):
        if not self.has_nonmotor:
            return np.zeros(len(rows),dtype=float)
        x=optional_matrix(rows,self.keys)
        answer=np.asarray(self.model.predict(x),dtype=float)-self.empty_baseline
        # All-missing clinical observations get zero clinical-specific
        # correction even when a model would have assigned a constant.
        present=x[:,len(self.keys):].sum(axis=1)>0
        return np.where(present,np.clip(answer,-3.,3.),0.)

def predict_revision(motor_predictions,rows,residual,gamma,add_bias=True):
    burden,risk=motor_predictions
    shift=(residual.bias if add_bias else 0.)
    correction=residual.correction(rows)*float(gamma)
    # Nonmotor branch changes severity, while nonzero-risk remains
    # fixed. The Brier guardrail is therefore explicit but vacuous.
    return (np.maximum(0.,np.asarray(burden)+shift+correction),
            np.asarray(risk,dtype=float).copy())

def select_verified(base_model,residual,tune,audit):
    """Propose one correction on DEV; audit it once; rollback to pure Motor.

    Candidate tuning is based only on MAE, with a predeclared margin.
    A plain bias correction is separately evaluated to distinguish
    calibration from nonmotor predictive gain.
    """
    base_dev=base_model.predict(tune)
    base_audit=base_model.predict(audit)
    baseline_dev=score(base_dev,tune)
    baseline_audit=score(base_audit,audit)
    proposals=[(score(predict_revision(base_dev,tune,residual,g),tune)["mae"],g)
               for g in DEV_GAMMAS]
    dev_best,gamma=min(proposals,key=lambda a:(a[0],a[1]))
    audit_candidate=score(predict_revision(base_audit,audit,residual,gamma),audit)
    accepted=(dev_best+0.01<baseline_dev["mae"] and
              audit_candidate["mae"]+0.01<baseline_audit["mae"] and
              audit_candidate["brier"]<=baseline_audit["brier"]+0.005)
    return (gamma if accepted else 0.),bool(accepted),{
        "proposal":"nonmotor_oof_residual",
        "candidate_gamma":gamma,"accepted":bool(accepted),
        "audit_proposals":1,"dev_baseline_mae":baseline_dev["mae"],
        "dev_candidate_mae":dev_best,
        "audit_baseline_mae":baseline_audit["mae"],
        "audit_candidate_mae":audit_candidate["mae"],
        "audit_baseline_brier":baseline_audit["brier"],
        "audit_candidate_brier":audit_candidate["brier"],
        "fit_oof_mae":residual.oof_fit_mae,
        "fit_oof_bias_mae":residual.oof_bias_fit_mae,
        "fit_residual_bias":residual.bias}

def comparison(name,reference,candidate,rows,fold,horizon):
    return {"reference":name[0],"candidate":name[1],
            **paired_bootstrap(reference,candidate,rows,
                seed=SEED+100*fold+horizon)}

def execute(args):
    if not 0<=args.arm_id<20 or not 4<=args.epochs<=100:
        raise ValueError("Bad arm or epoch count")
    held,variant=divmod(args.arm_id,4)
    root=args.data_root.resolve()
    out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing local PPMI or frozen cohort manifest")
    if out==root or root in out.parents or out==args.manifest_db.resolve().parent:
        raise ValueError("Unsafe output location")
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    stage("before_load_cohorts",arm=args.arm_id)
    people,source=load_cohorts(root,args.manifest_db)
    stage("after_load_cohorts",patients=len(people))
    evidence,qc,_=evidence_tables(root,people)
    stage("after_evidence_tables",patients=len(evidence))
    results=[];paired=[];audit_log=[];coverage=[]
    for task in TASKS:
        stage("task_start",cohort=task[0],horizon=task[1],variant=variant)
        subset=[p for p in people if p["cohort"]==task[0]]
        target,_=future_site_pairs(subset,task[1])
        stage("after_target_pairs",count=len(target))
        rows=data_rows(subset,target,evidence,task)
        stage("after_data_rows",count=len(rows))
        fit,tune,audit,eval_rows=split_data(rows,task,held)
        stage("after_split",fit=len(fit),tune=len(tune),
              audit=len(audit),eval=len(eval_rows))
        if min(map(len,(fit,tune,audit,eval_rows)))<10:
            coverage.append({"cohort":task[0],"horizon":task[1],
                             "status":"insufficient_split"})
            continue
        keys=optional_keys(fit)
        stage("before_motor_fit",epochs=args.epochs,features=len(keys))
        motor=motor_fit(fit,subset,held,task[1],args.epochs)
        stage("after_motor_fit")
        motor_pred=motor.predict(eval_rows)
        stage("after_motor_predict")
        stage("before_hgb_fit")
        predictions={
            "motor_wm":motor_pred,
            "motor_hgb":TimeTabular("hgb").fit(fit,()).predict(eval_rows),
            "multimodal_hgb":TimeTabular("hgb").fit(fit,keys).predict(eval_rows),
        }
        stage("after_hgb_fit")
        if variant>=1:
            stage("before_early_fusion_fit")
            early=DirectWM(keys,style="early",epochs=args.epochs,
                           seed=SEED+100*held+task[1]).fit(
                               fit,get_aux(subset,{r["pid"] for r in fit}))
            predictions["early_multimodal"]=early.predict(eval_rows)
            stage("after_early_fusion_fit")
        residual=None
        if variant>=2:
            stage("before_crossfit_residual")
            residual=CrossFitClinicalResidual(keys).fit(
                fit,subset,task,held,args.epochs)
            stage("after_crossfit_residual",inner_splits=len(residual.splits))
            predictions["bias_only"]=predict_revision(
                motor_pred,eval_rows,residual,0.)
            predictions["nonmotor_residual"]=predict_revision(
                motor_pred,eval_rows,residual,1.)
        if variant==3:
            stage("before_verifier")
            gamma,accepted,decision=select_verified(motor,residual,tune,audit)
            stage("after_verifier",accepted=bool(accepted))
            # Rollback means exact incumbent Motor, not a silent
            # bias-only update without an independent audit.
            predictions["verified_residual"]=(
                predict_revision(motor_pred,eval_rows,residual,gamma)
                if accepted else motor_pred)
            audit_log.append({"cohort":task[0],"horizon":task[1],
                              "outer_fold":held,**decision})
        primary={0:"motor_wm",1:"early_multimodal",
                 2:"nonmotor_residual",3:"verified_residual"}[variant]
        for method,pred in predictions.items():
            results.append({"cohort":task[0],"horizon":task[1],
                "outer_fold":held,"arm":args.arm_id,"method":method,
                "is_primary":int(method==primary),
                "patients":len(eval_rows),
                "positive":int(labels(eval_rows)[2].sum()),
                "optional_panel_size":len(keys),
                "mean_observed_visits":float(np.mean(
                    [len(r["history"]) for r in eval_rows])),
                **score(pred,eval_rows)})
        comparisons=[
            ("motor_hgb","motor_wm"),
            ("multimodal_hgb","motor_wm"),
        ]
        for candidate in ("early_multimodal","bias_only",
                          "nonmotor_residual","verified_residual"):
            if candidate in predictions:
                comparisons.append(("motor_wm",candidate))
                if candidate in ("nonmotor_residual","verified_residual"):
                    comparisons.append(("multimodal_hgb",candidate))
        if "bias_only" in predictions and "nonmotor_residual" in predictions:
            comparisons.append(("bias_only","nonmotor_residual"))
        for reference,candidate in comparisons:
            paired.append({"cohort":task[0],"horizon":task[1],
                           "outer_fold":held,"arm":args.arm_id,
                           **comparison((reference,candidate),
                                        predictions[reference],
                                        predictions[candidate],eval_rows,
                                        held,task[1])})
        coverage.append({"cohort":task[0],"horizon":task[1],
            "outer_fold":held,"status":"complete",
            "fit":len(fit),"dev":len(tune),"audit":len(audit),
            "evaluation":len(eval_rows),"optional_features":len(keys),
            "fit_real_visit_pairs":len(get_aux(subset,{r["pid"] for r in fit})),
            "inner_oof_splits":residual.splits if residual else [],
            "eligible_nonmotor_fit":sum(any(k in r["optional"] for k in keys)
                                       for r in fit)})
        write_csv(out/"method_metrics.csv",results)
        write_csv(out/"paired_bootstrap.csv",paired)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audit_log,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task),"elapsed_sec":time.perf_counter()-start},indent=2)+"\n")
        print("PHASE77_TASK_COMPLETE",task,"arm",args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":77,"arm":args.arm_id,"outer_fold":held,
        "strategy":variant,"completed_tasks":len(coverage),
        "walltime_s":time.perf_counter()-start,
        "PPMI_source":source["source_motor_table"],"qc":qc,
        "scope":[
         "Internal exploratory frozen Prodromal TRAIN and strict PD only",
         "No patient in DEV/AUDIT/OUTER enters WM or OOF residual training",
         "Residuals trained on fit-only threefold out-of-fold motor forecasts",
         "No individual clinical outputs, raw data or Qwen prompts exported",
         "True Motor pre-cutoff history and BL optional clinical values only",
         "Risk head unchanged by residual severity correction; Brier comparison is vacuous",
         "DEV selects one bounded candidate; independent AUDIT may veto",
         "This is not autonomous RSI or a learned treatment counterfactual",
         "Previously consumed final TEST never accessed"
        ],"finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print(json.dumps({"status":"success","arm":args.arm_id,
                      "tasks":len(coverage),"output":str(out)}))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=18)
    args=p.parse_args()
    faulthandler.enable(file=sys.stderr,all_threads=True)
    faulthandler.dump_traceback_later(900,repeat=True,file=sys.stderr)
    stage("main_enter",arm=args.arm_id)
    try:
        execute(args)
    finally:
        faulthandler.cancel_dump_traceback_later()
if __name__=="__main__":
    main()
