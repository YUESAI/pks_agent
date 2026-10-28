#!/usr/bin/env python3
"""Phase73: patient-disjoint prospective-cutoff PPMI WM/Doctor/Verifier.

Five patient folds × four standalone CPU strategies. Every job fits exactly
the same FIT-only Ridge reference, single continuous-time WM, and 3-seed
ensemble; arm 3 additionally asks *real Qwen* for one previously measured
baseline clinical test and applies a DEV/AUDIT-gated checkpoint revision.
The same-patient matched reference metrics are computed WITHIN EACH job.

No raw patient data, names, prompts, patient IDs, or forecasts are written.
No visits on/after earliest acceptable target date may enter the predictor.
This is an exploratory retrospective test, not intervention/causal RSI.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs,HIDX
from scripts.phase64_evidence import evidence_tables
from scripts.phase36_masked_site_feasibility import ALL,VISIBLE
from scripts.phase72_ctlatent_wm import (
    EnsembleWM,build_auxiliary,fixed_ridge_features)
from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase60_cpu_multistep_wm import auc

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
VINDEX=tuple(ALL.index(s) for s in VISIBLE)
CUTOFF={12:8,24:16}
TOLERANCE={12:3,24:4}
MAX_KEYS=10


def patient_fold(pid,task):
    digest=hashlib.sha256(f"phase66|{SEED}|{task}|{pid}".encode()).hexdigest()
    return int(digest[:12],16)%5


def stable(pid,salt):
    return hashlib.sha256(f"phase73|{salt}|{pid}".encode()).hexdigest()


def data_rows(people,truth,raw_evidence,task):
    mapping={p["pid"]:p for p in people}
    horizon=task[1];cutoff=CUTOFF[horizon]
    rows=[]
    for entry in truth:
        person=mapping[entry["pid"]]
        base=person["baseline_month"]
        earlier=[v for v in person["visits"] if
                 0<=v["month"]-base<=cutoff]
        if not earlier or earlier[0]["month"]!=base:
            raise AssertionError("Lost original BL visit")
        history=[{"month":int(v["month"]),
             "scores":{name:int(v["score"][j])
                       for name,j in zip(VISIBLE,VINDEX)
                       if v["status"][j]==1}}
             for v in earlier]
        if any(x["month"]-base>cutoff for x in history):
            raise AssertionError("Future measurement in patient history")
        # Reproduce Phase62's selected verified future onsite target:
        eligible=[v for v in person["visits"][1:]
          if abs(v["month"]-base-horizon)<=TOLERANCE[horizon]
          and v["mask_type"]=="complete_33_verified_total"]
        eligible.sort(key=lambda v:abs(v["month"]-base-horizon))
        if not eligible or (len(eligible)>1 and
            abs(eligible[0]["month"]-base-horizon)==
            abs(eligible[1]["month"]-base-horizon)):
            raise AssertionError("Ambiguous strict future label")
        target=eligible[0]
        if not target["month"]-base>cutoff:
            raise AssertionError("Prediction cutoff touches target visit")
        site=[int(target["score"][i]) for i in HIDX]
        if sum(site)!=entry["target_burden"]:
            raise AssertionError("Future site burden target not matched")
        optional={k:float(v) for k,v in raw_evidence[entry["pid"]].items()
                  if not k.startswith(("BL_","R01_"))}
        rows.append({"pid":entry["pid"],"anchor":base,"horizon":horizon,
                     "history":history,"optional":optional,
                     "released":{},"site":site})
    if len({r["pid"] for r in rows})!=len(rows):
        raise AssertionError("Duplicate subject in horizon")
    return rows


def split_data(rows,task,held):
    outer=[r for r in rows if patient_fold(r["pid"],task)==held]
    pool=sorted((r for r in rows if patient_fold(r["pid"],task)!=held),
                key=lambda r:stable(r["pid"],"inner"))
    keep=max(20,int(len(pool)*.30))
    dev=pool[-keep:]
    fit=pool[:-keep]
    mid=len(dev)//2
    tune,audit=dev[:mid],dev[mid:]
    groups=[{r["pid"] for r in g} for g in (fit,tune,audit,outer)]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4)):
        raise AssertionError("Patient leaks across FIT DEV AUDIT OUTER")
    return fit,tune,audit,outer


def chosen_optional_keys(fit):
    support=Counter(k for r in fit for k in r["optional"])
    # Entire optional alphabet is determined strictly by FIT. No
    # feature/outcome selection or leakage from heldout patients.
    return tuple(k for k,c in sorted(support.items(),key=lambda x:(-x[1],x[0]))
                 if c>=8)[:MAX_KEYS]


def fixed_release(rows,keys):
    returned=[]
    for record in rows:
        copy=dict(record)
        available=sorted(k for k in keys if k in record["optional"])
        copy["released"]=({available[0]:record["optional"][available[0]]}
                          if available else {})
        returned.append(copy)
    return returned


def heuristic_gain(wm,record,available,fit_support):
    """FIT-only empirical optional-score variations as *hypothetical* advice.

    Not conditional acquisition utility, not real simulated measurements.
    The only data released to Doctor after choosing is the real recorded
    original clinical test.
    """
    if not available:return {}
    contexts=[]
    offsets={}
    for key in available:
        options=fit_support.get(key,[])
        if not len(options):continue
        start=len(contexts)
        for val in options:
            example=dict(record)
            example["released"]={key:float(val)}
            contexts.append(example)
        offsets[key]=(start,len(contexts))
    if not contexts:return {k:0. for k in available}
    means,_,_=wm.infer(contexts,"ensemble")
    result={}
    for key in available:
        if key not in offsets:result[key]=0.;continue
        a,b=offsets[key]
        result[key]=float(np.var(means[a:b]))
    return result


def qwen_release(rows,keys,doctor,wm,fit_support,task_label):
    result=[];invalid=0;called=0
    for record in rows:
        copied=dict(record)
        available=sorted(k for k in keys if k in record["optional"])
        if not available:
            copied["released"]={};result.append(copied);continue
        advice=heuristic_gain(wm,record,available,fit_support)
        # LLM NEVER receives current patient values, target scores or PID.
        messages=[{"role":"system","content":
            "Research-only selection of an actually recorded clinical baseline test. "
            "Choose one allowed name; output only JSON {\"ask\":\"NAME\"}. "
            "Never claim a future prediction is a measurement."},
          {"role":"user","content":
            "Task (months): "+str(task_label)+
            ". Available true previously recorded test names: "+
            json.dumps(available)+
            ". Hypothetical FIT-model sensitivity scores (higher better, "
            "not actual observations): "+
            json.dumps({k:round(float(advice[k]),5) for k in available})+
            '. Respond with exactly {"ask":"ONE_ALLOWED_NAME"}.'}]
        reply=doctor.generate(messages);called+=1
        selected=extract_action(reply,set(available))
        if selected is None:
            invalid+=1
            selected=min(available,key=lambda k:stable(
                record["pid"],"fallback"+k))
        # Only this line releases the ACTUALLY measured test.
        copied["released"]={selected:record["optional"][selected]}
        result.append(copied)
    return result,{"qwen_prompts":called,"invalid_qwen_actions":invalid}


def fit_reference(fit,keys):
    X=fixed_ridge_features(fixed_release(fit,keys),keys)
    y=np.asarray([sum(r["site"]) for r in fit],dtype=float)
    positive=(y>0).astype(int)
    regression=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    regression.fit(X,y)
    if len(set(positive))>=2:
        binary=make_pipeline(StandardScaler(),LogisticRegression(
            C=.3,max_iter=1000,random_state=SEED))
        binary.fit(X,positive)
    else:binary=None
    prior=float((positive.sum()+1)/(len(positive)+2))
    return regression,binary,prior


def reference_predict(reference,rows,keys):
    reg,bin_model,prior=reference
    X=fixed_ridge_features(rows,keys)
    burden=np.maximum(0,np.asarray(reg.predict(X),dtype=float))
    risk=(bin_model.predict_proba(X)[:,1] if bin_model else
          np.full(len(rows),prior))
    return burden,np.clip(risk,0,1),np.zeros(len(rows))


def select_revision(wm,dev_rows,audit_rows):
    """Predeclared checkpoint 0 vs ensemble, one independent audit.

    The same REAL Qwen-selected item is used for both candidates.
    One candidate reaches audit; no repeated audit querying.
    """
    def loss(rs,member):
        prediction=wm.infer(rs,member)
        target=np.array([sum(r["site"]) for r in rs],float)
        risk=(target>0).astype(float)
        return (float(np.mean(np.abs(prediction[0]-target))),
                float(np.mean((prediction[1]-risk)**2)))
    base_dev=loss(dev_rows,0)
    candidate_dev=loss(dev_rows,"ensemble")
    base_audit=loss(audit_rows,0)
    candidate_audit=loss(audit_rows,"ensemble")
    accepted=(candidate_dev[0]+.003<base_dev[0] and
              candidate_audit[0]+.003<base_audit[0] and
              candidate_audit[1]<=base_audit[1]+.01)
    chosen="ensemble" if accepted else 0
    return chosen,{"candidate":"ensemble","incumbent":"member0",
        "accepted":bool(accepted),
        "dev_baseline_mae":base_dev[0],
        "dev_candidate_mae":candidate_dev[0],
        "audit_baseline_mae":base_audit[0],
        "audit_candidate_mae":candidate_audit[0],
        "audit_baseline_brier":base_audit[1],
        "audit_candidate_brier":candidate_audit[1],
        "audited_proposals":1,"same_qwen_evidence_for_both_checkpoints":True}


def measure(pred,rows):
    y=np.asarray([sum(r["site"]) for r in rows],float)
    binary=(y>0).astype(float)
    yh,p,variance=pred
    return {"mae":float(np.mean(np.abs(yh-y))),
       "rmse":float(np.sqrt(np.mean((yh-y)**2))),
       "brier":float(np.mean((p-binary)**2)),
       "auroc":auc(p.tolist(),binary.astype(int).tolist()),
       "ensemble_disagreement":float(np.mean(variance))}


def contrast(reference,candidate,rows,seed,n_boot=500):
    y=np.asarray([sum(r["site"]) for r in rows],float)
    first=np.asarray(reference[0]);second=np.asarray(candidate[0])
    delta=np.abs(second-y)-np.abs(first-y)
    rng=np.random.default_rng(seed)
    bs=np.array([np.mean(delta[rng.integers(0,len(y),len(y))])
                 for i in range(n_boot)])
    return {"delta_mae":float(np.mean(delta)),
            "ci_low":float(np.quantile(bs,.025)),
            "ci_high":float(np.quantile(bs,.975))}


def csv_rows(path,rows):
    if not rows:return
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("data-root","manifest-db","report-dir","model-dir"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--arm-id",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=12)
    args=ap.parse_args()
    if not 0<=args.arm_id<20 or not 2<=args.epochs<=100:
        raise ValueError("20 arms and 2..100 epochs only")
    held,arm=divmod(args.arm_id,4)
    root=args.data_root.resolve();out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("PPMI inputs or frozen TRAIN manifest missing")
    if out==root or root in out.parents or out==args.manifest_db.resolve().parent:
        raise ValueError("Unsafe output directory")
    started=time.time()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,people)
    doctor=TransformersDoctor(args.model_dir,max_new_tokens=40,device="cpu") if arm==3 else None
    metrics=[];paired=[];audit_info=[];coverage=[]
    for task in TASKS:
        subset=[p for p in people if p["cohort"]==task[0]]
        future,_=future_site_pairs(subset,task[1])
        rows=data_rows(subset,future,evidence,task)
        fit,dev,audit,evaluation=split_data(rows,task,held)
        if min(map(len,(fit,dev,audit,evaluation)))<10:continue
        keys=chosen_optional_keys(fit)
        fitids={r["pid"] for r in fit}
        # Extra FIT-only self-supervised trajectories can train the
        # recurrent WM but cannot contribute invented outcome labels.
        all_ids={r["pid"] for r in rows}
        extraids={p["pid"] for p in subset
                  if p["pid"] not in all_ids
                  and patient_fold(p["pid"],task)!=held
                  and int(stable(p["pid"],"extra")[:2],16)%10<7}
        auxiliary=build_auxiliary([p for p in subset
                                  if p["pid"] in fitids|extraids])
        model=EnsembleWM(keys,epochs=args.epochs,members=3,
                         seed=SEED+held*100+task[1]).fit(fit,auxiliary)
        ridge=fit_reference(fit,keys)
        fit_support={}
        for key in keys:
            values=np.array([r["optional"][key] for r in fit
                             if key in r["optional"]],float)
            fit_support[key]=(np.unique(np.quantile(values,[.1,.5,.9]))
                              if len(values) else [])
        fixed=fixed_release(evaluation,keys)
        predictions={
            "ridge_fixed":reference_predict(ridge,fixed,keys),
            "wm_single_fixed":model.infer(fixed,0),
            "wm_ensemble_fixed":model.infer(fixed,"ensemble"),
        }
        qstats={"qwen_prompts":0,"invalid_qwen_actions":0}
        verifier={"accepted":False,"audited_proposals":0}
        if arm==3:
            qdev,s1=qwen_release(dev,keys,doctor,model,fit_support,task[1])
            qaudit,s2=qwen_release(audit,keys,doctor,model,fit_support,task[1])
            qevaluation,s3=qwen_release(evaluation,keys,doctor,model,
                                       fit_support,task[1])
            chosen,verifier=select_revision(model,qdev,qaudit)
            predictions["wm_single_qwen"]=model.infer(qevaluation,0)
            predictions["wm_ensemble_qwen"]=model.infer(qevaluation,"ensemble")
            predictions["wm_verified_qwen"]=model.infer(qevaluation,chosen)
            qstats={"qwen_prompts":s1["qwen_prompts"]+s2["qwen_prompts"]+s3["qwen_prompts"],
                    "invalid_qwen_actions":s1["invalid_qwen_actions"]+s2["invalid_qwen_actions"]+s3["invalid_qwen_actions"]}
            audit_info.append({"task":str(task),**verifier})
        main_method={0:"ridge_fixed",1:"wm_single_fixed",
                     2:"wm_ensemble_fixed",3:"wm_verified_qwen"}[arm]
        for method,values in predictions.items():
            metrics.append({"cohort":task[0],"horizon":task[1],
                    "outer_fold":held,"arm_id":args.arm_id,
                    "is_primary":int(method==main_method),
                    "method":method,"patients":len(evaluation),
                    "positive":int(sum(sum(r["site"])>0 for r in evaluation)),
                    "mean_observed_visits":float(np.mean([len(r["history"]) for r in evaluation])),
                    "optional_features":len(keys),"fit_supervised":len(fit),
                    "self_supervised_visit_pairs":len(auxiliary),
                    **measure(values,evaluation)})
        def paired_vs(a,b):
            return {"cohort":task[0],"horizon":task[1],"fold":held,
                    "arm_id":args.arm_id,"reference":a,"candidate":b,
                    **contrast(predictions[a],predictions[b],evaluation,
                               seed=SEED+held*17+task[1])}
        paired.append(paired_vs("ridge_fixed","wm_single_fixed"))
        paired.append(paired_vs("wm_single_fixed","wm_ensemble_fixed"))
        if arm==3:
            paired.append(paired_vs("wm_ensemble_fixed","wm_ensemble_qwen"))
            paired.append(paired_vs("wm_single_qwen","wm_verified_qwen"))
        coverage.append({"cohort":task[0],"horizon":task[1],
                   "fold":held,"supervised_fit":len(fit),
                   "dev":len(dev),"audit":len(audit),"evaluation":len(evaluation),
                   "auxiliary_fit_pairs":len(auxiliary),
                   "n_with_at_least_2_past_visits":sum(
                       len(r["history"])>=2 for r in evaluation),
                   "n_with_3plus_past_visits":sum(
                       len(r["history"])>=3 for r in evaluation),
                   **qstats})
    out.mkdir(parents=True,exist_ok=False)
    csv_rows(out/"method_metrics.csv",metrics)
    csv_rows(out/"paired_bootstrap.csv",paired)
    csv_rows(out/"coverage.csv",coverage)
    (out/"audit.json").write_text(json.dumps(audit_info,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
          "phase":73,"arm_id":args.arm_id,"outer_fold":held,"main_arm":arm,
          "elapsed_sec":round(time.time()-started,2),
          "total_qwen_prompts":doctor.total_prompts if doctor else 0,
          "multimodal_qc":quality,
          "protocol":"Patient-disjoint, exact horizon cutoff; all in-job model baselines evaluated on identical fixed clinical evidence. Actual Qwen evidence reused for both verifier candidates.",
          "restrictions":["Exploratory frozen Prodromal TRAIN + strict PD",
             "24m at month16 and 12m at month8 prediction cutoff",
             "No post-cutoff visits ever passed to prediction",
             "Qwen optional baseline test acquisition is retrospective",
             "Verifier performs one allowlisted model checkpoint gate",
             "No autonomous self-modifying RSI, no causal intervention",
             "No patient-level records, prompts or identifiers exported"]
          },indent=2)+"\n")
    print(json.dumps({"status":"success","arm":args.arm_id,
            "completed_tasks":len(coverage),"report":str(out)}))
if __name__=="__main__":
    main()
