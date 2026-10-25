#!/usr/bin/env python3
"""Phase68 masked neural-WM formal 5-fold x 4-policy CPU benchmark. Twenty independent sbatch jobs.

Policies: deterministic belief planner, plain Qwen, WM-guided Qwen,
WM-guided Qwen + independent DEV/AUDIT verifier-gated temperature/style.
No neural model weight rewriting. All data stays on cluster.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables,panel_items,episode_rows,PatientTool
from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase68_multimodal_neural_wm import MultimodalMaskedWM
from scripts.phase60_cpu_multistep_wm import auc

TASKS=(("prodromal",12),("prodromal",24),("PD",12))
POLICIES=("greedy","qwen_plain","qwen_wm","qwen_verified")
SEED=20261018

def fold(pid,task):
    # Same fixed outcome-blind fold for all 20 jobs.
    h=hashlib.sha256(f"phase66|{SEED}|{task}|{pid}".encode()).hexdigest()
    return int(h[:12],16)%5

def config(arm):
    if not 0<=arm<20:raise ValueError("Invalid arm")
    return arm//4,POLICIES[arm%4]

def messages(remaining,observed,scores,policy):
    content=("You are a research clinical-evidence request agent, not a clinician. "
        "Select exactly ONE item from the allowed list. Do not infer or "
        "fabricate scores, diagnoses, patient identity, future outcomes. "
        'Return only {"ask":"ITEM"}. Allowed: '+
        json.dumps(sorted(remaining))+". Measured evidence so far: "+
        json.dumps(observed,sort_keys=True)+".")
    if policy in ("qwen_wm","qwen_verified"):
        best=sorted(remaining,key=lambda k:(-scores[k],k))[:8]
        content+=("\nModel-estimated expected uncertainty reduction (NOT "
                  "actual measured evidence): "+
                  json.dumps({k:round(scores[k],6) for k in best})+".")
    return [{"role":"system","content":"Strict tool-name request as one JSON object."},
            {"role":"user","content":content}]

def request_episode(r,wm,items,q,policy,doctor,style="variance",temperature=1.):
    available=set(items)&set(r["measured"])
    tool=PatientTool(r["measured"],available,min(q,len(available)))
    invalid=0
    while len(tool.observed)<tool.budget:
        remaining=tool.available-set(tool.observed)
        vals=wm.action_scores(tool.observed,remaining,style,temperature)
        best=min(remaining,key=lambda k:(-vals[k],k))
        if policy=="greedy":action=best
        else:
            raw=doctor.generate(messages(remaining,tool.observed,vals,policy))
            action=extract_action(raw,set(remaining))
            if action is None:
                invalid+=1
                # WM-free, fixed fallback for plain Qwen.
                action=(min(remaining,key=lambda k:hashlib.sha256(
                    f"phase68plain|{r['pid']}|{len(tool.observed)}|{k}".encode()
                    ).hexdigest()) if policy=="qwen_plain" else best)
        answer=tool.request(action)
        if answer["provenance"]!="actually_measured_BL_or_pre_target_R01":
            raise AssertionError("Unmeasured evidence escaped")
    return (*wm.outcomes(tool.observed,temperature),len(tool.observed),invalid)

def verifier(wm,tune,audit,items,q):
    """One candidate chosen on TUNE, accepted/rejected ONCE on AUDIT.

    This optimizes a deterministic proxy; no claim of real-Qwen DEV fitness.
    """
    candidates=[(style,temp) for style in ("variance","entropy")
                for temp in (.7,1.,1.3,1.6)]
    incumbent=("variance",1.)
    def mse(rows,cfg):
        predictions=[request_episode(r,wm,items,q,"greedy",None,*cfg)[0]
                     for r in rows]
        return sum((p-r["burden"])**2 for p,r in zip(predictions,rows))/len(rows)
    first=mse(tune,incumbent)
    options=[(mse(tune,x),x) for x in candidates]
    best,choice=min(options)
    before=mse(audit,incumbent);after=mse(audit,choice)
    accepted=(first-best>1e-4 and before-after>1e-4)
    chosen=choice if accepted else incumbent
    return chosen,{"candidate_style":choice[0],"candidate_temperature":choice[1],
        "selected_style":chosen[0],"selected_temperature":chosen[1],
        "accepted":accepted,"audited_proposals":1,
        "tune_before":first,"tune_candidate":best,
        "audit_before":before,"audit_candidate":after}

def writecsv(path,rows):
    if rows:
        with path.open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)

def run(args):
    held_fold,policy=config(args.arm_id)
    root=args.data_root.resolve();out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing inputs")
    if out==root or root in out.parents or out==args.manifest_db.resolve().parent:
        raise ValueError("Unsafe output")
    started=time.perf_counter()
    patients,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,patients)
    doctor=(TransformersDoctor(args.model_dir,args.max_new_tokens,device="cpu")
            if policy!="greedy" else None)
    metrics=[];trace=[];coverage=[]
    for task in TASKS:
        subset=[p for p in patients if p["cohort"]==task[0]]
        targets,_=future_site_pairs(subset,task[1])
        rows=episode_rows(subset,targets,evidence,*task)
        if len(rows)<60:
            coverage.append({"task":str(task),"status":"too_few","n":len(rows)})
            continue
        fit=[r for r in rows if fold(r["pid"],task)!=held_fold]
        eval_rows=[r for r in rows if fold(r["pid"],task)==held_fold]
        # Split fit by patient hash; inner tune/audit never used in WM fit.
        sorted_fit=sorted(fit,key=lambda r:hashlib.sha256(
            f"phase68dev|{r['pid']}".encode()).hexdigest())
        n=len(sorted_fit);n_dev=max(20,int(n*.30))
        fit_rows=sorted_fit[:-n_dev]
        tune=sorted_fit[-n_dev:-(n_dev//2)]
        audit=sorted_fit[-(n_dev//2):]
        ids=[{r["pid"] for r in grp} for grp in (fit_rows,tune,audit,eval_rows)]
        if any(ids[i]&ids[j] for i in range(4) for j in range(i+1,4)):
            raise AssertionError("Patient leakage")
        if min(map(len,(fit_rows,tune,audit,eval_rows)))<10:
            coverage.append({"task":str(task),"status":"small_split","n":len(rows)})
            continue
        fit_ids={r["pid"] for r in fit_rows}
        all_labeled_ids={r["pid"] for r in rows}
        # Extra unlabelled observation profiles belong ONLY to outer FIT
        # by the same subject hash; never use heldout or DEV IDs.
        extra=[{"pid":p["pid"],"measured":evidence[p["pid"]]} for p in subset
               if p["pid"] not in all_labeled_ids and
               fold(p["pid"],task)!=held_fold and
               int(hashlib.sha256(f"phase66dev|{p['pid']}".encode()
                   ).hexdigest()[:8],16)%10<7]
        # Panel 4 is fullest available baseline+recorded R01, may contain
        # patient-specific missing values, never fabricated.
        keys=sorted(set().union(*(set(r["measured"]) for r in fit_rows+extra)))
        allowed=panel_items(4,keys)
        # FIT-side unlabelled BL and permitted earlier R01 measurements only.
        # No future motor visit is relabelled BL; no heldout patient
        # contributes to outcome or observation WM fitting.
        wm=MultimodalMaskedWM(allowed,epochs=args.epochs,
                q=args.questions,seed=SEED+held_fold).fit(fit_rows,extra)
        choice=("variance",1.)
        ver={"accepted":False,"audited_proposals":0}
        if policy=="qwen_verified":
            choice,ver=verifier(wm,tune,audit,allowed,args.questions)
        result={}
        for r in eval_rows:
            result[r["pid"]]=request_episode(
                r,wm,allowed,args.questions,policy,doctor,*choice)
        values=np.array([result[r["pid"]][0] for r in eval_rows])
        truth=np.array([r["burden"] for r in eval_rows])
        risks=np.array([result[r["pid"]][1] for r in eval_rows])
        labels=np.array([r["label"] for r in eval_rows])
        pos=int(labels.sum());n=len(eval_rows)
        quota=max(1,int(.2*n+.5))
        order=sorted(range(n),key=lambda i:(-risks[i],eval_rows[i]["pid"]))
        found=int(labels[order[:quota]].sum())
        metrics.append({"cohort":task[0],"horizon":task[1],"fold":held_fold,
            "policy":policy,"patients":n,"positive":pos if pos>=10 else "<10",
            "captured":found if found>=10 else "<10",
            "recall_at_20":found/pos if min(pos,found)>=10 else "",
            "mae":float(np.mean(np.abs(values-truth))),
            "rmse":float(np.mean((values-truth)**2)**.5),
            "auroc":auc(risks.tolist(),labels.tolist()),
            "brier":float(np.mean((risks-labels)**2)),
            "actual_questions":sum(result[r["pid"]][2] for r in eval_rows),
            "invalid_qwen_actions":sum(result[r["pid"]][3] for r in eval_rows),
            "panel_features":len(allowed)})
        coverage.append({"task":str(task),"status":"evaluated",
             "eligible_labeled":len(rows),"train_labeled":len(fit_rows),
             "train_unlabelled":len(extra),"tune":len(tune),"audit":len(audit),
             "outer_eval":n})
        trace.append({"task":str(task),**ver})
    out.mkdir(parents=True,exist_ok=False)
    writecsv(out/"fold_metrics.csv",metrics)
    writecsv(out/"cohort_coverage.csv",coverage)
    (out/"verifier_trace.json").write_text(json.dumps(trace,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
        "arm_id":args.arm_id,"fold":held_fold,"policy":policy,
        "elapsed_seconds":round(time.perf_counter()-started,2),
        "qwen_generations":doctor.total_prompts if doctor else 0,
        "quality_flags":quality,
        "limits":["Exploratory original frozen Prodromal TRAIN and separate PD",
         "Neural masked observation heads trained on measured FIT BL/R01 only",
         "Neural non-motor feature observation heads use FIT-only quantile bins",
         "All actual acquired scores are BL or pre-target R01",
         "Conditional categorical model, not causal temporal dynamics",
         "WM advice estimated from class-conditioned historical measurements",
         "RSI is one-shot verifier of greedy proxy style and temperature",
         "DEV verifier does not rerun Qwen; cannot imply genuine online RSI",
         "No access to consumed original final test",
         "20 comparisons exploratory; no post-selection inference",
         "Only aggregate files exported"],
        "completed_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    print(json.dumps({"status":"success","arm":args.arm_id,
        "completed_tasks":len(metrics),"report":str(out)},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--questions",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=40)

    p.add_argument("--epochs",type=int,default=16)
    a=p.parse_args()
    if not 1<=a.questions<=12 or not 2<=a.epochs<=100:raise ValueError("Invalid config")
    run(a)

if __name__=="__main__":main()
