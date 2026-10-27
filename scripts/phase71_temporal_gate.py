#!/usr/bin/env python3
"""Phase71 5 folds x 4 variants: fixed-evidence temporal WM mechanism gate.

No LLM calls. Same patients, actual two observed clinical items,
same outcomes, in-fold static benchmark. Cohort separation and
tune/audit/outcome folds enforced before model fitting.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables,panel_items,episode_rows
from scripts.phase69_temporal_multimodal_wm import TemporalMultimodalWM,build_motor_transitions
from scripts.phase70_horizon_decision import HorizonFusion
from scripts.phase60_cpu_multistep_wm import auc

TASKS=(("prodromal",12),("prodromal",24),("PD",12))
BLENDS=(0.,.1,.2,.35,.5)
SEED=20261018

def fold(pid,task):
    # IDENTICAL to prior Phase66/70 outside evaluation split
    k=hashlib.sha256(f"phase66|{SEED}|{task}|{pid}".encode()).hexdigest()
    return int(k[:12],16)%5

def split(rows,held):
    outer=[r for r in rows if fold(r["pid"],(r["cohort"],r["horizon"]))==held]
    others=[r for r in rows if fold(r["pid"],(r["cohort"],r["horizon"]))!=held]
    ordered=sorted(others,key=lambda r:hashlib.sha256(
        f"phase71dev|{r['pid']}".encode()).hexdigest())
    ndev=max(20,int(.30*len(ordered)))
    fit=ordered[:-ndev]
    dev=ordered[-ndev:]
    mid=len(dev)//2
    tune,audit=dev[:mid],dev[mid:]
    if len(set().union(*[{r["pid"] for r in x} for x in (fit,tune,audit,outer)]))!=len(rows):
        raise AssertionError("Nonunique or repeated patient in split")
    return fit,tune,audit,outer

def fixed_evidence(rows,items,ranking,q):
    allowed=set(items)
    evidence={}
    for r in rows:
        available=r["measured"]
        selected=[k for k in ranking if k in allowed and k in available][:q]
        evidence[r["pid"]]={k:available[k] for k in selected}
    return evidence

def evaluate(wm,rows,ev,horizon,blend):
    predictor=HorizonFusion(wm,horizon,blend=blend)
    y=np.array([r["burden"] for r in rows],dtype=float)
    binary=np.array([r["label"] for r in rows],dtype=int)
    forecasts=[predictor.outcomes(ev[r["pid"]]) for r in rows]
    mean=np.array([t[0] for t in forecasts])
    p=np.array([t[1] for t in forecasts])
    return mean,p,y,binary

def score(pred):
    val,p,y,binary=pred
    return (float(np.mean(np.abs(val-y))),
            float(np.mean((p-binary)**2)))

def bootstrap(a,b,nboot=500,seed=SEED):
    ay,_,truth,_=a
    by,_,_,_=b
    delta=np.abs(by-truth)-np.abs(ay-truth)
    rng=np.random.default_rng(seed)
    z=np.array([np.mean(delta[rng.integers(0,len(delta),size=len(delta))])
                 for _ in range(nboot)])
    return float(np.mean(delta)),float(np.quantile(z,.025)),float(np.quantile(z,.975))

def metrics(task,held,variant,rows,pred,reference,blend,ntrans):
    yh,p,y,labels=pred
    mae,brier=score(pred)
    refmae,refbrier=score(reference)
    d,lo,hi=bootstrap(reference,pred,seed=SEED+held+task[1])
    return {"cohort":task[0],"horizon":task[1],"fold":held,"variant":variant,
     "patients":len(rows),"positives":int(labels.sum()) if labels.sum()>=10 else "<10",
     "blend":blend,"fit_transitions":ntrans,
     "mae":mae,"rmse":float(np.sqrt(np.mean((yh-y)**2))),
     "brier":brier,"auroc":auc(p.tolist(),labels.tolist()),
     "matched_static_mae":refmae,"matched_static_brier":refbrier,
     "delta_mae_vs_static":d,"paired_ci_low":lo,"paired_ci_high":hi}

def csv_write(path,rows):
    if rows:
        with path.open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--questions",type=int,default=2)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--temporal-epochs",type=int,default=6)
    args=p.parse_args()
    if not 0<=args.arm_id<20 or args.questions!=2:
        raise ValueError("Arm 0..19; frozen q=2")
    held=args.arm_id//4
    variant=args.arm_id%4
    root=args.data_root.resolve();out=args.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not args.manifest_db.is_file():
        raise FileNotFoundError("Missing original inputs")
    if root==out or root in out.parents or out==args.manifest_db.resolve().parent:
        raise ValueError("Unsafe result location")
    t0=time.perf_counter()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,qc,_=evidence_tables(root,people)
    results=[];audit_reports=[];coverage=[]
    for task in TASKS:
        subset=[r for r in people if r["cohort"]==task[0]]
        truth,_=future_site_pairs(subset,task[1])
        rows=episode_rows(subset,truth,evidence,*task)
        if len(rows)<60:continue
        fit,tune,audit,evaluation=split(rows,held)
        if min(map(len,(fit,tune,audit,evaluation)))<10:continue
        label_ids={r["pid"] for r in rows}
        fit_ids={r["pid"] for r in fit}
        # Extra unlabeled profiles: fit-side fold only, never tune/audit/test.
        extra=[{"pid":r["pid"],"measured":evidence[r["pid"]]}
             for r in subset if r["pid"] not in label_ids
             and fold(r["pid"],task)!=held
             and int(hashlib.sha256(f"phase66dev|{r['pid']}".encode()).hexdigest()[:8],16)%10<7]
        eligible=set(fit_ids)|{r["pid"] for r in extra}
        transitions=build_motor_transitions(subset,eligible)
        keys=sorted(set().union(*(set(r["measured"]) for r in fit+extra)))
        allowed=panel_items(4,keys)
        counts={k:sum(k in r["measured"] for r in fit) for k in allowed}
        # No target labels used in acquisition policy.
        ranking=sorted(allowed,key=lambda k:(-counts[k],k))
        ev={**fixed_evidence(tune,allowed,ranking,2),
            **fixed_evidence(audit,allowed,ranking,2),
            **fixed_evidence(evaluation,allowed,ranking,2)}
        seed=SEED+held
        static=TemporalMultimodalWM(allowed,epochs=args.epochs,
                    temporal_epochs=0,seed=seed,q=2).fit(fit,extra,())
        baseline=evaluate(static,evaluation,ev,task[1],0.)
        if variant==0:
            trained=static;blend=0.
        else:
            trained=TemporalMultimodalWM(allowed,epochs=args.epochs,
                   temporal_epochs=args.temporal_epochs,seed=seed,q=2).fit(
                       fit,extra,transitions)
            blend={1:0.,2:.35,3:0.}[variant]
        if variant==3:
            # DEV chooses one candidate; AUDIT independently gates
            # that single candidate against 0 with a Brier constraint.
            tune_e={k:ev[k] for k in [r["pid"] for r in tune]}
            audit_e={k:ev[k] for k in [r["pid"] for r in audit]}
            scored=[(score(evaluate(trained,tune,tune_e,task[1],w)),w)
                    for w in BLENDS]
            (devmae,devbrier),candidate=min(scored,key=lambda x:(x[0][0],x[0][1],x[1]))
            before=score(evaluate(trained,audit,audit_e,task[1],0.))
            after=score(evaluate(trained,audit,audit_e,task[1],candidate))
            accepted=(candidate>0 and after[0]+.005<before[0] and
                      after[1]<=before[1]+.005)
            blend=candidate if accepted else 0.
            audit_reports.append({"cohort":task[0],"horizon":task[1],
              "candidate_blend":candidate,"selected_blend":blend,
              "accepted":accepted,"dev_candidate_mae":devmae,
              "audit_mae_baseline":before[0],"audit_mae_candidate":after[0],
              "audit_brier_baseline":before[1],"audit_brier_candidate":after[1]})
        result=evaluate(trained,evaluation,ev,task[1],blend)
        results.append(metrics(task,held,variant,evaluation,result,baseline,blend,len(transitions)))
        coverage.append({"cohort":task[0],"horizon":task[1],"fold":held,
            "eligible":len(rows),"fit":len(fit),"unlabelled_fit":len(extra),
            "tune":len(tune),"audit":len(audit),"eval":len(evaluation),
            "transitions":len(transitions),"measured_panel_features":len(allowed)})
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/"fold_metrics.csv",results)
    csv_write(out/"coverage.csv",coverage)
    (out/"audit.json").write_text(json.dumps(audit_reports,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
       "arm_id":args.arm_id,"fold":held,"variant":variant,
       "elapsed_seconds":time.perf_counter()-t0,
       "source_qc":qc,
       "protocol":"fixed real two-item evidence, matched within-job static WM, FIT-only transitions, DEV/AUDIT gate",
       "limits":["Exploratory TRAIN only","No Qwen and no RSI; temporal mechanism study",
       "True R01 offset still estimated by six months when R01 evidence used",
       "No actual causal intervention or external validation",
       "Only aggregate results exported"],
       "completed":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    print(json.dumps({"completed":len(results),"arm":args.arm_id,"output":str(out)}))
if __name__=="__main__":main()
