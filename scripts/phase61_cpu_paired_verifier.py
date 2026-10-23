#!/usr/bin/env python3
"""Phase61: CPU-only paired longitudinal WM/verifier ablation on TRAIN.

Recorded same-V04 visible-item masking, not a causal intervention.
No Qwen generation: isolates temporal fusion and calibration from Doctor policy.
The original Phase44 TEST is consumed. Only aggregate reports are exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.phase45_world_model import ObservationWorldModel,PatientEvidenceService
from scripts.phase59_grounded_collaboration import JointBelief
from scripts.phase59_temporal_prior import fit_prior,forecast_prior
from scripts.phase60_cpu_multistep_wm import split_train,auc,stable_hash

SEED=20261018
WEIGHTS=(0.,.25,.5,.75,1.)
TEMPS=(.7,1.,1.3,1.6,2.)
STYLES=("information_gain","risk_reduction","uncertainty")
ARMS=("random_fixed","observation_fixed","temporal_fixed",
      "observation_calibrated","temporal_verified","temporal_shuffled")


def safe_count(n):
    return n if n>=10 else "<10"


def audit_split(dev,seed=SEED):
    tune=[];audit=[]
    for y in (0,1):
        group=sorted((r for r in dev if r["label"]==y),
                     key=lambda r:stable_hash("phase61|audit|"+r["pid"],seed))
        cut=max(1,min(len(group)-1,int(.6*len(group))))
        tune+=group[:cut];audit+=group[cut:]
    assert not ({r["pid"] for r in tune}&{r["pid"] for r in audit})
    if len(tune)<25 or len(audit)<20:raise ValueError("Insufficient DEV")
    return tune,audit


def episode(row,world,prior,cfg,questions,random_order=False):
    weight,temp,style=cfg
    belief=JointBelief(world,prior,weight if prior is not None else 0.,temp)
    evidence=PatientEvidenceService(row["features"],world.items,questions)
    fixed=(sorted(world.items,key=lambda name:stable_hash(
        "phase61|random|"+row["pid"]+"|"+name))
           if random_order else None)
    for _ in range(questions):
        remaining=tuple(k for k in world.items if k not in evidence.observed)
        if random_order:
            chosen=next(k for k in fixed if k in remaining)
        else:
            chosen=min(remaining,key=lambda k:(
                -belief.candidate_value(evidence.observed,k,style),k))
        event=evidence.request(chosen)
        if event["provenance"]!="observed_V04_masking_simulation":
            raise AssertionError("Fabricated measured answer")
    if len(evidence.observed)!=questions:raise AssertionError("Query mismatch")
    return belief.posterior(evidence.observed)


def predict(rows,world,priors,cfg,q,random_order=False):
    return {r["pid"]:episode(r,world,priors.get(r["pid"]),
                              cfg,q,random_order) for r in rows}


def brier(rows,predictions):
    return sum((predictions[r["pid"]]-r["label"])**2
               for r in rows)/len(rows)


def search_and_audit(world,tune,audit,priors,configs,incumbent,q=8):
    """One frozen candidate from TUNE grid, then one independent AUDIT gate.

    Audit is consulted ONLY after candidate selection and never in a loop.
    The final outer evaluation is not consulted for mutation/acceptance.
    """
    before=brier(tune,predict(tune,world,priors,incumbent,q))
    candidate_loss,candidate=min(
        (brier(tune,predict(tune,world,priors,cfg,q)),cfg)
        for cfg in configs)
    audited_before=brier(audit,predict(audit,world,priors,incumbent,q))
    audited_after=brier(audit,predict(audit,world,priors,candidate,q))
    accepted=(before-candidate_loss>1e-4 and
              audited_before-audited_after>1e-4)
    selected=candidate if accepted else incumbent
    trace={"candidate":repr(candidate),"accepted":accepted,
           "reason":"improved_both_tiers" if accepted else "rollback",
           "dev_tune_base_brier":before,"dev_tune_candidate_brier":candidate_loss,
           "dev_audit_base_brier":audited_before,
           "dev_audit_candidate_brier":audited_after,
           "audited_proposals":1,
           "chosen_hash":hashlib.sha256(repr(selected).encode()).hexdigest()[:16]}
    return selected,trace


def shuffled_priors(outer,priors,seed=SEED):
    """Outcome-blind cyclic permutation, preserving missingness."""
    available=sorted((r["pid"] for r in outer
                      if priors.get(r["pid"]) is not None),
                     key=lambda pid:stable_hash("phase61|placebo|"+pid,seed))
    if len(available)<3:raise ValueError("Missing placebo cohort")
    return {pid:priors[other] for pid,other in zip(
        available,available[1:]+available[:1])}


def aggregate(rows,scores,arm,budget):
    n=len(rows)
    ys=[r["label"] for r in rows]
    ps=[scores[r["pid"]] for r in rows]
    pos=sum(ys);quota=max(1,int(.20*n+.5))
    ranked=sorted(range(n),key=lambda i:(-ps[i],stable_hash(rows[i]["pid"])))
    captured=sum(ys[i] for i in ranked[:quota])
    return {"arm":arm,"budget":budget,"n_patients":n,"positives":safe_count(pos),
            "site_quota":quota,"captured":safe_count(captured),
            "recall_at_20pct":captured/pos if min(pos,captured)>=10 else "",
            "auroc":auc(ps,ys),"brier":brier(rows,scores),
            "total_real_questions":n*budget,"fabricated_answers":0,
            "llm_generations":0}


def pair_bootstrap(rows,predictions,left,right,budget,reps,seed=SEED):
    """Same-subject bootstrap in RAM; no identifier/trajectory in exports."""
    n=len(rows);ys=[r["label"] for r in rows]
    a=[predictions[left][r["pid"]] for r in rows]
    b=[predictions[right][r["pid"]] for r in rows]
    def calc(draw):
        truth=[ys[i] for i in draw]
        la=[a[i] for i in draw];rb=[b[i] for i in draw]
        db=sum((v-y)**2-(u-y)**2
               for y,u,v in zip(truth,la,rb))/len(draw)
        al=auc(la,truth);ar=auc(rb,truth)
        da=ar-al if al is not None and ar is not None else None
        k=max(1,int(.20*len(draw)+.5))
        ka=sorted(range(len(draw)),key=lambda j:(-la[j],j))[:k]
        kb=sorted(range(len(draw)),key=lambda j:(-rb[j],j))[:k]
        ca=sum(truth[j] for j in ka);cb=sum(truth[j] for j in kb)
        dr=((cb-ca)/sum(truth)
            if sum(truth)>=10 and min(ca,cb)>=10 else None)
        return db,da,dr
    actual=calc(list(range(n)))
    rng=random.Random(seed+budget+int(hashlib.sha256(
        (left+"|"+right).encode()).hexdigest()[:8],16))
    draws=[[],[],[]]
    for _ in range(reps):
        for j,v in enumerate(calc([rng.randrange(n) for __ in range(n)])):
            if v is not None:draws[j].append(v)
    def interval(values):
        if len(values)<max(30,reps//4):return None,None
        values.sort()
        return values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]
    ci=[interval(v) for v in draws]
    return {"budget":budget,"left":left,"right":right,
            "delta_brier":actual[0],"brier_ci_low":ci[0][0],
            "brier_ci_high":ci[0][1],
            "delta_auc":actual[1],"auc_ci_low":ci[1][0],
            "auc_ci_high":ci[1][1],
            "delta_recall20":actual[2],"recall_ci_low":ci[2][0],
            "recall_ci_high":ci[2][1],"bootstrap":reps,
            "scope":"exploratory_FIXED_MODEL_subject_paired_no_refitting"}


def run(rows,items,priors,budgets=(4,8,12),bootstrap=600,seed=SEED):
    fit,dev,outer=split_train(rows,seed)
    tune,audit=audit_split(dev,seed)
    for r in dev+outer:
        if r["pid"] not in priors:raise ValueError("Missing prior mapping")
    world=ObservationWorldModel(items).fit(fit)
    fixed=(.5,1.,"information_gain")
    fusion_configs=[(w,t,style) for w in WEIGHTS for t in TEMPS
                    for style in STYLES]
    selected,trace=search_and_audit(
        world,tune,audit,priors,fusion_configs,fixed)
    observation,trace_obs=search_and_audit(
        world,tune,audit,{},[(0.,t,"information_gain") for t in TEMPS],
        (0.,1.,"information_gain"))
    placebo=shuffled_priors(outer,priors,seed)
    definitions={
        "random_fixed":((0.,1.,"information_gain"),{}),
        "observation_fixed":((0.,1.,"information_gain"),{}),
        "temporal_fixed":(fixed,priors),
        "observation_calibrated":(observation,{}),
        "temporal_verified":(selected,priors),
        "temporal_shuffled":(fixed,placebo),
    }
    metrics=[];paired=[]
    for q in budgets:
        if q<1 or q>len(items):raise ValueError("Invalid query budget")
        scores={}
        for arm in ARMS:
            cfg,source=definitions[arm]
            scores[arm]=predict(
                outer,world,source,cfg,q,random_order=(arm=="random_fixed"))
            metrics.append(aggregate(outer,scores[arm],arm,q))
        if bootstrap:
            for left,right in (
                ("observation_fixed","temporal_fixed"),
                ("temporal_fixed","temporal_verified"),
                ("observation_fixed","temporal_verified"),
                ("temporal_shuffled","temporal_fixed"),
                ("observation_fixed","observation_calibrated"),
            ):
                paired.append(pair_bootstrap(
                    outer,scores,left,right,q,bootstrap,seed))
    summary={
        "protocol":"Phase61 TRAIN-only CPU paired same-V04 masking",
        "seed":seed,
        "split_counts":{"fit":len(fit),"tune":len(tune),
                        "one_shot_audit":len(audit),"outer_eval":len(outer)},
        "outer_positive":safe_count(sum(r["label"] for r in outer)),
        "outer_temporal_history_available":sum(
            priors[r["pid"]] is not None for r in outer),
        "selected_verified_temporal":{"weight":selected[0],
                                      "temperature":selected[1],"style":selected[2]},
        "selected_verified_observation":{"temperature":observation[1]},
        "question_budgets":list(budgets),
        "eval_split_fingerprint":hashlib.sha256(
            "|".join(sorted(r["pid"] for r in outer)).encode()).hexdigest()[:16],
        "limitations":[
            "No Qwen generations or autonomous neural RSI in Phase61",
            "Same-V04 masking simulation; not prospective clinical action",
            "Original Phase44 TEST already consumed; TRAIN reuse exploratory",
            "Prior fit excludes all site DEV and outer EVAL people",
            "Only real measured visible items reach patient observation tool",
            "TUNE searches fixed grid, AUDIT independently gates exactly one proposal",
            "Placebo permutes prior blindly to observed labels",
            "Bootstrap holds fitted models fixed; prior method exploration uncorrected",
            "Only aggregate outputs; small captured counts/recalls suppressed",
        ],
    }
    return metrics,paired,[trace,trace_obs],summary


def csv_write(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--temporal-epochs",type=int,default=8)
    p.add_argument("--bootstrap",type=int,default=600)
    p.add_argument("--budgets",type=int,nargs="+",default=[4,8,12])
    a=p.parse_args()
    if a.temporal_epochs<2 or a.bootstrap<0:raise ValueError("Invalid full job")
    root=a.data_root.resolve();manifest=a.manifest_db.resolve()
    output=a.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing frozen TRAIN data")
    if output==root or root in output.parents or output==manifest.parent:
        raise ValueError("Unsafe aggregate export directory")
    from scripts.phase45_wm_rsi_agents import load_train
    started=time.perf_counter()
    rows,items,_,_=load_train(root,manifest)
    _,dev,outer=split_train(rows)
    prohibited={r["pid"] for r in dev+outer}
    wm,index,device,model_metadata=fit_prior(
        root,manifest,prohibited,epochs=a.temporal_epochs)
    priors={r["pid"]:forecast_prior(wm,index,device,r["pid"])
            for r in dev+outer}
    del wm
    metrics,paired,traces,metadata=run(
        rows,items,priors,budgets=a.budgets,bootstrap=a.bootstrap)
    metadata["temporal_fit"]=model_metadata
    metadata["wall_seconds"]=round(time.perf_counter()-started,2)
    metadata["finished_utc"]=datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/"arm_metrics.csv",metrics)
    csv_write(output/"paired_bootstrap.csv",paired)
    (output/"verifier_trace.json").write_text(
        json.dumps(traces,indent=2)+"\n",encoding="utf8")
    (output/"summary.json").write_text(
        json.dumps(metadata,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":"success","output":str(output),
                      "metrics":len(metrics),"paired":len(paired),
                      "temporal_candidate_accepted":traces[0]["accepted"],
                      "temperature_candidate_accepted":traces[1]["accepted"]},
                     indent=2))


if __name__=="__main__":main()
