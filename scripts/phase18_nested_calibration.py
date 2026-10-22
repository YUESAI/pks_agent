#!/usr/bin/env python3
"""Phase 18 — Nested calibration and paired cost/risk audit (PPMI TRAIN only).

Outer five-fold patient CV is entirely inside the existing, frozen PPMI TRAIN.
Inside each outer fit set, split subjects again (stratified by label and BL
anchor): fit a numeric NB on proper-fit subjects, and calibrate the resulting
STATIC-IG partial-score trajectories on disjoint calibration subjects.

Compare no-stop, raw-score-threshold, and isotonic-calibrated stopping using
the same outer held-out subjects. No validation/test/external rows are queried.
No person-level rows, IDs, scores, dates, or predictions are written.

ALL calibration is exploratory: outcomes, observation coding and motor states
are unvalidated, and a 0.90 score threshold is NOT a safe clinical rule.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase13_numeric_policy_cv import (
    bin_value, load_training, train_model
)
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase10_leakage_splits import auc
from scripts.phase15_motor_budget import expected_gain
from scripts.budgeted_env_v0 import ACTION_SPECS

SPACES=("motor_allowed","motor_excluded")
POLICIES=("no_stop","raw_0.90_stop","calibrated_0.90_stop")


def sigmoid(score):
    return 1.0/(1.0+math.exp(-max(-500,min(500,score))))


def calibration_split(records, seed, fraction=0.30):
    """Partition outer-fit patients by label/anchor, never by clinical score."""
    groups=defaultdict(list)
    for pid,label,anchor,values in records:
        groups[(label,bool(anchor))].append(pid)
    calibration=set()
    for (label,anchor),ids in sorted(groups.items()):
        ranked=sorted(ids,key=lambda pid:hashlib.sha256(
            f"{seed}|{label}|{anchor}|{pid}".encode()).hexdigest())
        n=len(ranked)
        if n<=2:
            raise ValueError("Outer fit stratum too small for calibration")
        take=max(1,min(n-2,round(n*fraction)))
        calibration.update(ranked[:take])
    proper=[(pid,label,anchor,values) for pid,label,anchor,values
            in records if pid not in calibration]
    cal=[(pid,label,anchor,values) for pid,label,anchor,values
         in records if pid in calibration]
    if set(p for p,_,_,_ in proper)&set(p for p,_,_,_ in cal):
        raise ValueError("Proper fit and calibration overlap")
    if {label for _,label,_,_ in proper}!={"PD","HC"} or \
       {label for _,label,_,_ in cal}!={"PD","HC"}:
        raise ValueError("Nested calibration requires both labels in each partition")
    return proper,cal


def static_order(model,success,space):
    actions=[a for a in ACTION_SPECS if a!="age" and
             not(space=="motor_excluded" and a=="motor_exam")]
    return sorted(actions,key=lambda a:(
        -expected_gain(model,a,0.)*success[a]/
        max(1,ACTION_SPECS[a]["cost"]),a))


def trace_static(model,success,values,space,budget):
    """Return score trajectory after each paid request (same for every stop rule).

    Stored exclusively in RAM. A stop policy can inspect only prefix snapshots.
    Each tuple: (log_odds, spent, valid_paid_results).
    """
    score=0.
    spent=0
    valid_paid=0

    # The same age background is requested by all policies (zero cost).
    age_bin=bin_value("age",values.get("age"))
    if age_bin is not None:
        score+=math.log(model["age"]["PD"][age_bin]/
                        model["age"]["HC"][age_bin])
    result=[]
    for action in static_order(model,success,space):
        price=ACTION_SPECS[action]["cost"]
        if spent+price>budget:
            continue
        spent+=price
        bucket=bin_value(action,values.get(action))
        if bucket is not None:
            score+=math.log(model[action]["PD"][bucket]/
                            model[action]["HC"][bucket])
            valid_paid+=1
        result.append((score,spent,valid_paid))
    if not result:
        result.append((score,0,0))
    return result


class IsotonicCalibrator:
    """Weighted isotonic probability map, fitted without any evaluation patients.

    Class-balanced weights describe a balanced-prior score, NOT population PD
    prevalence, individual prognosis, or a validated clinical risk.
    """
    def __init__(self,triples):
        # triples are (score, label_PD_as_bool, positive_weight)
        if not triples:
            raise ValueError("Calibration observations missing")
        items=sorted(triples,key=lambda row:row[0])
        blocks=[]
        i=0
        while i<len(items):
            x=items[i][0]
            n=0.;pos=0.
            while i<len(items) and items[i][0]==x:
                score,truth,w=items[i]
                n+=w
                pos+=w*int(truth)
                i+=1
            blocks.append([x,x,n,pos])
            while len(blocks)>1 and (blocks[-2][3]/blocks[-2][2] >
                                     blocks[-1][3]/blocks[-1][2]):
                new=blocks.pop()
                old=blocks.pop()
                blocks.append([old[0],new[1],old[2]+new[2],old[3]+new[3]])
        self.upper=[b[1] for b in blocks]
        self.prob=[min(.98,max(.02,b[3]/b[2])) for b in blocks]

    def predict(self,score):
        position=bisect.bisect_left(self.upper,score)
        return self.prob[min(position,len(self.prob)-1)]


def fit_calibrator(model,success,calibration,space,budget):
    """Equal total weight to PD and HC, equal total weight per patient."""
    label_count=Counter(label for _,label,_,_ in calibration)
    if min(label_count["PD"],label_count["HC"])<5:
        raise ValueError("Insufficient patients per class to calibrate")
    training=[]
    for pid,label,anchor,values in calibration:
        sequence=trace_static(model,success,values,space,budget)
        steps=[score for score,cost,got in sequence if got>=1]
        if not steps:
            continue
        weight=.5/label_count[label]/len(steps)
        training.extend((score,label=="PD",weight) for score in steps)
    if not training:
        raise ValueError("Calibration has no paid observations")
    return IsotonicCalibrator(training)


def choose_snapshot(trajectory,policy,calibrator=None,threshold=.90):
    """Never stop unless a paid measurement has returned numeric evidence."""
    if policy not in POLICIES:
        raise ValueError("Unknown early-stop policy")
    for score,cost,n_valid in trajectory:
        if n_valid<1 or policy=="no_stop":
            continue
        probability=(sigmoid(score) if policy=="raw_0.90_stop" else
                     calibrator.predict(score))
        if max(probability,1-probability)>=threshold:
            return score,cost,n_valid,True
    score,cost,n_valid=trajectory[-1]
    return score,cost,n_valid,False


def balanced_accuracy(rows):
    """Rows: (truth_is_PD, diagnostic_score, cost, stopped, final_probability)."""
    pd=sum(truth for truth,_,_,_,_ in rows)
    hc=len(rows)-pd
    if not pd or not hc:
        raise ValueError("Missing PD or HC in metric")
    tp=sum(score>=0 and truth for truth,score,cost,stopped,p in rows)
    tn=sum(score<0 and not truth for truth,score,cost,stopped,p in rows)
    return .5*(tp/pd+tn/hc)


def summarize(rows):
    n=len(rows)
    if not n:
        raise ValueError("Empty evaluation group")
    pd=sum(y for y,_,_,_,_ in rows)
    hc=n-pd
    avgcost=sum(cost for y,score,cost,stopped,p in rows)/n
    pred=[(score,y) for y,score,cost,stopped,p in rows]
    brier={}
    for label in (True,False):
        subset=[(prob-float(y))**2 for y,score,cost,stopped,prob
                in rows if y==label]
        brier[label]=sum(subset)/len(subset)
    return {
        "n":n,"n_PD":pd,"n_HC":hc,
        "balanced_accuracy":round(balanced_accuracy(rows),4),
        "auroc":round(auc(pred),4),
        "mean_cost":round(avgcost,3),
        "stopped_pct":round(100*sum(stopped for y,s,c,stopped,p in rows)/n,2),
        "class_balanced_brier":round((brier[True]+brier[False])/2,5),
    }


def paired_delta(reference,alternative,bootstraps,seed):
    """Stratified paired bootstrap, resample identical subjects across variants."""
    if reference.keys()!=alternative.keys():
        raise ValueError("Unequal patient populations in bootstrap")
    keys=sorted(reference)
    ids_pd=[pid for pid in keys if reference[pid][0]]
    ids_hc=[pid for pid in keys if not reference[pid][0]]
    if len(ids_pd)<2 or len(ids_hc)<2:
        raise ValueError("Too few patients per class")
    for pid in keys:
        if reference[pid][0]!=alternative[pid][0]:
            raise ValueError("Inconsistent labels")
    rand=random.Random(seed)
    metrics=("balanced_accuracy","auroc","mean_cost")
    point_a=summarize([reference[pid] for pid in keys])
    point_b=summarize([alternative[pid] for pid in keys])
    samples={name:[] for name in metrics}
    for iteration in range(bootstraps):
        ids=[ids_pd[rand.randrange(len(ids_pd))] for _ in ids_pd]
        ids += [ids_hc[rand.randrange(len(ids_hc))] for _ in ids_hc]
        first=summarize([reference[pid] for pid in ids])
        second=summarize([alternative[pid] for pid in ids])
        for name in metrics:
            samples[name].append(second[name]-first[name])
    output=[]
    for metric in metrics:
        sorted_values=sorted(samples[metric])
        def quantile(q):
            return sorted_values[min(len(sorted_values)-1,
                     max(0,int(q*(len(sorted_values)-1))))]
        output.append({
            "metric":metric,"difference":round(point_b[metric]-point_a[metric],4),
            "lower_95":round(quantile(.025),4),
            "upper_95":round(quantile(.975),4),
            "bootstrap_replicates":bootstraps,
        })
    return output


def write(path,header,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def audit(source,engine,manifest,splits,output,folds,budget,seed,bootstraps):
    source,manifest,splits,output=[Path(p).expanduser().resolve() for p
                                  in (source,manifest,splits,output)]
    if not all(path.is_file() for path in (source,manifest,splits)):
        raise FileNotFoundError("Source, manifest, or frozen split DB missing")
    if output in (source,manifest,splits) or folds<2 or budget<1 or bootstraps<100:
        raise ValueError("Invalid output, folds, budget, bootstrap count")
    data=load_training(source,engine,manifest,splits)
    records=[(pid,label,anchor,values) for pid,(label,anchor,values)
             in sorted(data.items())]
    fold_of=assign_folds(records,folds,seed)
    evaluated=defaultdict(dict)
    model_metadata=[]
    for held in range(folds):
        outer_fit=[row for row in records if fold_of[row[0]]!=held]
        outer_test=[row for row in records if fold_of[row[0]]==held]
        proper,calibration=calibration_split(outer_fit,seed+1009*held)
        model,success=train_model([(label,anchor,values)
                                   for pid,label,anchor,values in proper])
        model_metadata.append({
            "fold":held,"proper_fit":len(proper),
            "calibration":len(calibration),"outer_evaluation":len(outer_test),
            "calibration_HC":sum(row[1]=="HC" for row in calibration),
        })
        for space in SPACES:
            fitted=fit_calibrator(model,success,calibration,space,budget)
            for pid,label,anchor,values in outer_test:
                trajectory=trace_static(model,success,values,space,budget)
                for policy in POLICIES:
                    score,cost,n_valid,stopped=choose_snapshot(
                        trajectory,policy,fitted)
                    calibrated_probability=fitted.predict(score)
                    evaluated[(space,policy)][pid]=(
                        label=="PD",score,cost,stopped,calibrated_probability)

    report=[]
    pairs=[]
    for space in SPACES:
        reference=evaluated[(space,"no_stop")]
        for policy in POLICIES:
            cases=evaluated[(space,policy)]
            if len(cases)!=len(records):
                raise ValueError("Outer folds did not cover each TRAIN patient")
            report.append({
                "action_space":space,"policy":policy,
                "evaluated_population":"PPMI_TRAIN_nested_out_of_fold",
                "max_budget":budget,**summarize(list(cases.values()))})
            if policy=="no_stop":
                continue
            for row in paired_delta(reference,cases,bootstraps,
                                    seed+(11 if space=="motor_allowed" else 19)+
                                    (1 if policy=="raw_0.90_stop" else 2)):
                pairs.append({
                    "action_space":space,
                    "comparison":policy+"_minus_no_stop",**row,
                })
    output.mkdir(parents=True,exist_ok=True)
    write(output/"nested_stopping_metrics.csv",
        ["action_space","policy","evaluated_population","max_budget",
         "n","n_PD","n_HC","balanced_accuracy","auroc","mean_cost",
         "stopped_pct","class_balanced_brier"],report)
    write(output/"paired_cost_risk_uncertainty.csv",
        ["action_space","comparison","metric","difference",
         "lower_95","upper_95","bootstrap_replicates"],pairs)
    write(output/"nested_calibration_counts.csv",
        ["fold","proper_fit","calibration","outer_evaluation","calibration_HC"],
        model_metadata)
    meta={
        "status":"exploratory_nested_PPMI_TRAIN_only",
        "population_size":len(records),"outer_folds":folds,
        "max_budget":budget,"stop_score_threshold":.90,
        "seed":seed,"bootstrap_replicates":bootstraps,
        "calibration":"weighted isotonic regression on disjoint partial-score trajectories",
        "calibration_split_fraction":.30,
        "comparison":"static IG with no stop vs raw-score and calibrated 0.90 stop",
        "cautions":[
            "PPMI val/test and BioFIND patient measurements are NOT accessed.",
            "The 0.90 threshold has been inspected in Phase17; this is NOT a fresh confirmation.",
            "Isotonic calibration enforces a balanced prior, not population PD prevalence.",
            "Calibration folds have relatively few HC subjects; risk estimates are unstable.",
            "No threshold is justified as a safe clinical operating point.",
            "Early stopping may change AUROC and error even when accuracy appears unchanged.",
            "Motor score is treatment-state mixed and labels are retrospective candidates.",
            "No patient IDs, individual values, predictions, or trajectories are exported.",
            "The separate local databases must stay on the authorized cluster.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--split-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase18_calibration"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=300)
    args=p.parse_args()
    result=audit(args.source_db,args.source_engine,args.manifest_db,
                 args.split_db,args.report_dir,args.folds,args.budget,
                 args.seed,args.bootstraps)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Upload aggregate Phase18 reports only, never patient-level databases.")

if __name__=="__main__":
    main()
