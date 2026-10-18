#!/usr/bin/env python3
"""Phase 28: TRAIN-only longitudinal Motor signal and multi-step readiness audit.

Uses only Phase23 frozen TRAIN patients. Do NOT query validation or test
patients. Reports contain small-cell-suppressed aggregates, NEVER individual
IDs, values, dates or trajectories.

This is a descriptive feasibility audit, NOT a World Model or verified
clinical progression metric. Medication UNSPECIFIED does not mean untreated.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sqlite3
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,median

HORIZONS={6:(4,8),12:(9,15),24:(20,28)}
BASELINE_BINS=("0_to_3","4_to_7","8_to_15","16_plus","no_baseline_motor")
CHANGE_BINS=("decrease_10plus","decrease_5_to_9","decrease_3_to_4",
 "decrease_1_to_2","unchanged","increase_1_to_2","increase_3_to_4",
 "increase_5_to_9","increase_10plus")

def severity_bin(value):
    if value is None:return "no_baseline_motor"
    if value<=3:return "0_to_3"
    if value<=7:return "4_to_7"
    if value<=15:return "8_to_15"
    return "16_plus"

def change_bin(value):
    if value<=-10:return "decrease_10plus"
    if value<=-5:return "decrease_5_to_9"
    if value<=-3:return "decrease_3_to_4"
    if value<0:return "decrease_1_to_2"
    if value==0:return "unchanged"
    if value<=2:return "increase_1_to_2"
    if value<=4:return "increase_3_to_4"
    if value<=9:return "increase_5_to_9"
    return "increase_10plus"

def direction(value):
    return "increase" if value>0 else "decrease" if value<0 else "unchanged"

def suppression(n,minimum):
    return str(n) if n==0 or n>=minimum else "<"+str(minimum)

def quantile(values,q):
    ordered=sorted(values)
    if not ordered:raise ValueError("Empty quantile input")
    pos=(len(ordered)-1)*q;lo=int(pos);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]*(hi-pos)+ordered[hi]*(pos-lo)

def read_train_manifest(manifest):
    """The WHERE p.split='train' SQL restriction is explicit on BOTH queries."""
    db=sqlite3.connect("file:"+str(Path(manifest).resolve())+"?mode=ro",uri=True)
    try:
        people={}
        rows=db.execute(
            "SELECT p.pid,p.baseline_month,p.baseline_motor,"
            "r6.status,r12.status,r24.status FROM patients p "
            "LEFT JOIN readiness r6 ON r6.pid=p.pid AND r6.horizon=6 "
            "LEFT JOIN readiness r12 ON r12.pid=p.pid AND r12.horizon=12 "
            "LEFT JOIN readiness r24 ON r24.pid=p.pid AND r24.horizon=24 "
            "WHERE p.split='train' ORDER BY p.pid").fetchall()
        for pid,month,value,r6,r12,r24 in rows:
            if not pid or pid in people or any(s is None for s in (r6,r12,r24)):
                raise ValueError("Incomplete or repeated TRAIN registry entry")
            if value is not None and (
                not math.isfinite(float(value)) or not 0<=float(value)<=132):
                raise ValueError("Invalid baseline motor score")
            people[pid]={"anchor":month,"baseline":None if value is None else float(value),
                         "readiness":{6:r6,12:r12,24:r24},"targets":{}}
        if not people:raise ValueError("Missing frozen TRAIN subjects")
        future=db.execute(
            "SELECT t.pid,t.horizon,t.baseline_month,t.target_month,"
            "t.months_apart,t.baseline_motor,t.target_motor,t.delta_motor "
            "FROM targets t JOIN patients p ON p.pid=t.pid "
            "WHERE p.split='train' AND t.horizon IN (6,12,24) "
            "ORDER BY t.pid,t.horizon").fetchall()
        for pid,h,bl_month,target_month,gap,bl_score,target_score,delta in future:
            p=people.get(pid)
            if p is None:raise ValueError("Target outside TRAIN split")
            if h in p["targets"] or p["readiness"][h]!="paired":
                raise ValueError("Duplicate or readiness-inconsistent target")
            start,end=HORIZONS[h]
            values=(bl_score,target_score,delta)
            if (p["anchor"] is None or p["baseline"] is None or
                bl_month!=p["anchor"] or
                any(v is None or not math.isfinite(float(v)) for v in values) or
                not (0<=float(bl_score)<=132 and 0<=float(target_score)<=132) or
                abs(float(bl_score)-p["baseline"])>1e-6 or
                abs(float(target_score)-float(bl_score)-float(delta))>1e-6 or
                gap!=target_month-bl_month or not start<=gap<=end):
                raise ValueError("Inconsistent frozen target values/gap")
            p["targets"][h]={"gap":int(gap),"score":float(target_score),
                             "delta":float(delta)}
        for p in people.values():
            for h in HORIZONS:
                if (p["readiness"][h]=="paired")!=(h in p["targets"]):
                    raise ValueError("Target/readiness mismatch")
        return people
    finally:db.close()

def write_csv(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)

def n_stats(values,minimum):
    if len(values)<minimum:
        return {"mean":"","median":"","q25":"","q75":"","sd":""}
    mu=mean(values)
    return {"mean":round(mu,3),"median":round(median(values),3),
            "q25":round(quantile(values,.25),3),"q75":round(quantile(values,.75),3),
            "sd":round(math.sqrt(mean((x-mu)**2 for x in values)),3)}

def audit(manifest,output,minimum=10):
    manifest=Path(manifest).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if not manifest.is_file() or output==manifest or minimum<10:
        raise ValueError("Missing manifest, unsafe output or min-cell-count <10")
    people=read_train_manifest(manifest)
    readiness=Counter();baseline_count=Counter();baseline_vs_ready=Counter()
    overlap=Counter();gaps=Counter();changes=Counter()
    delta_by_baseline=defaultdict(list);all_delta=[];coverage=Counter()
    transitions=Counter()
    for p in people.values():
        group=severity_bin(p["baseline"]);baseline_count[group]+=1
        available=frozenset(p["targets"])
        overlap[tuple(h for h in HORIZONS if h in available)]+=1
        for h in HORIZONS:
            readiness[(h,p["readiness"][h])]+=1
            baseline_vs_ready[(group,h,"paired" if h in available else "not_paired")]+=1
            if h in available:
                gaps[(h,p["targets"][h]["gap"])]+=1
        if 12 in available:
            delta=p["targets"][12]["delta"]
            changes[change_bin(delta)]+=1
            delta_by_baseline[group].append(delta);all_delta.append(delta)
        if 12 in available and 24 in available:
            delta12=p["targets"][12]["delta"]
            delta24=p["targets"][24]["score"]-p["targets"][12]["score"]
            transitions[(direction(delta12),direction(delta24))]+=1
            coverage["12_and_24_months"]+=1
            if delta12*delta24<0:coverage["nonzero_direction_reversal"]+=1
        if all(h in available for h in HORIZONS):
            coverage["6_and_12_and_24_months"]+=1
    output.mkdir(parents=True,exist_ok=True)
    files={}
    files["train_readiness_by_horizon.csv"]=(
        ["horizon_months","readiness_status","train_patients"],
        [dict(horizon_months=h,readiness_status=s,
              train_patients=suppression(n,minimum))
         for (h,s),n in sorted(readiness.items())])
    files["train_joint_horizon_coverage.csv"]=(
        ["available_target_months","train_patients"],
        [dict(available_target_months="+".join(map(str,k)) if k else "none",
              train_patients=suppression(n,minimum))
         for k,n in sorted(overlap.items())])
    severity=[]
    for group in BASELINE_BINS:
        total=baseline_count[group];yes=baseline_vs_ready[group,12,"paired"]
        no=baseline_vs_ready[group,12,"not_paired"]
        can_report_rate=total>=minimum and all(x==0 or x>=minimum for x in (yes,no))
        severity.append({
            "baseline_motor_bin":group,"train_patients":suppression(total,minimum),
            "paired_12m":suppression(yes,minimum),"not_paired_12m":suppression(no,minimum),
            "paired_rate_pct":round(100*yes/total,2) if can_report_rate else ""})
    files["baseline_severity_vs_12m_observability.csv"]=(
        ["baseline_motor_bin","train_patients","paired_12m","not_paired_12m",
         "paired_rate_pct"],severity)
    files["train_12m_change_distribution.csv"]=(
        ["change_category","patients"],
        [dict(change_category=k,patients=suppression(changes[k],minimum))
         for k in CHANGE_BINS])
    change_rows=[]
    for group in BASELINE_BINS:
        vals=delta_by_baseline[group];stat=n_stats(vals,minimum)
        change_rows.append(dict(baseline_motor_bin=group,
              paired_patients=suppression(len(vals),minimum),
              mean_delta=stat["mean"],median_delta=stat["median"],
              q25_delta=stat["q25"],q75_delta=stat["q75"],sd_delta=stat["sd"]))
    files["train_12m_change_by_baseline.csv"]=(
        ["baseline_motor_bin","paired_patients","mean_delta","median_delta",
         "q25_delta","q75_delta","sd_delta"],change_rows)
    files["train_observed_month_gaps.csv"]=(
        ["target_horizon","actual_gap_months","train_patients"],
        [dict(target_horizon=h,actual_gap_months=gap,
              train_patients=suppression(gaps[h,gap],minimum))
         for h,(start,end) in HORIZONS.items() for gap in range(start,end+1)])
    files["train_multistep_direction.csv"]=(
        ["BL_to_12_direction","12_to_24_direction","train_patients"],
        [dict(BL_to_12_direction=first,12_to_24_direction=second,
              train_patients=suppression(transitions[first,second],minimum))
         for first in ("increase","unchanged","decrease")
         for second in ("increase","unchanged","decrease")])
    files["train_multistep_coverage.csv"]=(
        ["metric","train_patients"],
        [dict(metric=key,train_patients=suppression(coverage[key],minimum))
         for key in ("12_and_24_months","6_and_12_and_24_months",
                     "nonzero_direction_reversal")])
    for filename,(cols,rows) in files.items():
        write_csv(output/filename,cols,rows)
    result={
        "status":"TRAIN_ONLY_longitudinal_signal_and_multistep_readiness_audit",
        "baseline_train_population":len(people),
        "strict_12m_paired_train_patients":len(all_delta),
        "strict_24m_paired_train_patients":sum(24 in p["targets"] for p in people.values()),
        "strict_12m_and_24m_paired_train_patients":
            suppression(coverage["12_and_24_months"],minimum),
        "all_three_6_12_24_paired_train_patients":
            suppression(coverage["6_and_12_and_24_months"],minimum),
        "windows_months":{"6":[4,8],"12":[9,15],"24":[20,28]},
        "train_12m_change_summary":n_stats(all_delta,minimum),
        "limitations":[
            "Only frozen Phase23 TRAIN records are queried. PPMI val/test remain unexamined.",
            "Future target availability is used for aggregate QC, never as baseline input.",
            "Closest-in-window visits are not exactly scheduled 6/12/24 month dates.",
            "Medication UNSPECIFIED cannot prove fixed medication state.",
            "Repeated-score fluctuations are not a measured noise ceiling.",
            "3/5/10 point change bins are descriptive, NOT validated progression thresholds.",
            "Selection into observed followup may correlate with baseline health.",
            "No patient-level data, individual scores, dates or trajectories exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase28_longitudinal_signal"))
    p.add_argument("--min-cell-count",type=int,default=10)
    args=p.parse_args()
    print(json.dumps(audit(args.manifest_db,args.report_dir,args.min_cell_count),
                     ensure_ascii=False,indent=2))
    print("Upload only aggregate Phase28 reports. Keep private manifest on cluster.")

if __name__=="__main__":
    main()
