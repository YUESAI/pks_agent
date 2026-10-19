#!/usr/bin/env python3
"""Phase42: follow-up missingness bounds for R01 -> future V04 risk ranking.

Locked PPMI Prodromal TRAIN only. All patients with eligible real R01
partial Motor exams (27 observable items and six unrateable) are ranked
on their R01 observable sum. Only a subset have eligible later V04
outcomes. For patients WITHOUT a V04 label, do not invent one: compute
the sharp best/worst-case identification bounds on future-positive
precision@k and recall@k, treating every missing binary outcome as
individually unknown. A future V04 outcome is NOT a same-day R01 label
and ranking does NOT estimate the causal benefit of a clinic referral.

No trained models, no validation/test reads, and no row-level exports.
Ties use the same seeded SHA256 rule as Phase41. All calculations are
exploratory on the repeatedly studied PPMI TRAIN population.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path


sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase36_masked_site_feasibility import (
    VISIBLE,HIDDEN,parse_rows,unique_by_patient
)
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase22_strict_motor_pairs import one_file

def load_train_pairs(root,manifest):
    """Match Phase41's exact R01 / later V04 eligibility without importing it.

    Preserve the Phase36 record parsing and conservative one-exam-per-event
    restriction. Retain identifiers and scores only in local process memory.
    """
    anchors=train_anchors(manifest)
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    qc=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,qc),qc)
    remote={}
    onsite={}
    for (pid,event),(month,fields,total) in selected.items():
        if any(fields[name][0]!="valid" for name in VISIBLE):
            qc[event+"_visible_incomplete"]+=1
            continue
        visible=sum(fields[name][1] for name in VISIBLE)
        if event=="R01":
            if total is None and all(fields[name][0]=="unable" for name in HIDDEN):
                remote[pid]=(month,visible)
            else:
                qc["R01_not_six_unrateable"]+=1
        elif event=="V04":
            if (total is not None and
                all(fields[name][0]=="valid" for name in HIDDEN) and
                visible+sum(fields[name][1] for name in HIDDEN)==total):
                onsite[pid]=(month,visible,
                             sum(fields[name][1] for name in HIDDEN))
            else:
                qc["V04_not_complete_same_visit"]+=1
    paired=[]
    for pid in sorted(remote.keys() & onsite.keys()):
        rmonth,rscore=remote[pid]
        vmonth,vscore,hidden=onsite[pid]
        if vmonth<=rmonth:
            qc["V04_not_after_R01"]+=1
            continue
        paired.append({
            "pid":pid,"remote_sum":rscore,
            "future_visible_sum":vscore,"future_six":hidden,
            "gap":vmonth-rmonth,
        })
    return paired,remote,onsite,qc,source,len(anchors)


BUDGETS=(.10,.20,.30,.40,.50)
BINS=(("0",0,0),("1_to_3",1,3),("4_to_7",4,7),("8plus",8,108))
MAX_HIDDEN=24


def quota(n,fraction):
    if n<0 or not 0<=fraction<=1:raise ValueError("Invalid budget")
    return max(0,min(n,int(math.floor(n*fraction+.5))))


def priority_hash(pid,seed):
    # Matches Phase41's tie-break on identical risk values.
    return hashlib.sha256(
        (f"Phase41|{seed}|{pid}").encode("utf-8")).hexdigest()


def ranked_ids(registry,seed):
    return sorted(registry,key=lambda pid:(
        -registry[pid]["visible"],priority_hash(pid,seed)))


def topk(registry,budget,seed):
    return set(ranked_ids(registry,seed)[:quota(len(registry),budget)])


def build_registry(remote,paired):
    """All 229 eligible R01 subjects retained, not just complete cases.

    remote maps pid -> (month, R01 visible score).
    paired is the Phase41 strict future-V04 patient subset with
    {'pid','remote_sum','future_six',...} rows.
    """
    registry={}
    for pid,entry in remote.items():
        if not pid or not isinstance(entry,(list,tuple)) or len(entry)!=2:
            raise ValueError("Invalid R01 subject/score")
        _,score=entry
        score=float(score)
        if not math.isfinite(score) or not 0<=score<=108:
            raise ValueError("Invalid visible Motor total")
        registry[pid]={"visible":score,"future_positive":None}
    visited=set()
    for row in paired:
        pid=row["pid"]
        if pid in visited or pid not in registry:
            raise ValueError("Inconsistent or duplicate complete-case patient")
        visited.add(pid)
        if abs(float(row["remote_sum"])-registry[pid]["visible"])>1e-6:
            raise ValueError("R01 scores differ between full and paired views")
        y=float(row["future_six"])
        if not math.isfinite(y) or not (0<=y<=MAX_HIDDEN):
            raise ValueError("Future six-item score outside 0..24")
        registry[pid]["future_positive"]=bool(y>0)
    if not visited or len(registry)<=len(visited):
        raise ValueError("Expected both observed and missing future-V04 labels")
    return registry


def exact_bounds(registry,chosen):
    """Mathematically sharp bounds with NO assumption about missing labels.

    a = selected known positives, u = selected unlabeled,
    p = all known positives, v = unselected unlabeled.
    Precision lower a/K; upper (a+u)/K.
    Recall lower a/(p+v), upper (a+u)/(p+u).
    They arise by assigning all unobserved labels adversely/favorably.
    """
    if not chosen or not chosen.issubset(registry):
        raise ValueError("Empty or invalid selection")
    known_p=sum(x["future_positive"] is True for x in registry.values())
    known_n=sum(x["future_positive"] is False for x in registry.values())
    unknown=len(registry)-known_p-known_n
    a=sum(registry[pid]["future_positive"] is True for pid in chosen)
    b=sum(registry[pid]["future_positive"] is False for pid in chosen)
    u=len(chosen)-a-b
    v=unknown-u
    if known_p<=0 or min(a,b,u,v)<0:
        raise ValueError("Invalid label accounting")
    k=len(chosen)
    return {
        "selected_n":k,
        "observed_positive_selected":a,
        "observed_negative_selected":b,
        "unknown_selected":u,
        "unknown_not_selected":v,
        "known_positive_total":known_p,
        "known_negative_total":known_n,
        "unknown_total":unknown,
        "precision_lower":a/k,
        "precision_upper":(a+u)/k,
        "recall_lower":a/(known_p+v),
        "recall_upper":(a+u)/(known_p+u),
        "observed_case_selected_n":a+b,
        "observed_case_recall":a/known_p,
    }


def safe_count(n,min_count):
    return str(n) if n==0 or n>=min_count else f"<{min_count}"


def fmt_pct(value):
    return round(100*value,2)


def missingness_by_score(registry,minimum):
    rows=[]
    for label,left,right in BINS:
        patients=[r for r in registry.values() if left<=r["visible"]<=right]
        n=len(patients)
        complete=sum(r["future_positive"] is not None for r in patients)
        missing=n-complete
        positive=sum(r["future_positive"] is True for r in patients)
        rows.append({
            "R01_visible_score_bin":label,
            "R01_total_patients":safe_count(n,minimum),
            "later_V04_available":safe_count(complete,minimum),
            "later_V04_missing":safe_count(missing,minimum),
            "known_future_positive":safe_count(positive,minimum),
            "observability_pct":fmt_pct(complete/n) if n>=minimum else "",
        })
    return rows


def run_calibration(registry,seed=20261018,tie_repeats=100,minimum=10):
    if minimum<10 or tie_repeats<20:
        raise ValueError("Unsafe small-cell threshold or tie repeats")
    n=len(registry)
    known_p=sum(r["future_positive"] is True for r in registry.values())
    unknown=sum(r["future_positive"] is None for r in registry.values())
    if n<minimum or known_p<minimum or unknown<minimum:
        raise ValueError("Not enough subjects/positives/missing outcomes")
    rows=[];tie_rows=[]
    for budget in BUDGETS:
        chosen=topk(registry,budget,seed)
        bounds=exact_bounds(registry,chosen)
        k=bounds["selected_n"]
        rows.append({
            "budget_pct":int(100*budget),
            "eligible_R01":n,
            "exact_topk":k,
            "selected_known_future_positive":
                safe_count(bounds["observed_positive_selected"],minimum),
            "selected_without_V04_label":
                safe_count(bounds["unknown_selected"],minimum),
            "future_positive_precision_lower_pct":fmt_pct(bounds["precision_lower"]),
            "future_positive_precision_upper_pct":fmt_pct(bounds["precision_upper"]),
            "future_positive_recall_lower_pct":fmt_pct(bounds["recall_lower"]),
            "future_positive_recall_upper_pct":fmt_pct(bounds["recall_upper"]),
            "known_label_recall_pct":fmt_pct(bounds["observed_case_recall"]),
            "random_expected_recall_pct":fmt_pct(k/n),
            "worst_case_lower_exceeds_random_expected":
                bounds["recall_lower"]>k/n,
        })
        trials=[exact_bounds(registry,topk(registry,budget,seed+j))
                for j in range(tie_repeats)]
        tie_rows.append({
            "budget_pct":int(100*budget),
            "tie_seed_repeats":tie_repeats,
            "known_positive_selected_min":
                safe_count(min(v["observed_positive_selected"] for v in trials),minimum),
            "known_positive_selected_max":
                safe_count(max(v["observed_positive_selected"] for v in trials),minimum),
            "unknown_selected_min":
                safe_count(min(v["unknown_selected"] for v in trials),minimum),
            "unknown_selected_max":
                safe_count(max(v["unknown_selected"] for v in trials),minimum),
            "worst_case_recall_lowest_pct":
                fmt_pct(min(v["recall_lower"] for v in trials)),
            "best_case_recall_highest_pct":
                fmt_pct(max(v["recall_upper"] for v in trials)),
        })
    return rows,tie_rows


def save(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def audit(root,manifest,out,seed=20261018,tie_repeats=100,minimum=10):
    root,manifest,out=[Path(p).expanduser().resolve()
                       for p in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Original PPMI source or private manifest missing")
    if out==manifest or out==root or root in out.parents:
        raise ValueError("Unsafe report directory")
    if minimum<10 or tie_repeats<20:
        raise ValueError("Unsafe audit settings")
    paired,remote,site,qc,source,train_registry=load_train_pairs(root,manifest)
    registry=build_registry(remote,paired)
    budget,ties=run_calibration(registry,seed,tie_repeats,minimum)
    positive=sum(r["future_positive"] is True for r in registry.values())
    missing=sum(r["future_positive"] is None for r in registry.values())
    n=len(registry)
    out.mkdir(parents=True,exist_ok=True)
    save(out/"missing_outcome_budget_bounds.csv",[
        "budget_pct","eligible_R01","exact_topk",
        "selected_known_future_positive","selected_without_V04_label",
        "future_positive_precision_lower_pct",
        "future_positive_precision_upper_pct",
        "future_positive_recall_lower_pct",
        "future_positive_recall_upper_pct",
        "known_label_recall_pct","random_expected_recall_pct",
        "worst_case_lower_exceeds_random_expected"],budget)
    save(out/"tie_seed_bound_sensitivity.csv",[
        "budget_pct","tie_seed_repeats",
        "known_positive_selected_min","known_positive_selected_max",
        "unknown_selected_min","unknown_selected_max",
        "worst_case_recall_lowest_pct","best_case_recall_highest_pct"],ties)
    save(out/"followup_completeness_by_R01_score.csv",[
        "R01_visible_score_bin","R01_total_patients",
        "later_V04_available","later_V04_missing",
        "known_future_positive","observability_pct"],
        missingness_by_score(registry,minimum))
    save(out/"source_quality_flags.csv",["check","records_or_patients"],[
        {"check":k,"records_or_patients":safe_count(v,minimum)}
        for k,v in sorted(qc.items())])
    summary={
        "status":"TRAIN_only_all_R01_future_V04_unknown_label_bounds",
        "frozen_train_registry":train_registry,
        "all_eligible_R01_patients":n,
        "future_V04_observed":n-missing,
        "future_V04_missing":missing,
        "observed_future_positive":positive,
        "all_R01_future_positive_prevalence_lower_pct":
            fmt_pct(positive/n),
        "all_R01_future_positive_prevalence_upper_pct":
            fmt_pct((positive+missing)/n),
        "tie_seed_repeats":tie_repeats,
        "seed":seed,"budgets_percent":[int(100*x) for x in BUDGETS],
        "missingness_assumption":"NONE: each missing V04 binary outcome may be 0 or 1",
        "caveats":[
            "Sharp bounds characterize unknown future-V04 labels, NOT hidden same-day R01 truth.",
            "Worst-case precision and recall are population identification bounds, not confidence intervals.",
            "A lower bound exceeding random expected recall is NOT a statistical significance test.",
            "Later V04 completeness may be informative; no missing at random assumption used.",
            "Repeated TRAIN explorations are not independent prospective validation.",
            "Tie breaks use SHA256 of patient IDs only in local RAM, never exported.",
            "All analyses are descriptive; no actual referral cost/utility estimated.",
            "Only frozen PPMI TRAIN subjects; val/test remain unopened.",
            "No subject-level score, date, label, ranking, or identifier exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+chr(10))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase42_missing_followup_bounds"))
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--tie-repeats",type=int,default=100)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.data_root,a.manifest_db,a.report_dir,
                          a.seed,a.tie_repeats,a.min_cell_count),
                     indent=2,ensure_ascii=False))
    print("Upload aggregate Phase42 reports ONLY; never patient-level data.")

if __name__=="__main__":
    main()
