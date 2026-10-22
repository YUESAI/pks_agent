#!/usr/bin/env python3
"""Phase 10: label-aware availability shortcut audit + immutable patient splits.

Read source v2 DB and Phase 9 manifest. Write a LOCAL SQLite split index,
PLUS AGGREGATE reports. No patient-level data may leave the cluster.

This is a retrospective research diagnostic audit, NOT prospective validation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS=("train","val","test")
def open_source(path,engine):
    if engine=="duckdb":
        try:import duckdb
        except ImportError as exc:raise RuntimeError("Install duckdb") from exc
        return duckdb.connect(str(path),read_only=True)
    return sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True)

def split_patients(groups,seed):
    """Stratify by label+anchor, split at patient level; stable across runs."""
    results={}
    for (cohort,label,anchored),patients in sorted(groups.items()):
        if cohort=="BioFIND":
            for pid in patients:results[(cohort,pid)]="external_test"
            continue
        ordered=sorted(patients,key=lambda pid:hashlib.sha256(
            (str(seed)+"|"+cohort+"|"+label+"|"+str(anchored)+"|"+pid).encode()).hexdigest())
        n=len(ordered)
        n_train=round(n*0.70)
        n_val=round(n*0.15)
        if n>=3:
            n_train=max(1,min(n-2,n_train))
            n_val=max(1,min(n-n_train-1,n_val))
        for i,pid in enumerate(ordered):
            results[(cohort,pid)]=("train" if i<n_train else
                "val" if i<n_train+n_val else "test")
    return results

def auc(scores):
    """Rank-based AUROC of list (log-odds, true_is_PD); exact tied ranks."""
    positives=sum(y for _,y in scores);negatives=len(scores)-positives
    if not positives or not negatives:return None
    ordered=sorted(scores,key=lambda p:p[0])
    total_rank=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        avg_rank=(i+1+j)/2.
        total_rank+=avg_rank*sum(y for _,y in ordered[i:j])
        i=j
    return (total_rank-positives*(positives+1)/2)/(positives*negatives)

def classify_naive_bayes(rows, actions):
    """Fit a fixed Laplace-smoothed Bernoulli availability-only baseline."""
    stats={lab:Counter() for lab in ("PD","HC")}
    ns=Counter()
    for label,ready in rows:
        ns[label]+=1
        stats[label].update(ready)
    if not all(ns.values()):return None
    probabilities={lab:{action:(stats[lab][action]+1)/(ns[lab]+2)
                         for action in actions} for lab in ns}
    return probabilities

def score_nb(model, ready):
    # Equal class priors on purpose: report a balanced diagnostic baseline.
    scores={}
    for lab in ("PD","HC"):
        value=0.
        for action,p in model[lab].items():
            value+=math.log(p if action in ready else 1-p)
        scores[lab]=value
    return scores["PD"]-scores["HC"]

def evaluate(model, rows):
    scored=[(score_nb(model,ready), label=="PD") for label,ready in rows]
    tp=sum(score>=0 and truth for score,truth in scored)
    fp=sum(score>=0 and not truth for score,truth in scored)
    tn=sum(score<0 and not truth for score,truth in scored)
    fn=sum(score<0 and truth for score,truth in scored)
    def div(a,b):return a/b if b else None
    pd_recall=div(tp,tp+fn);hc_recall=div(tn,tn+fp)
    balanced=((pd_recall+hc_recall)/2 if pd_recall is not None and hc_recall is not None else None)
    def f1(a,b,c):return div(2*a,2*a+b+c)
    fpd=f1(tp,fp,fn);fhc=f1(tn,fp,fn)
    macro=((fpd+fhc)/2 if fpd is not None and fhc is not None else None)
    return dict(n=len(rows),n_PD=tp+fn,n_HC=tn+fp,
                balanced_accuracy=round(balanced,4) if balanced is not None else "",
                macro_f1=round(macro,4) if macro is not None else "",
                auroc=round(auc(scored),4) if auc(scored) is not None else "")

def safe_count(n,minimum):
    return str(n) if n==0 or n>=minimum else f"<{minimum}"

def write(path,fields,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def audit(source_path,source_engine,manifest_path,split_path,out,seed,min_count):
    source_path,manifest_path,split_path,out=(
        source_path.resolve(),manifest_path.resolve(),split_path.resolve(),out.resolve())
    if not source_path.is_file() or not manifest_path.is_file():
        raise ValueError("Missing local source DB or manifest DB")
    if split_path.exists():raise FileExistsError("Split DB already exists; choose a new path")
    if out==split_path or split_path==source_path or split_path==manifest_path:
        raise ValueError("Output paths must be distinct")
    source=open_source(source_path,source_engine)
    man=sqlite3.connect("file:"+str(manifest_path)+"?mode=ro",uri=True)
    try:
        labels={(cohort,pid):label for cohort,pid,label in source.execute(
            "SELECT cohort,patient_id,candidate_label FROM restricted_labels "
            "WHERE candidate_label IN ('PD','HC')").fetchall()}
        episodes=man.execute(
            "SELECT cohort,patient_id,baseline_month FROM episode_candidates").fetchall()
        catalogs=defaultdict(set)
        for cohort,action in man.execute("SELECT cohort,action_key FROM action_catalog"):
            catalogs[cohort].add(action)
        groups=defaultdict(list)
        seen=set()
        anchor_count=Counter()
        for cohort,pid,anchor in episodes:
            key=(cohort,pid)
            if key in seen:raise ValueError("Duplicate episode patient")
            seen.add(key)
            if key not in labels:raise ValueError("Episode is missing its restricted candidate label")
            present=bool(anchor)
            groups[(cohort,labels[key],present)].append(pid)
            anchor_count[(cohort,labels[key],present)]+=1
        if set(labels)!=seen:
            raise ValueError("Source candidate labels and manifest population are inconsistent")
        assigned=split_patients(groups,seed)
        mask=defaultdict(set)
        duplicate_action=Counter()
        for cohort,pid,action,status in man.execute(
            "SELECT cohort,patient_id,action_key,status FROM episode_actions"):
            key=(cohort,pid)
            if key not in seen or action not in catalogs[cohort]:
                raise ValueError("Manifest has orphan episode action or unknown catalog action")
            duplicate_action[(cohort,pid,action)]+=1
            if status=="ready":mask[key].add(action)
        if any(n>1 for n in duplicate_action.values()):
            raise ValueError("Duplicate action row per patient/action")
    finally:
        source.close();man.close()

    split_path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(split_path)
    try:
        db.execute("""CREATE TABLE patient_splits(
            cohort TEXT,patient_id TEXT,split TEXT,
            PRIMARY KEY(cohort,patient_id))""")
        db.executemany("INSERT INTO patient_splits VALUES(?,?,?)",
                       [(cohort,pid,split) for (cohort,pid),split in sorted(assigned.items())])
        db.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT)")
        db.executemany("INSERT INTO metadata VALUES(?,?)",
                       [("status","candidate_unvalidated"),
                        ("seed",str(seed)),
                        ("algorithm","SHA256_stratified_cohort_label_anchor_70_15_15"),
                        ("created_utc",datetime.now(timezone.utc).isoformat())])
        db.commit()
    finally:db.close()

    out.mkdir(parents=True,exist_ok=True)
    totals=Counter((cohort,assigned[key],labels[key]) for key in seen for cohort in [key[0]])
    splits=[dict(cohort=c,split=s,label=l,patients=safe_count(n,min_count))
            for (c,s,l),n in sorted(totals.items())]
    bias=[]
    for cohort,options in sorted(catalogs.items()):
        counts=Counter(l for key,l in labels.items() if key in seen and key[0]==cohort)
        for action in sorted(options):
            ready=Counter(labels[key] for key in seen if key[0]==cohort and action in mask[key])
            d_pd=counts["PD"];d_hc=counts["HC"]
            r_pd=ready["PD"];r_hc=ready["HC"]
            display=(r_pd==0 or r_pd>=min_count) and (r_hc==0 or r_hc>=min_count)
            p_pd=r_pd/d_pd if d_pd else None
            p_hc=r_hc/d_hc if d_hc else None
            bias.append(dict(
                cohort=cohort,action=action,PD_total=d_pd,HC_total=d_hc,
                PD_ready=safe_count(r_pd,min_count),HC_ready=safe_count(r_hc,min_count),
                PD_ready_pct=round(100*p_pd,2) if display and p_pd is not None else "",
                HC_ready_pct=round(100*p_hc,2) if display and p_hc is not None else "",
                availability_gap_percentage_points=round(100*abs(p_pd-p_hc),2)
                  if display and p_pd is not None and p_hc is not None else "",
                single_action_shortcut_balanced_accuracy=round(0.5+abs(p_pd-p_hc)/2,4)
                  if display and p_pd is not None and p_hc is not None else "",
            ))
    baseline=[]
    # Fit availability-only model on PPMI train, never on external-test labels.
    train=[(labels[k],mask[k]) for k,s in assigned.items() if k[0]=="PPMI" and s=="train"]
    model=classify_naive_bayes(train,sorted(catalogs["PPMI"]))
    if model:
        for split in SPLITS:
            data=[(labels[k],mask[k]) for k,s in assigned.items() if k[0]=="PPMI" and s==split]
            baseline.append(dict(cohort="PPMI",split=split,**evaluate(model,data)))
    write(out/"split_summary.csv",["cohort","split","label","patients"],splits)
    write(out/"action_availability_bias.csv",
          ["cohort","action","PD_total","HC_total","PD_ready","HC_ready",
           "PD_ready_pct","HC_ready_pct","availability_gap_percentage_points",
           "single_action_shortcut_balanced_accuracy"],bias)
    write(out/"availability_only_baseline.csv",
          ["cohort","split","n","n_PD","n_HC","balanced_accuracy","macro_f1","auroc"],baseline)
    anchor=[dict(cohort=c,label=l,has_BL_month_anchor=bool(has),
                 patients=safe_count(n,min_count))
            for (c,l,has),n in sorted(anchor_count.items())]
    write(out/"anchor_by_label.csv",
          ["cohort","label","has_BL_month_anchor","patients"],anchor)
    summary={
        "status":"candidate_unvalidated",
        "patients_total":len(seen),
        "split_rows_local_only":len(assigned),
        "seed":seed,
        "split_rules":"PPMI patient-stratified label+anchor 70/15/15; BioFIND external_test only",
        "availability_only_baseline":"Laplace-smoothed Bernoulli Naive Bayes, equal class priors",
        "warnings":[
            "Availability can be strongly predictive of diagnosis; Doctor should not receive internals of unavailability.",
            "No patient IDs are exported in reports; local split SQLite contains identifiers.",
            "Splits are NOT clinically validated; cohort and label assignment policies require audit.",
            "Source and manifest DB are protected research artifacts; keep them on cluster.",
            "No model should be trained before independently verifying outcome labeling and timestamp semantics.",
            "Patient splits may still be affected by site/patient selection and cohort-shift confounding.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db",required=True,type=Path)
    parser.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--split-db",type=Path,default=Path("outputs/episode_splits/splits_v0.sqlite"))
    parser.add_argument("--report-dir",type=Path,default=Path("outputs/leakage_report"))
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    if args.min_cell_count<10:parser.error("--min-cell-count must be >=10")
    summary=audit(args.source_db,args.source_engine,args.manifest_db,
                  args.split_db,args.report_dir,args.seed,args.min_cell_count)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print("Upload only aggregate reports. NEVER upload split, manifest or source DB.")

if __name__=="__main__":main()
