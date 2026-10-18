#!/usr/bin/env python3
"""Phase 12: train-only budgeted Doctor–Patient interface benchmark.

Measures fixed/random request policies' coverage and cost, NOT diagnostic
accuracy. Reports contain aggregates only; no raw values or patient IDs.
"""
import argparse
import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.budgeted_env_v0 import BudgetedDoctorPatientEnv, ACTION_SPECS

ORDERS = {
    "fixed_clinical": ("age","sleep_item","autonomic_item",
                       "cognitive_test","motor_exam","smell_test"),
    "fixed_reverse": ("smell_test","motor_exam","cognitive_test",
                      "autonomic_item","sleep_item","age"),
}

def action_order(policy, seed, pid):
    if policy!="random":
        return ORDERS[policy]
    digest=hashlib.sha256((str(seed)+"|"+pid).encode()).digest()
    seq=list(ACTION_SPECS)
    random.Random(int.from_bytes(digest[:8],"big")).shuffle(seq)
    return seq

def play(manifest,source,engine,pid,policy,budget,seed):
    stats=Counter()
    with BudgetedDoctorPatientEnv(manifest,source,pid,
                                  budget=budget,source_engine=engine) as env:
        for action in action_order(policy,seed,pid):
            result=env.request(action)
            if result["status"]=="insufficient_budget":
                continue
            stats["requested_"+action]+=1
            stats["status_"+result["status"]]+=1
            if result["status"]=="recorded":
                stats["recorded_"+action]+=1
        stats["cost"]=env.spent
        stats["n_requested"]=len(env.requested)
        stats["patients_with_results"]=int(stats["status_recorded"]>0)
        stats["patients"]=1
    return stats

def write(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)

def run(source,engine,manifest,splits,output,budget=5,seed=20261018,minimum=10):
    source,manifest,splits,output=map(
        lambda x:Path(x).expanduser().resolve(),(source,manifest,splits,output))
    if not all(f.is_file() for f in (source,manifest,splits)):
        raise ValueError("Missing source, manifest or split DB")
    if budget<0 or minimum<10:
        raise ValueError("Budget must be nonnegative and minimum count >=10")
    conn=sqlite3.connect("file:"+str(splits)+"?mode=ro",uri=True)
    try:
        people=[pid for cohort,pid,split in conn.execute(
            "SELECT cohort,patient_id,split FROM patient_splits")
            if cohort=="PPMI" and split=="train"]
    finally:
        conn.close()
    if not people:raise ValueError("No PPMI TRAIN patients")
    policies=("fixed_clinical","fixed_reverse","random")
    summary_rows=[];action_rows=[]
    for policy in policies:
        aggregate=Counter()
        for pid in people:
            aggregate.update(play(manifest,source,engine,pid,policy,budget,seed))
        n=aggregate["patients"]
        total=aggregate["n_requested"]
        summary_rows.append({
            "policy":policy,"evaluated_split":"PPMI_train_only",
            "n_patients":n,"budget":budget,
            "mean_cost":round(aggregate["cost"]/n,3),
            "mean_actions":round(total/n,3),
            "mean_results":round(aggregate["status_recorded"]/n,3),
            "patients_with_at_least_one_result_pct":round(
                100*aggregate["patients_with_results"]/n,2),
            "unavailable_rate_pct":round(100*aggregate["status_not_available"]/total,2)
                if total else "",
        })
        for action in ACTION_SPECS:
            req=aggregate["requested_"+action]
            got=aggregate["recorded_"+action]
            action_rows.append({"policy":policy,"action":action,
                "requested":req,"returned":got,
                "return_rate_pct":round(100*got/req,2) if req>=minimum else ""})
    output.mkdir(parents=True,exist_ok=True)
    write(output/"policy_service_metrics.csv",
          ["policy","evaluated_split","n_patients","budget","mean_cost",
           "mean_actions","mean_results","patients_with_at_least_one_result_pct",
           "unavailable_rate_pct"],summary_rows)
    write(output/"action_request_metrics.csv",
          ["policy","action","requested","returned","return_rate_pct"],action_rows)
    report={
        "status":"train_only_interface_smoke_test",
        "n_train_patients":len(people),"budget":budget,"policies":list(policies),
        "warnings":[
            "This is an interface and coverage smoke test, not diagnosis accuracy.",
            "No PPMI val/test or BioFIND episodes are played.",
            "Motor ON/OFF/UNSPECIFIED variants are internally selected by precedence;",
            " corresponding clinical values are not interchangeable.",
            "Unknown or unrecorded observations must not be interpreted as negatives.",
            "The replay is retrospective and cannot reconstruct diagnosis-time order.",
            "No patient IDs or individual outcomes are exported.",
        ],
        "created_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db",type=Path,required=True)
    parser.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--split-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,default=Path("outputs/phase12_smoke"))
    parser.add_argument("--budget",type=int,default=5)
    parser.add_argument("--seed",type=int,default=20261018)
    a=parser.parse_args()
    print(json.dumps(run(a.source_db,a.source_engine,a.manifest_db,a.split_db,
                         a.report_dir,a.budget,a.seed),ensure_ascii=False,indent=2))
    print("Upload aggregate reports only, not patient-level databases")

if __name__=="__main__":
    main()
