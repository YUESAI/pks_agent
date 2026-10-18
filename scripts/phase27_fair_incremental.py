#!/usr/bin/env python3
"""Phase 27: fair incremental-value audit of baseline multimodal predictors.

Research-only, TRAIN-only nested cross-validation in PPMI Prodromal subjects.
Outer folds exactly match Phase24/26. Ridge training, median imputation,
standardization and penalty selection use *only the fitting portion* of a
fold. All models use the same penalty grid and the same inner fold IDs.

Primary hyperparameter objective is inner-fold RMSE for the future Motor
score, fixed before viewing Phase27 results. MAE remains secondary.
Do NOT query Phase23 val/test observations or target values. Outputs are
aggregates only; no patient identifiers, raw values or predictions exported.

Age excluded until source/measurement-time provenance is established.
UNSPECIFIED medication state does not prove treatment equivalence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import (
    load_train_only,fixed_folds,metrics,severity_bin
)
from scripts.phase26_multimodal_oof import (
    EXTRA,fit_multi,predict_multi,load_features,bootstrap_differences
)

FEATURE_SETS={
    "motor_ridge":(),
    "motor_moca":("moca_total",),
    "motor_core":("moca_total","rbd_vivid_dreams","scopa_aut_item1"),
    "motor_core_upsit":tuple(EXTRA),
}
ALL_MODELS=("persistence",)+tuple(FEATURE_SETS)
PENALTIES=(0.01,0.1,1.0,10.0,100.0)
INNER_FOLDS=3


def load_dataset(manifest,source,engine):
    """Only PPMI TRAIN subjects with a 12-month target are loaded."""
    rows,registry,pair_counts=load_train_only(manifest)
    ids=[pid for pid,bl,target in rows]
    anchors={}
    with sqlite3.connect("file:"+str(manifest)+"?mode=ro",uri=True) as db:
        for start in range(0,len(ids),200):
            batch=ids[start:start+200]
            marks=",".join("?" for _ in batch)
            for pid,month in db.execute(
                "SELECT pid,baseline_month FROM patients WHERE split='train' "
                "AND pid IN ("+marks+")",batch):
                if pid in anchors:raise ValueError("Duplicate baseline month")
                anchors[pid]=month
    if set(anchors)!=set(ids):
        raise ValueError("Missing TRAIN baseline month for eligible patient")
    extras,status_counts,age_provenance,upsit_provenance=load_features(
        source,engine,ids,anchors)
    return ([(pid,x,y,extras[pid]) for pid,x,y in rows],
            registry,pair_counts,status_counts)


def tune_penalty(outer_fit,model,seed,outer_fold):
    """Fit no hyperparameters on the outer evaluation fold."""
    names=FEATURE_SETS[model]
    mini_rows=[(pid,x,y) for pid,x,y,features in outer_fit]
    inner=fixed_folds(mini_rows,INNER_FOLDS,seed+1009*(outer_fold+1))
    scores={}
    for penalty in PENALTIES:
        sum_squared=0.
        n=0
        for held in range(INNER_FOLDS):
            fit=[row for row in outer_fit if inner[row[0]]!=held]
            evaluate=[row for row in outer_fit if inner[row[0]]==held]
            if not evaluate:raise ValueError("Empty nested fitting fold")
            params=fit_multi(fit,names,penalty=penalty)
            for pid,x,y,features in evaluate:
                pred=predict_multi(params,x,features)
                sum_squared+=(pred-y)**2
                n+=1
        if n!=len(outer_fit):raise ValueError("Lost inner CV patient")
        scores[penalty]=math.sqrt(sum_squared/n)
    # Stable tie-break: stronger regularization when inner performance ties.
    chosen=min(PENALTIES,key=lambda p:(scores[p],-p))
    return chosen,scores[chosen]


def write_csv(path,fields,rows):
    with path.open("w",encoding="utf-8",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit(manifest,source,engine,output,folds=5,seed=20261018,
          bootstraps=500,minimum=10):
    manifest,source,output=[Path(p).expanduser().resolve()
                            for p in (manifest,source,output)]
    if not manifest.is_file() or not source.is_file():
        raise FileNotFoundError("Phase23 manifest or clinical DB not found")
    if output in (manifest,source) or folds<2 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid audit arguments")
    dataset,registry,pair_counts,feature_status=load_dataset(
        manifest,source,engine)
    if len(dataset)<folds*10:
        raise ValueError("Too few eligible TRAIN patients for nested CV")
    outer=fixed_folds([(pid,x,y) for pid,x,y,_ in dataset],folds,seed)
    scores={model:{} for model in ALL_MODELS}
    hyperparameters=[]
    fold_counts=[]
    for held in range(folds):
        fit=[r for r in dataset if outer[r[0]]!=held]
        test=[r for r in dataset if outer[r[0]]==held]
        if not fit or not test:raise ValueError("Empty outer CV fold")
        fold_counts.append({
            "fold":held,"fit_patients":len(fit),
            "heldout_train_patients":len(test)})
        fitted={}
        for model in FEATURE_SETS:
            penalty,inner_rmse=tune_penalty(fit,model,seed,held)
            fitted[model]=fit_multi(fit,FEATURE_SETS[model],penalty=penalty)
            hyperparameters.append({
                "outer_fold":held,"model":model,
                "selected_penalty":penalty,
                "fit_only_inner_cv_rmse":round(inner_rmse,5)})
        for pid,x,y,extra in test:
            if pid in scores["persistence"]:
                raise ValueError("Patient evaluated more than once")
            scores["persistence"][pid]=(y,x)
            for model,params in fitted.items():
                scores[model][pid]=(y,predict_multi(params,x,extra))
    if any(len(scores[model])!=len(dataset) for model in ALL_MODELS):
        raise ValueError("Missing outer-fold prediction")
    metric_rows=[]
    for model in ALL_MODELS:
        result=metrics(list(scores[model].values()))
        metric_rows.append({
            "model":model,
            "population":"PPMI_Prodromal_TRAIN_nested_OOF",
            **{k:round(v,5) if isinstance(v,float) else v
               for k,v in result.items()},
        })
    comparisons=[]
    for target in ("persistence","motor_ridge"):
        for idx,model in enumerate(ALL_MODELS):
            if model==target or model=="persistence":continue
            if target=="motor_ridge" and model=="motor_ridge":continue
            for result in bootstrap_differences(
                scores[target],scores[model],bootstraps,
                seed+(101 if target=="motor_ridge" else 503)+31*idx):
                comparisons.append({
                    "model":model,"reference":target,
                    "comparison":model+"_minus_"+target,**result})
    bins=("0_to_3","4_to_7","8_to_15","16_plus")
    baseline={pid:x for pid,x,y,extra in dataset}
    severity=[]
    for group in bins:
        ids=[pid for pid,x in baseline.items() if severity_bin(x)==group]
        if len(ids)<minimum:continue
        for model in ALL_MODELS:
            m=metrics([scores[model][pid] for pid in ids])
            severity.append({
                "baseline_motor_bin":group,"model":model,"n":len(ids),
                "mae":round(m["mae"],5),"rmse":round(m["rmse"],5)})
    coverage=[
        {"feature":feature,"status":status,
         "patients":str(count) if count==0 or count>=minimum else f"<{minimum}"}
        for (feature,status),count in sorted(feature_status.items())]
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"nested_multimodal_oof.csv",
        ["model","population","n","mae","rmse","mean_signed_error",
         "observed_mean_target"],metric_rows)
    write_csv(output/"incremental_paired_differences.csv",
        ["model","reference","comparison","metric","difference",
         "lower_95","upper_95","bootstrap_replicates"],comparisons)
    write_csv(output/"inner_penalty_selection.csv",
        ["outer_fold","model","selected_penalty","fit_only_inner_cv_rmse"],
        hyperparameters)
    write_csv(output/"severity_stratified_errors.csv",
        ["baseline_motor_bin","model","n","mae","rmse"],severity)
    write_csv(output/"train_feature_status.csv",
        ["feature","status","patients"],coverage)
    write_csv(output/"outer_cv_fold_counts.csv",
        ["fold","fit_patients","heldout_train_patients"],fold_counts)
    summary={
        "status":"EXPLORATORY_train_only_nested_multimodal_incremental_value",
        "task":"PPMI prodromal baseline to 12 +/- 3-month Motor total",
        "train_pairs":len(dataset),"outer_folds":folds,
        "inner_folds":INNER_FOLDS,"penalty_grid":list(PENALTIES),
        "selection_metric":"inner 3-fold RMSE; stronger penalty breaks exact ties",
        "seed":seed,"bootstrap_replicates":bootstraps,
        "models":list(ALL_MODELS),
        "patient_splits":"unchanged Phase23; outer folds unchanged Phase24",
        "future_target_readiness_as_predictor":False,
        "age_included":False,
        "data_safety":[
            "Only Phase23 TRAIN observed 12-month pairs and TRAIN BL/SC clinical rows queried.",
            "Only proper inner training patients fit imputers, scales and regressions.",
            "No validation or test patient observations, scores or outcomes queried.",
            "No patient IDs, raw clinical values or per-subject predictions exported.",
            "Repeated use of TRAIN folds is exploratory; paired intervals are not independent final confirmation.",
            "Baselines are linear; a non-linear World Model has not been justified yet.",
            "Missing UPSIT remains imputed; absence can be informative.",
            "Medication UNSPECIFIED does not establish unchanged treatment.",
            "Single-visit questionnaire item coding still needs clinical codebook checks.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase27_fair_incremental"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=500)
    p.add_argument("--min-cell-count",type=int,default=10)
    args=p.parse_args()
    result=audit(args.manifest_db,args.source_db,args.source_engine,
                 args.report_dir,args.folds,args.seed,args.bootstraps,
                 args.min_cell_count)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Upload only aggregate Phase27 reports. Never upload private patient DBs.")


if __name__=="__main__":
    main()
