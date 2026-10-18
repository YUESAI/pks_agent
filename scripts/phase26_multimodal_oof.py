#!/usr/bin/env python3
"""Phase 26: leakage-controlled baseline multimodal Motor regression (TRAIN only).

Prediction target: PPMI Prodromal 12 +/- 3 month NP3TOT, evaluated using the
SAME five TRAIN folds as Phase24. Never query any validation/test target or
clinical observation. Select BL MoCA/RBD/SCOPA (month exactly BL) and UPSIT
at strictly pre-BL SC (1..6 months) using Phase25 quality checks.

Age is excluded: Phase25 observed 0 date-eligible ages. A separate aggregate
provenance audit investigates why, without treating undated age as BL evidence.

Ridge models use training-fold-only imputation/standardization and fitted
outcomes. Future target presence, actual visit gap and missing-future status
never enter a model. Only aggregates are written; research use only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import (
    load_train_only, fixed_folds, fit as motor_fit,
    predict as motor_predict, metrics as base_metrics, severity_bin,
)
from scripts.phase25_baseline_feature_audit import (
    source_rows, choose_feature, month_number
)

EXTRA=("moca_total", "rbd_vivid_dreams", "scopa_aut_item1", "upsit_total")
FEATURE_SETS={
    "motor_moca":("moca_total",),
    "motor_core":("moca_total","rbd_vivid_dreams","scopa_aut_item1"),
    "motor_core_upsit":EXTRA,
}
MODELS=("persistence","motor_ridge",*FEATURE_SETS)


def load_features(clinical,engine,train_ids,anchors):
    """Only TRAIN patient IDs; Phase25 temporal/QC gate reused unchanged."""
    observations=source_rows(clinical,engine,train_ids)
    values={}
    status_counts=Counter()
    age_provenance=Counter()
    upsit_provenance=Counter()
    for pid in train_ids:
        anchor=anchors[pid]
        features={}
        for concept in EXTRA:
            status,value=choose_feature(concept,observations[pid].get(concept,{}),anchor)
            status_counts[(concept,status)]+=1
            features[concept]=value if status in ("ready_BL","ready_SC") else None
        values[pid]=features
        age_bl=observations[pid].get("age_at_visit",{}).get("BL",[])
        if not age_bl:
            age_provenance["no_BL_age_row"]+=1
        else:
            seen=set()
            for raw,month,quality,imputed in age_bl:
                m=month_number(month)
                if m is None: seen.add("undated_BL_age_row")
                elif anchor is None: seen.add("missing_BL_anchor")
                elif m==anchor: seen.add("same_month_BL_age")
                elif m<anchor: seen.add("earlier_month_BL_age")
                else: seen.add("later_month_BL_age")
            for outcome in seen:
                age_provenance[outcome]+=1
        upsit_sc=observations[pid].get("upsit_total",{}).get("SC",[])
        if not upsit_sc:
            upsit_provenance["no_SC_UPSIT_row"]+=1
        else:
            seen=set()
            for raw,month,quality,imputed in upsit_sc:
                m=month_number(month)
                if m is None: seen.add("SC_month_unparseable")
                elif anchor is None: seen.add("missing_BL_anchor")
                elif m==anchor: seen.add("same_month_as_BL")
                elif m>anchor: seen.add("after_BL_month")
                elif anchor-m>6: seen.add("earlier_than_6_months")
                else: seen.add("strictly_prior_within_6_months")
            for outcome in seen:
                upsit_provenance[outcome]+=1
    return values,status_counts,age_provenance,upsit_provenance


def features_for(x, extra, medians, center, scale):
    """Predictor matrix depends ONLY on BL motor and selected BL features."""
    numbers=[float(x)]
    for name in extra:
        value=extra[name]
        numbers.append(medians[name] if value is None else float(value))
    # No missingness flag: avoid learning collection-pattern shortcuts.
    return [(v-c)/s for v,c,s in zip(numbers,center,scale)]


def solve_linear_system(matrix, rhs):
    """Partial-pivot Gaussian elimination; small ridge system, no dependency."""
    n=len(rhs)
    A=[list(matrix[i])+[float(rhs[i])] for i in range(n)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(A[j][i]))
        if abs(A[pivot][i])<1e-10:
            raise ValueError("Ill-conditioned ridge regression system")
        A[i],A[pivot]=A[pivot],A[i]
        scale=A[i][i]
        for j in range(i,n+1):A[i][j]/=scale
        for row in range(n):
            if row==i:continue
            ratio=A[row][i]
            for j in range(i,n+1):
                A[row][j]-=ratio*A[i][j]
    return [A[i][n] for i in range(n)]


def fit_multi(train, names, penalty=.25):
    """Fit delta-score ridge using only TRAIN-FOLD outcomes, no tuning."""
    if len(train)<10 or penalty<=0:
        raise ValueError("Need >=10 fitting patients and fixed positive penalty")
    medians={}
    for name in names:
        observed=sorted(row[3].get(name) for row in train
                        if row[3].get(name) is not None)
        medians[name]=(observed[(len(observed)-1)//2] if observed else 0.)
    raw=[[float(x)]+[
        medians[name] if extras.get(name) is None else float(extras[name])
        for name in names] for _,x,y,extras in train]
    width=1+len(names)
    centers=[mean(row[j] for row in raw) for j in range(width)]
    sd=[math.sqrt(mean((row[j]-centers[j])**2 for row in raw))
        for j in range(width)]
    sd=[v if v>1e-8 else 1. for v in sd]
    X=[[1.]+[(row[j]-centers[j])/sd[j] for j in range(width)]
       for row in raw]
    target=[float(y)-float(x) for pid,x,y,ex in train]
    p=width+1
    gram=[[sum(r[i]*r[j] for r in X) for j in range(p)] for i in range(p)]
    for j in range(1,p):
        gram[j][j]+=penalty*len(train)
    rhs=[sum(r[i]*y for r,y in zip(X,target)) for i in range(p)]
    coeffs=solve_linear_system(gram,rhs)
    return {"medians":medians,"centers":centers,"scale":sd,
            "coefficients":coeffs,"names":names}


def predict_multi(params, x, extras):
    vals=[float(x)]+[
        params["medians"][name] if extras.get(name) is None
        else float(extras[name]) for name in params["names"]]
    normalized=[(v-c)/s for v,c,s in
                zip(vals,params["centers"],params["scale"])]
    delta=params["coefficients"][0]+sum(
        a*b for a,b in zip(params["coefficients"][1:],normalized))
    return min(132.,max(0.,float(x)+delta))


def bootstrap_differences(ref,other,iterations,seed):
    """Paired change in MAE and RMSE, other minus persistence."""
    if ref.keys()!=other.keys():raise ValueError("Predictions not patient-paired")
    ids=sorted(ref)
    triples=[]
    for pid in ids:
        y0,p0=ref[pid];y1,p1=other[pid]
        if y0!=y1:raise ValueError("Patient labels and outcomes mismatched")
        triples.append((y0,p0,p1))
    if len(triples)<10 or iterations<100:raise ValueError("Too few bootstrap samples")
    def differences(rows):
        mae=mean(abs(p1-y)-abs(p0-y) for y,p0,p1 in rows)
        rmse=math.sqrt(mean((p1-y)**2 for y,p0,p1 in rows))-math.sqrt(
            mean((p0-y)**2 for y,p0,p1 in rows))
        return mae,rmse
    point=differences(triples)
    rng=random.Random(seed)
    bootstrap=[[],[]]
    for iteration in range(iterations):
        sample=[triples[rng.randrange(len(triples))] for _ in triples]
        values=differences(sample)
        for metric in (0,1):bootstrap[metric].append(values[metric])
    result=[]
    for j,name in enumerate(("MAE","RMSE")):
        items=sorted(bootstrap[j])
        def quantile(q):
            position=(len(items)-1)*q
            low=int(position);high=min(low+1,len(items)-1)
            return items[low]*(high-position)+items[high]*(position-low)
        result.append({"metric":name,"difference":round(point[j],5),
                       "lower_95":round(quantile(.025),5),
                       "upper_95":round(quantile(.975),5),
                       "bootstrap_replicates":iterations})
    return result


def export(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,fieldnames=fields)
        w.writeheader();w.writerows(rows)


def audit(manifest,source,engine,output,folds=5,seed=20261018,
          bootstraps=500,minimum=10):
    manifest,source,output=(Path(path).expanduser().resolve()
                            for path in (manifest,source,output))
    if not manifest.is_file() or not source.is_file() or output in (manifest,source):
        raise ValueError("Missing input or unsafe output directory")
    if folds<2 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid folds/bootstrap/minimum")
    train_rows,registry,count=load_train_only(manifest)
    ids=[pid for pid,x,y in train_rows]
    anchors={}
    # Read TRAIN-only baseline-month anchors. No val/test rows.
    import sqlite3
    with sqlite3.connect("file:"+str(manifest)+"?mode=ro",uri=True) as db:
        for start in range(0,len(ids),200):
            part=ids[start:start+200]
            q=",".join("?" for _ in part)
            anchors.update(db.execute(
                "SELECT pid,baseline_month FROM patients WHERE split='train' "
                "AND pid IN ("+q+")",part).fetchall())
    if set(anchors)!=set(ids):
        raise ValueError("One or more TRAIN baseline anchors not found")
    extras,status_counts,age_provenance,upsit_provenance=load_features(
        source,engine,ids,anchors)
    dataset=[(pid,x,y,extras[pid]) for pid,x,y in train_rows]
    fold_by_id=fixed_folds(train_rows,folds,seed)
    oof={name:{} for name in MODELS}
    fold_counts=[]
    for held in range(folds):
        fit_rows=[r for r in dataset if fold_by_id[r[0]]!=held]
        held_rows=[r for r in dataset if fold_by_id[r[0]]==held]
        baseline_params=motor_fit([(pid,x,y) for pid,x,y,ex in fit_rows])
        multi={name:fit_multi(fit_rows,FEATURE_SETS[name])
               for name in FEATURE_SETS}
        fold_counts.append({"fold":held,"fit_n":len(fit_rows),
                            "heldout_train_n":len(held_rows)})
        for pid,x,y,ex in held_rows:
            for name in MODELS:
                if name=="persistence":prediction=x
                elif name=="motor_ridge":
                    prediction=motor_predict("ridge_delta",baseline_params,x)
                else:prediction=predict_multi(multi[name],x,ex)
                oof[name][pid]=(y,prediction)
    if any(len(p)!=len(train_rows) for p in oof.values()):
        raise ValueError("Out-of-fold coverage incomplete")
    metrics_rows=[]
    paired_rows=[]
    for idx,name in enumerate(MODELS):
        stats=base_metrics(list(oof[name].values()))
        metrics_rows.append({"model":name,"population":"PPMI_Prodromal_TRAIN_OOF",
                             **{k:round(v,5) if isinstance(v,float) else v
                                for k,v in stats.items()}})
        if name!="persistence":
            paired_rows.extend({"model":name,"comparison":name+"_minus_persistence",
                                **r} for r in bootstrap_differences(
                                    oof["persistence"],oof[name],bootstraps,seed+103*idx))
    rows=[]
    for concept in EXTRA:
        for kind in ("ready_BL","ready_SC","not_recorded","date_rejected",
                     "quality_rejected","range_rejected","ambiguous_repeat"):
            rows.append({"feature":concept,"status":kind,
                         "n_train_pairs":str(status_counts[(concept,kind)])
                         if status_counts[(concept,kind)]>=minimum or
                            status_counts[(concept,kind)]==0 else f"<{minimum}"})
    age_rows=[{"audit":"age_BL_month_provenance","category":name,
               "patients":str(value) if value>=minimum or value==0 else f"<{minimum}"}
              for name,value in sorted(age_provenance.items())]
    upsit_rows=[{"audit":"UPSIT_SC_month_provenance","category":name,
                 "patients":str(value) if value>=minimum or value==0 else f"<{minimum}"}
                for name,value in sorted(upsit_provenance.items())]
    output.mkdir(parents=True,exist_ok=True)
    export(output/"multimodal_train_oof.csv",
           ["model","population","n","mae","rmse","mean_signed_error",
            "observed_mean_target"],metrics_rows)
    export(output/"multimodal_paired_vs_persistence.csv",
           ["model","comparison","metric","difference","lower_95","upper_95",
            "bootstrap_replicates"],paired_rows)
    export(output/"paired_train_feature_coverage.csv",
           ["feature","status","n_train_pairs"],rows)
    export(output/"age_upsit_date_provenance.csv",
           ["audit","category","patients"],age_rows+upsit_rows)
    export(output/"train_cv_fold_counts.csv",
           ["fold","fit_n","heldout_train_n"],fold_counts)
    summary={
        "status":"EXPLORATORY_PPMI_prodromal_TRAIN_multimodal_predictive_check",
        "n_train_pairs":len(train_rows),"folds":folds,"seed":seed,
        "models":list(MODELS),"features_excluded":["age_at_visit"],
        "validated_baseline_feature_window":"MoCA/RBD/SCOPA BL anchored; UPSIT pre-BL SC <=6 months",
        "fit_only_on_inner_training_fold":True,
        "folds_match_phase24":True,
        "ridge_penalty_fixed_before_results":0.25,
        "no_missingness_indicators_in_model":True,
        "privacy_and_limitations":[
            "Only TRAIN patient IDs, 12-month target scores and their baseline observations are queried.",
            "No PPMI validation/test clinical measurements or future scores opened.",
            "No patient-level data, individual targets or predictions are exported.",
            "Age excluded pending investigation of BL-event undated source records.",
            "UPSIT missingness is substantial; model imputes from fitting fold only.",
            "Within-calendar-month ordering remains unverified.",
            "Questionnaire numeric code and medication-state semantics remain provisional.",
            "Follow-up may be informatively missing; OOF applies to observed-target subset.",
            "This is not a clinically validated progression or World Model assessment.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase26_multimodal_oof"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=500)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.manifest_db,a.source_db,a.source_engine,
                          a.report_dir,a.folds,a.seed,a.bootstraps,
                          a.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload aggregate Phase26 reports only. Keep private DBs on cluster.")

if __name__=="__main__":
    main()
