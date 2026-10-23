#!/usr/bin/env python3
"""Phase62 CPU formal patient-disjoint multi-cohort future-site prediction.

Independent PD, HC and frozen-TRAIN Prodromal 6/12/24-month tasks.
Predict future measured six-hands-on burden FROM TRUE BL ONLY.
Four fixed baselines, 5-fold stratified patient-level OOF within cohort.
This is exploratory frozen PPMI retrospective prediction, not a Doctor trial.
"""
from __future__ import annotations
import argparse,csv,json,math,random,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs,GROUPS,VIDX
from scripts.phase60_cpu_multistep_wm import auc

SEED=20261018
METHODS=("fit_mean","baseline_visible_sum","masked_ridge","masked_forest")
HORIZONS=(6,12,24)


def features(patients):
    x=[]
    for p in patients:
        v=p["baseline"]
        # Values that are UR/absent are np.nan, NOT a measured 0.
        values=[float(z) if s==1 else np.nan
                for z,s in zip(v["score"],v["status"])]
        mask=[float(s) for s in v["status"]]
        x.append((values,mask))
    scored=np.asarray([r[0] for r in x],dtype=float)
    statuses=np.asarray([r[1] for r in x],dtype=float)
    # BL only; mask value 2 denotes UR, not zero.
    # Fixed prespecified 27-item visible score sum, but missing implies 0
    # only within a numerical feature accompanied by visibility indicators.
    vis=np.nansum(scored[:,VIDX],axis=1).reshape(-1,1)
    full=np.column_stack((scored,statuses))
    return {"fit_mean":vis,"baseline_visible_sum":vis,
            "masked_ridge":full,"masked_forest":full}


def score_metrics(ground,preds,method,cohort,horizon):
    y=np.asarray([r["target_burden"] for r in ground],dtype=float)
    positive=np.asarray([r["target_positive"] for r in ground],dtype=int)
    p=np.maximum(0.,np.asarray(preds,dtype=float))
    if len(p)!=len(y) or not np.isfinite(p).all():
        raise AssertionError("Incomplete or nonfinite OOF predictions")
    # Risk transformation is predefined, monotone, not calibrated.
    risk=1.-np.exp(-p)
    npos=int(positive.sum())
    k=max(1,int(.20*len(y)+.5))
    ordering=sorted(range(len(y)),key=lambda i:(-risk[i],i))
    caught=sum(positive[i] for i in ordering[:k])
    return {
        "cohort":cohort,"horizon_months":horizon,"method":method,
        "patients":len(y),"positive":npos if npos>=10 else "<10",
        "quota20":k,"captured":caught if caught>=10 else "<10",
        "recall20":caught/npos if caught>=10 and npos>=10 else "",
        "MAE_six_sum":float(np.mean(np.abs(p-y))),
        "RMSE_six_sum":float(np.sqrt(np.mean((p-y)**2))),
        "AUROC_nonzero":auc(risk.tolist(),positive.tolist()),
        "risk_brier_uncalibrated":float(np.mean((risk-positive)**2)),
        "real_baseline_only":True,
    }


def fold_predict(records,folds):
    labels=np.asarray([r["target_positive"] for r in records])
    class_size=np.bincount(labels,minlength=2)
    if min(class_size)<folds:
        return None,"insufficient_class_count_for_stratified_CV"
    dataset=features(records)
    target=np.asarray([r["target_burden"] for r in records],dtype=float)
    split=StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED)
    all_pred={method:np.full(len(records),np.nan) for method in METHODS}
    sizes=[]
    for fold,(train,test) in enumerate(split.split(np.zeros(len(records)),labels)):
        if set(train)&set(test):raise AssertionError("CV leakage")
        if len(test)<5:raise AssertionError("Insufficient fold size")
        base=target[train].mean()
        all_pred["fit_mean"][test]=base
        models={
            "baseline_visible_sum":make_pipeline(
                SimpleImputer(strategy="median",add_indicator=True),
                StandardScaler(),Ridge(alpha=10.)),
            "masked_ridge":make_pipeline(
                SimpleImputer(strategy="constant",fill_value=-1.),
                StandardScaler(),Ridge(alpha=10.)),
            "masked_forest":make_pipeline(
                SimpleImputer(strategy="constant",fill_value=-1.),
                RandomForestRegressor(
                    n_estimators=200,min_samples_leaf=8,max_features=.6,
                    random_state=SEED+fold,n_jobs=2)),
        }
        for method,model in models.items():
            model.fit(dataset[method][train],target[train])
            all_pred[method][test]=model.predict(dataset[method][test])
        sizes.append({"fold":fold,"train":len(train),"heldout":len(test),
                      "train_positive":int(labels[train].sum()),
                      "heldout_positive":int(labels[test].sum())})
    return all_pred,sizes


def paired_mae_ci(records,preds,left="baseline_visible_sum",
                  right="masked_ridge",n_boot=600,seed=SEED):
    target=np.asarray([r["target_burden"] for r in records])
    difference=(np.abs(np.maximum(0.,preds[right])-target)-
                np.abs(np.maximum(0.,preds[left])-target))
    rng=random.Random(seed)
    samples=sorted(float(np.mean(
        difference[[rng.randrange(len(records)) for _ in records]]))
        for _ in range(n_boot))
    return {"MAE_right_minus_left":float(difference.mean()),
            "bootstrap_CI_low":samples[int(.025*(n_boot-1))],
            "bootstrap_CI_high":samples[int(.975*(n_boot-1))],
            "bootstrap_reps":n_boot,
            "comparison":right+"_minus_"+left,
            "disclaimer":"No model refit or multiplicity correction"}


def evaluate(patients,horizons=(6,12,24),folds=5,bootstrap=600):
    metrics=[];contrasts=[];folds_log=[];capacity=[]
    for group in GROUPS:
        selected=[p for p in patients if p["cohort"]==group]
        for horizon in horizons:
            records,_=future_site_pairs(selected,horizon)
            n=len(records);npos=sum(p["target_positive"] for p in records)
            reason="ok"
            if n<max(40,folds*8):
                reason="insufficient_samples"
            elif min(npos,n-npos)<max(10,folds):
                reason="insufficient_positive_or_negative_labels"
            capacity.append({"cohort":group,"horizon":horizon,
                             "eligible":n if n>=10 else "<10",
                             "positive":npos if npos>=10 else "<10",
                             "model_status":reason})
            if reason!="ok":continue
            # No patient from another cohort participates in this fold's fit.
            oof,detail=fold_predict(records,folds)
            if oof is None:
                capacity[-1]["model_status"]=detail;continue
            for entry in detail:
                folds_log.append({"cohort":group,"horizon":horizon,**entry})
            for name in METHODS:
                metrics.append(score_metrics(records,oof[name],name,group,horizon))
            contrasts.append({"cohort":group,"horizon_months":horizon,
                              **paired_mae_ci(records,oof,n_boot=bootstrap)})
    return metrics,contrasts,folds_log,capacity


def save_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--horizon",type=int,choices=HORIZONS,required=True)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--bootstrap",type=int,default=600)
    a=p.parse_args()
    if a.folds<3 or a.bootstrap<100:raise ValueError("Invalid full run")
    out=a.report_dir.resolve();root=a.data_root.resolve()
    if out==root or root in out.parents or out==a.manifest_db.resolve().parent:
        raise ValueError("Private source cannot be an output destination")
    started=time.perf_counter()
    patients,source=load_cohorts(root,a.manifest_db)
    metrics,paired,foldlog,capacity=evaluate(
        patients,(a.horizon,),a.folds,a.bootstrap)
    out.mkdir(parents=True,exist_ok=False)
    save_csv(out/"oof_metrics.csv",metrics)
    save_csv(out/"paired_MAE_bootstrap.csv",paired)
    save_csv(out/"fold_sizes.csv",foldlog)
    save_csv(out/"eligibility.csv",capacity)
    report={"status":"Phase62 exploratory CPU multicohort strict BL to verified future V04-style six-item",
            "horizon":a.horizon,"cohorts":list(GROUPS),
            "methods":list(METHODS),"folds":a.folds,
            "eligible_tasks":sum(r["model_status"]=="ok" for r in capacity),
            "model_tasks_skipped":sum(r["model_status"]!="ok" for r in capacity),
            "elapsed_seconds":round(time.perf_counter()-started,2),
            "source_table":source["source_motor_table"],
            "limitations":[
                "Prodromal uses only original frozen TRAIN; original VAL/TEST not queried",
                "New PD/HC development cohorts never mixed with Prodromal",
                "Retrospective horizon chosen as 6/12/24 months; labels are future actually scored six-item burden",
                "Strict same-medication-state followup, not a causal effect",
                "All inputs drawn from patient BL only, no target date, target event or label in features",
                "Missing and 101/UR not treated as measured zero; mask retained for masked models",
                "Risk mapping 1-exp(-predicted burden) uncalibrated",
                "Paired patient bootstrap holds trained models fixed",
                "Patients without complete future label excluded; possible followup selection bias",
                "Only aggregate CSV/JSON export; no patient features or identifiers exported",
                "Not yet full Qwen Doctor/RSI nor causal WM simulation"
            ],"created_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"complete","horizon":a.horizon,
                      "task_metrics":len(metrics),"capacity":capacity,
                      "output":str(out)},indent=2))


if __name__=="__main__":main()
