#!/usr/bin/env python3
"""Phase24 PPMI Prodromal Motor regression; TRAIN-only out-of-fold evaluation.

Read Phase23 private SQLite manifest read-only. Only TRAIN patients' 12-month
target scores are queried. All reports are aggregate: no IDs, individual
scores, predictions or subject dates. UNSPECIFIED drug state is not proof
of matching treatment. This is exploratory, NOT clinical prognosis.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sqlite3
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean,median

MODELS=("persistence","mean_delta","median_delta","ridge_delta")
BUCKETS=("0_to_3","4_to_7","8_to_15","16_plus")

def severity_bin(x):
    if x<4:return "0_to_3"
    if x<8:return "4_to_7"
    if x<16:return "8_to_15"
    return "16_plus"

def load_train_only(path):
    """Protected SQL: NO val/test target measurement is selected."""
    db=sqlite3.connect("file:"+str(path.resolve())+"?mode=ro",uri=True)
    try:
        cols={r[1] for r in db.execute("PRAGMA table_info(targets)")}
        if not {"pid","horizon","baseline_motor","target_motor","months_apart"}.issubset(cols):
            raise ValueError("Requires Phase23 longitudinal manifest schema")
        registry=dict(db.execute("SELECT split,COUNT(*) FROM patients GROUP BY split").fetchall())
        counts=dict(db.execute(
            "SELECT p.split,COUNT(*) FROM patients p "
            "JOIN targets t ON t.pid=p.pid AND t.horizon=12 GROUP BY p.split").fetchall())
        rows=db.execute(
            "SELECT p.pid,t.baseline_motor,t.target_motor,t.months_apart "
            "FROM patients p JOIN targets t ON t.pid=p.pid "
            "WHERE p.split='train' AND t.horizon=12 ORDER BY p.pid").fetchall()
        duplicates=db.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT t.pid) "
            "FROM patients p JOIN targets t ON t.pid=p.pid "
            "WHERE p.split='train' AND t.horizon=12").fetchone()[0]
        if duplicates:raise ValueError("Duplicate patient targets")
    finally:db.close()
    clean=[]
    for pid,x,y,gap in rows:
        if not pid or x is None or y is None or gap is None:
            raise ValueError("Incomplete training pair")
        x,y=float(x),float(y)
        if not (math.isfinite(x) and math.isfinite(y) and
                0<=x<=132 and 0<=y<=132 and 9<=gap<=15):
            raise ValueError("Invalid score or 12 +/- 3 month target gap")
        clean.append((pid,x,y))
    if not clean or len(clean)!=counts.get("train",0):
        raise ValueError("Missing or inconsistent TRAIN pairs")
    return clean,registry,counts

def fixed_folds(rows,k,seed):
    """Fold assignment uses only patient identity and BL motor severity."""
    strata=defaultdict(list)
    for pid,x,y in rows:strata[severity_bin(x)].append(pid)
    assignment={}
    for group in BUCKETS:
        ordered=sorted(strata[group],key=lambda pid:hashlib.sha256(
            f"{seed}|prodromal|motor12|{group}|{pid}".encode()).digest())
        for i,pid in enumerate(ordered):assignment[pid]=i%k
    if len(assignment)!=len(rows) or any(
            sum(f==fold for f in assignment.values())==0 for fold in range(k)):
        raise ValueError("Invalid or empty cross-validation fold")
    return assignment

def fit(rows,ridge_penalty_per_patient=1.0):
    """Model 12-month score change using ONLY baseline values + fit targets."""
    if len(rows)<3:raise ValueError("Not enough fitting pairs")
    x=[b for pid,b,y in rows]
    changes=[y-b for pid,b,y in rows]
    a=mean(changes);center=mean(x)
    cov=sum((v-center)*(d-a) for v,d in zip(x,changes))
    variance=sum((v-center)**2 for v in x)
    return {"mean_delta":a,"median_delta":median(changes),
            "ridge_delta_slope":cov/(variance+ridge_penalty_per_patient*len(rows)),
            "mean_baseline":center}

def predict(model,params,x):
    if model=="persistence":v=x
    elif model=="mean_delta":v=x+params["mean_delta"]
    elif model=="median_delta":v=x+params["median_delta"]
    elif model=="ridge_delta":
        v=x+params["mean_delta"]+params["ridge_delta_slope"]*(x-params["mean_baseline"])
    else:raise ValueError("Unknown baseline model")
    return max(0.,min(132.,v))

def metrics(rows):
    if not rows:raise ValueError("Empty metric group")
    errors=[p-y for y,p in rows]
    return {"n":len(rows),"mae":mean(abs(e) for e in errors),
            "rmse":math.sqrt(mean(e*e for e in errors)),
            "mean_signed_error":mean(errors),
            "observed_mean_target":mean(y for y,p in rows)}

def paired_bootstrap(base,other,iterations,seed):
    if base.keys()!=other.keys():raise ValueError("Unpaired patient sets")
    ids=sorted(base)
    if len(ids)<10 or iterations<100:raise ValueError("Insufficient bootstrap size")
    pairs=[]
    for pid in ids:
        y0,p0=base[pid];y1,p1=other[pid]
        if y0!=y1:raise ValueError("Unmatched target value")
        pairs.append((y0,p0,p1))
    delta=mean(abs(p1-y)-abs(p0-y) for y,p0,p1 in pairs)
    rng=random.Random(seed);draws=[]
    for i in range(iterations):
        sample=[pairs[rng.randrange(len(pairs))] for j in range(len(pairs))]
        draws.append(mean(abs(p1-y)-abs(p0-y) for y,p0,p1 in sample))
    draws.sort()
    def quantile(f):
        pos=(len(draws)-1)*f;lo=int(pos);hi=min(lo+1,len(draws)-1)
        return draws[lo]*(hi-pos)+draws[hi]*(pos-lo)
    return delta,quantile(.025),quantile(.975)

def export(path,header,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)

def audit(manifest,output,folds=5,seed=20261018,bootstraps=500,min_count=10):
    manifest=Path(manifest).expanduser().resolve()
    output=Path(output).expanduser().resolve()
    if not manifest.is_file() or output==manifest or folds<2 or bootstraps<100 or min_count<10:
        raise ValueError("Invalid private manifest, report path, folds or privacy parameter")
    records,registry,counts=load_train_only(manifest)
    if len(records)<folds*5:raise ValueError("Not enough TRAIN pairs for CV")
    assigned=fixed_folds(records,folds,seed)
    predictions={model:{} for model in MODELS}
    fold_rows=[]
    for held in range(folds):
        fitting=[r for r in records if assigned[r[0]]!=held]
        evaluated=[r for r in records if assigned[r[0]]==held]
        params=fit(fitting)
        fold_rows.append(dict(fold=held,fit_patients=len(fitting),
                              evaluated_train_patients=len(evaluated)))
        for pid,x,y in evaluated:
            for model in MODELS:
                if pid in predictions[model]:raise ValueError("Repeated patient in held-out folds")
                predictions[model][pid]=(y,predict(model,params,x))
    if any(len(predictions[m])!=len(records) for m in MODELS):
        raise ValueError("Incomplete out-of-fold predictions")
    model_rows=[]
    for model,values in predictions.items():
        result=metrics(list(values.values()))
        model_rows.append(dict(model=model,evaluated_population="PPMI_prodromal_TRAIN_OOF",
                               **{k:round(v,5) if isinstance(v,float) else v
                                  for k,v in result.items()}))
    comparisons=[]
    for index,model in enumerate(MODELS):
        if model=="persistence":continue
        d,lo,hi=paired_bootstrap(predictions["persistence"],predictions[model],
                                  bootstraps,seed+index*100)
        comparisons.append(dict(comparison=model+"_minus_persistence",metric="MAE",
                                difference=round(d,5),lower_95=round(lo,5),
                                upper_95=round(hi,5),bootstrap_replicates=bootstraps))
    severity=[]
    baseline_by_pid={pid:x for pid,x,y in records}
    for group in BUCKETS:
        ids=[pid for pid,x in baseline_by_pid.items() if severity_bin(x)==group]
        if len(ids)<min_count:continue
        for model in MODELS:
            result=metrics([predictions[model][pid] for pid in ids])
            severity.append(dict(baseline_score_range=group,model=model,n=result["n"],
                                 mae=round(result["mae"],5),rmse=round(result["rmse"],5)))
    output.mkdir(parents=True,exist_ok=True)
    export(output/"train_oof_motor_metrics.csv",
           ["model","evaluated_population","n","mae","rmse","mean_signed_error",
            "observed_mean_target"],model_rows)
    export(output/"paired_vs_persistence.csv",
           ["comparison","metric","difference","lower_95","upper_95",
            "bootstrap_replicates"],comparisons)
    export(output/"severity_stratified_train_errors.csv",
           ["baseline_score_range","model","n","mae","rmse"],severity)
    export(output/"train_cv_fold_counts.csv",
           ["fold","fit_patients","evaluated_train_patients"],fold_rows)
    summary={
        "status":"TRAIN_ONLY_exploratory_longitudinal_regression_baselines",
        "task":"PPMI Prodromal 12 +/- 3 month NP3TOT prediction from baseline NP3TOT",
        "train_pair_count":len(records),"folds":folds,"seed":seed,
        "patient_registry_counts":registry,
        "pair_counts_by_split_for_integrity_only":counts,
        "baseline_models":list(MODELS),"bootstrap_replicates":bootstraps,
        "uses_future_actual_month_gap_as_feature":False,
        "limitations":[
            "Only TRAIN 12-month target scores are selected; validation/test remain unopened.",
            "Observed-target subset may be selected by disease severity or study retention.",
            "UNSPECIFIED does not guarantee matched treatment between visits.",
            "No patient IDs, raw motor scores, predictions or actual visit dates exported.",
            "This is exploratory prediction of recorded score, not clinical progression.",
            "Target gap is checked for eligibility only, never used for prediction.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase24_motor_baselines"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=500)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    print(json.dumps(audit(a.manifest_db,a.report_dir,a.folds,
                          a.seed,a.bootstraps,a.min_cell_count),
                     indent=2,ensure_ascii=False))
    print("Upload aggregate Phase24 reports only, never the private manifest.")

if __name__=="__main__":main()
