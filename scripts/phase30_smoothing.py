#!/usr/bin/env python3
"""Phase30: PPMI TRAIN-only 12-to-24 Motor smoothing vs ridge."""
import argparse,csv,json,sqlite3,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import fixed_folds,metrics
from scripts.phase26_multimodal_oof import fit_multi,predict_multi
from scripts.phase29_sequential import load,model_rows,tune,pair_diff

WEIGHTS=(0.,.125,.25,.375,.5,.625,.75,.875,1.)
MODELS=("last_value","midpoint","weighted_blend","ridge_last","ridge_bl_last")
def choose_weight(train):
    if len(train)<10:raise ValueError("Insufficient training rows")
    return min(WEIGHTS,key=lambda w:(mean(
        (w*r["bl"]+(1-w)*r["last"]-r["target"])**2 for r in train),
        abs(w-.5),w))

def selection_counts(manifest):
    db=sqlite3.connect("file:"+str(manifest)+"?mode=ro",uri=True)
    try:
        rows=db.execute("""SELECT t12.baseline_motor,t12.target_motor,
          CASE WHEN t24.pid IS NULL THEN 0 ELSE 1 END
          FROM patients p
          JOIN targets t12 ON t12.pid=p.pid AND t12.horizon=12
          LEFT JOIN targets t24 ON t24.pid=p.pid AND t24.horizon=24
          WHERE p.split='train'""").fetchall()
    finally:db.close()
    groups={0:[],1:[]}
    for bl,last,flag in rows:
        if bl is None or last is None:raise ValueError("Missing 12m score")
        groups[flag].append((bl,last))
    return [{"has_24m":bool(key),"n":len(group),
             "mean_BL":round(mean(x for x,y in group),3),
             "mean_12m":round(mean(y for x,y in group),3)}
            for key,group in groups.items() if group]

def write(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run(manifest,report,folds=5,seed=20261018,bootstraps=500):
    manifest,report=Path(manifest).resolve(),Path(report).resolve()
    if not manifest.is_file() or manifest==report or folds<2 or bootstraps<100:
        raise ValueError("Invalid input or arguments")
    data=load(manifest)
    outer=fixed_folds([(r["pid"],r["bl"],r["target"]) for r in data],folds,seed)
    pred={m:{} for m in MODELS};weights=[];foldscores=[]
    for held in range(folds):
        training=[r for r in data if outer[r["pid"]]!=held]
        test=[r for r in data if outer[r["pid"]]==held]
        weight=choose_weight(training)
        fitted={}
        for m in ("ridge_last","ridge_bl_last"):
            penalty=tune(training,m,held,seed)
            rows,names=model_rows(training,m)
            fitted[m]=fit_multi(rows,names,penalty)
        weights.append({"fold":held,"n_fit":len(training),
                        "n_test":len(test),"baseline_weight":weight})
        for r in test:
            prediction={"last_value":r["last"],
                "midpoint":(r["bl"]+r["last"])/2,
                "weighted_blend":weight*r["bl"]+(1-weight)*r["last"]}
            for m in fitted:
                prediction[m]=predict_multi(
                    fitted[m],r["last"],{"baseline":r["bl"],"gap12":r["gap12"]})
            for m,value in prediction.items():
                pred[m][r["pid"]]=(r["target"],value)
        for m in MODELS:
            stat=metrics([pred[m][r["pid"]] for r in test])
            foldscores.append({"fold":held,"model":m,
                               "n":stat["n"],"mae":round(stat["mae"],5),
                               "rmse":round(stat["rmse"],5)})
    if any(len(v)!=len(data) for v in pred.values()):
        raise ValueError("Incomplete train out-of-fold predictions")
    totals=[{"model":m,**{k:round(v,5) if isinstance(v,float) else v
             for k,v in metrics(list(pred[m].values())).items()}}
            for m in MODELS]
    comparisons=[]
    for reference in ("last_value","midpoint","weighted_blend"):
        for model in MODELS:
            if model==reference:continue
            comparisons.extend({"reference":reference,"model":model,**row}
                for row in pair_diff(pred[reference],pred[model],bootstraps,
                                    seed+len(reference)*31+len(model)))
    qc=selection_counts(manifest)
    report.mkdir(parents=True,exist_ok=True)
    write(report/"model_metrics.csv",["model","n","mae","rmse",
         "mean_signed_error","observed_mean_target"],totals)
    write(report/"paired_differences.csv",["reference","model","metric",
          "difference","lower_95","upper_95"],comparisons)
    write(report/"chosen_weights.csv",["fold","n_fit","n_test","baseline_weight"],weights)
    write(report/"fold_metrics.csv",["fold","model","n","mae","rmse"],foldscores)
    write(report/"followup_selection.csv",["has_24m","n","mean_BL","mean_12m"],qc)
    meta={"status":"TRAIN_ONLY_exploratory_smoothing_vs_transition",
          "complete_sequences":len(data),"candidate_weights":list(WEIGHTS),
          "outer_folds":folds,"seed":seed,
          "notes":["TRAIN only; no validation/test data queried.",
          "Weighted blend selected on fitting fold only.",
          "12-to-24 target availability audited but not a predictor.",
          "UNSPECIFIED medication code does not mean controlled treatment.",
          "No patient-level records or prediction outputs.",
          "Smoothing is not evidence of learned disease dynamics."],
          "generated_utc":datetime.now(timezone.utc).isoformat()}
    (report/"summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase30_smoothing"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=500)
    a=p.parse_args()
    print(json.dumps(run(a.manifest_db,a.report_dir,a.folds,a.seed,a.bootstraps),indent=2))
if __name__=="__main__":main()
