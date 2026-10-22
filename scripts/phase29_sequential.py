#!/usr/bin/env python3
"""Phase29 exploratory sequential Motor prediction, frozen PPMI TRAIN only.
Only baseline and 12-month measurements enter predictors. Never upload DB.
"""
import argparse,csv,json,math,sqlite3,sys,random
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import fixed_folds,metrics
from scripts.phase26_multimodal_oof import fit_multi,predict_multi

RIDGE={"ridge_last":("last",()),"ridge_bl_last":("last",("baseline",)),
       "ridge_bl_last_gap":("last",("baseline","gap12"))}
MODELS=("last_value","baseline_value","linear_extrapolation")+tuple(RIDGE)
LAMBDA=(.01,.1,1.,10.,100.)

def load(path):
    db=sqlite3.connect("file:"+str(path)+"?mode=ro",uri=True)
    try:
        rows=db.execute("""SELECT p.pid,p.baseline_month,p.baseline_motor,
          a.baseline_month,a.target_month,a.months_apart,a.baseline_motor,
          a.target_motor,a.delta_motor,b.baseline_month,b.target_month,
          b.months_apart,b.baseline_motor,b.target_motor,b.delta_motor
          FROM patients p
          JOIN targets a ON a.pid=p.pid AND a.horizon=12
          JOIN targets b ON b.pid=p.pid AND b.horizon=24
          WHERE p.split='train' ORDER BY p.pid""").fetchall()
    finally:db.close()
    result=[]
    for v in rows:
        pid,anchor,bl,a0,a1,g1,ab,y12,d1,b0,b1,g2,bb,y24,d2=v
        if any(x is None for x in v):raise ValueError("Missing longitudinal field")
        if not (anchor==a0==b0 and a1-anchor==g1 and b1-anchor==g2
                and 9<=g1<=15 and 20<=g2<=28 and b1>a1
                and abs(bl-ab)<1e-6 and abs(bl-bb)<1e-6
                and abs(y12-ab-d1)<1e-6 and abs(y24-bb-d2)<1e-6
                and all(math.isfinite(float(x)) and 0<=x<=132
                        for x in (bl,y12,y24))):
            raise ValueError("Inconsistent target provenance")
        result.append({"pid":pid,"bl":float(bl),"last":float(y12),
                       "gap12":int(g1),"target":float(y24),
                       "future_gap":int(b1-a1)})
    if len({x["pid"] for x in result})!=len(result):
        raise ValueError("Duplicate patients")
    return result

def matrix(rows,model):
    main,names=RIDGE[model]
    return [(r["pid"],r["bl"] if main=="baseline" else r["last"],
             r["target"],{"baseline":r["bl"],"gap12":r["gap12"]})
            for r in rows],names

def tune(rows,model,seed,outer):
    fold=fixed_folds([(r["pid"],r["bl"],r["target"]) for r in rows],3,
                     seed+1009*(outer+1))
    data,names=matrix(rows,model)
    scores={}
    for penalty in LAMBDA:
        squared=0
        for held in range(3):
            fit=[r for r in data if fold[r[0]]!=held]
            test=[r for r in data if fold[r[0]]==held]
            if not fit or not test:raise ValueError("Empty inner fold")
            fitted=fit_multi(fit,names,penalty)
            squared+=sum((predict_multi(fitted,x,extra)-y)**2
                         for pid,x,y,extra in test)
        scores[penalty]=math.sqrt(squared/len(rows))
    return min(LAMBDA,key=lambda p:(scores[p],-p))

def predict_simple(model,r):
    if model=="last_value":return r["last"]
    if model=="baseline_value":return r["bl"]
    if model=="linear_extrapolation":
        return max(0.,min(132.,r["last"]+(r["last"]-r["bl"])*
                              (24-r["gap12"])/r["gap12"]))
    raise ValueError("Unknown baseline")

def pair_diff(a,b,reps,seed):
    if a.keys()!=b.keys():raise ValueError("Unpaired patients")
    items=[(a[k][0],a[k][1],b[k][1]) for k in sorted(a)]
    if any(a[k][0]!=b[k][0] for k in a):raise ValueError("Mismatched outcomes")
    def delta(samples,metric):
        if metric=="MAE":
            return mean(abs(y-p1)-abs(y-p0) for y,p0,p1 in samples)
        return math.sqrt(mean((y-p1)**2 for y,p0,p1 in samples))-math.sqrt(
            mean((y-p0)**2 for y,p0,p1 in samples))
    rng=random.Random(seed);out=[]
    for metric in ("MAE","RMSE"):
        values=[]
        for i in range(reps):
            sample=[items[rng.randrange(len(items))] for j in items]
            values.append(delta(sample,metric))
        values.sort()
        out.append({"metric":metric,"difference":round(delta(items,metric),5),
                    "lower_95":round(values[int(.025*(reps-1))],5),
                    "upper_95":round(values[int(.975*(reps-1))],5)})
    return out

def save(path,headers,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)

def run(manifest,out,folds=5,seed=20261018,bootstrap=500):
    manifest,out=Path(manifest).resolve(),Path(out).resolve()
    if not manifest.is_file() or out==manifest or folds<2 or bootstrap<100:
        raise ValueError("Invalid source or parameters")
    rows=load(manifest)
    if len(rows)<folds*10:raise ValueError("Too few TRAIN 3-point sequences")
    outer=fixed_folds([(r["pid"],r["bl"],r["target"]) for r in rows],folds,seed)
    predictions={m:{} for m in MODELS};penalties=[]
    for fold in range(folds):
        fit=[r for r in rows if outer[r["pid"]]!=fold]
        held=[r for r in rows if outer[r["pid"]]==fold]
        learned={}
        for model in RIDGE:
            penalty=tune(fit,model,seed,fold)
            data,names=matrix(fit,model)
            learned[model]=fit_multi(data,names,penalty)
            penalties.append({"fold":fold,"model":model,"penalty":penalty})
        for r in held:
            pid=r["pid"];y=r["target"]
            for m in MODELS:
                if m in RIDGE:
                    kind,names=RIDGE[m]
                    x=r["bl"] if kind=="baseline" else r["last"]
                    p=predict_multi(learned[m],x,{"baseline":r["bl"],"gap12":r["gap12"]})
                else:p=predict_simple(m,r)
                predictions[m][pid]=(y,p)
    if any(len(v)!=len(rows) for v in predictions.values()):
        raise ValueError("Missing held-out predictions")
    metric_rows=[dict(model=m,**{k:round(v,5) if isinstance(v,float) else v
                  for k,v in metrics(list(predictions[m].values())).items()})
                  for m in MODELS]
    paired=[]
    for ref in ("last_value","ridge_last"):
        for m in MODELS:
            if m==ref:continue
            paired.extend(dict(model=m,reference=ref,**d) for d in pair_diff(
                predictions[ref],predictions[m],bootstrap,seed+len(m)*17+len(ref)))
    gap_counts=Counter(r["future_gap"] for r in rows)
    out.mkdir(parents=True,exist_ok=True)
    save(out/"model_metrics.csv",["model","n","mae","rmse",
         "mean_signed_error","observed_mean_target"],metric_rows)
    save(out/"paired_differences.csv",
         ["model","reference","metric","difference","lower_95","upper_95"],paired)
    save(out/"penalty_by_fold.csv",["fold","model","penalty"],penalties)
    save(out/"actual_future_gap_audit.csv",["gap_months","patients"],[
         {"gap_months":gap,"patients":n if n>=10 else "<10"}
         for gap,n in sorted(gap_counts.items())])
    summary={"status":"TRAIN_ONLY_12_to_24_exploratory","train_sequences":len(rows),
             "folds":folds,"seed":seed,"models":list(MODELS),
             "future_gap_as_feature":False,
             "warnings":["Only frozen TRAIN patients queried",
             "Complete-case selection bias; unknown medication state",
             "12 and 24 months are visit windows, not exact appointment dates",
             "No patient level data exported; not clinical progression validation"],
             "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase29_sequential"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=500)
    a=p.parse_args()
    print(json.dumps(run(a.manifest_db,a.report_dir,a.folds,a.seed,a.bootstraps),indent=2))
if __name__=="__main__":main()
