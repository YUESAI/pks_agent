"""Phase58 ten matched decision policies, frozen Prodromal TRAIN only.

Exactly 10 fixed strategies; all share Phase56 five patient folds and
strict R01->later V04 pairs, scored in one OOF pass. No TEST/VAL.
No patient-level exports. Models fit outer TRAIN only; neural epoch and
Markov regularization select on inner development. Other choices fixed
in advance. Retrospective V04 MONTH is given to temporal predictors.
"""
from __future__ import annotations
import argparse,csv,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from scripts.phase36_masked_site_feasibility import ALL,VISIBLE,HIDDEN
from scripts.phase54_masked_sequences import load_train
from scripts.phase54_evaluate import split_patients,select_dev,train_model
from scripts.phase56_remote_to_site_wm import (
    find_pairs,wm_predict,markov_predict,choose_markov_for_site,
    outcomes,HIDX,VIDX
)
from scripts.phase55_markov_baseline import ItemMarkov
from scripts.phase58_paired_metrics import compute_matrix

METHODS=(
    "visible27_sum",
    "visible27_nonzero",
    "visible27_ridge",
    "visible27_logistic",
    "visible27_naive_bayes",
    "markov_tuned",
    "latent_wm_64",
    "latent_wm_128",
    "wm_markov_blend",
    "visible_wm_blend",
)

def feature(pair):
    r=pair[1]
    if not all(r["status"][j]==1 for j in VIDX):
        raise ValueError("Unmeasured R01 visible item")
    return np.array([int(r["score"][j]) for j in VIDX],dtype=np.float64)

def label(pair):
    site=pair[2]
    if not all(site["status"][j]==1 for j in HIDX):
        raise ValueError("Future six label unobserved")
    return int(any(site["score"][j]>0 for j in HIDX))

def fit_features(train_pairs):
    if len(train_pairs)<20:raise ValueError("Insufficient TRAIN paired labels")
    x=np.stack([feature(p) for p in train_pairs.values()])
    y=np.array([label(p) for p in train_pairs.values()],dtype=np.float64)
    if not 0<y.sum()<len(y):raise ValueError("Need both classes")
    mu=x.mean(0)
    sd=np.maximum(x.std(0),.5)
    z=np.column_stack((np.ones(len(x)),(x-mu)/sd))
    identity=np.eye(z.shape[1]);identity[0,0]=0.
    # Fixed ridge penalty (no development/outer optimization).
    beta=np.linalg.solve(z.T@z+20.*identity,z.T@y)
    # Fixed L2 logistic penalty and iteration count for audit reproducibility.
    gamma=np.zeros(z.shape[1])
    for iteration in range(200):
        logits=np.clip(z@gamma,-20.,20.)
        p=1/(1+np.exp(-logits))
        grad=z.T@(p-y)/len(y)+.08*(identity@gamma)
        curvature=(p*(1-p))[:,None]*z
        h=z.T@curvature/len(y)+.08*identity+1e-5*np.eye(z.shape[1])
        step=np.linalg.solve(h,grad)
        gamma-=step
        if np.linalg.norm(step)<1e-7:break
    # Dirichlet category likelihood p(item rating | outcome)
    counts=np.ones((2,len(VIDX),5),dtype=np.float64)
    for row,goal in zip(x.astype(int),y.astype(int)):
        counts[goal,np.arange(len(VIDX)),row]+=1
    probs=counts/counts.sum(-1,keepdims=True)
    prior=float((y.sum()+1)/(len(y)+2))
    return (mu,sd,beta,gamma,probs,prior)

def fitted_scores(model,pairs):
    mu,sd,beta,gamma,probs,prior=model
    result={n:{} for n in ("visible27_ridge","visible27_logistic",
                            "visible27_naive_bayes")}
    for pid,pair in pairs.items():
        row=feature(pair)
        z=np.r_[1.,(row-mu)/sd]
        ridge=float(z@beta)
        logistic=float(1/(1+np.exp(-np.clip(z@gamma,-20,20))))
        # Use log-likelihood ratio to avoid underflow.
        integer=row.astype(int)
        logodds=np.log(prior/(1-prior))+sum(
            np.log(probs[1,np.arange(len(VIDX)),integer])-
            np.log(probs[0,np.arange(len(VIDX)),integer]))
        nb=float(1/(1+np.exp(-np.clip(logodds,-30,30))))
        result["visible27_ridge"][pid]=ridge
        result["visible27_logistic"][pid]=logistic
        result["visible27_naive_bayes"][pid]=nb
    return result

def risk_from_items(items):
    return 1.-float(np.prod(np.asarray(items)[:,0]))

def fit_wm_on_fold(inner,dev,outer,dev_pairs,width,epochs,fold,device,seed):
    best=None
    for e in sorted({q for q in (2,5,10,epochs) if q<=epochs}):
        start=time.perf_counter()
        candidate=train_model(inner,width,e,seed+fold*103+width*11,device)
        pred=wm_predict(candidate,dev_pairs,device)
        nll=np.mean([v["nll"] for v in outcomes(dev_pairs,pred).values()])
        if best is None or (nll,e)<(best[0],best[1]):best=(nll,e)
    final=train_model(outer,width,best[1],seed+fold*103+width*11,device)
    return final,{"chosen_epoch":best[1],"inner_dev_six_nll":round(float(best[0]),6),
                  "parameter_count":sum(p.numel() for p in final.parameters())}

def run(patients,output,epochs=25,folds=5,seed=20261018,
        bootstrap=500,device_name="cpu"):
    if folds!=5 or not 2<=epochs<=100:raise ValueError("Only fixed five-fold protocol")
    device=torch.device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    pairs,quality=find_pairs(patients)
    if len(pairs)<50:raise ValueError("Insufficient strict R01->V04 pairs")
    assignment=split_patients(patients,folds,seed)
    predictions={name:{} for name in METHODS}
    fold_report=[]
    overall_start=time.perf_counter()
    for fold in range(folds):
        started=time.perf_counter()
        outer=[p for p in patients if assignment[p["pid"]]!=fold]
        held=[p for p in patients if assignment[p["pid"]]==fold]
        inner,dev=select_dev(outer,seed+fold)
        training_pairs={p["pid"]:pairs[p["pid"]] for p in outer
                        if p["pid"] in pairs}
        development={p["pid"]:pairs[p["pid"]] for p in dev
                     if p["pid"] in pairs}
        testing={p["pid"]:pairs[p["pid"]] for p in held
                 if p["pid"] in pairs}
        if min(len(training_pairs),len(development),len(testing))<5:
            raise ValueError("Too few paired R01 people in fold")
        features=fit_features(training_pairs)
        fitted=fitted_scores(features,testing)
        for pid,pair in testing.items():
            x=feature(pair)
            predictions["visible27_sum"][pid]=float(x.sum())
            predictions["visible27_nonzero"][pid]=float((x>0).sum())
        for name,scores in fitted.items():predictions[name].update(scores)

        mode,alpha,markov_dev=choose_markov_for_site(inner,dev,development)
        markov=ItemMarkov(alpha,mode).fit(outer)
        markov_probs=markov_predict(markov,testing)
        for pid,probs in markov_probs.items():
            predictions["markov_tuned"][pid]=risk_from_items(probs)

        trained={}
        for width in (64,128):
            model,log=fit_wm_on_fold(inner,dev,outer,development,width,epochs,
                                     fold,device,seed)
            probs=wm_predict(model,testing,device)
            for pid,p in probs.items():
                predictions[f"latent_wm_{width}"][pid]=risk_from_items(p)
            trained[width]=log
            del model
        # Fixed, predeclared convex mixtures; no post-hoc weight tuning.
        for pid in testing:
            a=predictions["latent_wm_64"][pid]
            b=predictions["markov_tuned"][pid]
            c=predictions["visible27_logistic"][pid]
            predictions["wm_markov_blend"][pid]=.5*a+.5*b
            predictions["visible_wm_blend"][pid]=.5*a+.5*c
        fold_report.append({
            "outer_fold":fold,"outer_fit_people":len(outer),
            "inner_fit_people":len(inner),"inner_dev_people":len(dev),
            "outer_heldout_R01_pairs":len(testing),
            "train_labeled_R01_pairs":len(training_pairs),
            "dev_labeled_R01_pairs":len(development),
            "wm64_epochs":trained[64]["chosen_epoch"],
            "wm128_epochs":trained[128]["chosen_epoch"],
            "wm64_parameters":trained[64]["parameter_count"],
            "wm128_parameters":trained[128]["parameter_count"],
            "markov_mode":mode,"markov_alpha":alpha,
            "fold_elapsed_seconds":round(time.perf_counter()-started,2)})
    if any(set(output)!=set(pairs) for output in predictions.values()):
        raise AssertionError("Incomplete OOF or patient cohort mismatch")
    labels={pid:label(pair) for pid,pair in pairs.items()}
    ids=sorted(pairs)
    stats,paired,overlap=compute_matrix(
        predictions,labels,ids,seed,nboot=bootstrap)
    output=Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    def csv_out(name,rows):
        with (output/name).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    csv_out("ten_strategy_metrics.csv",stats)
    csv_out("paired_bootstrap.csv",paired)
    csv_out("topk_overlap.csv",overlap)
    csv_out("fold_training_compute.csv",fold_report)
    summary={
        "status":"Phase58_exploratory_TRAIN_only_ten_matched_policies",
        "n_strategies":len(METHODS),"strategies":list(METHODS),
        "R01_to_verified_later_V04_patients":len(pairs),
        "positive_future_hidden_six":sum(labels.values()),
        "outer_folds":folds,"bootstrap_draws":bootstrap,
        "total_elapsed_seconds":round(time.perf_counter()-overall_start,2),
        "device":str(device),"seed":seed,
        "caveats":[
            "TRAIN-only repeated exploration; original final TEST consumed.",
            "Future recorded V04 month is known as retrospective forecast horizon.",
            "All strategies have identical patient cohort and top-20%-quota.",
            "27-item visible-score methods are full R01 available measurements.",
            "WM belief dynamics are lightweight, not full Dreamer 4 pretraining.",
            "No Qwen, VLM, Doctor action cost or RSI in this benchmark.",
            "Mixtures are fixed 50/50; not learned on evaluated outer patients.",
            "Top-k recall and AUROC bootstrap are paired on patients; no retraining per bootstrap.",
            "No identifiers, scores, selected patient IDs or individual predictions exported."
        ],"generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--bootstrap",type=int,default=500)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    data,metadata=load_train(a.data_root,a.manifest_db)
    print(json.dumps(run(data,a.report_dir,epochs=a.epochs,
                         bootstrap=a.bootstrap,device_name=a.device),indent=2))
    print("Share aggregate reports ONLY. No patient-level clinical data.")
if __name__=="__main__":main()
