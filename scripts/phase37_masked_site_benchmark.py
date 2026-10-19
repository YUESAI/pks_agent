#!/usr/bin/env python3
"""Phase37: TRAIN-only same-visit masked-remote MDS-UPDRS III benchmark.

At a valid PPMI V04 exam, use ONLY the 27 remotely assessable Motor items
to predict the SUM of the six contemporaneous hidden rigidity/stability items.
Truth is the six actual V04 item scores at the SAME visit, never a later exam.
Require all 33 scores numeric 0..4 and sum(scores)==NP3TOT before inclusion.

Five-fold TRAIN-only outer CV, 3-fold inner ridge penalty selection.
Compare always-zero, fit-fold mean hidden sum, 27-item Ridge and a Ridge
using only the visible sum. No PPMI val/test patients are queried.
No identifiers, raw scores, patient-level forecasts or dates are exported.

This simulated masking benchmark does not establish real R01 deployment
performance; real remote versus onsite feature distributions can differ.
"""
from __future__ import annotations

import argparse,csv,json,math,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import fixed_folds,metrics
from scripts.phase26_multimodal_oof import fit_multi,predict_multi
from scripts.phase29_sequential import pair_diff
from scripts.phase36_masked_site_feasibility import (
    ALL,HIDDEN,VISIBLE,parse_rows,unique_by_patient
)
from scripts.phase31_motor_repetitions import train_anchors
from scripts.phase22_strict_motor_pairs import one_file,safe,save

MODELS=("hidden_zero","hidden_train_mean","ridge_visible_sum","ridge_visible_27")
RIDGE={
    "ridge_visible_sum":(),
    "ridge_visible_27":tuple(VISIBLE),
}
PENALTIES=(.01,.1,1.,10.,100.)
HIDDEN_MAX=24
INNER_FOLDS=3


def load_site_episodes(root,manifest):
    """Reuse Phase36 exact quality gate. R01 is never used as V04 supervision."""
    source=one_file(root/"PPMI","MDS-UPDRS_Part_III_")
    anchors=train_anchors(manifest)
    qc=Counter()
    selected=unique_by_patient(parse_rows(source,anchors,qc),qc)
    episodes=[]
    for (pid,event),(_,fields,total) in selected.items():
        if event!="V04":continue
        if total is None or any(label!="valid" for label,value in fields.values()):
            qc["V04_incomplete_or_unrated"]+=1
            continue
        values={k:value for k,(label,value) in fields.items()}
        if sum(values.values())!=total:
            qc["V04_total_mismatch"]+=1
            continue
        visible=sum(values[name] for name in VISIBLE)
        hidden=sum(values[name] for name in HIDDEN)
        if visible+hidden!=total or not 0<=hidden<=HIDDEN_MAX:
            raise ValueError("Invalid hidden-item supervision")
        episodes.append({
            "pid":pid,"visible":float(visible),
            "hidden":float(hidden),"full_total":float(total),
            "extra":{name:float(values[name]) for name in VISIBLE},
        })
    if len({r["pid"] for r in episodes})!=len(episodes):
        raise ValueError("Repeated person in same-visit supervision")
    return sorted(episodes,key=lambda r:r["pid"]),qc,source


def train_matrix(rows,names):
    # x=visible total, y=full total means the ridge learns the hidden delta.
    # No hidden item or full total enters predictors; they are fit labels only.
    return [(r["pid"],r["visible"],r["full_total"],r["extra"]) for r in rows]


def clip_hidden(pred_full,visible):
    return max(0.,min(float(HIDDEN_MAX),float(pred_full)-float(visible)))


def tune_ridge(fitting,model,seed,outer_fold):
    names=RIDGE[model]
    foldmap=fixed_folds(
        [(r["pid"],r["visible"],r["full_total"]) for r in fitting],
        INNER_FOLDS,seed+1009*(outer_fold+1))
    data=train_matrix(fitting,names)
    results={}
    for penalty in PENALTIES:
        sse=0.;evaluated=0
        for held in range(INNER_FOLDS):
            proper=[r for r in data if foldmap[r[0]]!=held]
            heldout=[r for r in data if foldmap[r[0]]==held]
            if not proper or not heldout:
                raise ValueError("Empty inner fold")
            fit=fit_multi(proper,names,penalty=penalty)
            for pid,visible,full,features in heldout:
                predicted=clip_hidden(
                    predict_multi(fit,visible,features),visible)
                true_hidden=full-visible
                sse+=(predicted-true_hidden)**2
                evaluated+=1
        if evaluated!=len(fitting):raise ValueError("Missing inner predictions")
        results[penalty]=math.sqrt(sse/evaluated)
    chosen=min(PENALTIES,key=lambda p:(results[p],-p))
    return chosen,results[chosen]


def roc_auc(paired,positive):
    """Mann-Whitney AUC, accounting for ties. Class is hidden sum > 0."""
    ranked=sorted((float(pred),pid) for pid,(truth,pred) in paired.items())
    npos=sum(truth>0 for truth,pred in paired.values())
    nneg=len(paired)-npos
    if not npos or not nneg:return None
    total_ranks=0.;position=0
    while position<len(ranked):
        end=position+1
        while end<len(ranked) and ranked[end][0]==ranked[position][0]:
            end+=1
        average_rank=((position+1)+end)/2
        total_ranks+=average_rank*sum(
            paired[ranked[j][1]][0]>0 for j in range(position,end))
        position=end
    return (total_ranks-npos*(npos+1)/2)/(npos*nneg)


def report(manifest,root,out,folds=5,seed=20261018,bootstraps=500,minimum=10):
    manifest,root,out=[Path(x).expanduser().resolve() for x in (manifest,root,out)]
    if not manifest.is_file() or not (root/"PPMI").is_dir():
        raise FileNotFoundError("Missing private manifest or original PPMI source")
    if out==manifest or root==out or root in out.parents:
        raise ValueError("Unsafe report directory")
    if folds<2 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid folds/bootstrap/privacy threshold")
    episodes,qc,source=load_site_episodes(root,manifest)
    if len(episodes)<folds*10:
        raise ValueError("Insufficient same-visit labeled TRAIN samples")
    assignments=fixed_folds(
        [(r["pid"],r["visible"],r["full_total"]) for r in episodes],
        folds,seed)
    pred={model:{} for model in MODELS}
    fit_penalty=[];fold_sizes=[]
    for held in range(folds):
        proper=[r for r in episodes if assignments[r["pid"]]!=held]
        heldout=[r for r in episodes if assignments[r["pid"]]==held]
        if not proper or not heldout:raise ValueError("Empty outer fold")
        mean_hidden=mean(r["hidden"] for r in proper)
        fitted={}
        for model,names in RIDGE.items():
            penalty,inner_rmse=tune_ridge(proper,model,seed,held)
            fitted[model]=fit_multi(
                train_matrix(proper,names),names,penalty=penalty)
            fit_penalty.append({
                "fold":held,"model":model,
                "chosen_penalty":penalty,
                "inner_rmse_hidden":round(inner_rmse,5)})
        fold_sizes.append({"fold":held,"fit_n":len(proper),"heldout_n":len(heldout)})
        for r in heldout:
            if r["pid"] in pred["hidden_zero"]:
                raise ValueError("Patient in multiple outer folds")
            outcome=r["hidden"]
            scores={
                "hidden_zero":0.,
                "hidden_train_mean":mean_hidden,
            }
            for name,fit in fitted.items():
                scores[name]=clip_hidden(
                    predict_multi(fit,r["visible"],r["extra"]),
                    r["visible"])
            for name,value in scores.items():
                pred[name][r["pid"]]=(outcome,value)
    if any(len(pred[name])!=len(episodes) for name in MODELS):
        raise ValueError("Incomplete held-out predictions")
    outcomes=Counter("hidden_zero" if r["hidden"]==0 else "hidden_positive"
                     for r in episodes)
    metric_rows=[]
    for name in MODELS:
        m=metrics(list(pred[name].values()))
        auc=roc_auc(pred[name],True)
        metric_rows.append({
            "model":name,"population":"PPMI_Prodromal_TRAIN_V04_masked_OOF",
            "n":m["n"],"hidden_mae":round(m["mae"],5),
            "hidden_rmse":round(m["rmse"],5),
            "signed_hidden_error":round(m["mean_signed_error"],5),
            "nonzero_hidden_AUROC":round(auc,5) if auc is not None else "",
        })
    comparisons=[]
    for reference in ("hidden_zero","hidden_train_mean","ridge_visible_sum"):
        for idx,model in enumerate(MODELS):
            if model==reference:continue
            for res in pair_diff(pred[reference],pred[model],bootstraps,
                                 seed+173*idx+19*MODELS.index(reference)):
                comparisons.append({"reference":reference,"model":model,**res})
    subgroup=[]
    for group in ("hidden_zero","hidden_positive"):
        ids=[r["pid"] for r in episodes
             if (r["hidden"]==0)==(group=="hidden_zero")]
        if len(ids)<minimum:continue
        for name in MODELS:
            m=metrics([pred[name][pid] for pid in ids])
            subgroup.append({
                "true_hidden_group":group,"model":name,
                "n":len(ids),"hidden_mae":round(m["mae"],5),
                "hidden_rmse":round(m["rmse"],5)})
    out.mkdir(parents=True,exist_ok=True)
    save(out/"masked_site_oof_metrics.csv",[
        "model","population","n","hidden_mae","hidden_rmse",
        "signed_hidden_error","nonzero_hidden_AUROC"],metric_rows)
    save(out/"masked_site_paired_differences.csv",[
        "reference","model","metric","difference","lower_95","upper_95"],comparisons)
    save(out/"masked_site_inner_selection.csv",[
        "fold","model","chosen_penalty","inner_rmse_hidden"],fit_penalty)
    save(out/"masked_site_fold_sizes.csv",[
        "fold","fit_n","heldout_n"],fold_sizes)
    save(out/"masked_site_outcome_groups.csv",[
        "true_hidden_group","patients"],[
            {"true_hidden_group":group,"patients":safe(outcomes[group],minimum)}
            for group in ("hidden_zero","hidden_positive")])
    save(out/"masked_site_error_by_outcome.csv",[
        "true_hidden_group","model","n","hidden_mae","hidden_rmse"],subgroup)
    save(out/"quality_flags.csv",["check","rows_or_patients"],[
        {"check":k,"rows_or_patients":safe(v,minimum)}
        for k,v in sorted(qc.items())])
    meta={
        "status":"TRAIN_ONLY_simulated_same_V04_masked_remote_Motor_benchmark",
        "eligible_train_V04_patients":len(episodes),
        "hidden_six_zero_patients":outcomes["hidden_zero"],
        "hidden_six_positive_patients":outcomes["hidden_positive"],
        "visible_item_count":len(VISIBLE),"hidden_item_count":len(HIDDEN),
        "primary_regression_metric":"hidden MAE",
        "secondary_metrics":["hidden RMSE","nonzero_hidden AUROC"],
        "ridge_penalty_selection":"inner 3-fold RMSE of hidden six sum",
        "folds":folds,"seed":seed,"bootstraps":bootstraps,
        "input_source":source.name,
        "caveats":[
            "Simulated hiding at V04 is NOT an observed remote R01 same-day trial.",
            "Real R01 and V04 visible-item distributions differ; transportability not established.",
            "No PPMI val/test patients or longitudinal future targets queried.",
            "The true hidden six and recorded NP3TOT are labels/quality checks only.",
            "All preprocessing and penalties fit strictly within TRAIN CV folds.",
            "Strong zero-score prevalence makes the hidden_zero model essential.",
            "No clinical deployment or real examination-cost policy is inferred.",
            "Only aggregate results exported; patient-level values remain on cluster.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(meta,indent=2)+chr(10),
                                     encoding="utf-8")
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase37_masked_site_benchmark"))
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--bootstraps",type=int,default=500)
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(report(args.manifest_db,args.data_root,args.report_dir,
                            args.folds,args.seed,args.bootstraps,
                            args.min_cell_count),indent=2,ensure_ascii=False))
    print("Upload ONLY Phase37 aggregate reports; keep private data local.")

if __name__=="__main__":
    main()
