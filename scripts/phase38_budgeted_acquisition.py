#!/usr/bin/env python3
"""Phase38: budgeted simulated acquisition of masked six Motor item scores.

At a complete V04 onsite visit, simulate only the 27 remotely observable
features being available. An "acquisition" reveals the six hidden item
scores from THAT SAME V04 examination, thereby resolving the residual
hidden-sum error exactly. This is an IDEALIZED oracle reveal, NOT a real
remote-to-onsite clinical intervention, price model, or same-day R01 study.

Use frozen PPMI Prodromal TRAIN patients only. Ridge estimators and their
hyperparameters are learned within five outer folds with three-fold inner
tuning. For a preset visit budget, rank held-out patients WITHOUT looking
at hidden scores; compare random, raw visible sum, Ridge visible sum and
Ridge 27-item ranking. An impossible perfect-information oracle provides
an upper bound only. No private IDs, scores, dates or choices are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase24_motor_baselines import fixed_folds
from scripts.phase26_multimodal_oof import fit_multi,predict_multi
from scripts.phase37_masked_site_benchmark import (
    load_site_episodes,train_matrix,tune_ridge,clip_hidden,RIDGE
)

BUDGETS=(0.,.10,.20,.30,.40,.50,1.)
POLICIES=("random","visible_sum","ridge_visible_sum","ridge_visible_27",
          "oracle_upper_bound")
INNER_FOLDS=3


def hash_rank(seed,pid,salt):
    return hashlib.sha256(
        (str(seed)+"|Phase38|"+salt+"|"+pid).encode()).hexdigest()


def rank_indices(rows,policy,seed,fold):
    """Rank heldout rows using available risk scores, not hidden truth.

    Only oracle_upper_bound reads hidden labels: it is an UNACHIEVABLE
    descriptive bound, never an actionable policy.
    """
    if policy=="random":
        return sorted(range(len(rows)),
                      key=lambda i:hash_rank(seed,rows[i]["pid"],"random"))
    if policy=="oracle_upper_bound":
        return sorted(range(len(rows)),key=lambda i:(
            -rows[i]["hidden"],hash_rank(seed,rows[i]["pid"],"tie")))
    field={
        "visible_sum":"visible",
        "ridge_visible_sum":"risk_visible_sum",
        "ridge_visible_27":"risk_visible_27",
    }[policy]
    return sorted(range(len(rows)),key=lambda i:(
        -rows[i][field],hash_rank(seed,rows[i]["pid"],"tie")))


def oof_risk_rows(episodes,folds,seed):
    if len(episodes)<folds*10:
        raise ValueError("Too few eligible TRAIN onsite exams")
    outer=fixed_folds(
        [(r["pid"],r["visible"],r["full_total"]) for r in episodes],
        folds,seed)
    predictions=[]
    metadata=[]
    for held in range(folds):
        fitting=[r for r in episodes if outer[r["pid"]]!=held]
        heldout=[r for r in episodes if outer[r["pid"]]==held]
        if not fitting or not heldout:raise ValueError("Empty outer fold")
        models={}
        for name,names in RIDGE.items():
            penalty,rmse=tune_ridge(fitting,name,seed,held)
            models[name]=fit_multi(
                train_matrix(fitting,names),names,penalty=penalty)
            metadata.append({
                "fold":held,"model":name,"selected_penalty":penalty,
                "inner_hidden_rmse":round(rmse,5),
                "fit_patients":len(fitting),"heldout_patients":len(heldout)})
        for r in heldout:
            predictions.append({
                "pid":r["pid"],"fold":held,
                "hidden":r["hidden"],"visible":r["visible"],
                "risk_visible_sum":clip_hidden(
                    predict_multi(models["ridge_visible_sum"],
                                  r["visible"],r["extra"]),r["visible"]),
                "risk_visible_27":clip_hidden(
                    predict_multi(models["ridge_visible_27"],
                                  r["visible"],r["extra"]),r["visible"]),
            })
    if len(predictions)!=len(episodes) or len(set(x["pid"] for x in predictions))!=len(episodes):
        raise ValueError("Incomplete patient-heldout predictions")
    return predictions,metadata


def selected_at_budget(rows,policy,budget,seed):
    """Apply the same fixed fractional quota independently in each fold."""
    grouped={}
    for r in rows:
        grouped.setdefault(r["fold"],[]).append(r)
    chosen=set()
    for fold,part in grouped.items():
        ranked=rank_indices(part,policy,seed,fold)
        count=int(math.floor(budget*len(part)+0.5))
        for index in ranked[:count]:
            chosen.add(part[index]["pid"])
    return chosen


def aggregate(rows,chosen):
    """Unselected patients have zero imputed hidden burden.

    Selected patients have exactly observed hidden values (ideal reveal).
    The error is therefore the sum of true hidden burden left unrevealed.
    """
    n=len(rows)
    positive=sum(r["hidden"]>0 for r in rows)
    missed=sum(r["hidden"] for r in rows if r["pid"] not in chosen)
    captured=sum(r["hidden"]>0 for r in rows if r["pid"] in chosen)
    return {
        "n":n,"selected":len(chosen),"positives":positive,
        "captured":captured,"missed_hidden_sum":missed,
        "residual_mae":missed/n,
        "positive_recall":captured/positive if positive else None,
    }


def pair_bootstrap(rows,alternative,reference,reps,seed):
    """Paired CI of remaining absolute score error; frozen OOF choices.

    Does NOT refit models or re-rank each bootstrap sample.
    """
    if reps<100:raise ValueError("At least 100 bootstrap replicates required")
    alt,ref=alternative,reference
    values=[
        (r["hidden"] if r["pid"] not in alt else 0.,
         r["hidden"] if r["pid"] not in ref else 0.)
        for r in rows
    ]
    point=mean(a-b for a,b in values)
    rng=random.Random(seed)
    draws=[]
    for iteration in range(reps):
        draws.append(mean(
            (lambda z:z[0]-z[1])(values[rng.randrange(len(values))])
            for _ in values))
    draws.sort()
    def q(frac):
        pos=(len(draws)-1)*frac
        lo=int(pos);hi=min(lo+1,len(draws)-1)
        return draws[lo]*(hi-pos)+draws[hi]*(pos-lo)
    return point,q(.025),q(.975)


def write_csv(path,columns,records):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(records)


def run(root,manifest,out,folds=5,seed=20261018,
        bootstraps=500,minimum=10):
    root,manifest,out=[Path(p).expanduser().resolve()
                       for p in (root,manifest,out)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI raw source or private manifest")
    if out==manifest or out==root or root in out.parents:
        raise ValueError("Report directory may not overwrite source data")
    if folds<2 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid audit parameters")
    episodes,qc,source=load_site_episodes(root,manifest)
    rows,inner=oof_risk_rows(episodes,folds,seed)
    if len(rows)<minimum:raise ValueError("Too few onsite exams")
    pcount=sum(r["hidden"]>0 for r in rows)
    if pcount<minimum:raise ValueError("Insufficient hidden-positive patients")
    decisions={}
    summary_rows=[]
    for budget in BUDGETS:
        for policy in POLICIES:
            chosen=selected_at_budget(rows,policy,budget,seed)
            decisions[(policy,budget)]=chosen
            stat=aggregate(rows,chosen)
            summary_rows.append({
                "policy":policy,"requested_budget_fraction":budget,
                "selected_patients":stat["selected"],
                "actual_acquisition_fraction":round(stat["selected"]/len(rows),5),
                "hidden_positive_captured":stat["captured"] if
                  stat["captured"]>=minimum or stat["captured"]==0 else f"<{minimum}",
                "positive_recall":round(stat["positive_recall"],5),
                "residual_hidden_mae":round(stat["residual_mae"],5),
                "relative_mae_reduction_from_zero":round(
                    1-stat["residual_mae"]/(sum(r["hidden"] for r in rows)/len(rows)),5),
                "reference_only":policy=="oracle_upper_bound",
            })
    contrasts=[]
    for budget in BUDGETS:
        if budget==0 or budget==1:continue
        for index,policy in enumerate(
                ("visible_sum","ridge_visible_sum","ridge_visible_27")):
            for ref in ("random","visible_sum"):
                if policy==ref:continue
                point,lo,hi=pair_bootstrap(
                    rows,decisions[(policy,budget)],decisions[(ref,budget)],
                    bootstraps,seed+int(100*budget)*41+index*7+len(ref))
                contrasts.append({
                    "requested_budget_fraction":budget,
                    "policy":policy,"reference":ref,
                    "delta_residual_mae":round(point,5),
                    "lower_95":round(lo,5),"upper_95":round(hi,5),
                    "bootstraps":bootstraps,
                })
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"budgeted_acquisition_curve.csv",[
        "policy","requested_budget_fraction","selected_patients",
        "actual_acquisition_fraction","hidden_positive_captured",
        "positive_recall","residual_hidden_mae",
        "relative_mae_reduction_from_zero","reference_only"],summary_rows)
    write_csv(out/"budgeted_paired_contrasts.csv",[
        "requested_budget_fraction","policy","reference",
        "delta_residual_mae","lower_95","upper_95","bootstraps"],contrasts)
    write_csv(out/"ridge_fold_model_selection.csv",[
        "fold","model","selected_penalty","inner_hidden_rmse",
        "fit_patients","heldout_patients"],inner)
    write_csv(out/"source_quality_flags.csv",["check","count"],[
        {"check":key,"count":str(count) if count>=minimum or count==0
         else f"<{minimum}"} for key,count in sorted(qc.items())])
    meta={
        "status":"TRAIN_only_simulated_V04_budgeted_six_item_reveal",
        "eligible_train_V04_patients":len(rows),
        "hidden_positive_train_patients":pcount,
        "budgets_fraction":list(BUDGETS),
        "policies":list(POLICIES),
        "outer_folds":folds,"inner_folds":INNER_FOLDS,"seed":seed,
        "bootstraps":bootstraps,
        "primary_outcome":"remaining MAE of hidden-six sum after idealized same-visit acquisition",
        "privacy_and_interpretation":[
            "Zero-acquisition patients have six hidden items estimated as zero.",
            "Acquired patients are assumed to reveal true same-visit hidden six items perfectly.",
            "This is a simulated V04 masking game, not observed real R01 clinical referral.",
            "No numeric monetary costs or real inspection durations are assumed.",
            "An oracle_upper_bound uses test labels only as an unattainable reference.",
            "For deployable policies, held-out labels are never used for acquisition ranking.",
            "Ridge fitting and penalty selection never access outer held-out labels.",
            "Bootstrap intervals condition on fixed out-of-fold choices and do not refit models.",
            "Real R01 distribution shift and selective clinic followup remain unresolved.",
            "No patient-level IDs, records, choices, predictions or dates are exported.",
            "Only frozen Phase23 TRAIN patients queried; validation/test unopened.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2)+chr(10),encoding="utf-8")
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase38_budgeted_acquisition"))
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--bootstraps",type=int,default=500)
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(run(args.data_root,args.manifest_db,args.report_dir,
                         args.folds,args.seed,args.bootstraps,
                         args.min_cell_count),ensure_ascii=False,indent=2))
    print("Upload Phase38 aggregate report ONLY. No private patient data.")


if __name__=="__main__":
    main()
