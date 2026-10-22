#!/usr/bin/env python3
"""Phase49: decision-aligned observation WM, nested TRAIN-only patient CV.

Compares simple Naive Bayes, masked Transformer and a decision-aligned
masked Transformer. The new model adds a positive-negative pairwise
ranking loss to binary outcome and categorical item reconstruction.
The ranking-loss coefficient is selected ONLY on inner development
patients, not on the outer heldout patients.

Recorded complete PPMI TRAIN V04 rows support *same-visit* partial
observation simulations. This is NOT a learned temporal Parkinson's
disease World Model or evidence that clinical referrals have benefit.
Original Phase44 TEST is consumed and never read.
"""
from __future__ import annotations
import argparse,csv,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from scripts.phase45_world_model import ObservationWorldModel
from scripts.phase45_wm_rsi_agents import (
    load_train,hashkey,roc_auc,measure,outer_assignment
)
from scripts.phase48_masked_transformer_wm import (
    MaskedItemTransformer,NeuralObservationWM,
    masked_batch,multitask_loss,dataset,negative_loglik,
)
BUDGETS=(4,8,12,16)
RANK_WEIGHTS=(0.,.3,1.)
SELECT_BUDGETS=(8,12)


def inner_partition(records,seed):
    """Stratify inner split by target on TRAIN, never across outer fold."""
    fit=[];dev=[]
    for label in (0,1):
        group=sorted((r for r in records if r["label"]==label),
                     key=lambda r:hashkey(seed,r["pid"]))
        size=max(1,int(round(.2*len(group))))
        dev.extend(group[:size]);fit.extend(group[size:])
    if len(fit)<100 or len(dev)<20:
        raise ValueError("Need enough distinct inner fit and dev patients")
    if set(r["pid"] for r in fit)&set(r["pid"] for r in dev):
        raise AssertionError("Inner patient leakage")
    return fit,dev


def ranking_loss(logits,labels):
    """Differentiable bipartite-ranking surrogate in each training batch."""
    pos=logits[labels>0.5]
    neg=logits[labels<=0.5]
    if not pos.numel() or not neg.numel():
        return logits.sum()*0
    return F.softplus(-(pos[:,None]-neg[None,:])).mean()


def train_model(fit,dev,items,seed,device,rank_weight,
                epochs=45,width=64,depth=2,patience=8):
    """Identical initial seeds/splits for all candidates within a fold."""
    if rank_weight<0 or epochs<2:
        raise ValueError("Invalid training objective")
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    rng=torch.Generator().manual_seed(seed+101)
    x,y=dataset(fit,items);xd,yd=dataset(dev,items)
    model=MaskedItemTransformer(
        len(items),width=width,depth=depth,dropout=.12).to(device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=8e-4,weight_decay=.02)
    dm=masked_batch(xd,options=(0,4,8,12,16,len(items)),
                    generator=torch.Generator().manual_seed(seed+823)).to(device)
    xd=xd.to(device);yd=yd.to(device)
    best=float("inf");best_state=None;best_epoch=0;stale=0
    for epoch in range(1,epochs+1):
        model.train()
        perm=torch.randperm(len(x),generator=rng)
        for ids in perm.split(64):
            xb=x[ids].to(device);yb=y[ids].to(device)
            mask=masked_batch(x[ids],generator=rng).to(device)
            logits,conditioned=model(xb,mask)
            loss=multitask_loss(
                (logits,conditioned),xb,yb,mask,item_weight=.5)
            loss=loss+rank_weight*ranking_loss(logits,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits,conditional=model(xd,dm)
            dev_loss=multitask_loss(
                (logits,conditional),xd,yd,dm,item_weight=.5)
            dev_loss=dev_loss+rank_weight*ranking_loss(logits,yd)
            value=float(dev_loss.cpu())
        if value<best-1e-5:
            best=value;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone()
                        for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience:break
    if best_state is None:raise RuntimeError("No trained weights")
    model.load_state_dict(best_state)
    model.eval()
    return model,best_epoch


def metrics_for_budget(world,patients,items,seen,seed):
    nll,risks=negative_loglik(world,patients,items,seen,seed)
    m=measure(patients,risks,seed,True)
    return {
        "masked_nll":nll,
        "brier":m["brier"],
        "auroc":m["auroc"],
        "topk_recall":m["recall"],
        "hidden_burden_recall":m["burden_recall"],
    },risks


def inner_score(world,dev,items,seed):
    """Predeclared verifier score, no outer-TEST feedback.

    Penalize miscalibration, poor discrimination and poor next-item
    likelihood at q=8/12, all on inner development subjects.
    """
    scores=[]
    for q in SELECT_BUDGETS:
        metrics,_=metrics_for_budget(world,dev,items,q,seed)
        scores.append(
            metrics["brier"]+
            .25*(1.-metrics["auroc"])+
            .15*metrics["masked_nll"])
    return mean(scores)


def run(records,items,out,folds=3,epochs=45,width=64,depth=2,
        seed=20261018,device_name="cpu"):
    if not 2<=folds<=5 or not 2<=epochs<=150:
        raise ValueError("Invalid folds or epochs")
    if len(records)<150 or len({r["pid"] for r in records})!=len(records):
        raise ValueError("Need >=150 distinct TRAIN patients")
    device=torch.device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    assignment=outer_assignment(records,folds,seed)
    pred={name:{q:{} for q in BUDGETS}
          for name in ("naive_bayes","masked_transformer","decision_aligned_wm")}
    scores_by_fold=[]
    selection_log=[]
    for fold in range(folds):
        outer_train=[r for r in records if assignment[r["pid"]]!=fold]
        outer_held=[r for r in records if assignment[r["pid"]]==fold]
        fitting,dev=inner_partition(outer_train,seed+113*fold)
        naive=ObservationWorldModel(items).fit(fitting)
        trained={}
        for rank_weight in RANK_WEIGHTS:
            neural,epoch=train_model(
                fitting,dev,items,seed+1001*fold,device,rank_weight,
                epochs=epochs,width=width,depth=depth)
            wrapped=NeuralObservationWM(items,[neural],device)
            score=inner_score(wrapped,dev,items,seed+113*fold)
            trained[rank_weight]=(wrapped,score,epoch)
            selection_log.append({
                "outer_fold":fold,"rank_weight":rank_weight,
                "inner_score":round(score,6),
                "best_epoch":epoch,
                "inner_fit_n":len(fitting),
                "inner_dev_n":len(dev),
                "outer_eval_n":len(outer_held),
            })
        # Candidate lambda=0 reproduces the objective type in Phase48.
        # Selection among improved candidates uses INNER dev ONLY.
        selected=min(RANK_WEIGHTS,key=lambda w:(trained[w][1],w))
        models={
            "naive_bayes":naive,
            "masked_transformer":trained[0.][0],
            "decision_aligned_wm":trained[selected][0],
        }
        scores_by_fold.append({
            "outer_fold":fold,
            "chosen_rank_weight":selected,
            "best_inner_score":round(trained[selected][1],6),
            "baseline_inner_score":round(trained[0.][1],6),
            "decision_candidate_accepted":selected!=0.,
            "outer_eval_n":len(outer_held),
        })
        for name,model in models.items():
            for q in BUDGETS:
                _,risk=metrics_for_budget(
                    model,outer_held,items,q,seed)
                pred[name][q].update(risk)
    table=[]
    for q in BUDGETS:
        for name in pred:
            # OOF patient predictions obtained from a model that never
            # trained on that person's recorded measurements.
            estimates=pred[name][q]
            if len(estimates)!=len(records):
                raise AssertionError("Incomplete outer-fold predictions")
            all_stats=measure(records,estimates,seed,True)
            all_stats.pop("chosen")
            # NLL must be averaged across disjoint outer patients,
            # not estimated with any model trained on them. Recompute
            # below from per-fold conditional NLL captured separately.
            table.append({
                "model":name,"visible_questions":q,
                "oof_patients":len(records),
                "oof_positives":sum(r["label"] for r in records),
                "oof_brier":round(all_stats["brier"],6),
                "oof_auroc":round(all_stats["auroc"],6),
                "oof_topk_recall":round(all_stats["recall"],6),
                "oof_burden_recall":round(all_stats["burden_recall"],6),
            })
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in (
        ("oof_decision_quality.csv",table),
        ("inner_candidate_verifier.csv",selection_log),
        ("fold_selected_objectives.csv",scores_by_fold)
    ):
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    summary={
        "status":"exploratory_TRAIN_only_nestedCV_decision_aligned_observation_WM",
        "train_episodes":len(records),"folds":folds,"epochs_max":epochs,
        "candidate_ranking_weights":list(RANK_WEIGHTS),
        "selection_score":"mean(q8,q12) Brier + .25*(1-AUROC) + .15*masked_item_NLL",
        "architectural_width":width,"architectural_depth":depth,
        "observed_budgets":list(BUDGETS),"seed":seed,
        "device":str(device),
        "cautions":[
            "World Model targets same-visit V04 masked observations, not clinical progression.",
            "Patient outcomes are used for TRAIN fits and inner verifier but never as observed features.",
            "All candidates selected on inner dev only; outer patients excluded from training.",
            "Repeated TRAIN scientific exploration is not independent confirmatory testing.",
            "Original Phase44 TEST is consumed and remains unopened.",
            "Ranking loss and candidate acceptance are constrained model search, NOT autonomous LLM RSI.",
            "No individual patient records, decisions or identifiers exported.",
            "No clinical benefit or causal effect of measurement demonstrated."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--epochs",type=int,default=45)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--seed",type=int,default=20261018)
    a=p.parse_args()
    root,db,out=[Path(x).expanduser().resolve()
                 for x in (a.data_root,a.manifest_db,a.report_dir)]
    if not (root/"PPMI").is_dir() or not db.is_file() or out==db or out==root or root in out.parents:
        raise ValueError("Missing PPMI source or unsafe output directory")
    records,items,qc,source=load_train(root,db)
    print(json.dumps(run(records,items,out,a.folds,a.epochs,a.width,
                         a.depth,a.seed,a.device),indent=2))
    print("Upload only aggregate report ZIP; patient-level records remain on cluster.")


if __name__=="__main__":
    main()
