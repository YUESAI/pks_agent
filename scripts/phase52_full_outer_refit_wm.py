#!/usr/bin/env python3
"""Phase52: full-outer-fit 5/10-fold patient CV for Phase50 longitudinal WM.

Within each outer TRAIN fold, an INNER train/dev split selects only the
number of epochs. A FRESH model is then trained on ALL outer-training
people, for that fixed epoch count, without consulting outer-heldout data.

Input: frozen PPMI Prodromal TRAIN BL/12m/24m sequences from Phase29.
No original VAL or consumed TEST. Only aggregated reports leave cluster.
This fixes the small actual fitting-set issue; it does NOT increase the
number of independent trajectories or make Dreamer/JEPA state-of-the-art.
"""
from __future__ import annotations
import argparse,csv,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.phase50_clinical_latent_wm import (
    ClinicalLatentDynamics,batch,loss_fn,fingerprint,fit_one
)

METHODS=("baseline_value","last_value","fixed_50_50_smoothing",
         "linear_extrapolation","latent_imagination_wm")


def assign_folds(records,nfolds,seed):
    if nfolds not in (3,5,10):
        raise ValueError("Outer folds must be 3, 5 or 10")
    names=[r["pid"] for r in records]
    if len(names)!=len(set(names)) or len(names)<nfolds*12:
        raise ValueError("Too few distinct frozen TRAIN patients")
    ordered=sorted(names,key=lambda p:fingerprint(seed,p))
    return {pid:i % nfolds for i,pid in enumerate(ordered)}


def choose_inner(records,seed):
    ordered=sorted(records,key=lambda r:fingerprint(seed,r["pid"]))
    count=max(12,int(round(.2*len(ordered))))
    if len(ordered)-count<24:
        raise ValueError("Too few inner fitting patients")
    return ordered[count:],ordered[:count]


def refit_on_full_outer_train(records,epochs,seed,width,device):
    """No heldout/development data fed as labels outside outer TRAIN.

    Epoch count is fixed by a separate inner development subset.
    Training from scratch is essential: continuing the inner-fit
    checkpoint would expose fewer records to early optimizer updates.
    """
    if not 1<=epochs<=250:raise ValueError("Invalid fixed epoch count")
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    generator=torch.Generator().manual_seed(seed+937)
    values=batch(records,device)
    model=ClinicalLatentDynamics(width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=.015)
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(records),generator=generator).split(64):
            index=ids.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model,*(value[index] for value in values))
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite WM refit objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
    model.eval()
    return model


def score_baseline(name,row):
    if name=="baseline_value":return row["bl"]
    if name=="last_value":return row["last"]
    if name=="fixed_50_50_smoothing":return .5*(row["bl"]+row["last"])
    if name=="linear_extrapolation":
        return max(0.,min(132.,row["last"]+
                   (row["last"]-row["bl"])*(24-row["gap"])/row["gap"]))
    if name=="latent_imagination_wm":return row["latent"]
    raise ValueError("Unknown baseline")


def summarize_predictions(predictions,fold_sizes,seed,bootstraps):
    observations=list(predictions.values())
    if not observations:raise ValueError("Empty outer predictions")
    metrics=[]
    for method in METHODS:
        residuals=[score_baseline(method,r)-r["truth"] for r in observations]
        metrics.append({
            "model":method,"n":len(observations),
            "mae":round(mean(abs(e) for e in residuals),5),
            "rmse":round(math.sqrt(mean(e*e for e in residuals)),5),
            "signed_bias":round(mean(residuals),5)
        })
    delta=[abs(r["latent"]-r["truth"])-
           abs(.5*(r["bl"]+r["last"])-r["truth"]) for r in observations]
    rng=random.Random(seed+401)
    draws=[]
    for _ in range(bootstraps):
        draws.append(mean(delta[rng.randrange(len(delta))]
                          for __ in delta))
    draws.sort()
    def quantile(q):
        at=q*(len(draws)-1)
        lo=int(at);hi=min(lo+1,len(draws)-1)
        return draws[lo]*(hi-at)+draws[hi]*(at-lo)
    paired=[{
        "comparison":"latent_imagination_wm - fixed_50_50_smoothing",
        "delta_mae":round(mean(delta),5),
        "bootstrap_lower_95":round(quantile(.025),5),
        "bootstrap_upper_95":round(quantile(.975),5),
        "bootstrap_draws":bootstraps,
        "patients":len(delta)
    }]
    by_fold=[]
    for fold,n in sorted(fold_sizes.items()):
        sample=[r for r in observations if r["outer_fold"]==fold]
        if len(sample)!=n:raise AssertionError("Fold sample missing")
        for name in ("fixed_50_50_smoothing","latent_imagination_wm"):
            by_fold.append({
                "outer_fold":fold,"model":name,"heldout_n":n,
                "mae":round(mean(abs(score_baseline(name,r)-r["truth"])
                                 for r in sample),5)
            })
    return metrics,paired,by_fold


def write_table(path,rows):
    if not rows:raise ValueError("Empty aggregate table")
    with path.open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluate(rows,out,folds=5,members=3,epochs=60,seed=20261018,
             width=64,device_name="cpu",bootstraps=500):
    if not 1<=members<=5 or not 2<=epochs<=250 or bootstraps<100:
        raise ValueError("Invalid member, epoch or bootstrap settings")
    if len(rows)<100 or len({r["pid"] for r in rows})!=len(rows):
        raise ValueError("Need distinct TRAIN longitudinal trajectories")
    device=torch.device(device_name)
    if device.type=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    assignment=assign_folds(rows,folds,seed)
    predictions={};fit_audit=[];fold_sizes={}
    for fold in range(folds):
        outer_train=[r for r in rows if assignment[r["pid"]]!=fold]
        held=[r for r in rows if assignment[r["pid"]]==fold]
        inner_fit,inner_dev=choose_inner(outer_train,seed+fold*199)
        fit_ids={r["pid"] for r in inner_fit}
        dev_ids={r["pid"] for r in inner_dev}
        outer_ids={r["pid"] for r in outer_train}
        held_ids={r["pid"] for r in held}
        if fit_ids & dev_ids or outer_ids & held_ids or fit_ids|dev_ids!=outer_ids:
            raise AssertionError("Invalid nested patient partition")
        if len(held)<12:raise AssertionError("Outer fold too small")
        fold_sizes[fold]=len(held)
        ensemble=[]
        for member in range(members):
            member_seed=seed+fold*1009+member*9176
            # Select training duration using only inner TRAIN development labels.
            _inner_model,best_epoch,dev_mae=fit_one(
                inner_fit,inner_dev,member_seed,width,epochs,device)
            del _inner_model
            # Fix duration and restart on ALL non-heldout patients.
            final=refit_on_full_outer_train(
                outer_train,best_epoch,member_seed,width,device)
            ensemble.append(final)
            fit_audit.append({
                "outer_fold":fold,"ensemble_member":member,
                "outer_train_n":len(outer_train),
                "inner_fit_n":len(inner_fit),
                "inner_dev_n":len(inner_dev),
                "final_refit_n":len(outer_train),
                "heldout_n":len(held),
                "selected_epochs":best_epoch,
                "inner_dev_mae":round(dev_mae,5),
                "final_refit_uses_full_outer_train":True,
            })
        actual=batch(held,device)
        with torch.inference_mode():
            estimate=torch.stack([
                member.forecast(actual[0],actual[1],actual[3])[0]
                for member in ensemble]).mean(0).cpu().tolist()
        for row,value in zip(held,estimate):
            pid=row["pid"]
            if pid in predictions or not 0<=float(value)<=132:
                raise AssertionError("Duplicate or invalid out-of-fold forecast")
            predictions[pid]={
                "truth":row["target"],"latent":float(value),
                "last":row["last"],"bl":row["bl"],
                "gap":row["gap12"],"outer_fold":fold
            }
    if len(predictions)!=len(rows):
        raise AssertionError("Incomplete out-of-fold predictions")
    metrics,paired,by_fold=summarize_predictions(
        predictions,fold_sizes,seed,bootstraps)
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    write_table(out/"oof_metrics.csv",metrics)
    write_table(out/"paired_smoothing_contrast.csv",paired)
    write_table(out/"fold_training_sizes.csv",fit_audit)
    write_table(out/"fold_aggregate_mae.csv",by_fold)
    summary={
        "status":"exploratory_TRAIN_only_nested_epoch_select_full_outer_refit",
        "n_independent_train_trajectories":len(rows),
        "outer_folds":folds,
        "ensemble_members_per_fold":members,
        "max_epochs_for_inner_selection":epochs,
        "selected_epochs_fixed_before_outer_refit":True,
        "full_outer_training_patients_per_fold":
            sorted({a["outer_train_n"] for a in fit_audit}),
        "inner_selection_fit_patients_per_fold":
            sorted({a["inner_fit_n"] for a in fit_audit}),
        "outer_heldout_patients_per_fold":
            sorted(fold_sizes.values()),
        "seed":seed,"device":str(device),
        "warnings":[
            "More folds increase fit size per model, not independent patient count.",
            "Each inner development set is for selecting epochs only; final refit uses all non-heldout people.",
            "No outer heldout labels are used for fitting, early stopping or hyperparameter selection.",
            "The comparison with the old 3fold/non-refit Phase50 run changes two factors; not a causal ablation.",
            "This repeats exploratory TRAIN analyses; no independent confirmatory generalization claim.",
            "Original Phase44 final TEST was consumed and remains untouched.",
            "Not a full Dreamer4 reproduction or intervention simulator.",
            "Patient IDs, scores, trajectories and individual forecasts stay local in RAM and are never exported."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--folds",choices=(3,5,10),type=int,default=5)
    p.add_argument("--members",type=int,default=3)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--bootstraps",type=int,default=500)
    p.add_argument("--seed",type=int,default=20261018)
    args=p.parse_args()
    manifest=args.manifest_db.expanduser().resolve()
    out=args.report_dir.expanduser().resolve()
    if not manifest.is_file() or out==manifest:
        raise FileNotFoundError("Private TRAIN manifest not available or unsafe output")
    from scripts.phase29_sequential import load
    records=load(manifest) # hard-coded SQL WHERE split='train'
    print(json.dumps(evaluate(
        records,out,args.folds,args.members,args.epochs,args.seed,
        args.width,args.device,args.bootstraps),indent=2))
    print("Upload AGGREGATE Phase52 report only, never the private manifest.")

if __name__=="__main__":main()
