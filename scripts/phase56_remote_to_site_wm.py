"""Phase56: real R01 (27 visible, six 101/UR) -> later verified V04.

Only frozen Prodromal TRAIN and patient-disjoint five-fold OOF.
Predict actual measured onsite six scores WITHOUT reading those scores,
target item masks, or future site values at inference. The V04 month
is supplied as a retrospective forecast horizon, not a prospective
appointment prediction. Models: Phase54 latent WM, inner-tuned
Dirichlet item Markov, V04 site-item marginal.

Original Phase44 TEST consumed. No patient-level records exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE
from scripts.phase54_masked_sequences import load_train
from scripts.phase54_temporal_wm import forecast
from scripts.phase54_evaluate import split_patients,select_dev,train_model
from scripts.phase55_markov_baseline import ItemMarkov

HIDX=tuple(ALL.index(k) for k in HIDDEN)
VIDX=tuple(ALL.index(k) for k in VISIBLE)
METHODS=("latent_wm","tuned_item_markov","site_item_marginal")
ALPHAS=(2.,10.,50.,200.)


def find_pairs(patients):
    """Only strict actual R01 and verified later V04 within one TRAIN person."""
    pairs={}
    qc=Counter()
    for person in patients:
        visits=person["visits"]
        remote=[v for v in visits if v["event"]=="R01"
                and all(v["status"][j]==1 for j in VIDX)
                and all(v["status"][j]==2 for j in HIDX)
                and v.get("documented_total") is None]
        site=[v for v in visits if v["event"]=="V04"
              and all(s==1 for s in v["status"])
              and v.get("documented_total") is not None
              and sum(v["score"])==v["documented_total"]]
        if len(remote)!=1 or len(site)!=1:
            qc["not_exactly_one_strict_R01_and_verified_V04"]+=1
            continue
        r,v=remote[0],site[0]
        if v["month"]<=r["month"]:
            qc["V04_not_after_R01"]+=1
            continue
        if not (visits[0]["event"]=="BL" and
                visits[0]["month"]<r["month"]):
            qc["missing_true_BL_before_R01"]+=1
            continue
        pairs[person["pid"]]=(visits[0],r,v)
    return pairs,qc


def masked_input(pair):
    """Dummy forecast target contains NO future values, mask or event type."""
    bl,r,v=pair
    blank={"month":v["month"],"score":(0,)*33,
           "status":(0,)*33,"event":"UNOBSERVED_FUTURE"}
    return {"visits":[bl,r,blank]}


def wm_predict(model,pairs,device):
    predictions={}
    for pid,pair in pairs.items():
        example=masked_input(pair)
        p,*_=forecast(model,[example],device)
        probabilities=p[0,-1,HIDX,:].numpy().astype(np.float64)
        if probabilities.shape!=(6,5):raise AssertionError("Six-item shape")
        predictions[pid]=probabilities
    return predictions


def markov_predict(model,pairs):
    result={}
    for pid,(bl,r,v) in pairs.items():
        latest=[None]*33;month=[None]*33
        for visit in (bl,r):
            for i,(value,status) in enumerate(zip(visit["score"],visit["status"])):
                if status==1:
                    latest[i]=int(value);month[i]=int(visit["month"])
        probs=[]
        for item in HIDX:
            gap=None if month[item] is None else v["month"]-month[item]
            probs.append(model.predict(item,latest[item],gap))
        result[pid]=np.asarray(probs,dtype=np.float64)
    return result


def site_prior(train_people):
    """Categorical 6-item V04 distribution; fit patients only."""
    count=np.ones((6,5),dtype=np.float64)
    for patient in train_people:
        for v in patient["visits"]:
            if v["event"]!="V04" or not all(v["status"][j]==1 for j in HIDX):
                continue
            for h,j in enumerate(HIDX):
                count[h,int(v["score"][j])]+=1
    return count/count.sum(-1,keepdims=True)


def outcomes(pairs,predictions):
    metrics={}
    for pid,(bl,r,site) in pairs.items():
        p=np.asarray(predictions[pid],dtype=np.float64)
        if p.shape!=(6,5) or not np.isfinite(p).all():
            raise AssertionError("Invalid forecast probabilities")
        if not np.allclose(p.sum(-1),1.,atol=1e-5):
            raise AssertionError("Probabilities not normalized")
        truth=[int(site["score"][j]) for j in HIDX]
        nll=mean(-math.log(max(1e-10,float(p[h,v])))
                 for h,v in enumerate(truth))
        # Approximate joint nonzero risk: factorized item predictions,
        # not an independently trained calibrated patient outcome head.
        risk=1.-float(np.prod(p[:,0]))
        metrics[pid]={"nll":nll,"risk":risk,
                      "positive":int(sum(truth)>0)}
    return metrics


def choose_markov_for_site(inner,dev,dev_pairs):
    choices=[]
    for mode in ("pooled","gap"):
        for alpha in ALPHAS:
            model=ItemMarkov(alpha,mode).fit(inner)
            pred=markov_predict(model,dev_pairs)
            loss=mean(x["nll"] for x in outcomes(dev_pairs,pred).values())
            choices.append((loss,mode,alpha))
    if not choices:raise ValueError("No inner R01->V04 development pairs")
    loss,mode,alpha=min(choices,key=lambda x:(x[0],x[1],x[2]))
    return mode,alpha,loss


def auc(rows):
    yes=[r["risk"] for r in rows if r["positive"]==1]
    no=[r["risk"] for r in rows if r["positive"]==0]
    if not yes or not no:return None
    return sum((a>b)+.5*(a==b) for a in yes for b in no)/(len(yes)*len(no))


def rank_stats(pairs,seed):
    rows=list(pairs.items())
    ordered=sorted(rows,key=lambda t:(-t[1]["risk"],hashlib.sha256(
        f"Phase56|{seed}|{t[0]}".encode()).hexdigest()))
    k=min(len(ordered),max(1,int(.2*len(ordered)+.5)))
    pos=sum(x["positive"] for _,x in rows)
    captured=sum(x["positive"] for _,x in ordered[:k])
    return {"eligible":len(rows),"positive":pos,"selected":k,
            "captured_positive":captured,
            "topk_recall":captured/pos if pos else None,
            "auroc":auc([v for _,v in rows]),
            "patient_mean_nll":mean(x["nll"] for _,x in rows)}


def run(patients,out,epochs=25,width=64,folds=5,
        seed=20261018,device_name="cpu",bootstraps=500):
    if not 2<=epochs<=100 or width not in (16,32,64,96,128):
        raise ValueError("Invalid hyperparameters")
    if bootstraps<100:raise ValueError("Need at least 100 bootstrap samples")
    device=torch.device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    all_pairs,criteria=find_pairs(patients)
    if len(all_pairs)<40:
        raise ValueError("Too few strict R01-to-verified V04 TRAIN pairs")
    assignment=split_patients(patients,folds,seed)
    predictions={m:{} for m in METHODS}
    fold_log=[]
    for fold in range(folds):
        outer=[p for p in patients if assignment[p["pid"]]!=fold]
        held=[p for p in patients if assignment[p["pid"]]==fold]
        inner,dev=select_dev(outer,seed+fold)
        dev_pairs={p["pid"]:all_pairs[p["pid"]] for p in dev
                   if p["pid"] in all_pairs}
        held_pairs={p["pid"]:all_pairs[p["pid"]] for p in held
                    if p["pid"] in all_pairs}
        if len(dev_pairs)<5 or len(held_pairs)<5:
            raise ValueError("Insufficient inner-dev/outer R01 pairs in fold "+str(fold))
        candidates=sorted({q for q in (2,5,10,epochs) if q<=epochs})
        trial=[]
        for n in candidates:
            model=train_model(inner,width,n,seed+fold*103,device)
            loss=mean(x["nll"] for x in outcomes(
                dev_pairs,wm_predict(model,dev_pairs,device)).values())
            trial.append((loss,n))
        chosen_loss,selected_epoch=min(trial)
        neural=train_model(outer,width,selected_epoch,seed+fold*103,device)
        mode,alpha,markov_dev=choose_markov_for_site(inner,dev,dev_pairs)
        markov=ItemMarkov(alpha,mode).fit(outer)
        priors=site_prior(outer)
        fold_preds={
            "latent_wm":wm_predict(neural,held_pairs,device),
            "tuned_item_markov":markov_predict(markov,held_pairs),
            "site_item_marginal":{pid:priors for pid in held_pairs},
        }
        for method,item_probs in fold_preds.items():
            scored=outcomes(held_pairs,item_probs)
            if set(predictions[method])&set(scored):
                raise AssertionError("OOF person used twice")
            predictions[method].update(scored)
        fold_log.append({
            "fold":fold,"outer_fit_patients":len(outer),
            "inner_fit_patients":len(inner),"inner_dev_patients":len(dev),
            "dev_remote_site_pairs":len(dev_pairs),
            "heldout_remote_site_pairs":len(held_pairs),
            "wm_epochs":selected_epoch,"wm_inner_site_nll":round(chosen_loss,6),
            "markov_gap_mode":mode,"markov_alpha":alpha,
            "markov_inner_site_nll":round(markov_dev,6)})
    if any(set(predictions[m])!=set(all_pairs) for m in METHODS):
        raise AssertionError("Incomplete or leaked patient OOF results")
    stats=[{"method":method,**{key:(round(v,6) if isinstance(v,float) else v)
                               for key,v in rank_stats(predictions[method],seed).items()}}
           for method in METHODS]
    rng=random.Random(seed+10001)
    ids=sorted(all_pairs)
    paired=[]
    for method in ("tuned_item_markov","site_item_marginal"):
        difference=[predictions["latent_wm"][pid]["nll"]-
                    predictions[method][pid]["nll"] for pid in ids]
        draws=sorted(mean(difference[rng.randrange(len(ids))]
                          for _ in ids) for _ in range(bootstraps))
        def pct(q):
            idx=q*(len(draws)-1);lo=int(idx);hi=min(lo+1,len(draws)-1)
            return draws[lo]*(hi-idx)+draws[hi]*(idx-lo)
        paired.append({"comparison":"latent_wm_minus_"+method,
                       "delta_patient_mean_nll":round(mean(difference),6),
                       "bootstrap_ci_low_95":round(pct(.025),6),
                       "bootstrap_ci_high_95":round(pct(.975),6),
                       "bootstrap_reps":bootstraps})
    out=Path(out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("real_r01_to_v04_metrics.csv",stats),
                      ("fold_inner_selection.csv",fold_log),
                      ("paired_nll_contrast.csv",paired)):
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    summary={
        "status":"exploratory_TRAIN_only_real_R01_27_to_future_V04_hidden_six",
        "train_trajectories":len(patients),
        "eligible_real_remote_future_site_pairs":len(all_pairs),
        "outer_folds":folds,"seed":seed,
        "strict_cohort_protocol":"Phase56_item_mask_exact_one_R01_one_V04_BL_anchored",
        "strict_cohort_difference":"Do not assume Phase36 199 and Phase53 197 match",
        "source_qc_small_counts_suppressed":{
            k:(n if n>=10 else "<10") for k,n in sorted(criteria.items())},
        "caveats":[
            "R01 visible scores are real observations, V04 six targets months later.",
            "Model receives BL/R01 measured data and V04 MONTH only, no onsite outcomes.",
            "Target site month is supplied retrospectively; not prospective appointment time.",
            "Risk from product of six independent 0-score probabilities is uncalibrated.",
            "The same patient never crosses outer training and evaluation.",
            "All results on repeatedly explored TRAIN, not fresh final confirmation.",
            "Consumed Phase44 TEST and locked original cohort remain untouched.",
            "No PHI/patient-level predictions or source records exported.",
            "Not a causal medication model, Dreamer4 reproduction or LLM Doctor RSI."
        ],"generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--max-epochs",type=int,default=25)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    data,meta=load_train(a.data_root,a.manifest_db)
    result=run(data,a.report_dir,a.max_epochs,a.width,a.folds,a.device_name
               if hasattr(a,"device_name") else 20261018)
    print(json.dumps({"aggregate_data_meta":meta,"evaluation":result},indent=2))
    print("Upload aggregate CSV/JSON only; PPMI patient data stay on cluster.")


if __name__=="__main__":main()
