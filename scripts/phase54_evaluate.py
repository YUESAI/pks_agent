"""Phase54 5-fold patient-heldout longitudinal WM evaluation.

The known target visit month is used ONLY as a retrospective forecast
horizon; no future outcome, event or item mask enters the predictor.
All private patient-level arrays stay in memory. Reports are aggregates.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
import torch
from scripts.phase54_masked_sequences import load_train
from scripts.phase54_temporal_wm import MaskedMotorWM,patient_batch,observed_nll,forecast
from scripts.phase55_markov_baseline import ItemMarkov,choose_markov

def hashed(seed,key):
    return hashlib.sha256(f"phase54|{seed}|{key}".encode()).hexdigest()

def split_patients(rows,folds,seed):
    if folds not in (3,5) or len(rows)<folds*20:
        raise ValueError("Need >=20 people per fold")
    ids=[r["pid"] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate patient IDs")
    order=sorted(ids,key=lambda x:hashed(seed,x))
    return {p:i%folds for i,p in enumerate(order)}

def select_dev(rows,seed):
    sorted_rows=sorted(rows,key=lambda x:hashed(seed,x["pid"]))
    size=max(15,round(len(rows)*.15))
    return sorted_rows[size:],sorted_rows[:size]

def train_model(rows,width,epochs,seed,device):
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    model=MaskedMotorWM(width=width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.02)
    for epoch in range(epochs):
        order=torch.randperm(len(rows),generator=generator).tolist()
        model.train()
        for start in range(0,len(order),16):
            b=[rows[i] for i in order[start:start+16]]
            vals,status,months,valid=patient_batch(b,device)
            logits=model(vals,status,months,valid)
            loss=observed_nll(logits,vals,status,valid)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite WM loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
    return model

def validation_loss(model,rows,device):
    model.eval();scores=[]
    with torch.no_grad():
        for start in range(0,len(rows),16):
            b=rows[start:start+16]
            x,s,m,v=patient_batch(b,device)
            logits=model(x,s,m,v)
            probs=logits.log_softmax(-1)
            chosen=(s[:,1:]==1)&v[:,1:,None]
            per_item=-probs.gather(-1,x[:,1:,...,None]).squeeze(-1)
            numerator=(per_item*chosen).sum((1,2))
            denominator=chosen.sum((1,2)).clamp_min(1)
            scores.extend((numerator/denominator).cpu().tolist())
    return mean(scores)

def baselines(fitting):
    counts=torch.ones(33,5,dtype=torch.float64) # Laplace smoothing
    for person in fitting:
        for visit in person["visits"]:
            for k,(value,state) in enumerate(zip(visit["score"],visit["status"])):
                if state==1:counts[k,value]+=1
    return (counts/counts.sum(-1,keepdim=True)).float()

def evaluate_rows(held,model,prior,device):
    stats={key:[] for key in ("neural_wm","last_observed","train_marginal")}
    # Patient-normalized score: each patient contributes equal weight.
    for start in range(0,len(held),16):
        batch=held[start:start+16]
        neural,vals,status,valid=forecast(model,batch,device)
        for i,person in enumerate(batch):
            losses={k:[] for k in stats}
            latest=[None]*33
            for k in range(33):
                if person["visits"][0]["status"][k]==1:
                    latest[k]=int(person["visits"][0]["score"][k])
            for j in range(1,len(person["visits"])):
                target=person["visits"][j]
                for k,(value,st) in enumerate(zip(target["score"],target["status"])):
                    if st!=1:continue
                    v=int(value)
                    neural_p=float(neural[i,j-1,k,v].clamp_min(1e-8))
                    base=float(prior[k,v].clamp_min(1e-8))
                    persist=base if latest[k] is None else (
                        .85 if v==latest[k] else .15/4)
                    losses["neural_wm"].append(-math.log(neural_p))
                    losses["train_marginal"].append(-math.log(base))
                    losses["last_observed"].append(-math.log(persist))
                for k,(value,st) in enumerate(zip(target["score"],target["status"])):
                    if st==1:latest[k]=int(value)
            for key,values in losses.items():
                if not values:raise AssertionError("No next-visit measurable scores")
                stats[key].append(mean(values))
    return stats

def run(data,out,folds=5,width=64,max_epochs=25,seed=20261018,device_name="cpu"):
    if not 2<=max_epochs<=100 or width not in (16,32,64,96,128):
        raise ValueError("Invalid run settings")
    device=torch.device(device_name)
    if device.type=="cpu":torch.set_num_threads(min(4,torch.get_num_threads()))
    assignment=split_patients(data,folds,seed)
    results={name:[] for name in ("neural_wm","last_observed","train_marginal","tuned_item_markov")}
    sizes=[];epoch_records=[];markov_choices=[]
    for fold in range(folds):
        outer=[p for p in data if assignment[p["pid"]]!=fold]
        held=[p for p in data if assignment[p["pid"]]==fold]
        inner,dev=select_dev(outer,seed+fold)
        if ({p["pid"] for p in outer}&{p["pid"] for p in held} or
            {p["pid"] for p in inner}&{p["pid"] for p in dev}):
            raise AssertionError("Patient leakage")
        candidates=(2,5,10,max_epochs)
        candidates=sorted(set(e for e in candidates if e<=max_epochs))
        scores=[]
        for n in candidates:
            provisional=train_model(inner,width,n,seed+fold*103,device)
            score=validation_loss(provisional,dev,device)
            scores.append((score,n))
        best_loss,best_epoch=min(scores)
        # Fresh initialization and all OUTER TRAIN patients used for final fit.
        fitted=train_model(outer,width,best_epoch,seed+fold*103,device)
        priors=baselines(outer)
        fold_result=evaluate_rows(held,fitted,priors,device)
        markov_choice=choose_markov(inner,dev)
        markov=ItemMarkov(markov_choice["selected_alpha"],
                          markov_choice["selected_gap_mode"]).fit(outer)
        patient_markov=markov.score_patients(held)
        fold_result["tuned_item_markov"]=[
            patient_markov[person["pid"]] for person in held]
        markov_choices.append({
            "fold":fold,"outer_fit_patients":len(outer),
            "inner_fit_patients":len(inner),"inner_dev_patients":len(dev),
            **markov_choice})
        for name in results:results[name].extend(fold_result[name])
        sizes.append({"fold":fold,"outer_fit_n":len(outer),
                      "inner_epoch_selection_fit_n":len(inner),
                      "inner_dev_n":len(dev),"heldout_patients":len(held),
                      "best_epoch":best_epoch,
                      "inner_dev_nll":round(best_loss,5)})
        epoch_records.append(best_epoch)
    if any(len(v)!=len(data) for v in results.values()):
        raise AssertionError("Incomplete patient-fold evaluation")
    metrics=[{"method":k,"patients":len(values),
              "mean_patient_next_visit_item_nll":round(mean(values),6)}
             for k,values in results.items()]
    rng=random.Random(seed+77)
    contrasts=[]
    for comparator in ("last_observed","train_marginal","tuned_item_markov"):
        difference=[a-b for a,b in zip(results["neural_wm"],results[comparator])]
        draws=sorted(mean(difference[rng.randrange(len(difference))]
                         for _ in difference) for _ in range(400))
        def percentile(q):
            x=q*(len(draws)-1);i=int(x);j=min(i+1,len(draws)-1)
            return draws[i]*(j-x)+draws[j]*(x-i)
        contrasts.append({
            "comparison":"neural_wm_minus_"+comparator,
            "delta_patient_mean_nll":round(mean(difference),6),
            "bootstrap_ci95_low":round(percentile(.025),6),
            "bootstrap_ci95_high":round(percentile(.975),6),
            "bootstrap_reps":400})
    out=Path(out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        with (out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    write("patient_oof_item_nll.csv",metrics)
    write("fold_training_sizes.csv",sizes)
    write("fold_markov_selections.csv",markov_choices)
    write("paired_nll_contrast.csv",contrasts)
    summary={"status":"TRAIN_only_irregular_time_masked_item_temporal_wm",
             "patient_trajectories":len(data),"outer_folds":folds,
             "inner_epoch_selection":candidates,
             "additional_controls":"TRAIN-fit tuned Markov and matched patient bootstrap against marginal",
             "selected_epochs":epoch_records,"width":width,
             "evaluation":"retrospective time-conditioned next observed visit",
             "constraints":[
                 "No real prospective appointment forecasting: target visit time gap is given.",
                 "Markov transition prior and smoothing selected ONLY on inner development; refitted on outer TRAIN.",
                 "Only recorded score items contribute to next-visit scoring.",
                 "The complete measured visit is used for belief update after it is scored.",
                 "No patient rows, identifiers or individual losses exported.",
                 "All data are frozen TRAIN; consumed Phase44 TEST untouched.",
                 "Not a treatment causal world model or full Dreamer4 reproduction."],
             "created_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data-root","manifest-db","report-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--max-epochs",type=int,default=25)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    data,meta=load_train(a.data_root,a.manifest_db)
    result=run(data,a.report_dir,a.folds,a.width,a.max_epochs,
               device_name=a.device)
    print(json.dumps({"data_capacity":meta,"wm_eval":result},indent=2))
    print("Share only aggregate report files, never patient-level records.")
if __name__=="__main__":main()
