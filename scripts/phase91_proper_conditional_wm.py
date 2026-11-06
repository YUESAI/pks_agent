#!/usr/bin/env python3
"""Phase91: proper-scored missingness-aware conditional Motor world model.

Only THREE NEW candidate jobs, on EXACT Phase84-90 global 80% FIT /
20% repeatedly-used DEV split. Phase90 static/dynamic comparisons
already exist, so do not waste jobs rerunning those baselines.

0 Static full33 mixture + proper-scored posterior after full real visible27.
1 Static full33 mixture + randomized observed-subset posterior training.
2 Continuous-time full33 mixture + same randomized subset posterior training.

All conditioning values are real full33 FIT visit measurements.
The 27 observed visible values are only used at the target visit in
the separate retrospective masked recovery task, NEVER in future forecast.
The hidden-six total target never enters model input.
Training includes both future prior and conditional posterior proper
scores; random masks simulate missing-at-visit observations by dropping
values, not by inventing tests. No independent test claims.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.phase62_ppmi_multicohort_audit import load_cohorts,VIDX,HIDX
from scripts.phase72_ctlatent_wm import sequence_features
from scripts.phase76_reliability_gated_wm import labels,write_csv
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase85_correlated_site_wm import (
    read_tasks,fingerprint,SEED,TASKS)
from scripts.phase87_visit_anchored_progression_wm import (
    merge_fit_auxiliary)
from scripts.phase90_masked_joint33_wm import (
    Full33Net,hidden_total_posterior,masked_joint_loglik,
    auxiliary_partial_targets,attach_true_future)

# Prespecified developmental weights, NOT fit by selecting on the DEV set.
CONDITIONAL_WEIGHT=.50
PARTIAL_WEIGHT=.50
FULL_JOINT_WEIGHT=1.
RPS_WEIGHT=1.5
OBSERVED_SIZES=(7,14,21,27)
EVAL_OBSERVED_SIZES=(7,14,27)
MASK_SALT="phase91-real-visit-mask-v1"


def real_visible(full33):
    """True visible27 ratings from a completed 33-item exam."""
    if full33.ndim!=2 or full33.shape[1]!=33:
        raise ValueError("Expected real full33 target [B,33]")
    return full33[:,list(VIDX)]


def training_masks(visible,rng,mode):
    """Drop some actually observed visible items; masked entries=-100.

    The same per-patient mask applies to all mixture components. No
    unobserved clinical values are generated or guessed.
    """
    if visible.ndim!=2 or visible.shape[1]!=27:
        raise ValueError("Expected 27 future VISIBLY measured items")
    if mode not in ("full","subsets"):
        raise ValueError("Unknown mask mode")
    if torch.any((visible<0)|(visible>4)):
        raise ValueError("Training future full33 ratings must be measured")
    if mode=="full":
        return visible.clone()
    rows=[]
    for i in range(len(visible)):
        count=int(rng.choice(OBSERVED_SIZES))
        indices=rng.choice(27,size=count,replace=False)
        vector=torch.full_like(visible[i],-100)
        vector[indices]=visible[i,indices]
        rows.append(vector)
    return torch.stack(rows)


def deterministic_eval_mask(rows,visible,n_observed):
    """Same patient/horizon subset across all candidate arms.

    Identifiers are used for deterministic private in-memory selection
    only; no IDs are written to disk.
    """
    if n_observed not in EVAL_OBSERVED_SIZES:
        raise ValueError("Unsupported fixed evaluation information budget")
    if visible.shape!=(len(rows),27):
        raise ValueError("Mask inputs and real visits are misaligned")
    out=torch.full_like(visible,-100)
    for i,row in enumerate(rows):
        key=f"{MASK_SALT}|{row['pid']}|{row['horizon']}"
        priorities=[
            (hashlib.sha256(f"{key}|{j}".encode()).hexdigest(),j)
            for j in range(27)]
        keep=sorted(priorities)[:n_observed]
        indices=[j for _,j in keep]
        out[i,indices]=visible[i,indices]
    return out


def proper_ranked_score(p,y):
    """Mean RPS over categorical 0..24 future six-site sum."""
    if p.ndim!=2 or p.shape[1]!=25 or y.shape!=(len(p),):
        raise ValueError("Expected [B,25] posterior and [B] burden labels")
    truth=(torch.arange(25,device=p.device)[None,:]>=y[:,None]).to(p.dtype)
    return ((p.cumsum(-1)-truth)**2).sum(-1).mean()/24.


class ConditionallyScoredWM:
    def __init__(self,dynamic=False,mask_mode="full",epochs=18,seed=SEED):
        if mask_mode not in ("full","subsets"):
            raise ValueError("Unknown conditional mask schedule")
        self.dynamic=bool(dynamic)
        self.mask_mode=mask_mode
        self.epochs=int(epochs)
        self.seed=int(seed)

    def fit(self,fit,aux):
        if len(fit)<30 or not aux:
            raise ValueError("Need supervised FIT and real FIT next visits")
        persons={str(r["pid"]) for r in fit}
        if any(str(a["pid"]) not in persons for a in aux):
            raise AssertionError("A DEV patient entered auxiliary training")
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=Full33Net(dynamic=self.dynamic)
        optimizer=torch.optim.AdamW(
            self.net.parameters(),lr=.0008,weight_decay=.025)

        X=sequence_features(fit,())
        site_scores,y,_=labels(fit)
        y=torch.as_tensor(y,dtype=torch.long)
        full=torch.as_tensor([r["future_full33"] for r in fit],
                             dtype=torch.long)
        if full.shape!=(len(fit),33) or torch.any((full<0)|(full>4)):
            raise AssertionError("FIT requires genuinely complete future visits")
        if not torch.equal(full[:,list(HIDX)],
                           torch.as_tensor(site_scores,dtype=torch.long)):
            raise AssertionError("True six-site target alignment failed")
        AX=sequence_features(aux,())
        partial=auxiliary_partial_targets(aux)
        if not torch.any(partial!=-100):
            raise AssertionError("No real observed auxiliary Motor targets")

        for epoch in range(self.epochs):
            self.net.train()
            order=rng.permutation(len(fit))
            for start in range(0,len(order),32):
                ii=order[start:start+32]
                mix,site=self.net(*[v[ii] for v in X])
                prior=hidden_total_posterior(mix,site)
                prior_nll=F.nll_loss(
                    prior.clamp_min(1e-12).log(),y[ii])
                prior_rps=proper_ranked_score(prior,y[ii])
                joint_nll=-masked_joint_loglik(
                    mix,site,full[ii]).mean()/33.

                # Distinct same-visit clinical task, trained ONLY on FIT.
                # Real observed future visible values are excluded from
                # the history encoder; they enter Bayes conditioning here.
                true_visible=real_visible(full[ii])
                observed=training_masks(true_visible,rng,self.mask_mode)
                posterior=hidden_total_posterior(mix,site,observed)
                posterior_nll=F.nll_loss(
                    posterior.clamp_min(1e-12).log(),y[ii])
                posterior_rps=proper_ranked_score(posterior,y[ii])

                # Auxiliary visits may have only some of 27 items
                # measured. Their masked marginal likelihood is valid
                # and never treats unknown six ratings as zeros.
                jj=rng.integers(0,len(aux),size=min(32,len(ii)))
                aux_mix,aux_site=self.net(*[v[jj] for v in AX])
                actual_count=partial[jj].ne(-100).sum(-1).clamp_min(1)
                partial_nll=(-masked_joint_loglik(
                    aux_mix,aux_site,partial[jj])/actual_count).mean()
                loss=(prior_nll+FULL_JOINT_WEIGHT*joint_nll+
                      RPS_WEIGHT*prior_rps+
                      PARTIAL_WEIGHT*partial_nll+
                      CONDITIONAL_WEIGHT*(
                          posterior_nll+RPS_WEIGHT*posterior_rps))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optimizer.step()

        self.net.eval()
        return self

    def predict_both(self,rows,true_full33=None,observed_budget=27):
        """Future forecast is computed strictly before visible target release."""
        with torch.inference_mode():
            mix,site=self.net(*sequence_features(rows,()))
            future=hidden_total_posterior(mix,site).numpy()
            if true_full33 is None:
                return future,None
            observed=real_visible(torch.tensor(true_full33,
                                      dtype=torch.long))
            if observed_budget==27:
                release=observed
            else:
                release=deterministic_eval_mask(
                    rows,observed,observed_budget)
            conditional=hidden_total_posterior(
                mix,site,release).numpy()
            return future,conditional

    def effective_components(self,rows):
        with torch.inference_mode():
            mix,_=self.net(*sequence_features(rows,()))
            weights=F.softmax(mix,dim=-1)
            entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
            return float(entropy.exp().mean())


def run(args):
    if args.arm not in (0,1,2) or not 4<=args.epochs<=100:
        raise ValueError("Allowed arms 0..2; epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve()
    out=args.output.resolve()
    if root==out or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Missing frozen manifest or unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    tasks=read_tasks(root,args.manifest_db)
    people,_=load_cohorts(root,args.manifest_db)
    source={str(p["pid"]):p for p in people}

    def new_model(seed):
        return ConditionallyScoredWM(
            dynamic=args.arm==2,
            mask_mode="full" if args.arm==0 else "subsets",
            epochs=args.epochs,seed=seed)

    prod_fit=(tasks[("prodromal",12)]["fit"]+
              tasks[("prodromal",24)]["fit"])
    prod_fit.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    prod_aux=merge_fit_auxiliary(tasks)
    trained=new_model(SEED+3000).fit(
        attach_true_future(prod_fit,source),prod_aux)
    print("PHASE91_PROD_FIT_DONE",args.arm,len(prod_fit),
          "real_aux_pairs",len(prod_aux),flush=True)

    reports=[];coverage=[]
    for task in TASKS:
        d=tasks[task]
        wm=(trained if task[0]=="prodromal" else new_model(
            SEED+312).fit(attach_true_future(d["fit"],source),
                           d["aux"]))

        # No target values are passed into future inference.
        future,_=wm.predict_both(d["dev"])
        prior_metrics=probabilities_scores(future,d["dev"])
        true_eval=attach_true_future(d["dev"],source)
        true_full=np.asarray([r["future_full33"] for r in true_eval],
                            dtype=np.int64)
        row={"cohort":task[0],"horizon":task[1],"arm":args.arm,
             "fit":len(d["fit"]),"dev":len(d["dev"]),
             "fit_fingerprint":fingerprint(d["fit"]),
             "dev_fingerprint":fingerprint(d["dev"]),
             "training_mask_mode":wm.mask_mode,
             "dynamic":int(wm.dynamic),
             "conditional_loss_weight":CONDITIONAL_WEIGHT,
             "effective_components":wm.effective_components(d["dev"]),
             "fit_real_aux_pairs":(len(prod_aux) if
                          task[0]=="prodromal" else len(d["aux"])),
             **prior_metrics}
        for n_observed in EVAL_OBSERVED_SIZES:
            _,conditioned=wm.predict_both(
                d["dev"],true_full33=true_full,
                observed_budget=n_observed)
            s=probabilities_scores(conditioned,d["dev"])
            prefix=f"same_visit_{n_observed}visible_"
            for key in ("mae_median","rmse_mean","rps","nll",
                        "brier","coverage80","width80"):
                row[prefix+key]=s[key]
            row[prefix+"rps_vs_unconditional"]=(
                s["rps"]-prior_metrics["rps"])

        reports.append(row)
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",reports)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE91_TASK_COMPLETE",task,"arm",args.arm,flush=True)

    (out/"summary.json").write_text(json.dumps({
        "phase":91,"arm":args.arm,
        "completed_tasks":len(coverage),
        "runtime_seconds":round(time.monotonic()-start,2),
        "protocol":"same Phase84–90 globally hashed 80% FIT/20% reused DEV",
        "new_mechanism":"FIT conditional proper scoring of exact hidden6 posterior",
        "observed_mask_sizes_training":(
            [27] if args.arm==0 else list(OBSERVED_SIZES)),
        "observed_mask_sizes_eval":list(EVAL_OBSERVED_SIZES),
        "baselines_reused":"Phase90 actual complete reports; do not rerun same models",
        "caveats":[
            "Future forecast reads only pre-cutoff history",
            "Retrospective same-visit masked reconstruction reads TRUE future visible27",
            "Two tasks have different information budgets, MUST NOT be merged",
            "Training conditionals use FIT complete future exams only",
            "FIT auxiliary visit likelihood is partial with -100 genuinely unmeasured",
            "Mask subset measurements are withheld, not fabricated",
            "No original consumed final TEST and no patient records exported",
            "20% DEV was repeatedly used; these scores are exploratory"],
        "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE91_SUCCESS",args.arm,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",required=True,type=int)
    p.add_argument("--epochs",default=18,type=int)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()
