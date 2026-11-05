#!/usr/bin/env python3
"""Phase90: partially observed joint 33-item longitudinal Motor world model.

One fixed, patient-disjoint 80% FIT / 20% repeatedly-used DEV split.
Four independent ordinary CPU arms (NO fivefold outer experiments):
0 Phase85 shared static six-hidden-site mixture baseline
1 Phase87 transition-supervised dynamic mixture baseline
2 NEW joint full33 mixture, static latent state
3 NEW joint full33 mixture, continuous-time latent state

Two distinctly labeled evaluation tasks:
- FUTURE forecast: P(six hidden scores total | pre-cutoff 27-item history).
- SAME-VISIT MASKED recovery: P(hidden six | pre-cutoff history,
  27 ACTUALLY MEASURED visible scores at target visit).
The latter is retrospective onsite-to-remote masking, NOT a future
forecast made before that visit; never conflate its metrics with the
first task.

All six hidden-site and 27 visible-site probabilities share the
SAME K-component future joint distribution. FIT auxiliary adjacent
visits provide true partial 27-item likelihood supervision. No DEV
target information enters training. No patient-level output or
previously consumed final TEST is ever written or accessed.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.phase36_masked_site_feasibility import ALL,HIDDEN,VISIBLE
from scripts.phase62_ppmi_multicohort_audit import (
    load_cohorts,HIDX,VIDX)
from scripts.phase73_matched_longitudinal_agent import TOLERANCE
from scripts.phase72_ctlatent_wm import sequence_features
from scripts.phase76_reliability_gated_wm import labels,write_csv
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase85_correlated_site_wm import (
    CorrelatedSiteWM,read_tasks,fingerprint,SEED,TASKS,
    correlated_total_distribution)
from scripts.phase86_continuous_time_progression_wm import (
    ContinuousTimeMixtureNet)
from scripts.phase87_visit_anchored_progression_wm import (
    VisitAnchoredWM,merge_fit_auxiliary)

assert len(ALL)==33 and len(HIDX)==6 and len(VIDX)==27
assert set(HIDX).isdisjoint(VIDX) and set(HIDX)|set(VIDX)==set(range(33))
COMPONENTS=4
WEIGHT_FULL_JOINT=1.
WEIGHT_AUX_PARTIAL=.5


def full_target_from_true_visit(row,source):
    """Find the SAME verified visit selected by established data_rows().

    The complete future 33 score values are used ONLY for FIT
    supervision or DEV evaluation. Never provided to pre-cutoff
    sequence_features(). HIDDEN scores are omitted in masked recovery.
    """
    p=source[str(row["pid"])]
    anchor=int(p["baseline_month"])
    horizon=int(row["horizon"])
    options=[v for v in p["visits"][1:]
             if abs(int(v["month"])-anchor-horizon)<=TOLERANCE[horizon]
             and v["mask_type"]=="complete_33_verified_total"]
    options.sort(key=lambda v:abs(int(v["month"])-anchor-horizon))
    if not options or (len(options)>1 and
       abs(int(options[0]["month"])-anchor-horizon)
       ==abs(int(options[1]["month"])-anchor-horizon)):
        raise AssertionError("Missing or ambiguous verified future target")
    selected=options[0]
    if int(selected["month"])<=int(row["history"][-1]["month"]):
        raise AssertionError("Future target is not after input cutoff")
    if any(selected["status"][i]!=1 for i in range(33)):
        raise AssertionError("Only actually rated full 33-item target permitted")
    full=tuple(int(selected["score"][i]) for i in range(33))
    if any(x<0 or x>4 for x in full):
        raise AssertionError("Invalid measured Motor score")
    if tuple(full[i] for i in HIDX)!=tuple(row["site"]):
        raise AssertionError("Future target conflicts with original six sites")
    return full


def attach_true_future(rows,source):
    """Creates a separate target-only field; never edits history or site."""
    return [{**row,"future_full33":full_target_from_true_visit(row,source)}
            for row in rows]


def auxiliary_partial_targets(aux):
    """True next-visit 27 visible items; 6 onsite hidden sites unmeasured."""
    arr=np.full((len(aux),33),-100,dtype=np.int64)
    for i,row in enumerate(aux):
        values=row["aux_motor"]
        if len(values)!=27:
            raise AssertionError("Unexpected 27-item auxiliary target")
        for j,index in enumerate(VIDX):
            v=int(values[j])
            if v!=-100 and not 0<=v<=4:
                raise AssertionError("Auxiliary item must be measured or -100")
            arr[i,index]=v
    return torch.as_tensor(arr,dtype=torch.long)


def masked_joint_loglik(mixture_logits,full_logits,targets):
    """Correct marginal log P(observed ratings), integrating missing items.

    Full score vector [B,33] uses -100 to mask genuinely unmeasured
    categories. The same component mixing distribution is shared by
    both full-33 and partial-visible-only observations.
    """
    if (mixture_logits.ndim!=2 or full_logits.ndim!=4
        or full_logits.shape[:2]!=mixture_logits.shape
        or full_logits.shape[2:]!=(33,5)
        or targets.shape!=(full_logits.shape[0],33)):
        raise ValueError("Expected [B,K], [B,K,33,5], [B,33]")
    valid=targets.ne(-100)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("No observed clinical ratings in example")
    if torch.any((targets[valid]<0)|(targets[valid]>4)):
        raise ValueError("Ratings must lie in 0..4")
    safe=targets.clamp(0,4)[:,None,:,None].expand(
        -1,full_logits.shape[1],-1,-1)
    log_site=F.log_softmax(full_logits,dim=-1).gather(
        -1,safe).squeeze(-1)
    joint=F.log_softmax(mixture_logits,dim=-1)+(
        log_site*valid[:,None,:]).sum(-1)
    return torch.logsumexp(joint,dim=1)


def hidden_total_posterior(mix_logits,full_logits,visible=None):
    """Compute p(Y_hidden=0..24), optionally conditional on observed visible27.

    Conditioning is only valid AFTER the target visit's visible27 scores
    have genuinely been observed. Hidden six never enter conditioning.
    """
    if full_logits.shape[2:]!=(33,5):
        raise ValueError("Full33 site logits required")
    weights=mix_logits
    if visible is not None:
        if visible.shape!=(len(mix_logits),27):
            raise ValueError("27 visible target observations expected")
        if torch.any((visible!=-100)&((visible<0)|(visible>4))):
            raise ValueError("Invalid future visible observation")
        partial=torch.full((len(mix_logits),33),-100,
                           device=visible.device,dtype=torch.long)
        partial[:,list(VIDX)]=visible
        valid=partial.ne(-100)
        indices=partial.clamp(0,4)[:,None,:,None].expand(
            -1,mix_logits.shape[1],-1,-1)
        site_log=F.log_softmax(full_logits,dim=-1).gather(
            -1,indices).squeeze(-1)
        # Exact posterior over components given only observed visible27.
        log_comp=F.log_softmax(mix_logits,dim=-1)+(
            site_log*valid[:,None,:]).sum(-1)
        weights=log_comp
    return correlated_total_distribution(
        weights,full_logits[:,:,list(HIDX),:])


class Full33Net(ContinuousTimeMixtureNet):
    """One joint observation model for hidden6 and visible27 scores."""
    def __init__(self,dynamic=False):
        super().__init__(components=COMPONENTS)
        self.dynamic=bool(dynamic)
        self.full33_head=nn.Linear(96,COMPONENTS*33*5)
    def forward(self,x,dt,valid,clinical,horizon):
        if horizon.shape!=(len(x),2):
            raise ValueError("Expected horizon [B,2]")
        gap=horizon[:,1:2]*3.
        if not bool(((gap>0)&(gap<=3.)).all()):
            raise ValueError("Post-target history forbidden")
        h=self._observed_state(x,dt,valid)
        if self.dynamic:
            h=self.evolve(h,gap)
        ctx=self.fuse(torch.cat((h,horizon),dim=-1))
        return (self.mixture(ctx),
                self.full33_head(ctx).reshape(-1,COMPONENTS,33,5))


class PartialObservationWM:
    def __init__(self,dynamic=False,epochs=18,seed=SEED):
        self.dynamic=bool(dynamic)
        self.epochs=int(epochs)
        self.seed=int(seed)
    def fit(self,fit,aux):
        if len(fit)<30 or not aux:
            raise ValueError("Need >=30 supervised FIT records and real next visits")
        ids={str(r["pid"]) for r in fit}
        if any(str(a["pid"]) not in ids for a in aux):
            raise AssertionError("Non-FIT patient in auxiliary trajectory")
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=Full33Net(dynamic=self.dynamic)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.0008,
                              weight_decay=.025)
        X=sequence_features(fit,())
        six,y,_=labels(fit)
        y=torch.as_tensor(y,dtype=torch.long)
        full=torch.as_tensor([r["future_full33"] for r in fit],
                             dtype=torch.long)
        if full.shape!=(len(fit),33):
            raise AssertionError("Expected true full33 future supervision")
        if not torch.equal(full[:,list(HIDX)],torch.as_tensor(six)):
            raise AssertionError("Six onsite target scores do not match")
        AX=sequence_features(aux,())
        partial=auxiliary_partial_targets(aux)
        support=torch.arange(25)[None,:]
        for epoch in range(self.epochs):
            self.net.train()
            order=rng.permutation(len(fit))
            for off in range(0,len(order),32):
                ix=order[off:off+32]
                mix,site=self.net(*[v[ix] for v in X])
                p=hidden_total_posterior(mix,site)
                nll=F.nll_loss(p.clamp_min(1e-12).log(),y[ix])
                # Average each observed full33 item log loss to keep
                # auxiliary and full-visit supervision comparable.
                joint=-masked_joint_loglik(mix,site,full[ix]).mean()/33.
                truth=(support>=y[ix,None]).to(p.dtype)
                ranked=(p.cumsum(-1)-truth).square().sum(-1).mean()/24.
                jj=rng.integers(0,len(aux),size=min(32,len(ix)))
                am,al=self.net(*[v[jj] for v in AX])
                observed=partial[jj].ne(-100).sum(-1).clamp_min(1)
                partial_nll=(-masked_joint_loglik(
                    am,al,partial[jj])/observed).mean()
                loss=(nll+WEIGHT_FULL_JOINT*joint+1.5*ranked+
                      WEIGHT_AUX_PARTIAL*partial_nll)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                opt.step()
        self.net.eval()
        return self
    def predict_distribution(self,rows,condition_visible=False):
        with torch.inference_mode():
            mix,site=self.net(*sequence_features(rows,()))
            observed=None
            if condition_visible:
                if any("future_full33" not in r for r in rows):
                    raise ValueError("Masked recovery requires actual current-visit observed visible27")
                observed=torch.tensor([
                    [r["future_full33"][j] for j in VIDX]
                    for r in rows],dtype=torch.long)
            return hidden_total_posterior(mix,site,observed).numpy()
    def effective_components(self,rows):
        with torch.inference_mode():
            mix,_=self.net(*sequence_features(rows,()))
            p=torch.softmax(mix,dim=1)
            return float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum(1)).mean())


def run(args):
    if args.arm not in range(4) or not 4<=args.epochs<=100:
        raise ValueError("Allowed arms 0..3 and 4..100 epochs")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve()
    out=args.output.resolve()
    if out==root or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output or missing original frozen TRAIN")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    tasks=read_tasks(root,args.manifest_db)
    people,_=load_cohorts(root,args.manifest_db)
    source={str(p["pid"]):p for p in people}
    prod_fit=(tasks[("prodromal",12)]["fit"]+
              tasks[("prodromal",24)]["fit"])
    prod_fit.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    prod_aux=merge_fit_auxiliary(tasks)
    if args.arm>=2:
        train=attach_true_future(prod_fit,source)
        shared=PartialObservationWM(dynamic=args.arm==3,
                     epochs=args.epochs,seed=SEED+3000).fit(train,prod_aux)
    elif args.arm==0:
        shared=CorrelatedSiteWM(args.epochs,seed=SEED+3000).fit(prod_fit)
    else:
        shared=VisitAnchoredWM(args.epochs,seed=SEED+3000,
                              aux_weight=.15).fit(prod_fit,prod_aux)
    reports=[];coverage=[]
    for task in TASKS:
        d=tasks[task]
        if task[0]=="prodromal":
            model=shared
        elif args.arm>=2:
            pd_train=attach_true_future(d["fit"],source)
            model=PartialObservationWM(dynamic=args.arm==3,
                          epochs=args.epochs,seed=SEED+312).fit(
                              pd_train,d["aux"])
        elif args.arm==0:
            model=CorrelatedSiteWM(args.epochs,seed=SEED+312).fit(
                d["fit"])
        else:
            model=VisitAnchoredWM(args.epochs,seed=SEED+312,
                                  aux_weight=.15).fit(d["fit"],d["aux"])
        future=model.predict_distribution(d["dev"])
        scored=probabilities_scores(future,d["dev"])
        row={"cohort":task[0],"horizon":task[1],
             "arm":args.arm,"fit":len(d["fit"]),"dev":len(d["dev"]),
             "fit_fingerprint":fingerprint(d["fit"]),
             "dev_fingerprint":fingerprint(d["dev"]),
             "fit_adjacent_real_visits":(len(prod_aux)
                      if task[0]=="prodromal" else len(d["aux"])),
             "effective_components":(
                 model.effective_components(d["dev"])
                 if isinstance(model,PartialObservationWM) else None)}
        row.update(scored)
        # This second outcome is a DIFFERENT masked-recovery task:
        # hidden6 are scored AFTER observing same-visit visible27.
        # No future observation is ever fed to forecast (above).
        conditional=None
        if isinstance(model,PartialObservationWM):
            masked=attach_true_future(d["dev"],source)
            posterior=model.predict_distribution(
                masked,condition_visible=True)
            conditional=probabilities_scores(posterior,d["dev"])
        for key in ("mae_median","rmse_mean","rps","nll",
                    "brier","coverage80","width80"):
            row["same_visit_conditioned_"+key]=(
                conditional[key] if conditional is not None else None)
        row["same_visit_reconstruction_rps_delta"]=(
            conditional["rps"]-scored["rps"] if conditional else None)
        reports.append(row)
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",reports)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE90_TASK_COMPLETE",task,args.arm,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":90,"arm":args.arm,"completed_tasks":len(coverage),
        "runtime_seconds":round(time.monotonic()-started,2),
        "model":("phase85_static" if args.arm==0 else
                 "phase87_transition" if args.arm==1 else
                 "masked_full33_static" if args.arm==2 else
                 "masked_full33_continuous_time"),
        "training":"patient-global 80% FIT, repeatedly explored 20% DEV",
        "evaluation_tasks":[
            "Future six-hidden-site forecast using pre-cutoff history only",
            "Same-visit masked six-hidden-site recovery after REAL future visible27 release"],
        "cautions":[
            "These two evaluation modes have DIFFERENT information budgets",
            "Only original FIT subjects used for learning all 33-site likelihoods",
            "Mixture components are not verified biological subtypes",
            "Same-visit reconstructed scores are not future forecasts",
            "Observed next visits are real clinical events; missing scores remain masked",
            "No patient-level identifiers/values exported; consumed final TEST excluded",
            "Repeatedly used DEV is not confirmatory evidence"],
        "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE90_SUCCESS",args.arm,flush=True)


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
