#!/usr/bin/env python3
"""Phase86: continuous-time latent progression with coherent six-site forecasts.

A fixed patient-global 80% FIT / 20% DEV development study.
Three standalone CPU arms:
  0 Phase85 shared-horizon correlated mixture (strong incumbent).
  1 Continuous-time neural latent progression before site-mixture decode.
  2 Identical progression model with a small flow-semigroup constraint.

Prodromal 12m/24m share FIT weights (different observed cutoffs).
PD12 is FIT-only separate. No new data, no previously used final TEST.
No causal/treatment/diagnosed subtype inference should be claimed.
"""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.phase72_ctlatent_wm import sequence_features
from scripts.phase76_reliability_gated_wm import labels,write_csv
from scripts.phase79_proper_scored_ordinal_wm import probabilities_scores
from scripts.phase85_correlated_site_wm import (
    MixtureSiteNet,CorrelatedSiteWM,correlated_total_distribution,
    joint_site_log_probability,read_tasks,fingerprint,SEED,TASKS,
    N_COMPONENTS)

SEMIGROUP_WEIGHT=.25
N_FLOW_STEPS=4


class ContinuousTimeMixtureNet(MixtureSiteNet):
    """Patient-specific latent drift between final observed visit and horizon.

    Existing GRU reads irregular measured Motor history. Neural controlled
    autonomous drift integrates hidden state from observed cutoff to
    the requested future month. Signed drift allows both improvement and
    progression; fixed-step Euler is a differentiable approximation,
    not a mechanistic PD simulator.
    """
    def __init__(self,components=N_COMPONENTS,latent=80):
        super().__init__(components=components)
        if self.cell.hidden_size!=latent:
            raise ValueError("Latent dimensions must match inherited GRU")
        self.flow_norm=nn.LayerNorm(latent)
        self.drift=nn.Sequential(nn.Linear(latent,96),nn.Tanh(),
                                 nn.Linear(96,latent))
        self.flow_strength=nn.Parameter(torch.tensor(-.5))

    def _observed_state(self,x,dt,valid):
        h=x.new_zeros((len(x),self.cell.hidden_size))
        rate=F.softplus(self.rate)
        for t in range(x.shape[1]):
            next_state=self.cell(self.observation(x[:,t]),
                                h*torch.exp(-rate*dt[:,t,None]))
            h=torch.where(valid[:,t,None],next_state,h)
        return h

    def evolve(self,h,years,steps=N_FLOW_STEPS):
        """Euler integration, with signed drift and positive step duration."""
        if steps<1:raise ValueError("Positive number of steps required")
        if years.ndim!=2 or years.shape!=(len(h),1):
            raise AssertionError("Invalid shape for time gaps")
        dt=years/steps
        strength=F.softplus(self.flow_strength)
        for _ in range(steps):
            velocity=torch.tanh(self.drift(self.flow_norm(h)))
            h=h+dt*strength*velocity
        return h

    def forward(self,x,dt,valid,clinical,horizon,
                compute_consistency=False):
        if horizon.shape!=(len(x),2):
            raise AssertionError("Expected true horizon and residual time gap")
        # Feature encoder: horizon=[absolute forecast months/36,
        # gap from last observed visit to target/36]. Express gap in years.
        remaining_years=horizon[:,1:2]*3.
        if (remaining_years<=0).any():
            raise AssertionError("Post-target observation forbidden")
        state=self._observed_state(x,dt,valid)
        future=self.evolve(state,remaining_years)
        context=self.fuse(torch.cat((future,horizon),dim=-1))
        logits=(self.mixture(context),
                self.future_sites(context).reshape(-1,self.components,6,5))
        if not compute_consistency:
            return logits
        # Semigroup numerical consistency for a constant autonomous
        # velocity field: integrate in two half-intervals, and penalize
        # disagreements from integrating directly in one interval.
        intermediate=self.evolve(state,remaining_years/2.)
        split_flow=self.evolve(intermediate,remaining_years/2.)
        penalty=(split_flow-future).square().mean()
        return logits[0],logits[1],penalty


class ProgressionWM:
    def __init__(self,epochs=18,seed=SEED,semigroup_weight=0.):
        self.epochs=epochs;self.seed=seed
        self.semigroup_weight=float(semigroup_weight)
        if self.semigroup_weight<0:raise ValueError("Negative penalty")

    def fit(self,fit):
        if len(fit)<30:raise ValueError("Insufficient FIT samples")
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.net=ContinuousTimeMixtureNet()
        optim=torch.optim.AdamW(self.net.parameters(),
                                lr=.0008,weight_decay=.025)
        X=sequence_features(fit,())
        site,y,_=labels(fit)
        site=torch.as_tensor(site,dtype=torch.long)
        y=torch.as_tensor(y,dtype=torch.long)
        support=torch.arange(25)[None,:]
        for epoch in range(self.epochs):
            order=rng.permutation(len(fit))
            self.net.train()
            for start in range(0,len(order),32):
                ii=order[start:start+32]
                output=self.net(*[x[ii] for x in X],
                      compute_consistency=self.semigroup_weight>0)
                if self.semigroup_weight:
                    mix,sites,flow_error=output
                else:
                    mix,sites=output
                    flow_error=None
                total=correlated_total_distribution(mix,sites)
                joint_nll=-joint_site_log_probability(
                    mix,sites,site[ii]).mean()
                total_nll=F.nll_loss(torch.log(total),y[ii])
                cdf=total.cumsum(-1)
                truth=(support>=y[ii,None]).to(cdf.dtype)
                rps=(cdf-truth).square().sum(-1).mean()/24.
                loss=total_nll+.2*joint_nll+1.5*rps
                if flow_error is not None:
                    loss=loss+self.semigroup_weight*flow_error
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                optim.step()
        self.net.eval()
        return self

    def predict_distribution(self,rows):
        with torch.inference_mode():
            mix,site=self.net(*sequence_features(rows,()))
            return correlated_total_distribution(mix,site).numpy()

    def diagnostics(self,rows):
        with torch.inference_mode():
            X=sequence_features(rows,())
            mix,site,consistency=self.net(*X,compute_consistency=True)
            probabilities=F.softmax(mix,-1)
            entropy=-(probabilities*torch.log(
                probabilities.clamp_min(1e-12))).sum(-1)
            return {
                "effective_components":float(entropy.exp().mean()),
                "semigroup_disagreement":float(consistency),
                "latent_flow_strength":float(
                    F.softplus(self.net.flow_strength))}


def run(args):
    if args.arm not in (0,1,2) or not 4<=args.epochs<=100:
        raise ValueError("Allowed arms 0..2, epochs 4..100")
    torch.backends.mkldnn.enabled=False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root=args.data_root.resolve();out=args.output.resolve()
    if root==out or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output or missing original frozen manifest")
    out.mkdir(parents=True,exist_ok=False)
    tasks=read_tasks(root,args.manifest_db)
    started=time.monotonic()
    combined=(tasks[("prodromal",12)]["fit"]+
              tasks[("prodromal",24)]["fit"])
    combined.sort(key=lambda r:(str(r["pid"]),int(r["horizon"])))
    if args.arm==0:
        shared=CorrelatedSiteWM(args.epochs,seed=SEED+3000).fit(combined)
    else:
        shared=ProgressionWM(args.epochs,seed=SEED+3000,
                            semigroup_weight=(
                                0. if args.arm==1 else SEMIGROUP_WEIGHT)).fit(
                                    combined)
    print("PHASE86_SHARED_FIT_DONE",args.arm,len(combined),flush=True)
    results=[];coverage=[]
    for task in TASKS:
        d=tasks[task]
        if task[0]=="prodromal":
            wm=shared
        else:
            seed=SEED+task[1]+300
            if args.arm==0:
                wm=CorrelatedSiteWM(args.epochs,seed=seed).fit(d["fit"])
            else:
                wm=ProgressionWM(args.epochs,seed=seed,
                      semigroup_weight=(
                          0. if args.arm==1 else SEMIGROUP_WEIGHT)).fit(
                              d["fit"])
        probabilities=wm.predict_distribution(d["dev"])
        scores=probabilities_scores(probabilities,d["dev"])
        diagnostics=(wm.diagnostics(d["dev"])
                      if isinstance(wm,ProgressionWM)
                      else {
                          "effective_components":
                              wm.effective_components(d["dev"]),
                          "semigroup_disagreement":None,
                          "latent_flow_strength":None})
        results.append({
            "cohort":task[0],"horizon":task[1],"arm":args.arm,
            "fit":len(d["fit"]),"dev":len(d["dev"]),
            "fit_fingerprint":fingerprint(d["fit"]),
            "dev_fingerprint":fingerprint(d["dev"]),
            **scores,**diagnostics})
        coverage.append({"cohort":task[0],"horizon":task[1],
                         "fit":len(d["fit"]),"dev":len(d["dev"]),
                         "status":"complete"})
        write_csv(out/"dev_metrics.csv",results)
        write_csv(out/"coverage.csv",coverage)
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","completed_tasks":len(coverage),
            "last_task":list(task)},indent=2)+"\n")
        print("PHASE86_TASK_COMPLETE",task,"arm",args.arm,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":86,"arm":args.arm,"completed_tasks":len(coverage),
        "runtime_sec":round(time.monotonic()-started,2),
        "fit_strategy":"global patient-disjoint 80/20; shared Prodromal 12/24 fit",
        "semigroup_weight":(
            SEMIGROUP_WEIGHT if args.arm==2 else 0.),
        "model":"Phase85 correlated mixture" if args.arm==0
                else "continuous-time latent progression mixture",
        "limitations":[
            "Exploratory previously reused DEV, never independent TEST",
            "Continuous-time neural latent state is not a verified biological state",
            "Semigroup loss is numerical regularization, not causal identification",
            "No patient-level scores or IDs are exported",
            "Previously consumed final TEST excluded"],
        "finished_utc":datetime.now(timezone.utc).isoformat()
    },indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({
        "status":"complete","completed_tasks":len(coverage)},indent=2)+"\n")
    print("PHASE86_SUCCESS",args.arm,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",required=True,type=int)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args())


if __name__=="__main__":
    main()
