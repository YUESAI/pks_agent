#!/usr/bin/env python3
"""Phase66: coherent class-conditioned clinical observation World Model.

Fits a smoothed finite latent (future burden class) mixture for measured
clinical evidence. It is a retrospective probabilistic observation WM, NOT
a learned disease dynamics simulator or a causal intervention model.

The entire conditional p(score_bin | future burden class) is fitted only
from TRAIN-labelled patients. Feature bins and marginal shrinkage can
additionally use FIT-fold baseline-only patients, never heldout IDs.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

N_CLASSES = 4
STYLES = ("variance","entropy")

def outcome_class(burden):
    y=float(burden)
    if y < 0 or not math.isfinite(y):raise ValueError("Bad clinical outcome")
    return 0 if y == 0 else 1 if y <= 1 else 2 if y <= 3 else 3

def entropy(p):
    return -sum(float(v)*math.log(max(1e-12,float(v))) for v in p if v>0)

class ConditionalClinicalWM:
    """Naive class-conditional categorical observation distribution.

    p(y,x1,...,xJ) = p(y) product_j p(binned measured xj|y).
    Missing/unrequested measurements simply contribute no likelihood.
    Given a current posterior, hypothetical answers update the same
    coherent probability distribution, never the actual PatientTool.
    """
    def __init__(self,items,alpha=5.,prior_alpha=1.):
        self.items=tuple(items)
        if not self.items or len(set(self.items))!=len(self.items):
            raise ValueError("Expected distinct real measurement names")
        if alpha<=0 or prior_alpha<=0:raise ValueError("Bad smoothing")
        self.alpha=float(alpha)
        self.prior_alpha=float(prior_alpha)

    def fit(self,labelled,unlabelled=()):
        labelled=list(labelled);unlabelled=list(unlabelled)
        if len(labelled)<30:raise ValueError("Insufficient independent labelled FIT")
        if len(set(r["pid"] for r in labelled))!=len(labelled):
            raise ValueError("Duplicate labelled person")
        label_ids={r["pid"] for r in labelled}
        if any(r.get("pid") in label_ids for r in unlabelled):
            raise ValueError("Overlapping labelled and unlabelled FIT")
        z=np.array([outcome_class(r["burden"]) for r in labelled],dtype=int)
        burden=np.array([r["burden"] for r in labelled],dtype=float)
        class_n=np.bincount(z,minlength=N_CLASSES).astype(float)
        self.prior=(class_n+self.prior_alpha)/(len(z)+N_CLASSES*self.prior_alpha)
        overall_mu=float(burden.mean())
        overall_m2=float(np.mean(burden*burden))
        self.mu=np.array([
            (float(burden[z==k].sum())+self.prior_alpha*overall_mu)/
            (class_n[k]+self.prior_alpha)
            for k in range(N_CLASSES)],dtype=float)
        self.m2=np.array([
            (float((burden[z==k]**2).sum())+self.prior_alpha*overall_m2)/
            (class_n[k]+self.prior_alpha)
            for k in range(N_CLASSES)],dtype=float)
        self.boundaries={}
        self.distributions={}
        self.measurement_support={}
        all_rows=labelled+unlabelled
        for key in self.items:
            available=np.array([float(r["measured"][key]) for r in all_rows
                                if key in r["measured"]],dtype=float)
            if available.size:
                if not np.isfinite(available).all():
                    raise ValueError("Bad measured clinical value")
                # Five bins max, quantile-derived only from FIT information.
                candidate=np.quantile(available,[.2,.4,.6,.8])
                edges=np.unique(candidate)
                # Remove bins whose edge equals maximum: last interval empty.
                edges=edges[edges<available.max()]
                edges=edges[edges>available.min()]
            else:
                edges=np.array([],dtype=float)
            self.boundaries[key]=edges
            nbins=len(edges)+1
            marginal=np.ones(nbins,dtype=float)
            for r in all_rows:
                if key in r["measured"]:
                    b=int(np.searchsorted(edges,float(r["measured"][key]),side="right"))
                    marginal[b]+=1
            marginal=marginal/marginal.sum()
            counts=np.array([self.alpha*marginal for _ in range(N_CLASSES)])
            for r,k in zip(labelled,z):
                if key in r["measured"]:
                    b=int(np.searchsorted(edges,float(r["measured"][key]),side="right"))
                    counts[k,b]+=1
            self.distributions[key]=counts/counts.sum(axis=1,keepdims=True)
            self.measurement_support[key]=int(sum(key in r["measured"]
                                              for r in labelled))
        self.fitted=True
        return self

    def posterior(self,observed,temperature=1.):
        if not getattr(self,"fitted",False):raise RuntimeError("Not fitted")
        if temperature<=0:raise ValueError("Invalid temperature")
        logw=np.log(self.prior)
        for key,value in observed.items():
            if key not in self.distributions:raise ValueError("Unknown observed tool")
            b=int(np.searchsorted(self.boundaries[key],float(value),side="right"))
            logw+=np.log(self.distributions[key][:,b])
        logw=logw/temperature
        p=np.exp(logw-logw.max())
        return p/p.sum()

    def outcomes(self,observed,temperature=1.):
        p=self.posterior(observed,temperature)
        mean=float(p@self.mu)
        risk=float(1.-p[0])
        return mean,risk

    def predictive_bin(self,observed,key,temperature=1.):
        if key in observed:raise PermissionError("Repeated measurement")
        post=self.posterior(observed,temperature)
        return post@self.distributions[key]

    def branch(self,post,key,index):
        likelihood=self.distributions[key][:,index]
        weights=post*likelihood
        weights=weights/weights.sum()
        return weights

    def class_variance(self,post):
        mu=float(post@self.mu)
        return max(0.,float(post@self.m2)-mu*mu)

    def action_scores(self,observed,available,style="variance",temperature=1.):
        if style not in STYLES:raise ValueError("Bad acquisition objective")
        p=self.posterior(observed,temperature)
        pre=self.class_variance(p) if style=="variance" else entropy(p)
        result={}
        for key in available:
            probs=p@self.distributions[key]
            future=0.
            for b,q in enumerate(probs):
                if q<=0:continue
                after=self.branch(p,key,b)
                future+=float(q)*(self.class_variance(after) if
                                  style=="variance" else entropy(after))
            result[key]=max(0.,pre-future)
        return result
