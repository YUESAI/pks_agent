"""Phase70: horizon-aware fusion of actual time-gap Motor WM predictions.

This module is observational, not causal. At a specified *prospective*
horizon it fuses the Phase69 static burden prediction and the six onsite
Motor-item probabilities from a FIT-trained time-conditioned head.
Hypothetical score draws are used ONLY inside the acquisition planner:
PatientTool still reveals recorded BL / eligible R01 observations only.

Every candidate action is evaluated with batch neural inference (CPU).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scripts.phase36_masked_site_feasibility import ALL, HIDDEN

SITE_INDEX=tuple(ALL.index(k) for k in HIDDEN)

def binary_entropy(p):
    p=np.clip(p,1.e-12,1.-1.e-12)
    return -p*np.log(p)-(1.-p)*np.log1p(-p)

class HorizonFusion:
    def __init__(self, wm, horizon, r01_offset=6, blend=.35):
        if not 1<=horizon<=36:
            raise ValueError("Requested horizon outside temporal WM range")
        if not 0<=r01_offset<horizon:
            raise ValueError("R01 visit must precede chosen target horizon")
        if not 0<=blend<=1:
            raise ValueError("Fusion must be a bounded mixture")
        self.wm=wm
        self.horizon=horizon
        self.r01_offset=r01_offset
        self.blend=float(blend)
        if not wm.fitted:
            raise RuntimeError("World model has not been fit")

    def _predict_many(self, evidence, temperature=1.):
        if temperature<=0:raise ValueError("Invalid probability temperature")
        if not evidence:
            return np.empty(0),np.empty(0)
        x=np.stack([self.wm._encode(obs) for obs in evidence]).astype(np.float32)
        dt=np.array([
            [max(1,self.horizon-self.r01_offset)
             if any(k.startswith("R01_") for k in obs) else self.horizon]
            for obs in evidence],dtype=np.float32)/36.
        with torch.inference_mode():
            tensor=torch.from_numpy(x)
            latent=self.wm.net.trunk(tensor)
            posterior=torch.softmax(
                self.wm.net.outcome(latent)/temperature,dim=-1).numpy()
            base_burden=posterior@self.wm.class_means
            base_risk=1.-posterior[:,0]
            if (self.blend>0 and
                getattr(self.wm,"n_transitions",0)>0):
                future_logits=self.wm.transition(
                    latent,torch.from_numpy(dt))
                predicted=torch.softmax(future_logits,dim=-1).numpy()
                six=predicted[:,SITE_INDEX,:]
                expected=np.sum(six*np.arange(5)[None,None,:],axis=(1,2))
                risk=1.-np.prod(six[:,:,0],axis=1)
                # The temporal head is trained only on observed Motor
                # evidence. Do not invoke its forecast from purely nonmotor
                # cues, for which its supervision has not been established.
                have_motor=np.array([
                    any(k.startswith(("BL_NP3","R01_NP3")) for k in obs)
                    for obs in evidence],dtype=np.float64)
                w=self.blend*have_motor
                base_burden=(1.-w)*base_burden+w*expected
                base_risk=(1.-w)*base_risk+w*risk
        return np.maximum(base_burden,0.),np.clip(base_risk,0.,1.)

    def outcomes(self,observed,temperature=1.):
        y,p=self._predict_many([observed],temperature)
        return float(y[0]),float(p[0])

    def action_scores(self,observed,remaining,style="variance",temperature=1.):
        if style not in ("variance","entropy"):
            raise ValueError("Unsupported research planning objective")
        remaining=tuple(sorted(remaining))
        if any(k in observed for k in remaining):
            raise PermissionError("Already acquired item not available")
        if not remaining:return {}
        with torch.inference_mode():
            tensor=torch.from_numpy(self.wm._encode(observed))[None,:]
            _,logit=self.wm.net(tensor)
            distributions=torch.softmax(logit[0],dim=-1).numpy()
        futures=[observed]
        ranges={}
        weights={}
        for key in remaining:
            i=self.wm.index[key]
            values=self.wm.reps[key]
            pr=distributions[i,:len(values)]
            pr=pr/pr.sum()
            start=len(futures)
            for value in values:
                simulated=dict(observed)
                simulated[key]=float(value)
                futures.append(simulated)
            ranges[key]=(start,len(futures))
            weights[key]=pr
        future_mean,future_risk=self._predict_many(futures,temperature)
        result={}
        for key in remaining:
            a,b=ranges[key]
            probs=weights[key]
            if style=="variance":
                expect=float(probs@future_mean[a:b])
                utility=float(np.sum(probs*(future_mean[a:b]-expect)**2))
            else:
                # True expected decrease in binary outcome uncertainty.
                # If conditional heads are mutually inconsistent this
                # can be negative, and is clipped to zero as a heuristic.
                utility=float(binary_entropy(future_risk[0])-
                              probs@binary_entropy(future_risk[a:b]))
            result[key]=max(0.,utility)
        return result
