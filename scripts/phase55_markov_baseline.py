"""Phase55: TRAIN-fit categorical item-transition baselines.

Score 101/UR and ordinary absent fields are never observations. Each
past observed Motor item may inform the next measured score. A
Dirichlet-shrunk conditional matrix provides an honest stronger
reference than fixed 85% "same score" persistence.
"""
from __future__ import annotations
import math
import numpy as np

N_ITEMS=33
GAP_MODES=("pooled","gap")


def group_gap(months):
    if months<=0:raise ValueError("Nonpositive item-to-item gap")
    if months<=8:return 0
    if months<=15:return 1
    return 2


class ItemMarkov:
    def __init__(self,prior_strength=20.0,gap_mode="pooled"):
        if prior_strength<=0 or gap_mode not in GAP_MODES:
            raise ValueError("Invalid transition baseline")
        self.alpha=float(prior_strength)
        self.gap_mode=gap_mode
        self.num_bins=1 if gap_mode=="pooled" else 3
        self.marg=None
        self.conditional=None

    def fit(self,patients):
        if not patients:raise ValueError("Empty fitting patients")
        marginal=np.ones((N_ITEMS,5),dtype=np.float64) # Laplace
        counts=np.zeros((self.num_bins,N_ITEMS,5,5),dtype=np.float64)
        for patient in patients:
            last=[None]*N_ITEMS
            last_month=[None]*N_ITEMS
            for encounter in patient["visits"]:
                current_month=int(encounter["month"])
                values=encounter["score"]
                status=encounter["status"]
                for j,(v,st) in enumerate(zip(values,status)):
                    if st!=1:continue
                    if not 0<=int(v)<=4:raise ValueError("Invalid measured score")
                    v=int(v)
                    marginal[j,v]+=1.
                    if last[j] is not None:
                        bucket=(group_gap(current_month-last_month[j])
                                if self.gap_mode=="gap" else 0)
                        counts[bucket,j,last[j],v]+=1.
                    last[j],last_month[j]=v,current_month
        self.marg=marginal/marginal.sum(-1,keepdims=True)
        denom=counts.sum(-1,keepdims=True)+self.alpha
        self.conditional=(counts+self.alpha*self.marg[None,:,None,:])/denom
        if not np.allclose(self.conditional.sum(-1),1.,atol=1e-8):
            raise AssertionError("Transition probability did not normalize")
        return self

    def predict(self,item,last_score=None,elapsed=None):
        if self.conditional is None:raise RuntimeError("Not fitted")
        if not 0<=item<N_ITEMS:raise ValueError("Unknown item")
        if last_score is None:
            return self.marg[item]
        if not 0<=last_score<=4:raise ValueError("Invalid observed score")
        if self.gap_mode=="gap":
            if elapsed is None:raise ValueError("Missing month interval")
            bucket=group_gap(elapsed)
        else:
            bucket=0
        return self.conditional[bucket,item,last_score]

    def score_patients(self,patients):
        """Return patient-level mean next-visit categorical NLL (RAM only)."""
        output={}
        for patient in patients:
            last=[None]*N_ITEMS
            last_month=[None]*N_ITEMS
            losses=[]
            for index,encounter in enumerate(patient["visits"]):
                curr=encounter["month"]
                for j,(value,st) in enumerate(zip(encounter["score"],
                                                  encounter["status"])):
                    if st!=1:continue
                    if index>0:
                        delta=curr-last_month[j] if last[j] is not None else None
                        prob=self.predict(j,last[j],delta)
                        losses.append(-math.log(max(1e-10,float(prob[value]))))
                    last[j],last_month[j]=int(value),curr
            if not losses:
                raise ValueError("Patient has no measured followup items")
            output[patient["pid"]]=sum(losses)/len(losses)
        return output


def choose_markov(inner_train,inner_dev,
                  alphas=(2.,10.,50.,200.),
                  gap_modes=GAP_MODES):
    """All parameters selected only on inner TRAIN development patients."""
    candidates=[]
    for mode in gap_modes:
        for alpha in alphas:
            model=ItemMarkov(alpha,mode).fit(inner_train)
            losses=model.score_patients(inner_dev)
            candidates.append((sum(losses.values())/len(losses),mode,alpha))
    score,mode,alpha=min(candidates,key=lambda x:(x[0],x[1],x[2]))
    return {"selected_gap_mode":mode,"selected_alpha":alpha,
            "inner_dev_patient_nll":score,
            "candidate_count":len(candidates)}
