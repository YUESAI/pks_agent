"""Phase58 matched OOF decision metrics; no patient-level exports.

Same fixed 20% selection quota, original Phase56 tie-break and
identical patient-bootstrap draws for all candidate strategies.
Bootstrap treats each sampled patient as a resampled unit; fitting
and strategy selection are NOT repeated inside bootstrap.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np

def tie_key(pid,seed):
    return int.from_bytes(hashlib.sha256(
        f"Phase56|{seed}|{pid}".encode()).digest()[:8],"big")

def auc(scores,labels):
    x=np.asarray(scores,dtype=np.float64)
    y=np.asarray(labels,dtype=np.int64)
    n=len(x);p=int(y.sum())
    if p==0 or p==n:return float("nan")
    order=np.argsort(x,kind="stable")
    sorted_x=x[order]
    _,first,counts=np.unique(sorted_x,return_index=True,return_counts=True)
    ranks=np.repeat(first+1+(counts-1)/2,counts)
    rank_sum=float(ranks[y[order]==1].sum())
    return (rank_sum-p*(p+1)/2)/(p*(n-p))

def select(scores,tie,budget=.2):
    score=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(score).all():raise ValueError("Nonfinite risk")
    n=len(score)
    count=min(n,max(1,int(budget*n+.5)))
    order=np.lexsort((np.asarray(tie,dtype=np.uint64),-score))
    return order[:count]

def quantile(values,prob):
    sorted_values=np.sort(np.asarray(values,dtype=np.float64))
    return float(np.quantile(sorted_values,prob))

def compute_matrix(scores,labels,ids,seed,nboot=500,budget=.2,
                   baseline="visible27_sum"):
    """Return ONLY aggregate tables. Caller retains private OOF data in RAM."""
    names=list(scores)
    if len(set(names))!=len(names) or baseline not in names:
        raise ValueError("Baseline missing or duplicate strategies")
    if len(set(ids))!=len(ids) or len(ids)<30:
        raise ValueError("Duplicate/insufficient patient records")
    y=np.asarray([labels[p] for p in ids],dtype=np.int64)
    if not set(y.tolist())<={0,1} or not 0<int(y.sum())<len(y):
        raise ValueError("Need binary mixed patient labels")
    if nboot<100:raise ValueError("Bootstrap repetitions too small")
    r={}
    for name,predictions in scores.items():
        if set(predictions)!=set(ids):
            raise ValueError("Unmatched per-patient OOF cohort for "+name)
        r[name]=np.asarray([predictions[p] for p in ids],dtype=np.float64)
        if not np.isfinite(r[name]).all():
            raise ValueError("Nonfinite score in "+name)
    hashes=np.asarray([tie_key(pid,seed) for pid in ids],dtype=np.uint64)
    selected={name:set(select(score,hashes,budget).tolist())
              for name,score in r.items()}
    def stat(values,truth,ties):
        idx=select(values,ties,budget)
        p=int(truth.sum())
        return (float(truth[idx].sum()/p) if p else float("nan"),
                auc(values,truth))
    base_stats={name:stat(values,y,hashes)
                for name,values in r.items()}
    rng=np.random.default_rng(seed+5801)
    comparisons=[(a,baseline) for a in names if a!=baseline]
    for pair in (("latent_wm_64","markov_tuned"),
                 ("latent_wm_128","markov_tuned"),
                 ("wm_markov_blend","markov_tuned"),
                 ("latent_wm_128","latent_wm_64")):
        if pair[0] in names and pair[1] in names and pair not in comparisons:
            comparisons.append(pair)
    bootstrap={pair:([],[]) for pair in comparisons}
    valid=0
    for _ in range(nboot):
        draw=rng.integers(0,len(ids),size=len(ids))
        outcome=y[draw]
        if not 0<int(outcome.sum())<len(outcome):
            continue
        tie=hashes[draw]
        measured={name:stat(values[draw],outcome,tie)
                  for name,values in r.items()}
        for pair,(rec,aroc) in bootstrap.items():
            left,right=pair
            rec.append(measured[left][0]-measured[right][0])
            aroc.append(measured[left][1]-measured[right][1])
        valid+=1
    if valid<100:raise ValueError("Too few valid patient bootstrap replicates")
    k=min(len(ids),max(1,int(budget*len(ids)+.5)))
    metrics=[]
    for name in names:
        chosen=selected[name]
        metrics.append({
            "strategy":name,"patients":len(ids),
            "positives":int(y.sum()),"selected_k":k,
            "captured_positive":int(y[list(chosen)].sum()),
            "recall_at_k":round(base_stats[name][0],6),
            "auroc":round(base_stats[name][1],6),
            "random_expected_positive":round(k*float(y.mean()),5),
            "random_expected_recall":round(k/len(ids),6)
        })
    paired=[]
    for (left,right),(rec,aroc) in bootstrap.items():
        paired.append({
            "candidate":left,"reference":right,"bootstrap_draws":valid,
            "delta_recall_at_k":round(base_stats[left][0]-base_stats[right][0],6),
            "recall_diff_ci95_low":round(quantile(rec,.025),6),
            "recall_diff_ci95_high":round(quantile(rec,.975),6),
            "delta_auroc":round(base_stats[left][1]-base_stats[right][1],6),
            "auroc_diff_ci95_low":round(quantile(aroc,.025),6),
            "auroc_diff_ci95_high":round(quantile(aroc,.975),6)
        })
    overlaps=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            intersection=len(selected[a]&selected[b])
            union=len(selected[a]|selected[b])
            overlaps.append({
                "strategy_a":a,"strategy_b":b,
                "selected_k":k,
                "selected_intersection":intersection if intersection>=10 or intersection==0 else "<10",
                "selected_jaccard":round(intersection/union,6)
                    if intersection>=10 or intersection==0 else "",
                "both_capture_positive":(
                    lambda count: count if count==0 or count>=10 else "<10"
                )(int(y[list(selected[a]&selected[b])].sum()))
                    if intersection>=10 or intersection==0 else "<10"
            })
    return metrics,paired,overlaps
