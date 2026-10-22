"""Phase59 independent inner verifier and allowlisted RSI config acceptance."""
import hashlib,json
from statistics import mean
from scripts.phase59_grounded_collaboration import (
    JointBelief,propose_evolution
)

def greedy_dev_score(records,world,prior_by_pid,config,q):
    weight,temp,style=config
    estimates=[]
    for row in records:
        prior=prior_by_pid.get(row["pid"])
        w=weight if prior is not None else 0.
        model=JointBelief(world,prior,w,temp)
        observed={}
        for _ in range(q):
            remaining=[item for item in world.items if item not in observed]
            chosen=sorted(remaining,key=lambda k:(
                -model.candidate_value(observed,k,style),k))[0]
            observed[chosen]=row["features"][chosen]
        risk=model.posterior(observed)
        estimates.append((risk,row["label"]))
    if not estimates:raise ValueError("No inner verifier people")
    return mean((risk-y)**2 for risk,y in estimates)

def accepted_revision(backend,world,dev,prior_by_pid,rounds,q):
    """Larger verifier score is worse; strict accept only if improved.

    An outer-evaluation patient is never passed to this method.
    Patient IDs/features are not sent to the LLM, only aggregate score.
    """
    current=(.5,1.,"information_gain")
    current_score=greedy_dev_score(dev,world,prior_by_pid,current,q)
    history=[]
    for i in range(1,rounds+1):
        old_hash=hashlib.sha256(repr(current).encode()).hexdigest()[:16]
        candidate=propose_evolution(
            backend,current,i,{"dev_brier":round(current_score,6),
                               "dev_patients":len(dev),"question_budget":q})
        if candidate is None:
            history.append({"round":i,"parent_hash":old_hash,
                            "proposed_hash":"","accepted":False,
                            "reason":"invalid_or_noop_proposal",
                            "verifier_before":round(current_score,6),
                            "verifier_candidate":""})
            continue
        candidate_score=greedy_dev_score(
            dev,world,prior_by_pid,candidate,q)
        accepted=candidate_score+1e-5<current_score
        history.append({
            "round":i,"parent_hash":old_hash,
            "proposed_hash":hashlib.sha256(repr(candidate).encode()).hexdigest()[:16],
            "accepted":accepted,
            "reason":"inner_dev_brier_improved" if accepted else
                     "verifier_rejected_no_improvement",
            "verifier_before":round(current_score,6),
            "verifier_candidate":round(candidate_score,6)})
        if accepted:current,current_score=candidate,candidate_score
    return current,history
