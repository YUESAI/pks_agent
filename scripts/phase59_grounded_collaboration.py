"""Phase59: verifier-grounded Doctor, observation likelihood, temporal prior.

Four identical-data policies: plain Qwen; Qwen+observation WM;
Qwen+observation/temporal WM; Qwen+dual WM+verifier-gated RSI.

Only REAL measured onsite V04 visible items are returned, under an
explicit same-visit masking SIMULATION. R01's contemporaneous site
scores are NOT reconstructed as real observations. Qwen never sees
the six hidden actual V04 scores, subject IDs or labels.
"""
from __future__ import annotations
import hashlib,json,math,re
from statistics import mean
from scripts.phase45_world_model import (
    PatientEvidenceService,entropy,sigmoid
)
from scripts.phase46_qwen_doctor import make_messages,extract_action

STYLES=("information_gain","risk_reduction","uncertainty")
WEIGHTS=(0.,.25,.5,.75,1.)
TEMPERATURES=(.7,1.,1.3)


def logit(p):
    z=max(1e-5,min(1-1e-5,float(p)))
    return math.log(z/(1-z))


class JointBelief:
    """Observation likelihood updated by optional temporal prior.

    The temporal prior is obtained using only earlier actual encounters.
    Same-visit item likelihoods use training-site ground truths only.
    Missing/101 items are NEVER passed to posterior as score zero.
    """
    def __init__(self,world,temporal_prior=None,weight=0.,temperature=1.):
        if not world.fitted or not 0<=weight<=1 or temperature<=0:
            raise ValueError("Invalid belief configuration")
        if weight and temporal_prior is None:
            raise ValueError("Temporal weight requires a prior")
        self.world=world
        self.items=world.items
        self.weight=float(weight)
        self.temperature=float(temperature)
        self.prior=float(temporal_prior) if temporal_prior is not None else float(world.prior)

    def posterior(self,observed):
        z=(1-self.weight)*logit(self.world.prior)+self.weight*logit(self.prior)
        for item,value in observed.items():
            if item not in self.world.conditional or value not in range(5):
                raise ValueError("Invalid actually observed clinical item")
            probs=self.world.conditional[item]
            z+=math.log(probs[1][value]/probs[0][value])
        return sigmoid(z/self.temperature)

    def predict_item(self,observed,item):
        if item in observed or item not in self.items:
            raise ValueError("Unavailable next measurement")
        risk=self.posterior(observed)
        choices=self.world.conditional[item]
        return tuple((1-risk)*choices[0][score]+risk*choices[1][score]
                     for score in range(5))

    def information_gain(self,observed,item):
        before=entropy(self.posterior(observed))
        probs=self.predict_item(observed,item)
        after=sum(p*entropy(self.posterior({**observed,item:i}))
                  for i,p in enumerate(probs))
        return max(0.,before-after)

    def expected_risk_reduction(self,observed,item):
        """Expected absolute posterior displacement as a planning heuristic.

        Not a causal disease intervention or measured patient outcome.
        """
        prior=self.posterior(observed)
        probs=self.predict_item(observed,item)
        return sum(p*abs(self.posterior({**observed,item:v})-prior)
                   for v,p in enumerate(probs))

    def uncertainty(self,observed,item):
        p=self.predict_item(observed,item)
        return -sum(x*math.log(max(x,1e-12)) for x in p)

    def candidate_value(self,observed,item,style):
        if style=="information_gain":return self.information_gain(observed,item)
        if style=="risk_reduction":return self.expected_risk_reduction(observed,item)
        if style=="uncertainty":return self.uncertainty(observed,item)
        raise ValueError("Invalid planner objective")


def prepare_messages(mode,remaining,observed,world,style):
    if mode not in ("plain","observation","dual","rsi"):
        raise ValueError("Unknown Doctor experiment arm")
    vals={k:world.candidate_value(observed,k,style) for k in remaining}
    is_plain=(mode=="plain")
    messages=make_messages(remaining,observed,world.posterior(observed),
                           vals,"llm_plain" if is_plain else "llm_wm")
    if not is_plain:
        # Label and purpose of the candidate metric must be true.
        description={
            "information_gain":"expected hidden-outcome entropy reduction",
            "risk_reduction":"expected absolute change in outcome risk",
            "uncertainty":"predictive entropy for the requested item"}
        messages[-1]["content"]+=(
            "\nCandidate score definition: "+description[style]+
            "; higher means greater value under this heuristic."
            "\nThese are MODEL estimates, not measured facts.")
        if mode in ("dual","rsi"):
            messages[-1]["content"]+=(
                "\nRisk prior uses a time-conditioned prediction trained"
                " ONLY on earlier visits; this is a retrospective study"
                " and does not imply a clinical intervention effect.")
    return messages,vals


def episode(row,world,doctor,questions,mode="observation",
            style="information_gain"):
    """Only the Patient tool owns true V04 measurements; LLM sees requests."""
    evidence=PatientEvidenceService(row["features"],world.items,questions)
    invalid=0;agreements=0
    for _ in range(questions):
        remaining=[item for item in world.items if item not in evidence.observed]
        messages,values=prepare_messages(mode,remaining,evidence.observed,
                                         world,style)
        proposal=doctor.generate(messages)
        action=extract_action(proposal,set(remaining))
        best=sorted(remaining,key=lambda a:(-values[a],a))[0]
        if action is None:
            invalid+=1
            action=best  # logged deterministic safety fallback
        agreements+=int(action==best)
        event=evidence.request(action)
        if event["provenance"]!="observed_V04_masking_simulation":
            raise AssertionError("Non-grounded source")
    return {"risk":world.posterior(evidence.observed),
            "invalid":invalid,"agreements":agreements,"queries":questions}


def score_results(rows,results,quota=.20,seed=20261018):
    if len(rows)!=len(results) or not rows:
        raise ValueError("Scores missing")
    aligned=[(r["pid"],r["label"],results[r["pid"]]["risk"])
             for r in rows]
    if len(set(x[0] for x in aligned))!=len(rows):
        raise ValueError("Duplicate subjects")
    positives=sum(y for _,y,_ in aligned)
    if not 0<positives<len(rows):
        raise ValueError("Need both classes for cohort metrics")
    brier=mean((risk-y)**2 for _,y,risk in aligned)
    from scripts.phase45_wm_rsi_agents import roc_auc
    roc=roc_auc([{"pid":pid,"label":lab} for pid,lab,p in aligned],
                {pid:p for pid,lab,p in aligned})
    n=min(len(rows),max(1,int(quota*len(rows)+.5)))
    ranked=sorted(aligned,key=lambda row:(-row[2],hashlib.sha256(
        f"phase59|{seed}|{row[0]}".encode()).hexdigest()))
    return {"patients":len(rows),"positive":positives,"site_quota":n,
            "captured_positives":sum(y for _,y,_ in ranked[:n]),
            "recall_at_20pct":sum(y for _,y,_ in ranked[:n])/positives,
            "auroc":roc,"brier":brier,
            "invalid_actions":sum(x["invalid"] for x in results.values()),
            "total_questions":sum(x["queries"] for x in results.values()),
            "greedy_agreement":sum(x["agreements"] for x in results.values())/
                               sum(x["queries"] for x in results.values())}


def parse_evolver(raw,current):
    """Strictly allowlisted versioned changes; no arbitrary Python execution."""
    possibilities=[str(raw).strip()]
    possibilities+=re.findall(r"\{[^{}]*\}",str(raw),re.S)
    for line in possibilities:
        try:obj=json.loads(line)
        except (ValueError,TypeError):continue
        if not isinstance(obj,dict) or set(obj)!={"weight","temperature","style"}:
            continue
        w=obj.get("weight");t=obj.get("temperature");style=obj.get("style")
        if (type(w) in (float,int) and type(t) in (float,int) and
            w in WEIGHTS and t in TEMPERATURES and style in STYLES):
            proposal=(float(w),float(t),style)
            if proposal!=current:return proposal
    return None


def propose_evolution(doctor,current,round_index,aggregate):
    """Only aggregate verifier scores, no private patient features or labels."""
    prompt=(
        "You are a research configuration proposer; NOT a patient clinician."
        " Improve a constrained clinical information acquisition policy."
        " You must not request patient-level data or edit arbitrary code."
        " Return only JSON with EXACT keys weight, temperature, style."
        " Allowed weight: [0,0.25,0.5,0.75,1]."
        " Allowed temperature: [0.7,1,1.3]."
        " Allowed style: [information_gain,risk_reduction,uncertainty]."
        " weight blends TRAIN-fitted temporal prior into a same-visit"
        " observation likelihood posterior; style changes Doctor guidance."
        " Current configuration: "+json.dumps({"weight":current[0],
            "temperature":current[1],"style":current[2]})+
        ". Current inner verifier summary (aggregate only): "+
        json.dumps(aggregate,sort_keys=True)+
        ". Proposal round "+str(round_index)+"."
        " Do not include any explanation."
    )
    raw=doctor.generate([{"role":"system","content":
        "Return a single valid allowlisted JSON configuration proposal."},
        {"role":"user","content":prompt}])
    return parse_evolver(raw,current)
