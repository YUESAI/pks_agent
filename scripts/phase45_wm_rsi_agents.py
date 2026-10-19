#!/usr/bin/env python3
"""Phase45 CPU-first: observation World Model + agents + verifier-guided RSI-lite.

Only frozen PPMI TRAIN. Mask complete same-visit V04 Motor examinations.
Inner development data select candidate policy and WM updates; independent
outer patient folds evaluate all systems. Patient-level data never exported.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase45_world_model import (
    ObservationWorldModel,PatientEvidenceService,query_policy
)

ALPHAS=(.25,1.,4.,16.)
TEMPERATURES=(.7,1.,1.5)
PLANNERS=("static","dynamic","variance","random")
CAPACITY=.20


def hashkey(seed,value):
    return hashlib.sha256(f"phase45|{seed}|{value}".encode()).hexdigest()


def quota(n):
    return max(1,min(n,int(math.floor(CAPACITY*n+.5))))


def outer_assignment(records,folds,seed):
    mapping={}
    for cls in (0,1):
        names=sorted((r["pid"] for r in records if r["label"]==cls),
                     key=lambda pid:hashkey(seed,pid))
        for i,pid in enumerate(names):mapping[pid]=i%folds
    if len(mapping)!=len(records) or any(i not in mapping.values() for i in range(folds)):
        raise ValueError("Invalid patient cross-validation assignment")
    return mapping


def load_train(root,manifest):
    from scripts.phase37_masked_site_benchmark import load_site_episodes
    from scripts.phase36_masked_site_feasibility import VISIBLE
    episodes,qc,source=load_site_episodes(root,manifest)
    records=[{"pid":r["pid"],
              "features":{item:int(r["extra"][item]) for item in VISIBLE},
              "label":int(r["hidden"]>0),"hidden":float(r["hidden"])}
             for r in episodes]
    if len({r["pid"] for r in records})!=len(records):
        raise ValueError("Repeated subject")
    return records,tuple(VISIBLE),qc,source.name


def risk_scores(world,records,items,config,questions,seed):
    result={}
    for r in records:
        patient=PatientEvidenceService(r["features"],items,questions)
        risk,n=query_policy(world,patient,config[2],questions,r["pid"],seed)
        if n!=questions or len(patient.observed)!=questions:
            raise RuntimeError("Patient request budget failure")
        result[r["pid"]]=risk
    return result


def selection(records,risk,seed):
    return set(sorted((r["pid"] for r in records),
                      key=lambda pid:(-risk[pid],hashkey(seed,pid)))[:quota(len(records))])


def roc_auc(records,risk):
    pos=[risk[r["pid"]] for r in records if r["label"]==1]
    neg=[risk[r["pid"]] for r in records if r["label"]==0]
    if not pos or not neg:return None
    return sum(1 if x>y else .5 if x==y else 0 for x in pos for y in neg)/(len(pos)*len(neg))


def measure(records,risk,seed,probabilistic):
    if set(risk)!={r["pid"] for r in records}:raise ValueError("Incomplete predictions")
    selected=selection(records,risk,seed)
    truth={r["pid"]:r for r in records}
    npos=sum(r["label"] for r in records)
    total_burden=sum(r["hidden"] for r in records)
    if npos==0:raise ValueError("No positive labels")
    captured=sum(truth[pid]["label"] for pid in selected)
    burden=sum(truth[pid]["hidden"] for pid in selected)
    return {
        "n":len(records),"positives":npos,"selected":len(selected),
        "captured":captured,"recall":captured/npos,
        "precision":captured/len(selected),
        "burden_recall":burden/total_burden if total_burden else 0.,
        "auroc":roc_auc(records,risk),
        "brier":mean((risk[r["pid"]]-r["label"])**2 for r in records)
                if probabilistic else None,
        "chosen":selected}


def inner_split(records,seed):
    train=[];dev=[]
    for label in (0,1):
        group=sorted((r for r in records if r["label"]==label),
                     key=lambda r:hashkey(seed,r["pid"]))
        n=max(1,int(round(.25*len(group))))
        dev+=group[:n];train+=group[n:]
    if len(train)<30 or len(dev)<15:return ValueError("Too few independent inner samples")
    return train,dev


def mutation(config,track,rng):
    options=[]
    if track in ("wm","joint"):
        options += [(a,config[1],config[2]) for a in ALPHAS if a!=config[0]]
        options += [(config[0],t,config[2]) for t in TEMPERATURES if t!=config[1]]
    if track in ("policy","joint"):
        options += [(config[0],config[1],m) for m in PLANNERS if m!=config[2]]
    return rng.choice(options)


def evolve(training,items,rounds,questions,seed,outer,track):
    fit,dev=inner_split(training,seed+1000*outer)
    rng=random.Random(seed+997*outer+{"policy":31,"wm":59,"joint":83}[track])
    incumbent=(1.,1.,"static")
    model_cache={}
    def model(config):
        key=config[:2]
        if key not in model_cache:
            model_cache[key]=ObservationWorldModel(
                items,alpha=key[0],temperature=key[1]).fit(fit)
        return model_cache[key]
    def fitness(config):
        scores=risk_scores(model(config),dev,items,config,questions,seed)
        s=measure(dev,scores,seed,True)
        return s["recall"]-.2*s["brier"]
    incumbent_fitness=fitness(incumbent)
    trace=[]
    for gen in range(1,rounds+1):
        candidate=mutation(incumbent,track,rng)
        objective=fitness(candidate)
        accepted=objective>incumbent_fitness+1e-9
        trace.append({"outer_fold":outer,"track":track,"generation":gen,
                      "candidate_alpha":candidate[0],
                      "candidate_temp":candidate[1],
                      "candidate_planner":candidate[2],
                      "candidate_dev_fitness":round(objective,6),
                      "incumbent_dev_fitness":round(incumbent_fitness,6),
                      "accepted":accepted})
        if accepted:incumbent,incumbent_fitness=candidate,objective
    return incumbent,trace


def conditional_bootstrap(records,a,b,reps,seed):
    rng=random.Random(seed)
    labels={r["pid"]:r["label"] for r in records}
    ids=list(labels);draws=[]
    for _ in range(reps):
        sample=[ids[rng.randrange(len(ids))] for unused in ids]
        npos=sum(labels[pid] for pid in sample)
        if npos:
            draws.append(sum(labels[pid]*(int(pid in a)-int(pid in b))
                             for pid in sample)/npos)
    draws.sort()
    if len(draws)<reps*.95:raise ValueError("Insufficient bootstrap outcomes")
    def quantile(p):
        at=p*(len(draws)-1);lo=int(at);hi=min(lo+1,len(draws)-1)
        return draws[lo]*(hi-at)+draws[hi]*(at-lo)
    return quantile(.025),quantile(.975)


def export(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(records,items,out,folds=3,questions=8,rounds=6,seed=20261018,
        bootstraps=300):
    out=Path(out).resolve()
    if len(records)<80 or len({r["pid"] for r in records})!=len(records):
        raise ValueError("Too few distinct TRAIN subjects")
    if not 1<=questions<=len(items) or not 2<=folds<=5 or rounds<1 or bootstraps<100:
        raise ValueError("Invalid experiment parameters")
    if len(set(r["label"] for r in records))<2:raise ValueError("Both classes required")
    assignment=outer_assignment(records,folds,seed)
    models=("random_topk","visible_sum_27","wm_full_27","random_questions",
            "static_questions","dynamic_questions","rsi_policy","rsi_wm","rsi_joint")
    oof={name:{} for name in models}
    traces=[];selected_configs=[]
    for fold in range(folds):
        train=[r for r in records if assignment[r["pid"]]!=fold]
        held=[r for r in records if assignment[r["pid"]]==fold]
        base=ObservationWorldModel(items).fit(train)
        configs={}
        for track in ("policy","wm","joint"):
            config,history=evolve(train,items,rounds,questions,seed,fold,track)
            configs[track]=config;traces.extend(history)
            selected_configs.append({"outer_fold":fold,"track":track,
                "alpha":config[0],"temperature":config[1],"planner":config[2],
                "fit_n":len(train),"held_n":len(held)})
        for r in held:
            pid=r["pid"]
            oof["random_topk"][pid]=int(hashkey(seed,"random|"+pid)[:12],16)/16**12
            oof["visible_sum_27"][pid]=sum(r["features"].values())
            oof["wm_full_27"][pid]=base.posterior(r["features"])
        for name,planner in (("random_questions","random"),
                             ("static_questions","static"),
                             ("dynamic_questions","dynamic")):
            oof[name].update(risk_scores(base,held,items,(1.,1.,planner),questions,seed))
        for track,config in configs.items():
            tuned=ObservationWorldModel(items,config[0],config[1]).fit(train)
            oof["rsi_"+track].update(
                risk_scores(tuned,held,items,config,questions,seed))
    results=[];chosen={}
    for name in models:
        probability=name not in ("random_topk","visible_sum_27")
        stat=measure(records,oof[name],seed,probability)
        chosen[name]=stat.pop("chosen")
        results.append({"model":name,
                        "questions_per_person":(0 if name=="random_topk" else
                            len(items) if name in ("visible_sum_27","wm_full_27") else questions),
                        **{key:round(val,6) if isinstance(val,float) else val
                           for key,val in stat.items()}})
    comparisons=[]
    for reference in ("random_topk","visible_sum_27","static_questions",
                      "dynamic_questions","rsi_policy","rsi_wm"):
        a,b=chosen["rsi_joint"],chosen[reference]
        truth={r["pid"]:r["label"] for r in records}
        point=sum(truth[p]*(int(p in a)-int(p in b)) for p in truth)/sum(truth.values())
        low,high=conditional_bootstrap(records,a,b,bootstraps,seed+len(reference))
        comparisons.append({"model":"rsi_joint","reference":reference,
                            "delta_recall":round(point,6),
                            "conditional_bootstrap_lower":round(low,6),
                            "conditional_bootstrap_upper":round(high,6)})
    out.mkdir(parents=True,exist_ok=True)
    for file,rows in (("model_metrics.csv",results),
                      ("paired_policy_contrasts.csv",comparisons),
                      ("evolution_trace.csv",traces),
                      ("selected_configs.csv",selected_configs)):
        export(out/file,rows)
    summary={
        "status":"EXPLORATORY_TRAIN_ONLY_same_V04_observation_WM_and_RSI_lite",
        "n_train_episodes":len(records),
        "hidden_six_positive":sum(r["label"] for r in records),
        "visible_items":len(items),"questions_per_agent":questions,
        "onsite_budget_fraction":CAPACITY,"outer_folds":folds,
        "rsi_generations":rounds,"seed":seed,"bootstraps":bootstraps,
        "rsi_scope":"offline proposals of WM alpha, temperature, and Doctor planner; inner verifier accepts or rejects",
        "caveats":[
            "World Model predicts SAME-VISIT observation distribution, not longitudinal Parkinson disease dynamics.",
            "Patient Agent releases only recorded V04 observable values, never hidden labels.",
            "V04 six-item acquisition is an idealized simulation, not an actual R01 referral.",
            "RSI-lite is constrained hyperparameter/policy mutation, not autonomous executable self-rewriting.",
            "Outer cross-validation uses frozen TRAIN people only; VAL and consumed TEST stay unopened.",
            "Q=27 baselines use more question actions than q-budget agents; this is an explicit cost/accuracy tradeoff.",
            "A successful train-only pilot is not a fresh independent evaluation.",
            "No patient-level clinical data, model predictions, or identifiers exported."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+chr(10))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--folds",type=int,default=3)
    p.add_argument("--questions",type=int,default=8)
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--bootstraps",type=int,default=300)
    a=p.parse_args()
    root,manifest,out=(Path(s).expanduser().resolve()
                       for s in (a.data_root,a.manifest_db,a.report_dir))
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing PPMI source or frozen manifest")
    if out==root or root in out.parents or out==manifest:
        raise ValueError("Unsafe output directory")
    records,items,qc,source=load_train(root,manifest)
    result=run(records,items,out,a.folds,a.questions,a.rounds,a.seed,a.bootstraps)
    print(json.dumps({**result,"source_csv":source,"quality_flag_categories":len(qc)},
                     indent=2,ensure_ascii=False))
    print("Upload aggregate report ONLY. Raw PPMI and patient SQLite remain private.")

if __name__=="__main__":
    main()
