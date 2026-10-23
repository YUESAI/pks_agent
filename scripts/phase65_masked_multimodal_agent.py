#!/usr/bin/env python3
"""Phase65: fit-fold random-mask trained outcome WM + empirical conditional item WM.

20 standalone CPU arms (5 panels x 4 policies), using real PPMI evidence
and optional offline Qwen. Methodological engineering pilot, NOT learned
causal dynamics or neural weight RSI. Patient identities/records never export.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables,panel_items,episode_rows,PatientTool
from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase64_multimodal_agent import fold_id,conversation
from scripts.phase60_cpu_multistep_wm import auc

SEED=20261018
POLICIES=("greedy","qwen_plain","qwen_wm","qwen_rsi")
TASKS=(("prodromal",12),("prodromal",24),("PD",12),("PD",24))
MIXES=(0.,.25,.5,.75,1.)
SCALES=(.8,1.,1.2)
STYLES=("variance","risk_shift")
BUCKET_COUNT=5


def stable(pid,key):
    return hashlib.sha256(f"phase65|{SEED}|{pid}|{key}".encode()).hexdigest()


def fold_segments(rows,task):
    fit=[r for r in rows if fold_id(r["pid"],task) in (2,3,4)]
    inner=sorted((r for r in rows if fold_id(r["pid"],task)==0),
                 key=lambda r:stable(r["pid"],"dev"))
    audit_cut=len(inner)//2
    tune,audit=inner[:audit_cut],inner[audit_cut:]
    evaluate=[r for r in rows if fold_id(r["pid"],task)==1]
    parts=[{r["pid"] for r in group} for group in (fit,tune,audit,evaluate)]
    if any(parts[i]&parts[j] for i in range(4) for j in range(i+1,4)):
        raise AssertionError("Person leakage across phase65 partitions")
    return fit,tune,audit,evaluate


class MaskedOutcomeWM:
    """Outcome ridge with stochastic observed-subset augmentation.

    Conditional observation probabilities are TRAIN-only empirical
    conditional tables by class-independent bin frequency; NOT a
    sequential generative patient disease-state model.
    """
    def __init__(self,items,q=4,repeats=12,alpha=20.,seed=SEED):
        self.items=tuple(items);self.q=q;self.repeats=repeats
        self.alpha=alpha;self.seed=seed

    def encode(self,measured):
        vals=np.zeros(len(self.items));masks=np.zeros(len(self.items))
        for j,item in enumerate(self.items):
            if item in measured:
                vals[j]=float(measured[item]);masks[j]=1.
        # Each feature is divided by fit observed feature stdev, precomputed.
        return np.concatenate((vals/self.scale,masks))

    def fit(self,fit,unlabelled=()):
        if not fit:raise ValueError("No supervision")
        self.scale=np.ones(len(self.items))
        self.lookup={}
        all_measurements=list(fit)+list(unlabelled)
        for j,item in enumerate(self.items):
            values=sorted(float(r["measured"][item]) for r in all_measurements
                          if item in r["measured"])
            if values:
                self.scale[j]=max(1.,float(np.std(values)))
                # Store actual observed clinical units. No fictitious 0..4
                # substitution for MoCA/UPSIT or other questionnaire fields.
                uniq,counts=np.unique(values,return_counts=True)
                cut=np.linspace(0,len(uniq),min(BUCKET_COUNT,len(uniq))+1,dtype=int)
                groups=[]
                for lo,hi in zip(cut[:-1],cut[1:]):
                    if lo==hi:continue
                    reps=uniq[lo:hi];weights=counts[lo:hi]
                    groups.append((float(np.average(reps,weights=weights)),
                                   float(weights.sum())))
                total=sum(p for v,p in groups)+len(groups)
                self.lookup[item]=[(v,(p+1.)/total) for v,p in groups]
            else:
                self.lookup[item]=[]
        generator=np.random.default_rng(self.seed)
        X=[];Y=[]
        # Inference sees at most q actual items, so train with an identical
        # cardinality distribution rather than fully revealed panels.
        for record in fit:
            observed=record["measured"]
            keys=sorted(k for k in self.items if k in observed)
            for draw in range(self.repeats):
                count=min(len(keys),int(generator.integers(0,self.q+1)))
                sampled=(generator.choice(keys,size=count,replace=False).tolist()
                         if count else [])
                X.append(self.encode({k:observed[k] for k in sampled}))
                Y.append(float(record["burden"]))
        self.scaler=StandardScaler().fit(np.asarray(X))
        self.reg=Ridge(alpha=self.alpha).fit(self.scaler.transform(X),Y)
        self.mean_burden=float(np.mean(Y))
        return self

    def estimate(self,measured):
        x=self.scaler.transform(self.encode(measured)[None,:])
        return max(0.,float(self.reg.predict(x)[0]))

    def action_values(self,observed,remaining,style="variance"):
        if style not in STYLES:raise ValueError("Unknown action criterion")
        prior=self.estimate(observed)
        values={}
        for key in remaining:
            possible=self.lookup[key]
            if not possible:values[key]=0.;continue
            estimates=[]
            for value,prob in possible:
                next_observed=dict(observed);next_observed[key]=value
                estimates.append((prob,self.estimate(next_observed)))
            center=sum(p*x for p,x in estimates)
            if style=="variance":
                score=sum(p*(x-center)**2 for p,x in estimates)
            else:
                score=sum(p*abs(x-prior) for p,x in estimates)
            values[key]=float(score)
        return values


def choose_fallback(remaining,observed,policy,values):
    if policy=="qwen_plain":
        return min(remaining,key=lambda key:stable(key,str(len(observed))))
    return min(remaining,key=lambda key:(-values.get(key,0.),key))


def episode(row,wm,allowed,q,policy,doctor=None,style="variance"):
    actions=set(allowed)&set(row["measured"])
    tool=PatientTool(row["measured"],actions,min(q,len(actions)))
    invalid=0;agreements=0
    while len(tool.observed)<tool.budget:
        remaining=tool.available-set(tool.observed)
        values=wm.action_values(tool.observed,remaining,style)
        best=choose_fallback(remaining,tool.observed,policy,values)
        if policy=="greedy":
            selected=best
        else:
            prompt=conversation(remaining,tool.observed,values,policy,
                                tool.budget-len(tool.observed))
            answer=doctor.generate(prompt)
            selected=extract_action(answer,set(remaining))
            if selected is None:
                selected=best
                invalid+=1
        agreements+=int(selected==best)
        event=tool.request(selected)
        if event["provenance"]!="actually_measured_BL_or_pre_target_R01":
            raise AssertionError("Model imagination reached patient evidence")
    return wm.estimate(tool.observed),len(tool.observed),invalid,agreements


def verifier(wm,tune,audit,allowed,q):
    """Allowlisted deterministic configuration proposal and one-shot veto.

    This is NOT LLM self-modification nor neural WM weights evolution.
    A single development-selected candidate is assessed ONCE on
    independent inner audit people; no audit-loop search.
    """
    cache={}
    def predictions(rows,style):
        key=(id(rows),style)
        if key not in cache:
            cache[key]=[(episode(r,wm,allowed,q,"greedy",style=style)[0],
                         r["burden"]) for r in rows]
        return cache[key]
    def score(rows,cfg):
        style,scale=cfg
        pair=predictions(rows,style)
        return sum((max(0.,x*scale)-y)**2 for x,y in pair)/len(pair)
    baseline=("variance",1.)
    possibilities=[(style,scale) for style in STYLES for scale in SCALES]
    current=score(tune,baseline)
    candidate=min(possibilities,key=lambda c:(score(tune,c),c))
    cand_score=score(tune,candidate)
    before=score(audit,baseline)
    after=score(audit,candidate)
    accepted=(cand_score+1e-4<current and after+1e-4<before)
    selected=candidate if accepted else baseline
    return selected,{"selected_style":selected[0],"selected_scale":selected[1],
                     "candidate_style":candidate[0],
                     "candidate_scale":candidate[1],
                     "accepted":accepted,
                     "tune_baseline_mse":current,"tune_candidate_mse":cand_score,
                     "audit_baseline_mse":before,"audit_candidate_mse":after,
                     "audit_proposals":1}


def metrics(rows,pred,q,task,arm):
    scores=np.array([pred[r["pid"]][0] for r in rows],dtype=float)
    truth=np.array([r["burden"] for r in rows],dtype=float)
    labels=np.array([r["label"] for r in rows],dtype=int)
    risk=1.-np.exp(-scores)
    ranking=sorted(range(len(rows)),key=lambda i:(-risk[i],stable(rows[i]["pid"],"tie")))
    quota=max(1,int(.2*len(rows)+.5))
    captured=int(labels[ranking[:quota]].sum());positive=int(labels.sum())
    return {"cohort":task[0],"horizon":task[1],"arm":arm,
            "patients":len(rows),"positive":positive if positive>=10 else "<10",
            "quota20":quota,"captured":captured if captured>=10 else "<10",
            "recall20":captured/positive if min(captured,positive)>=10 else "",
            "MAE":float(np.mean(np.abs(scores-truth))),
            "RMSE":float(np.mean((scores-truth)**2)**.5),
            "AUROC":auc(risk.tolist(),labels.tolist()),
            "Brier_uncalibrated":float(np.mean((risk-labels)**2)),
            "questions":sum(pred[r["pid"]][1] for r in rows),
            "invalid_qwen_actions":sum(pred[r["pid"]][2] for r in rows),
            "fallback_rate":(sum(pred[r["pid"]][2] for r in rows)/
                             max(1,sum(pred[r["pid"]][1] for r in rows))),
            "greedy_agreement":sum(pred[r["pid"]][3] for r in rows)/
                              max(1,sum(pred[r["pid"]][1] for r in rows))}


def paired_ci(rows,reference,candidate,bootstrap=500,seed=SEED):
    ref=np.array([reference[r["pid"]][0] for r in rows])
    cand=np.array([candidate[r["pid"]][0] for r in rows])
    truth=np.array([r["burden"] for r in rows])
    d=np.abs(cand-truth)-np.abs(ref-truth)
    rng=np.random.default_rng(seed)
    n=len(d)
    samples=np.array([float(np.mean(d[rng.integers(0,n,n)]))
                      for _ in range(bootstrap)])
    return {"paired_MAE_candidate_minus_greedy":float(d.mean()),
            "bootstrap_ci_low":float(np.quantile(samples,.025)),
            "bootstrap_ci_high":float(np.quantile(samples,.975)),
            "bootstrap_samples":bootstrap}


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--questions",type=int,default=4)
    p.add_argument("--mask-repeats",type=int,default=12)
    p.add_argument("--max-new-tokens",type=int,default=40)
    a=p.parse_args()
    if not 0<=a.arm_id<20 or not 1<=a.questions<=12 or a.mask_repeats<2:
        raise ValueError("Invalid Phase65 configuration")
    panel=a.arm_id//4;policy=POLICIES[a.arm_id%4]
    root=a.data_root.resolve();out=a.report_dir.resolve()
    if not (root/"PPMI").is_dir() or not a.manifest_db.is_file():
        raise FileNotFoundError("Missing offline PPMI data")
    if root==out or root in out.parents or out==a.manifest_db.resolve().parent:
        raise ValueError("Unsafe export directory")
    t0=time.perf_counter()
    people,meta=load_cohorts(a.data_root,a.manifest_db)
    evid,qc,coverage=evidence_tables(a.data_root,people)
    patient_lookup={p["pid"]:p for p in people}
    doctor=(TransformersDoctor(a.model_dir,a.max_new_tokens,device="cpu")
            if policy!="greedy" else None)
    outputs=[];contrasts=[];traces=[];counts=[]
    for task in TASKS:
        eligible=[p for p in people if p["cohort"]==task[0]]
        future,_=future_site_pairs(eligible,task[1])
        rows=episode_rows(eligible,future,evid,*task)
        if len(rows)<80:
            counts.append({"cohort":task[0],"horizon":task[1],
                           "eligible":len(rows),"status":"small_task"})
            continue
        fit,tune,audit,eval_rows=fold_segments(rows,task)
        if min(map(len,(fit,tune,audit,eval_rows)))<10:
            counts.append({"cohort":task[0],"horizon":task[1],
                           "eligible":len(rows),"status":"small_split"})
            continue
        all_keys=set().union(*(set(r["measured"]) for r in fit))
        allowed=panel_items(panel,all_keys)
        # Fit outcome head only on fit-labeled subjects. The empirical
        # score-support model also uses additional fit-side patients who
        # have no eligible strict future label, without using labels.
        fit_ids={r["pid"] for r in fit}
        held_ids={r["pid"] for r in tune+audit+eval_rows}
        eligible_ids={r["pid"] for r in rows}
        extra=[{"measured":evid[p["pid"]]} for p in eligible
               if p["pid"] not in eligible_ids and
               p["pid"] not in held_ids and
               fold_id(p["pid"],task) in (2,3,4)]
        wm=MaskedOutcomeWM(allowed,q=a.questions,
                           repeats=a.mask_repeats,seed=SEED+a.arm_id).fit(
                               fit,unlabelled=extra)
        revision=("variance",1.)
        trace={"accepted":False,"audit_proposals":0}
        if policy=="qwen_rsi":
            revision,trace=verifier(wm,tune,audit,allowed,a.questions)
        estimates={};ref={}
        for row in eval_rows:
            selected=episode(row,wm,allowed,a.questions,policy,doctor,
                             style=revision[0])
            estimates[row["pid"]]=(selected[0]*revision[1],*selected[1:])
            ref[row["pid"]]=episode(row,wm,allowed,a.questions,"greedy")
        outputs.append(metrics(eval_rows,estimates,a.questions,task,policy))
        contrasts.append({"cohort":task[0],"horizon":task[1],
                          **paired_ci(eval_rows,ref,estimates)})
        traces.append({"cohort":task[0],"horizon":task[1],**trace})
        counts.append({"cohort":task[0],"horizon":task[1],
                       "eligible":len(rows),"fit":len(fit),
                       "tune":len(tune),"audit":len(audit),
                       "eval":len(eval_rows),"unlabelled_fit":len(extra),
                       "panel_items":len(allowed),"status":"evaluated"})
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/"arm_metrics.csv",outputs)
    write_csv(out/"paired_greedy_bootstrap.csv",contrasts)
    write_csv(out/"cohort_sizes.csv",counts)
    (out/"verifier_trace.json").write_text(json.dumps(traces,indent=2)+"\n")
    report={"status":"Phase65_masked_multimodal_conditional_WM_engineering",
            "arm_id":a.arm_id,"panel":panel,"policy":policy,
            "runtime_seconds":round(time.perf_counter()-t0,2),
            "qwen_generations":doctor.total_prompts if doctor else 0,
            "qwen_tokens":doctor.total_generated_tokens if doctor else 0,
            "source_motor":meta["source_motor_table"],
            "quality_flags":qc,
            "scope":[
                "The learned outcome head is Ridge with stochastic observation masks, not a neural generative dynamics WM.",
                "Empirical clinical-unit item distributions are FIT-only and ignore item-conditional dependence.",
                "RSI is an allowlisted fixed-model calibration/policy-style selection verified once; not neural parameter coevolution.",
                "Baseline/R01 only, real recorded patient-tool responses; no future target leakage.",
                "Qwen plain fallback is label-free hash, not WM; every fallback separately counted.",
                "TRAIN-internal exploratory reused cohort; original Phase44 TEST consumed.",
                "No raw person-level records or prompts exported.",
            ],
            "generated_utc":datetime.now(timezone.utc).isoformat()}
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"completed","arm":a.arm_id,"tasks":len(outputs),
                      "dir":str(out)},indent=2))
if __name__=="__main__":
    main()
