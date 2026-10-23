#!/usr/bin/env python3
"""Phase64 CPU multimodal WM + real Qwen Doctor + constrained RSI.

20 jobs = 5 panels x 4 policies. Same frozen TRAIN Prodromal, PD separately.
Target: recorded future six motor-item burden at 12/24 months. This is
retrospective information acquisition on actually recorded BL/R01 data;
not a new clinical test, not causal policy learning or neural-weight RSI.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time,random
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables,panel_items,episode_rows,PatientTool,bucket
from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action
from scripts.phase60_cpu_multistep_wm import auc

PANELS=range(5)
POLICIES=("greedy","qwen_plain","qwen_wm","qwen_rsi")
TASKS=(("prodromal",12),("prodromal",24),("PD",12),("PD",24))
SEED=20261018

def fold_id(pid,task):
    h=hashlib.sha256(f"phase64|{SEED}|{task}|{pid}".encode()).hexdigest()
    return int(h[:8],16)%5

class ClinicalWM:
    """TRAIN-only masked ridge predictive WM + feature conditional generator.

    Prediction is a bounded future burden mean. Candidate feature values
    generated from TRAIN observed empirical distributions, NEVER real answers.
    """
    def __init__(self,items,alpha=10.):
        self.items=tuple(items);self.alpha=alpha
    def _encode(self,observed):
        out=[]
        for key in self.items:
            present=key in observed
            value=observed.get(key,0)
            if present: value=float(value)
            out.extend((value if present else 0.,float(present)))
        return out
    def fit(self,rows):
        self.distributions={}
        for key in self.items:
            count=np.ones(5)
            for row in rows:
                if key in row["measured"]:
                    count[bucket(key,row["measured"][key])]+=1
            self.distributions[key]=count/count.sum()
        x=np.asarray([self._encode(r["measured"]) for r in rows])
        y=np.asarray([r["burden"] for r in rows])
        self.scaler=StandardScaler().fit(x)
        self.model=Ridge(alpha=self.alpha).fit(self.scaler.transform(x),y)
        self.baseline=float(y.mean())
        return self
    def predict(self,observed):
        x=self.scaler.transform([self._encode(observed)])
        return max(0.,float(self.model.predict(x)[0]))
    def action_value(self,observed,key):
        current=self.predict(observed)
        probs=self.distributions[key]
        possible=[]
        for v,p in enumerate(probs):
            changed=dict(observed)
            # Bucket representatives are unverified hypothetical values, used
            # in local model evaluation only, NEVER passed to PatientTool.
            changed[key]=v
            possible.append((p,self.predict(changed)))
        expectation=sum(w*x for w,x in possible)
        return sum(w*(x-expectation)**2 for w,x in possible)
    def predict_risk(self,observed):
        return 1.-math.exp(-self.predict(observed))

def partition(rows,task):
    fit=[r for r in rows if fold_id(r["pid"],task)!=0]
    dev=[r for r in rows if fold_id(r["pid"],task)==0]
    ranked=sorted(dev,key=lambda r:hashlib.sha256(
        ("phase64dev|"+r["pid"]).encode()).hexdigest())
    half=len(ranked)//2
    tune,audit=ranked[:half],ranked[half:]
    if len(fit)<60 or min(len(tune),len(audit))<10:
        raise ValueError("Insufficient frozen prospective protocol folds")
    return fit,tune,audit

def conversation(available,observed,values,mode,budget):
    allowed=sorted(available)
    msg=("Research-only simulated historical evidence acquisition. "
         "Request one actually available BL or pre-target R01 clinical test. "
         "No diagnosis, future label or subject ID is available. "
         "Return ONLY strict JSON {\"ask\":\"ITEM\"}. "
         "Allowed actions: "+json.dumps(allowed)+
         ". Already released measured evidence: "+
         json.dumps(observed,sort_keys=True)+
         ". Remaining budget: "+str(budget)+".")
    if mode!="qwen_plain":
        best=sorted(available,key=lambda k:(-values[k],k))[:8]
        msg+=("\nWorld Model hypothetical expected predictive change scores "
              "(NOT observations): "+
              json.dumps({k:round(values[k],5) for k in best})+
              ". This guidance is uncertain.")
    return [{"role":"system","content":
             "Choose exactly one allowed clinical data request; no invented data."},
            {"role":"user","content":msg}]

def simulate(row,wm,allowed,q,policy,backend,weight=1.):
    available=set(allowed)&set(row["measured"])
    tool=PatientTool(row["measured"],available,min(q,len(available)))
    invalid=0
    while len(tool.observed)<tool.budget:
        remaining=tool.available-set(tool.observed)
        values={k:wm.action_value(tool.observed,k)*weight for k in remaining}
        best=min(remaining,key=lambda k:(-values[k],k))
        if policy=="greedy":selected=best
        else:
            prompt=conversation(remaining,tool.observed,values,policy,
                                tool.budget-len(tool.observed))
            answer=backend.generate(prompt)
            selected=extract_action(answer,set(remaining))
            if selected is None:
                selected=best;invalid+=1
        event=tool.request(selected)
        if event["provenance"]!="actually_measured_BL_or_pre_target_R01":
            raise AssertionError("Imagined answer entered evidence")
    return wm.predict(tool.observed),len(tool.observed),invalid

def verifier(wm,tune,audit,allowed,q):
    # Candidate revision is a safe linear calibration, not network weight editing.
    # Deterministic independent tune selects proposal; audit vetoes once.
    options=(.75,1.,1.25)
    def loss(rows,w):
        return sum((max(0.,w*simulate(
            r,wm,allowed,q,"greedy",None)[0])-r["burden"])**2
                   for r in rows)/len(rows)
    choices={w:loss(tune,w) for w in options}
    proposed=min(options,key=lambda w:(choices[w],w))
    before=loss(audit,1.)
    after=loss(audit,proposed)
    accepted=choices[proposed]+1e-5<choices[1.] and after+1e-5<before
    return (proposed if accepted else 1.),{
        "proposal":proposed,"accepted":accepted,
        "tune_base":choices[1.],"tune_candidate":choices[proposed],
        "audit_base":before,"audit_candidate":after,
        "audited_candidates":1}

def metrics(rows,pred,q,policy,task):
    vals=np.asarray([pred[r["pid"]][0] for r in rows])
    truth=np.asarray([r["burden"] for r in rows])
    binary=np.asarray([r["label"] for r in rows])
    risk=1-np.exp(-np.maximum(vals,0))
    order=sorted(range(len(rows)),key=lambda i:(-risk[i],rows[i]["pid"]))
    k=max(1,int(.2*len(rows)+.5))
    found=int(binary[order[:k]].sum())
    pos=int(binary.sum())
    return {"cohort":task[0],"horizon":task[1],"policy":policy,
            "patients":len(rows),"positive":pos if pos>=10 else "<10",
            "questions_mean":float(np.mean([pred[r["pid"]][1] for r in rows])),
            "invalid_calls":sum(pred[r["pid"]][2] for r in rows),
            "MAE":float(np.mean(np.abs(vals-truth))),
            "RMSE":float(np.mean((vals-truth)**2)**.5),
            "AUROC":auc(risk.tolist(),binary.tolist()),
            "Brier_uncalibrated":float(np.mean((risk-binary)**2)),
            "recall20":found/pos if found>=10 and pos>=10 else "",
            "captured":found if found>=10 else "<10"}

def write_csv(path,records):
    if not records:return
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--questions",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=32)
    a=p.parse_args()
    if not 0<=a.arm_id<20 or not 1<=a.questions<=12:
        raise ValueError("Invalid full experiment config")
    panel=a.arm_id//4
    policy=POLICIES[a.arm_id%4]
    if not (a.data_root/"PPMI").is_dir() or not a.manifest_db.is_file():
        raise FileNotFoundError("Missing PPMI input")
    out=a.report_dir.resolve();root=a.data_root.resolve()
    if out==root or root in out.parents or out==a.manifest_db.resolve().parent:
        raise ValueError("Unsafe output")
    started=time.perf_counter()
    patients,meta=load_cohorts(a.data_root,a.manifest_db)
    evid,quality,counts=evidence_tables(a.data_root,patients)
    backend=None
    if policy!="greedy":
        backend=TransformersDoctor(a.model_dir,a.max_new_tokens,device="cpu")
    reports=[];trace=[];sizes=[]
    for task in TASKS:
        people=[x for x in patients if x["cohort"]==task[0]]
        pairs,_=future_site_pairs(people,task[1])
        rows=episode_rows(people,pairs,evid,*task)
        if len(rows)<80:
            sizes.append({"task":str(task),"status":"insufficient","n":len(rows)})
            continue
        # Prespecified split: folds 2/3/4 FIT, fold 0 TUNE/AUDIT,
        # fold 1 solely evaluates the selected final system.
        # Reporting EVAL is a separate cohort from WM fit; tune and audit
        # are internal segments of the heldout fold and must be excluded
        # from final scoring (avoid selection contamination).
        # Evaluate instead on fixed fold-1, excluded from training below.
        fit=[r for r in rows if fold_id(r["pid"],task) in (2,3,4)]
        tune=[r for r in rows if fold_id(r["pid"],task)==0]
        audit=[r for r in rows if fold_id(r["pid"],task)==1]
        if min(len(fit),len(tune),len(audit))<10:continue
        eval_rows=sorted(audit,key=lambda r:r["pid"])
        # No tune on EVAL: verifier evaluation uses distinct half of fold 0;
        # EVAL is fold 1, so audit verifier on fold0 leftover.
        tune_sorted=sorted(tune,key=lambda r:r["pid"])
        mid=len(tune_sorted)//2
        inner_tune,inner_audit=tune_sorted[:mid],tune_sorted[mid:]
        if min(len(inner_tune),len(inner_audit))<5:continue
        all_keys=sorted(set().union(*(set(r["measured"]) for r in fit)))
        allowed=panel_items(panel,all_keys)
        wm=ClinicalWM(allowed).fit(fit)
        scale=1.;vtrace={"proposal":1.,"accepted":False}
        if policy=="qwen_rsi":
            scale,vtrace=verifier(wm,inner_tune,inner_audit,allowed,
                                  a.questions)
        scores={}
        for r in eval_rows:
            predicted,n,invalid=simulate(r,wm,allowed,a.questions,
                                         policy,backend)
            scores[r["pid"]]=(predicted*scale,n,invalid)
        reports.append(metrics(eval_rows,scores,a.questions,policy,task))
        trace.append({"task":str(task),**vtrace})
        sizes.append({"task":str(task),"fit":len(fit),
                      "tune":len(inner_tune),"audit":len(inner_audit),
                      "eval":len(eval_rows),
                      "eligible_all":len(rows),
                      "features":len(allowed)})
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/"metrics.csv",reports)
    write_csv(out/"cohort_sizes.csv",sizes)
    (out/"verifier_trace.json").write_text(json.dumps(trace,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
        "arm":a.arm_id,"panel":panel,"policy":policy,
        "source":meta["source_motor_table"],
        "modalities_read":quality,
        "model":"TRAIN-fit masked ridge conditional heuristic WM",
        "rsi":"deterministic verifier-gated calibration using greedy proxy; not LLM-proposed or coevolved neural weights",
        "qwen":"real local Qwen 4B CPU" if backend else "none",
        "time_seconds":time.perf_counter()-started,
        "llm_prompts":backend.total_prompts if backend else 0,
        "limits":["Exploratory reused frozen TRAIN and nonexternal PD",
                  "Historical acquisition simulation, not intervention",
                  "No patient-level exports or prospective clinical safety claim",
                  "Only actually measured BL/R01 scores tool-visible",
                  "Audit gating distinct from evaluation",
                  "R01 only on 12/24m, before target window"],
        "finished_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    print("SUCCESS",out)

if __name__=="__main__":main()
