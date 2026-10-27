#!/usr/bin/env python3
"""Phase72: twenty non-array CPUs = five patient folds x four model/policy arms.

0 ridge clinical baseline, 1 recurrent WM, 2 deep ensemble WM, 3 verified
ensemble-checkpoint and real Qwen single-item acquisition. Strictly FIT
model fitting, DEV selection, AUDIT single decision, OUTER untouched.
"""
import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs,HIDX
from scripts.phase64_evidence import evidence_tables
from scripts.phase36_masked_site_feasibility import ALL,VISIBLE
from scripts.phase72_ctlatent_wm import EnsembleWM,build_auxiliary,fixed_ridge_features
from scripts.phase46_qwen_doctor import TransformersDoctor,extract_action

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
VINDEX=[ALL.index(k) for k in VISIBLE]
KEYS=("MoCA_MCATOT","UPSIT_TOTAL_CORRECT")
def fold(pid,task):
    return int(hashlib.sha256(f"phase66|{SEED}|{task}|{pid}".encode()).hexdigest()[:12],16)%5
def prepare(patients,targets,evidence,task):
    pmap={p["pid"]:p for p in patients};rows=[]
    for t in targets:
        p=pmap[t["pid"]]
        # Strictly pre-target: R01 (or any other previous visit) is NOT
        # eligible when predicting from baseline. We deliberately freeze
        # decision time at BL for comparability and leakage safety.
        v=p["visits"][0]
        history=[{"month":int(v["month"]),"scores":{
            name:int(v["score"][i]) for name,i in zip(VISIBLE,VINDEX)
            if v["status"][i]==1}}]
        site=[int(w["score"][i]) for w in p["visits"][1:]
              if False for i in HIDX]
        # Phase62 target selector does not export individual six scores.
        # Select the identical strict future target again by exact
        # nearest-to-horizon month, skipping ambiguous ties.
        horizon=task[1];tol={12:3,24:4,6:2}[horizon]
        eligible=[w for w in p["visits"][1:] if
                  abs(w["month"]-p["baseline_month"]-horizon)<=tol and
                  w["mask_type"]=="complete_33_verified_total"]
        eligible.sort(key=lambda w:abs(w["month"]-p["baseline_month"]-horizon))
        if not eligible or (len(eligible)>1 and
            abs(eligible[0]["month"]-p["baseline_month"]-horizon)==
            abs(eligible[1]["month"]-p["baseline_month"]-horizon)):
            raise AssertionError("Strict target lost/ambiguous")
        site=[int(eligible[0]["score"][i]) for i in HIDX]
        if sum(site)!=int(t["target_burden"]):
            raise AssertionError("Future target disagreement")
        optional={k:float(evidence[p["pid"]][k]) for k in KEYS if k in evidence[p["pid"]]}
        rows.append(dict(pid=p["pid"],anchor=p["baseline_month"],horizon=horizon,
             history=history,optional=optional,released={},site=site))
    return rows
def split(rows,held):
    evaluation=[r for r in rows if fold(r["pid"],("prodromal",r["horizon"]))==held]
    # Called with task cohort hash below; this function unused.
    return evaluation
def partitions(rows,held,task):
    final=[r for r in rows if fold(r["pid"],task)==held]
    train=[r for r in rows if fold(r["pid"],task)!=held]
    train.sort(key=lambda r:hashlib.sha256(f"phase72dev|{r['pid']}".encode()).hexdigest())
    n=max(20,int(len(train)*.3))
    fit=train[:-n];dev=train[-n:];cut=len(dev)//2
    ids=[{r["pid"] for r in g} for g in (fit,dev[:cut],dev[cut:],final)]
    assert not any(ids[i]&ids[j] for i in range(4) for j in range(i+1,4))
    return fit,dev[:cut],dev[cut:],final
def choose_qwen(doctor,r):
    available=sorted(r["optional"])
    if not available:return {}
    prompt=[{"role":"system","content":'Research tool selection. Reply {"ask":"NAME"} only.'},
      {"role":"user","content":"Allowed previously measured BL clinical tests: "+
          json.dumps(available)+"; choose one to improve future Motor forecast."
          " Never invent test results."}]
    ans=extract_action(doctor.generate(prompt),set(available))
    if ans is None:ans=available[0]
    return {ans:r["optional"][ans]}
def with_release(rows,mode,doctor=None):
    ans=[]
    for r in rows:
        x=dict(r)
        if mode=="none":x["released"]={}
        elif mode=="fixed":
            keys=sorted(r["optional"]);x["released"]={keys[0]:r["optional"][keys[0]]} if keys else {}
        else:x["released"]=choose_qwen(doctor,r)
        ans.append(x)
    return ans
def evaluate(pred,rows):
    y=np.array([sum(r["site"]) for r in rows],float)
    labels=(y>0).astype(float)
    a,p,v=pred
    return {"mae":float(np.abs(a-y).mean()),"rmse":float(np.sqrt(np.mean((a-y)**2))),
      "brier":float(np.mean((p-labels)**2)),"mean_uncertainty":float(np.mean(v))}
def select_verified(wm,dev,audit,doctor):
    # Qwen observes exactly one ALREADY measured BL optional item.
    d=with_release(dev,"qwen",doctor);a=with_release(audit,"qwen",doctor)
    # Predeclared proposed model checkpoint: ensemble or member-0.
    options=("ensemble",0)
    def loss(rows,selection):
        return evaluate(wm.infer(rows,selection),rows)["mae"]
    baseline=loss(d,0);candidate=loss(d,"ensemble")
    before=loss(a,0);after=loss(a,"ensemble")
    accepted=(candidate+1e-3<baseline and after+1e-3<before)
    return ("ensemble" if accepted else 0),{
        "candidate":"ensemble","incumbent":"member0","accepted":accepted,
        "dev_incumbent_mae":baseline,"dev_candidate_mae":candidate,
        "audit_incumbent_mae":before,"audit_candidate_mae":after,
        "verifier_uses_real_qwen":True,"model_checkpoint_revision":True}
def save(path,records):
    if records:
        with path.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
def main():
    p=argparse.ArgumentParser()
    for k in ("data-root","manifest-db","report-dir","model-dir"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--arm-id",required=True,type=int)
    p.add_argument("--epochs",type=int,default=12)
    a=p.parse_args()
    if not 0<=a.arm_id<20:raise ValueError("Expected arm 0..19")
    held,arm=divmod(a.arm_id,4)
    start=time.time()
    patients,_=load_cohorts(a.data_root,a.manifest_db)
    evidence,_,_=evidence_tables(a.data_root,patients)
    doctor=TransformersDoctor(a.model_dir,32,device="cpu") if arm==3 else None
    results=[];audits=[];sizes=[]
    for task in TASKS:
        subset=[p for p in patients if p["cohort"]==task[0]]
        targets,_=future_site_pairs(subset,task[1])
        rows=prepare(subset,targets,evidence,task)
        fit,dev,audit,ev=partitions(rows,held,task)
        if min(map(len,(fit,dev,audit,ev)))<10:continue
        fitpid={r["pid"] for r in fit}
        # Shared latent training uses only FIT people's actual transitions.
        aux=build_auxiliary([p for p in subset if p["pid"] in fitpid])
        model=EnsembleWM(KEYS,members=3,epochs=a.epochs,
                         seed=SEED+held+task[1]).fit(fit,aux)
        # All non-Qwen arms have the SAME fixed evidence policy.
        eval_rows=with_release(ev,"qwen" if arm==3 else "fixed",doctor)
        if arm==0:
            training=with_release(fit,"fixed")
            reg=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            reg.fit(fixed_ridge_features(training,KEYS),
                    [sum(r["site"]) for r in training])
            prediction=np.maximum(0,reg.predict(fixed_ridge_features(eval_rows,KEYS)))
            # Conservative train-only constant prevalence; not calibrated.
            prevalence=(sum(sum(r["site"])>0 for r in fit)+1)/(len(fit)+2)
            estimates=(prediction,np.full(len(ev),prevalence),np.zeros(len(ev)))
        else:
            selector=0 if arm==1 else "ensemble"
            if arm==3:
                selector,ver=select_verified(model,dev,audit,doctor)
                audits.append({"task":str(task),**ver})
            estimates=model.infer(eval_rows,selector)
        measure=evaluate(estimates,eval_rows)
        results.append({"cohort":task[0],"horizon":task[1],"fold":held,
             "arm":arm,"n":len(ev),"fit_n":len(fit),"aux_transitions":len(aux),
             **measure})
        sizes.append({"task":str(task),"fit":len(fit),"dev":len(dev),
                      "audit":len(audit),"eval":len(ev),"aux":len(aux)})
    out=a.report_dir.resolve()
    if a.data_root.resolve() in out.parents or out==a.data_root.resolve():
        raise ValueError("Unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    save(out/"metrics.csv",results)
    save(out/"coverage.csv",sizes)
    (out/"audits.json").write_text(json.dumps(audits,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps({
         "phase":72,"arm_id":a.arm_id,"outer_fold":held,"strategy":arm,
         "elapsed_sec":time.time()-start,
         "qwen_prompts":doctor.total_prompts if doctor else 0,
         "limitations":["Exploratory frozen Prodromal TRAIN and strict PD only",
          "Real BL evidence only; future truth withheld from doctor",
          "Auxiliary transitions fit-patient only",
          "RSI is verifier-gated allowlisted checkpoint switch, not autonomous code mutation",
          "Arm3 Qwen-selected optional test differs from fixed non-Qwen arms",
          "Not a causal generative clinical world model or external validation",
          "No patient-level data exported"]},indent=2)+"\n")
    print(json.dumps({"completed":len(results),"arm":a.arm_id,"report":str(out)}))
if __name__=="__main__":main()
