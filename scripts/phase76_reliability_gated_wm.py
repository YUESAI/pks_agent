#!/usr/bin/env python3
"""Phase76: reliability-gated clinical CT-WM under matched information.

Twenty standalone CPUs = five original subject folds x four strategies.
0: fixed matched Ridge / HGB in motor-only and multimodal information sets;
1: motor-only direct-target CT WM vs motor-only HGB;
2: genuinely multimodal direct-target CT WM vs multimodal HGB;
3: paired motor-only and multimodal CT, plus audited L1-loss revision.
Every job retains matched baselines; arm3 reports all components.

The modality of each arm is explicit. For SAME-PANEL comparisons HGB,
Ridge, and CT-WM receive identical recorded clinical information and
pre-cutoff visible27 Motor chronology. Prior Phase74 CT-WM accidentally
ignored the optional baseline clinical fields; fixed here by releasing
ACTUALLY recorded FIT/EVAL optional values through the tensor encoder.
No Qwen, fabricated observation or unconstrained RSI.
No PHI or individual predictions exported.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,math,time,os
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.phase36_masked_site_feasibility import VISIBLE
from scripts.phase62_ppmi_multicohort_audit import load_cohorts,future_site_pairs
from scripts.phase64_evidence import evidence_tables
from scripts.phase73_matched_longitudinal_agent import data_rows,split_data
from scripts.phase72_ctlatent_wm import sequence_features,build_auxiliary
from scripts.phase60_cpu_multistep_wm import auc

SEED=20261018
TASKS=(("prodromal",12),("prodromal",24),("PD",12))
MAX_CLINICAL=64

def optional_keys(fit):
    counts=Counter(k for r in fit for k in r["optional"])
    return tuple(k for k,count in sorted(counts.items(),key=lambda v:(-v[1],v[0]))
                 if count>=10)[:MAX_CLINICAL]

def clinical_scale(key):
    return 40. if key.startswith(("MoCA","UPSIT")) else 4.

def engineered(rows,keys):
    """Only measured historical visible27 and measured BL clinical tests."""
    arr=[]
    for row in rows:
        hist=row["history"]
        if not hist:raise ValueError("Missing chronological original BL")
        if any(b["month"]<a["month"] for a,b in zip(hist,hist[1:])):
            raise AssertionError("Nonchronological origin features")
        first,last=hist[0],hist[-1]
        time=max(1.,float(last["month"]-first["month"]))
        out=[]
        for item in VISIBLE:
            observed=[(float(v["month"]),float(v["scores"][item]))
                      for v in hist if item in v["scores"]]
            first_val=observed[0][1] if observed else 0.
            last_val=observed[-1][1] if observed else 0.
            # PPMI longitudinal visits are sparsely and irregularly
            # measured per item. A feature can have zero or just one
            # measured value even when the *visit* itself is valid.
            # Do not index an empty observation series; singleton
            # measurements also have no estimable temporal slope.
            if len(observed)>=2:
                change=(last_val-first_val)/4.
                slope=(last_val-first_val)/max(
                    1.,observed[-1][0]-observed[0][0])
            else:
                change=0.
                slope=0.
            out.extend([first_val/4.,last_val/4.,change,
                        slope*6./4.,len(observed)/6.,float(bool(observed))])
        out.extend([len(hist)/6.,(last["month"]-first["month"])/36.,
                    (row["anchor"]+row["horizon"]-last["month"])/36.,
                    row["horizon"]/36.])
        for key in keys:
            value=row["optional"].get(key)
            out.extend([float(value)/clinical_scale(key) if value is not None else 0.,
                        float(value is not None)])
        arr.append(out)
    x=np.array(arr,dtype=np.float32)
    if not np.isfinite(x).all():raise AssertionError("Nonfinite predictors")
    return x

def neuralize(rows,keys):
    """Copy only actually recorded baseline clinical items into neural tensors.

    Phase74's neural model previously saw an empty released={} dict.
    This fixes the information mismatch while preserving missingness and
    the exact same per-patient clinical values as the tabular baselines.
    """
    result=[]
    for row in rows:
        updated=dict(row)
        updated["released"]={
            key:float(row["optional"][key]) for key in keys
            if key in row["optional"]
        }
        result.append(updated)
    return result


def labels(rows):
    site=np.asarray([r["site"] for r in rows],dtype=int)
    if site.shape!=(len(rows),6) or (site<0).any() or (site>4).any():
        raise ValueError("Six true clinical item ratings required")
    return site,site.sum(axis=1).astype(float),(site.sum(axis=1)>0).astype(int)

def score(pred,rows):
    mean,risk=pred
    _,y,z=labels(rows)
    mean=np.asarray(mean,dtype=float);risk=np.asarray(risk,dtype=float)
    if len(mean)!=len(rows) or len(risk)!=len(rows):
        raise AssertionError("Predictions not aligned")
    if not np.isfinite(mean).all() or not np.isfinite(risk).all():
        raise AssertionError("Nonfinite predictions")
    return {"mae":float(np.mean(np.abs(mean-y))),
            "rmse":float(np.mean((mean-y)**2)**.5),
            "brier":float(np.mean((risk-z)**2)),
            "auroc":auc(risk.tolist(),z.tolist())}

class TimeTabular:
    def __init__(self,kind):
        if kind not in ("ridge","hgb"):raise ValueError("Bad estimator")
        self.kind=kind
    def fit(self,rows,keys):
        self.keys=keys
        X=engineered(rows,keys)
        _,y,binary=labels(rows)
        if self.kind=="ridge":
            self.mean=make_pipeline(StandardScaler(),Ridge(alpha=20.))
            self.risk=make_pipeline(StandardScaler(),
                                   LogisticRegression(C=.15,max_iter=1000))
        else:
            self.mean=HistGradientBoostingRegressor(
                max_iter=180,max_leaf_nodes=9,min_samples_leaf=15,
                learning_rate=.04,l2_regularization=20.,random_state=SEED)
            self.risk=HistGradientBoostingClassifier(
                max_iter=130,max_leaf_nodes=7,min_samples_leaf=15,
                learning_rate=.04,l2_regularization=20.,random_state=SEED)
        self.mean.fit(X,y)
        self.single_class=len(set(binary))<2
        self.prior=float((binary.sum()+1)/(len(binary)+2))
        if not self.single_class:self.risk.fit(X,binary)
        return self
    def predict(self,rows):
        X=engineered(rows,self.keys)
        y=np.maximum(0.,np.asarray(self.mean.predict(X)))
        p=(np.full(len(rows),self.prior) if self.single_class
           else self.risk.predict_proba(X)[:,1])
        return y,np.clip(p,0,1)

class CTDirect(nn.Module):
    def __init__(self,nclinical,width=80,gated=False):
        super().__init__()
        self.nclinical=nclinical
        self.gated=gated
        if gated and nclinical:
            # Sigmoid(-3) ~ 0.047: untrusted optional features begin suppressed.
            self.clinical_gate_logits=nn.Parameter(torch.full((nclinical,),-3.))
        self.observation=nn.Sequential(nn.Linear(54,64),nn.GELU())
        self.decay_logits=nn.Parameter(torch.full((width,),-1.))
        self.cell=nn.GRUCell(64,width)
        self.context=nn.Sequential(nn.Linear(width+2*nclinical+2,96),
                                   nn.LayerNorm(96),nn.GELU(),
                                   nn.Dropout(.12))
        self.total=nn.Linear(96,1)
        self.binary=nn.Linear(96,1)
        self.site=nn.Linear(96,6*5)
        self.next_visible=nn.Linear(96,27*5)
    def forward(self,x,dt,valid,clinical,horizon):
        h=x.new_zeros((x.shape[0],self.cell.hidden_size))
        rate=torch.nn.functional.softplus(self.decay_logits)
        for t in range(x.shape[1]):
            next_h=self.cell(self.observation(x[:,t]),
                             h*torch.exp(-rate*dt[:,t,None]))
            h=torch.where(valid[:,t,None],next_h,h)
        if self.gated and self.nclinical:
            w=torch.sigmoid(self.clinical_gate_logits)
            clinical=torch.cat((clinical[:,:self.nclinical]*w,
                    clinical[:,self.nclinical:]*w),dim=1)
        z=self.context(torch.cat((h,clinical,horizon),dim=1))
        return (nn.functional.softplus(self.total(z).squeeze(1)),
                self.binary(z).squeeze(1),
                self.site(z).reshape(-1,6,5),
                self.next_visible(z).reshape(-1,27,5))

class DirectWM:
    def __init__(self,keys,style="balanced",epochs=18,seed=SEED):
        self.keys=keys;self.style=style;self.epochs=epochs;self.seed=seed
        if style not in ("balanced","early","gated","gated_mae"):
            raise ValueError("Unallowlisted WM revision")
    def fit(self,fit,aux):
        # Previous published exploratory phases used 4 threads. Allow new
        # constrained CPU jobs to opt into a single-thread backend when
        # the GRUCell native backward path exhibits pathological stalls.
        threads=int(os.environ.get("PKS_TORCH_THREADS","4"))
        if not 1<=threads<=8:
            raise ValueError("PKS_TORCH_THREADS must be 1..8")
        torch.set_num_threads(threads)
        torch.manual_seed(self.seed)
        gen=np.random.default_rng(self.seed)
        self.net=CTDirect(len(self.keys),gated=self.style.startswith("gated"))
        opt=torch.optim.AdamW(self.net.parameters(),lr=.0008,weight_decay=.025)
        inp=sequence_features(neuralize(fit,self.keys),self.keys)
        site,y,binary=labels(fit)
        site=torch.tensor(site,dtype=torch.long)
        y=torch.tensor(y,dtype=torch.float32)
        binary=torch.tensor(binary,dtype=torch.float32)
        if aux:
            ax=sequence_features(aux,self.keys)
            target=torch.tensor([r["aux_motor"] for r in aux],dtype=torch.long)
        else:ax=None;target=None
        weights=((.50,.35,.15) if self.style in ("balanced","early","gated") else
                 (.90,.05,.05))
        verbose=os.environ.get("PKS_TRAIN_PROGRESS")=="1"
        start_fit=time.monotonic()
        if verbose:
            print("WM_FIT_START "+json.dumps({
                "style":self.style,"epochs":self.epochs,
                "fit_people":len(fit),"aux_pairs":len(aux),
                "torch_threads":torch.get_num_threads(),
                "mkldnn":torch.backends.mkldnn.enabled}),flush=True)
        for epoch in range(self.epochs):
            epoch_start=time.monotonic()
            order=gen.permutation(len(fit))
            self.net.train()
            for start in range(0,len(fit),32):
                ii=order[start:start+32]
                batch=[a[ii] for a in inp]
                # Optional clinical observations may be absent; never
                # invent a score. Stochastic release masks an observed
                # clinical input only (already measured in FIT).
                if len(self.keys):
                    clinical=batch[3].clone()
                    n=len(self.keys)
                    dropout=(.65 if self.style.startswith("gated") else .15)
                    keep=torch.from_numpy((gen.random((len(ii),n))>dropout).astype(np.float32))
                    clinical[:,:n]*=keep
                    clinical[:,n:]*=keep
                    batch[3]=clinical
                total,risk,sites,_=self.net(*batch)
                huber=(torch.abs(total-y[ii]).mean()
                       if self.style=="gated_mae"
                       else torch.nn.functional.smooth_l1_loss(
                           total,y[ii],beta=1.5))
                ce=torch.nn.functional.cross_entropy(sites.reshape(-1,5),
                                                     site[ii].reshape(-1))
                bce=torch.nn.functional.binary_cross_entropy_with_logits(
                    risk,binary[ii])
                loss=weights[0]*huber+weights[1]*ce+weights[2]*bce
                if ax is not None and len(aux):
                    jj=gen.integers(0,len(aux),size=min(len(ii),32))
                    _,_,_,nextlogits=self.net(*[a[jj] for a in ax])
                    loss=loss+.12*torch.nn.functional.cross_entropy(
                        nextlogits.reshape(-1,5),target[jj].reshape(-1),
                        ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.)
                opt.step()
            if verbose:
                print("WM_EPOCH_DONE "+json.dumps({
                    "style":self.style,"epoch":epoch+1,
                    "epoch_seconds":round(time.monotonic()-epoch_start,3),
                    "total_seconds":round(time.monotonic()-start_fit,3)}),
                    flush=True)
        self.net.eval()
        return self
    def gate_summary(self):
        if not getattr(self.net,"gated",False) or not len(self.keys):
            return {"gate_count":0,"gate_mean":None,"gate_max":None}
        w=torch.sigmoid(self.net.clinical_gate_logits.detach()).numpy()
        return {"gate_count":len(w),"gate_mean":float(w.mean()),
                "gate_max":float(w.max())}
    def predict(self,rows):
        xs=sequence_features(neuralize(rows,self.keys),self.keys)
        self.net.eval()
        with torch.inference_mode():
            total,logit,_,_=self.net(*xs)
            return total.numpy().astype(float),torch.sigmoid(logit).numpy().astype(float)

def paired_bootstrap(reference,candidate,rows,seed,repeats=500):
    _,y,_=labels(rows)
    delta=np.abs(candidate[0]-y)-np.abs(reference[0]-y)
    rng=np.random.default_rng(seed)
    distribution=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()
                           for _ in range(repeats)])
    return {"delta_mae":float(delta.mean()),
            "ci_low":float(np.quantile(distribution,.025)),
            "ci_high":float(np.quantile(distribution,.975))}

def write_csv(path,records):
    if records:
        with path.open("w",newline="",encoding="utf8") as file:
            writer=csv.DictWriter(file,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)

def choose_revision(incumbent,proposal,tune,audit):
    before=score(incumbent.predict(tune),tune)
    after=score(proposal.predict(tune),tune)
    if after["mae"]+0.01>=before["mae"]:
        return incumbent,{"proposed":"mae_revision","accepted":False,
            "reason":"dev_not_improved","dev_before":before["mae"],
            "dev_proposed":after["mae"],"audit_checked":False}
    original=score(incumbent.predict(audit),audit)
    candidate=score(proposal.predict(audit),audit)
    accepted=(candidate["mae"]+.005<original["mae"] and
              candidate["brier"]<=original["brier"]+.01)
    return (proposal if accepted else incumbent),{
        "proposed":"mae_revision","accepted":bool(accepted),
        "reason":"audit_accept" if accepted else "audit_reject",
        "dev_before":before["mae"],"dev_proposed":after["mae"],
        "audit_checked":True,"audit_before":original["mae"],
        "audit_proposed":candidate["mae"],
        "audit_brier_before":original["brier"],
        "audit_brier_proposed":candidate["brier"]}

def run(args):
    if not 0<=args.arm_id<20:raise ValueError("Arm 0..19 required")
    fold,variant=divmod(args.arm_id,4)
    out=args.report_dir.resolve();root=args.data_root.resolve()
    if root==out or root in out.parents or not args.manifest_db.is_file():
        raise ValueError("Unsafe output or missing frozen TRAIN")
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    people,_=load_cohorts(root,args.manifest_db)
    evidence,quality,_=evidence_tables(root,people)
    results=[];contrasts=[];coverage=[];audits=[]
    for task in TASKS:
        patients=[p for p in people if p["cohort"]==task[0]]
        target,_=future_site_pairs(patients,task[1])
        rows=data_rows(patients,target,evidence,task)
        fit,tune,audit,held=split_data(rows,task,fold)
        if min(map(len,(fit,tune,audit,held)))<10:
            coverage.append({"task":str(task),"status":"insufficient"})
            continue
        keys=optional_keys(fit)
        fit_ids={r["pid"] for r in fit}
        # Longitudinal auxiliary supervision is FIT-person only.
        aux=build_auxiliary([p for p in patients if p["pid"] in fit_ids])
        baselines={
            "ridge_motor":TimeTabular("ridge").fit(fit,()),
            "hgb_motor":TimeTabular("hgb").fit(fit,()),
            "ridge_multimodal":TimeTabular("ridge").fit(fit,keys),
            "hgb_multimodal":TimeTabular("hgb").fit(fit,keys),
        }
        predictions={name:est.predict(held) for name,est in baselines.items()}
        # Exactly the same Motor-only incumbent across all four arms.
        motor=DirectWM((),style="balanced",epochs=args.epochs,
                       seed=SEED+fold+task[1]).fit(fit,aux)
        predictions["ct_motor"]=motor.predict(held)
        candidates={}
        if variant>=1:
            early=DirectWM(keys,style="early",epochs=args.epochs,
                    seed=SEED+fold+task[1]).fit(fit,aux)
            candidates["ct_early"]=early
            predictions["ct_early"]=early.predict(held)
        if variant>=2:
            gated=DirectWM(keys,style="gated",epochs=args.epochs,
                    seed=SEED+fold+task[1]).fit(fit,aux)
            candidates["ct_gated"]=gated
            predictions["ct_gated"]=gated.predict(held)
        if variant==3:
            proposed=DirectWM(keys,style="gated_mae",epochs=args.epochs,
                    seed=SEED+fold+task[1]).fit(fit,aux)
            # The incumbent is the same Motor-only CT WM. DEV prefilters
            # the gated revision and a separate AUDIT can veto it.
            selected,ver=choose_revision(motor,proposed,tune,audit)
            predictions["ct_verified"]=selected.predict(held)
            audits.append({"cohort":task[0],"horizon":task[1],
                           "fold":fold,**ver})
        primary={0:"ct_motor",1:"ct_early",
                 2:"ct_gated",3:"ct_verified"}[variant]
        for name,pred in predictions.items():
            panel=("motor_only" if name.endswith("_motor")
                   else "baseline_nonmotor_and_motor")
            gates=(candidates[name].gate_summary() if name in candidates
                   else selected.gate_summary() if name=="ct_verified"
                   else {"gate_count":0,"gate_mean":None,"gate_max":None})
            results.append({"cohort":task[0],"horizon":task[1],"fold":fold,
                "arm_id":args.arm_id,"model":name,"primary":name==primary,
                "patients":len(held),"positive":int(labels(held)[2].sum()),
                "panel":panel,
                "optional_features":0 if panel=="motor_only" else len(keys),
                "mean_history_visits":float(np.mean([len(r["history"]) for r in held])),
                **gates,**score(pred,held)})
        comparisons=[("hgb_motor","ct_motor"),
                     ("hgb_multimodal","ct_motor")]
        for candidate in ("ct_early","ct_gated","ct_verified"):
            if candidate in predictions:
                comparisons.extend([("ct_motor",candidate),
                         ("hgb_multimodal",candidate)])
        if "ct_early" in predictions and "ct_gated" in predictions:
            comparisons.append(("ct_early","ct_gated"))
        for reference,candidate in comparisons:
            contrasts.append({"cohort":task[0],"horizon":task[1],"fold":fold,
                "reference":reference,"candidate":candidate,
                **paired_bootstrap(predictions[reference],predictions[candidate],
                                   held,SEED+fold+task[1])})
        coverage.append({"cohort":task[0],"horizon":task[1],"fold":fold,
            "fit":len(fit),"dev":len(tune),"audit":len(audit),
            "evaluation":len(held),"real_fit_visit_pairs":len(aux),
            "n_eval_history2plus":sum(len(r["history"])>=2 for r in held),
            "n_eval_history3plus":sum(len(r["history"])>=3 for r in held),
            "status":"complete"})
        # Atomic task-level aggregate snapshots: survive a 12h time cap.
        write_csv(out/"method_metrics.csv",results)
        write_csv(out/"paired_bootstrap.csv",contrasts)
        write_csv(out/"coverage.csv",coverage)
        (out/"audit.json").write_text(json.dumps(audits,indent=2)+"\n")
        (out/"progress.json").write_text(json.dumps({
            "status":"partial","last_finished_task":list(task),
            "completed_tasks":len(coverage),"elapsed_seconds":round(
                time.perf_counter()-start,2)},indent=2)+"\n")
        print("PHASE76_TASK_DONE",task,"arm",args.arm_id,flush=True)
    (out/"summary.json").write_text(json.dumps({
        "phase":76,"arm_id":args.arm_id,"outer_fold":fold,"variant":variant,
        "status":"complete","completed_tasks":len(coverage),
        "elapsed_seconds":round(time.perf_counter()-start,2),
        "multimodal_quality":quality,
        "limits":["Frozen Prodromal TRAIN / strict PD only",
                  "Same actual visible27 longitudinal histories and baseline optional tests",
                  "No Qwen or autonomous RSI: only audited model-loss revision",
                  "Reliability gates on optional clinical input, with FIT-only dropout",
                  "No outcome/site from heldout used in FIT/DEV/AUDIT",
                  "Task-level reports persist before 12h walltime",
                  "Original consumed final TEST never accessed",
                  "No patient records exported"],
        "generated_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    (out/"progress.json").write_text(json.dumps({"status":"complete",
                  "completed_tasks":len(coverage)},indent=2)+"\n")
    print(json.dumps({"status":"success","arm":args.arm_id,
                      "completed_tasks":len(coverage),"path":str(out)}))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm-id",type=int,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=18)
    a=p.parse_args()
    if not 4<=a.epochs<=100:raise ValueError("Invalid epoch setting")
    run(a)
if __name__=="__main__":main()
