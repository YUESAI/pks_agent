#!/usr/bin/env python3
"""Phase59 real offline Qwen Doctor pilot: four parallel, fixed-split arms.

Same-V04 visible-item masking simulation (NOT real R01 intervention).
Fit/development/evaluation people are non-overlapping frozen TRAIN
participants. Doctor never sees hidden six or other unasked items.
RSI proposal sees aggregate DEV metrics only, not individual records.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.phase45_wm_rsi_agents import load_train
from scripts.phase45_world_model import ObservationWorldModel
from scripts.phase46_qwen_doctor import TransformersDoctor
from scripts.phase59_grounded_collaboration import (
    JointBelief,episode,score_results
)
from scripts.phase59_temporal_prior import fit_prior,forecast_prior
from scripts.phase59_rsi_verifier import accepted_revision

MODES=("plain","observation","dual","rsi")
SEED=20261018


def split(records,limit,seed=SEED):
    if not 12<=limit<=150:raise ValueError("Evaluate 12..150 people")
    if len(records)<200 or len({r["pid"] for r in records})!=len(records):
        raise ValueError("Expected distinct V04 TRAIN patients")
    ranked=sorted(records,key=lambda r:hashlib.sha256(
        f"phase59|{seed}|{r['pid']}".encode()).hexdigest())
    fit_end=int(.65*len(ranked))
    dev_end=int(.80*len(ranked))
    fit=ranked[:fit_end]
    dev=ranked[fit_end:dev_end]
    evaluation=ranked[dev_end:dev_end+limit]
    if len(evaluation)<12:raise ValueError("Not enough evaluation people")
    all_groups=[{r["pid"] for r in group} for group in (fit,dev,evaluation)]
    if any(all_groups[i]&all_groups[j] for i in range(3)
           for j in range(i+1,3)):
        raise AssertionError("Leaked patient across split")
    return fit,dev,evaluation


def safe_csv(path,records):
    if not records:return
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)


def execute(rows,items,mode,backend,questions,limit,
            priors=None,rsi_rounds=3,seed=SEED):
    """Pure experiment orchestration, allows synthetic fake Qwen tests."""
    if mode not in MODES or not 1<=questions<=len(items):
        raise ValueError("Invalid arm or question count")
    fit,dev,evaluation=split(rows,limit,seed)
    site=ObservationWorldModel(items).fit(fit)
    lookup=priors or {}
    config=(0.,1.,"information_gain")
    if mode in ("dual","rsi"):config=(.5,1.,"information_gain")
    changes=[]
    if mode=="rsi":
        if rsi_rounds<1 or rsi_rounds>8:raise ValueError("Too many RSI rounds")
        # Inner verifier only sees DEV data. LLM proposer sees aggregate Brier.
        config,changes=accepted_revision(
            backend,site,dev,lookup,rsi_rounds,questions)
    output={}
    started=time.perf_counter()
    for record in evaluation:
        pid=record["pid"]
        prior=lookup.get(pid)
        weight=config[0] if prior is not None else 0.
        belief=JointBelief(site,prior,weight,config[1])
        output[pid]=episode(record,belief,backend,questions,mode,
                            config[2])
    metrics=score_results(evaluation,output,seed=seed)
    metrics.update({"arm":mode,"question_budget":questions,
                    "weight":config[0],"temperature":config[1],
                    "style":config[2],
                    "evaluation_elapsed_seconds":round(time.perf_counter()-started,2)})
    metadata={"train_site_fit_patients":len(fit),
              "inner_verifier_dev_patients":len(dev),
              "outer_engineering_eval_patients":len(evaluation),
              "same_v04_hidden_positive_eval":sum(r["label"] for r in evaluation),
              "eligible_train_V04":len(rows),
              "temporal_prior_available_in_eval":
                sum(pid in lookup for pid in output),
              "selected_config_hash":hashlib.sha256(
                  repr(config).encode()).hexdigest()[:16]}
    return metrics,changes,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=MODES,required=True)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--questions",type=int,default=4)
    p.add_argument("--max-patients",type=int,default=24)
    p.add_argument("--max-new-tokens",type=int,default=48)
    p.add_argument("--temporal-epochs",type=int,default=8)
    p.add_argument("--rsi-rounds",type=int,default=3)
    options=p.parse_args()

    root=options.data_root.expanduser().resolve()
    manifest=options.manifest_db.expanduser().resolve()
    out=options.report_dir.expanduser().resolve()
    model_dir=options.model_dir.expanduser().resolve()
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing frozen TRAIN PPMI source")
    if root==out or root in out.parents or manifest==out or out==manifest.parent:
        raise ValueError("Unsafe aggregate report destination")
    if options.arm in ("dual","rsi") and options.temporal_epochs<2:
        raise ValueError("Need >=2 neural temporal fit epochs")
    rows,items,qc,source=load_train(root,manifest)
    fit,dev,evaluation=split(rows,options.max_patients)
    backend=TransformersDoctor(model_dir,options.max_new_tokens)
    start=time.perf_counter()
    priors={}
    temporal={}
    if options.arm in ("dual","rsi"):
        # Neither DEV nor evaluation patients may enter temporal fitting.
        prohibited={r["pid"] for r in dev+evaluation}
        wm,index,device,temporal=fit_prior(
            root,manifest,prohibited,epochs=options.temporal_epochs)
        for record in dev+evaluation:
            prior=forecast_prior(wm,index,device,record["pid"])
            if prior is not None:priors[record["pid"]]=prior
        del wm
    generations_before=backend.total_prompts
    metrics,history,split_meta=execute(
        rows,items,options.arm,backend,options.questions,
        options.max_patients,priors=priors,rsi_rounds=options.rsi_rounds)
    out.mkdir(parents=True,exist_ok=True)
    safe_csv(out/"arm_metrics.csv",[metrics])
    safe_csv(out/"rsi_verifier_trace.csv",history)
    report={
       "status":"Phase59_integrated_grounded_Qwen_temporal_WM_RSI_engineering_pilot",
       "arm":options.arm,
       "model_backbone":"Qwen3-4B-Instruct-2507",
       "model_local_directory":str(model_dir),
       "patient_split_protocol":"frozen_TRAIN_hash_65pct_fit_15pct_dev_20pct_eval",
       "seed":SEED,
       "questions":options.questions,
       "qwen_generations":backend.total_prompts-generations_before,
       "qwen_tokens_generated":backend.total_generated_tokens,
       "temporal_fit_metadata":temporal,
       "split":split_meta,
       "rsi_rounds":options.rsi_rounds if options.arm=="rsi" else 0,
       "rsi_accepted":sum(bool(x["accepted"]) for x in history),
       "wall_seconds":round(time.perf_counter()-start,2),
       "study_limits":[
          "Same-V04 onsite masking simulation; NOT real R01 hidden truth.",
          "LLM never receives patient ID or actual hidden six-item scores.",
          "All four arms share the same initial subject split and question count.",
          "Plain mode uses observation WM only as fixed post-question readout.",
          "Temporal model uses BL and actual retrospective V04 visit month ONLY.",
          "RSI proposer changes allowlisted guidance config; verifier is deterministic.",
          "RSI developer fitness uses a greedy proxy, not actual Qwen DEV rollouts.",
          "No model parameter rewriting or causal treatment intervention.",
          "Only previously explored TRAIN; original Phase44 TEST consumed.",
          "Parallel runs yield aggregate comparisons, not patient-paired CIs.",
          "No patient ID, text transcripts or per-patient scores exported."
       ],
       "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"metrics":metrics,"summary":report},indent=2))
    print("Only aggregate report may leave cluster.")


if __name__=="__main__":main()
