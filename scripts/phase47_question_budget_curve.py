#!/usr/bin/env python3
"""Phase47: single-pass Qwen Doctor cost curves at 4/8/12/16 questions.

Uses ONLY frozen TRAIN's complete V04 visits. The same fixed hash-disjoint
WM fitting and held-out cohort is used at every checkpoint. LLM models
generate once through the maximum budget and in-memory risk snapshots
are evaluated after every requested number of recorded visible items.
Patient ground-truth labels are never available to the Doctor or WM
inference. Phase44 TEST is consumed and untouched.

This is a same-visit V04 *masking simulation*, NOT longitudinal clinical
world modelling or proof of onsite referral value.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase45_wm_rsi_agents import load_train,measure,hashkey
from scripts.phase45_world_model import (
    ObservationWorldModel,PatientEvidenceService
)
from scripts.phase46_qwen_doctor import TransformersDoctor,llm_doctor_rollout

CONTROLLERS=("llm_plain","llm_wm","static_questions","dynamic_questions",
             "visible_sum_27","wm_full_27")


def patient_split(records,limit,seed):
    ordered=sorted(records,key=lambda r:hashkey(seed,r["pid"]))
    cutoff=max(30,int(.75*len(ordered)))
    training=ordered[:cutoff]
    heldout=sorted(ordered[cutoff:],key=lambda r:hashkey(seed+1,r["pid"]))[:limit]
    if len(heldout)<20:
        raise ValueError("Need at least 20 held-out TRAIN patients for this sweep")
    if not (0<sum(r["label"] for r in heldout)<len(heldout)):
        raise ValueError("Held-out sample must have two classes")
    if set(r["pid"] for r in training)&set(r["pid"] for r in heldout):
        raise ValueError("Leaked patient across fit/eval")
    return training,heldout


def baseline_snapshots(world,row,items,budgets,mode,seed):
    """One state-dependent sequence, risk after each budget prefix."""
    allowed=items
    if mode=="static_questions":
        order=world.static_order()
    elif mode=="random_questions":
        order=sorted(items,key=lambda k:hashkey(seed,row["pid"]+"|"+k))
    elif mode!="dynamic_questions":
        raise ValueError("Unknown non-LLM controller")
    patient=PatientEvidenceService(row["features"],allowed,max(budgets))
    values={};paths={}
    for step in range(1,max(budgets)+1):
        observed=patient.observed
        if mode=="dynamic_questions":
            remaining=[k for k in items if k not in observed]
            action=sorted(remaining,key=lambda k:
                          (-world.information_gain(observed,k),k))[0]
        else:
            action=order[step-1]
        patient.request(action)
        if step in budgets:
            values[step]=world.posterior(patient.observed)
            paths[step]=tuple(patient.observed)
    return values,paths


def evaluate_snapshots(fitting,held,items,backend,budgets,seed):
    world=ObservationWorldModel(items).fit(fitting)
    backend.world=world
    max_budget=max(budgets)
    risks={budget:{name:{} for name in CONTROLLERS}
           for budget in budgets}
    paths={budget:{name:{} for name in CONTROLLERS[:4]}
           for budget in budgets}
    invalid={mode:0 for mode in CONTROLLERS[:2]}
    generations=0
    for r in held:
        pid=r["pid"]
        score=sum(r["features"].values())
        full=world.posterior(r["features"])
        for budget in budgets:
            risks[budget]["visible_sum_27"][pid]=score
            risks[budget]["wm_full_27"][pid]=full
        for mode in ("llm_plain","llm_wm"):
            output=llm_doctor_rollout(
                world,r,backend,items,max_budget,mode,checkpoints=budgets)
            invalid[mode]+=output["invalid_action_fallbacks"]
            generations+=output["queries"]
            for budget in budgets:
                risks[budget][mode][pid]=output["risk_at_checkpoints"][budget]
                paths[budget][mode][pid]=output["selected_items"][:budget]
        for mode in ("static_questions","dynamic_questions"):
            snapshots,orders=baseline_snapshots(
                world,r,items,budgets,mode,seed)
            for budget in budgets:
                risks[budget][mode][pid]=snapshots[budget]
                paths[budget][mode][pid]=orders[budget]
    return risks,paths,invalid,generations


def aggregate(held,items,budgets,risks,paths,invalid,generations,seed):
    metrics=[];agreements=[]
    for budget in budgets:
        selected={}
        for name in CONTROLLERS:
            result=measure(held,risks[budget][name],seed,
                           name!="visible_sum_27")
            selected[name]=result.pop("chosen")
            metrics.append({
                "questions":len(items) if name.endswith("_27") else budget,
                "checkpoint_q":budget,
                "controller":name,
                "patients":result["n"],"hidden_positive":result["positives"],
                "selected":result["selected"],"captured_positive":result["captured"],
                "positive_recall":round(result["recall"],5),
                "positive_precision":round(result["precision"],5),
                "hidden_burden_recall":round(result["burden_recall"],5),
                "auroc":round(result["auroc"],5),
                "brier":round(result["brier"],5) if result["brier"] is not None else "",
                "llm_invalid_total_full_trajectory":invalid.get(name,""),
            })
        for a,b in (("llm_plain","llm_wm"),
                    ("llm_wm","dynamic_questions"),
                    ("llm_wm","static_questions"),
                    ("static_questions","dynamic_questions"),
                    ("llm_plain","dynamic_questions")):
            same_first=sum(paths[budget][a][r["pid"]][0]==
                           paths[budget][b][r["pid"]][0] for r in held)
            exact=sum(paths[budget][a][r["pid"]]==
                      paths[budget][b][r["pid"]] for r in held)
            shared=sum(len(set(paths[budget][a][r["pid"]])&
                           set(paths[budget][b][r["pid"]]))/budget for r in held)
            intersection=len(selected[a]&selected[b])
            union=len(selected[a]|selected[b])
            agreements.append({
                "checkpoint_q":budget,"controller_a":a,"controller_b":b,
                "patients":len(held),
                "first_question_agreement_pct":round(100*same_first/len(held),2),
                "full_sequence_agreement_pct":round(100*exact/len(held),2),
                "question_set_overlap_pct":round(100*shared/len(held),2),
                "topk_intersection":intersection,
                "topk_jaccard":round(intersection/union,5) if union else "",
            })
    return metrics,agreements


def write_csv(path,rows):
    if not rows:raise ValueError("Empty report")
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(records,items,model_dir,out,budgets=(4,8,12,16),limit=100,seed=20261018):
    budgets=tuple(sorted(set(budgets)))
    if not budgets or any(not 1<=q<=len(items) for q in budgets):
        raise ValueError("Budgets must be 1..visible item count")
    if limit<20 or limit>200:raise ValueError("Evaluation size 20..200")
    fitting,held=patient_split(records,limit,seed)
    backend=TransformersDoctor(model_dir,max_new_tokens=48)
    risks,paths,invalid,generations=evaluate_snapshots(
        fitting,held,items,backend,budgets,seed)
    metrics,agree=aggregate(held,items,budgets,risks,paths,invalid,
                            generations,seed)
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"cost_curve_metrics.csv",metrics)
    write_csv(out/"question_agreement_by_budget.csv",agree)
    summary={
        "status":"exploratory_TRAIN_only_V04_masked_Qwen_Doctor_cost_curves",
        "eligible_train_v04":len(records),
        "world_model_fit_patients":len(fitting),
        "heldout_patients":len(held),
        "heldout_positive":sum(r["label"] for r in held),
        "budget_checkpoints":list(budgets),
        "onsite_topk_fraction":0.2,
        "llm_generation_calls":generations,
        "llm_generated_tokens":backend.total_generated_tokens,
        "model_path":str(Path(model_dir).resolve()),
        "seed":seed,
        "warnings":[
            "V04 simulated masking only; not real R01 or patient outcome.",
            "The same 100 heldout TRAIN patients are reused at every checkpoint.",
            "LLM paths are generated once to the largest budget; risks saved at prefixes.",
            "No longitudinal state-transition World Model and no LLM RSI in Phase47.",
            "Full-27 controls use more recorded questions, not cost matched.",
            "No patient IDs, questions, scores or transcripts exported.",
            "Final Phase44 TEST consumed; all Phase47 evaluations use TRAIN only."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+chr(10))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,required=True)
    p.add_argument("--budgets",type=int,nargs="+",default=[4,8,12,16])
    p.add_argument("--max-patients",type=int,default=100)
    p.add_argument("--seed",type=int,default=20261018)
    a=p.parse_args()
    root,manifest,out=[Path(x).expanduser().resolve()
                       for x in (a.data_root,a.manifest_db,a.report_dir)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing source or private manifest")
    if root==out or root in out.parents or out==manifest:
        raise ValueError("Unsafe report directory")
    records,items,qc,source=load_train(root,manifest)
    summary=run(records,items,a.model_dir,out,tuple(a.budgets),
                a.max_patients,a.seed)
    print(json.dumps(summary,indent=2))
    print("Upload aggregate Phase47 ZIP ONLY; no patient-level exports.")
if __name__=="__main__":main()
