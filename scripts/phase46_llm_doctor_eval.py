#!/usr/bin/env python3
"""Phase46 real offline Qwen Doctor versus grounded WM Doctor controllers.

Uses only frozen PPMI TRAIN complete V04 exams (simulated remote questions).
The held-out final TEST was consumed in Phase44 and is NEVER reopened.
The evaluation set is chosen by fixed hash, not by hidden six-item labels.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.phase45_wm_rsi_agents import load_train,measure,hashkey
from scripts.phase45_world_model import (
    ObservationWorldModel,PatientEvidenceService,query_policy
)
from scripts.phase46_qwen_doctor import TransformersDoctor,llm_doctor_rollout

MODES=("llm_plain","llm_wm","static_questions","dynamic_questions",
       "visible_sum_27","wm_full_27")


def split_sample(rows,seed,limit):
    """Stratified deterministic sample from outer heldout; never use labels to select.

    The entire TRAIN-derived evaluation partition is fixed by stable hash.
    """
    ordered=sorted(rows,key=lambda r:hashkey(seed,r["pid"]))
    n=min(limit,len(ordered))
    if n<12:raise ValueError("Need >=12 heldout episodes")
    return ordered[:n]


def evaluate(rows,items,backend,questions,seed):
    predicted={m:{} for m in MODES}
    invalid=Counter();calls=Counter()
    paths={m:{} for m in MODES if m not in ("visible_sum_27","wm_full_27")}
    for row in rows:
        pid=row["pid"]
        for mode in MODES:
            if mode.startswith("llm_"):
                result=llm_doctor_rollout(
                    backend.world,row,backend,items,questions,mode)
                predicted[mode][pid]=result["risk"]
                invalid[mode]+=result["invalid_action_fallbacks"]
                calls[mode]+=result["queries"]
                paths[mode][pid]=result["selected_items"]
            elif mode in ("static_questions","dynamic_questions"):
                p=PatientEvidenceService(row["features"],items,questions)
                risk,n=query_policy(backend.world,p,
                    "static" if mode.startswith("static") else "dynamic",
                    questions,pid,seed)
                predicted[mode][pid]=risk
                paths[mode][pid]=tuple(p.observed)
            elif mode=="visible_sum_27":
                predicted[mode][pid]=sum(row["features"].values())
            else:
                predicted[mode][pid]=backend.world.posterior(row["features"])
    return predicted,invalid,calls,paths


def run(root,manifest,out,model_dir,questions=4,limit=36,
        seed=20261018,max_new_tokens=48):
    root,manifest,out,model_dir=[Path(x).expanduser().resolve()
                                for x in (root,manifest,out,model_dir)]
    if not (root/"PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Missing original PPMI data / frozen manifest")
    if root==out or root in out.parents or out==manifest:
        raise ValueError("Report location conflicts with private inputs")
    if not 1<=questions<=27 or not 12<=limit<=200:
        raise ValueError("Invalid question budget/evaluation cap")
    records,items,qc,source=load_train(root,manifest)
    # No label-based sampling: hash-only TRAIN patient partition.
    records=sorted(records,key=lambda r:hashkey(seed,r["pid"]))
    cutoff=max(30,int(.75*len(records)))
    fitting=records[:cutoff]
    holdout=split_sample(records[cutoff:],seed+1,limit)
    positives=sum(r["label"] for r in holdout)
    if positives==0 or positives==len(holdout):
        raise ValueError("Heldout sample has only one outcome class")
    world=ObservationWorldModel(items).fit(fitting)
    doctor=TransformersDoctor(model_dir,max_new_tokens=max_new_tokens)
    doctor.world=world
    predictions,invalid,calls,paths=evaluate(holdout,items,doctor,questions,seed)
    metrics=[]
    for mode in MODES:
        st=measure(holdout,predictions[mode],seed,
                   mode!="visible_sum_27")
        st.pop("chosen")
        metrics.append({
            "model":mode,
            "n":len(holdout),
            "selected":st["selected"],
            "positive_recall":round(st["recall"],5),
            "precision":round(st["precision"],5),
            "hidden_burden_recall":round(st["burden_recall"],5),
            "auroc":round(st["auroc"],5) if st["auroc"] is not None else "",
            "brier":round(st["brier"],5) if st["brier"] is not None else "",
            "per_patient_questions":len(items) if mode.endswith("_27")
                                     else questions,
            "invalid_llm_action_count":invalid[mode],
            "actual_llm_generations":calls[mode],
        })
    # Never serialize any individual patient's actions, identifiers or scores.
    diagnostic=[]
    for left,right in (
        ("llm_plain","llm_wm"),
        ("llm_plain","dynamic_questions"),
        ("llm_wm","dynamic_questions"),
        ("llm_wm","static_questions"),
        ("static_questions","dynamic_questions"),
    ):
        n=len(holdout)
        exact=sum(paths[left][r["pid"]]==paths[right][r["pid"]] for r in holdout)
        first=sum(paths[left][r["pid"]][0]==paths[right][r["pid"]][0] for r in holdout)
        overlap=sum(len(set(paths[left][r["pid"]]) & set(paths[right][r["pid"]])) /
                    questions for r in holdout)
        chosen_left=measure(holdout,predictions[left],seed,True)["chosen"]
        chosen_right=measure(holdout,predictions[right],seed,True)["chosen"]
        inter=len(chosen_left & chosen_right)
        union=len(chosen_left | chosen_right)
        diagnostic.append({
            "controller_a":left,"controller_b":right,
            "patients":n,"first_question_agreement_pct":round(100*first/n,2),
            "full_sequence_agreement_pct":round(100*exact/n,2),
            "question_set_overlap_pct":round(100*overlap/n,2),
            "topk_selected_overlap_n":inter,
            "topk_selected_jaccard":round(inter/union,5) if union else "",
        })
    out.mkdir(parents=True,exist_ok=True)
    with (out/"agent_behavior_agreement.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic[0]))
        w.writeheader();w.writerows(diagnostic)
    with (out/"model_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]))
        w.writeheader();w.writerows(metrics)
    result={
        "status":"TRAIN_only_Qwen3_Doctor_WM_partial_observation_pilot",
        "model_local_directory":str(model_dir),
        "model_config_name":doctor.config_id,
        "eligible_train_v04":len(records),
        "wm_fit_patients":len(fitting),
        "heldout_patients":len(holdout),
        "heldout_positive_patients":positives,
        "questions":questions,
        "site_acquisition_budget_pct":20,
        "llm_model_generations":doctor.total_prompts,
        "llm_total_generated_tokens":doctor.total_generated_tokens,
        "behavior_diagnostics":"agent_behavior_agreement.csv (aggregate, no patient identifiers)",
        "seed":seed,
        "limits":[
            "Doctor uses an actual local Qwen LLM; Patient and verifier are deterministic.",
            "WM is a same-visit observation generator, NOT longitudinal state dynamics.",
            "No LLM RSI in this stage; Phase45 RSI-lite runs as separate CPU experiment.",
            "LLM and heuristic questions matched at q; full-27 score uses 27 questions.",
            "Patient evidence is simulated masking of recorded complete V04 onsite exams.",
            "No true R01 intervention effect or prospective clinical validation.",
            "Evaluation uses exploratory frozen TRAIN subset only. Original TEST consumed.",
            "No private patient IDs, individual scores, chosen questions or transcripts exported."
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat()
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2)+chr(10))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True,type=Path)
    p.add_argument("--manifest-db",required=True,type=Path)
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--questions",type=int,default=4)
    p.add_argument("--max-patients",type=int,default=36)
    p.add_argument("--max-new-tokens",type=int,default=48)
    p.add_argument("--seed",type=int,default=20261018)
    a=p.parse_args()
    print(json.dumps(run(a.data_root,a.manifest_db,a.report_dir,a.model_dir,
                         a.questions,a.max_patients,a.seed,a.max_new_tokens),
                     indent=2,ensure_ascii=False))
    print("Upload only aggregate report ZIP; private data remain on cluster.")
if __name__=="__main__":
    main()
