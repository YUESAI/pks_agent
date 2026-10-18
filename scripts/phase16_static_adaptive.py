#!/usr/bin/env python3
"""Phase 16: test whether posterior adaptation improves over learned static order.

Run on locked PPMI TRAIN only, five-fold out-of-fold. Same numeric model,
patient folds, budgets and actions for every policy. Models/ordering priorities
are fitted on inner TRAIN folds only. No val/test/BioFIND rows are loaded.

Static IG learns one global per-action gain/cost ranking from the fit fold,
but DOES NOT change the ranking in response to patient observations.
Adaptive IG recomputes expected gain from the observed diagnostic posterior.

All exports aggregate; keep source/manifest/split databases private.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase13_numeric_policy_cv import (
    bin_value, evaluate, load_training, train_model
)
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase14_feature_attribution import paired_bootstrap
from scripts.phase15_motor_budget import EvidenceOracle, expected_gain
from scripts.budgeted_env_v0 import ACTION_SPECS

POLICIES = ("clinical", "motor_first", "static_ig", "adaptive_ig")
ACTION_SPACES = ("motor_allowed", "motor_excluded")
ORDERS = {
    "clinical": ("age", "sleep_item", "autonomic_item", "cognitive_test",
                 "motor_exam", "smell_test"),
    "motor_first": ("age", "motor_exam", "cognitive_test", "sleep_item",
                    "autonomic_item", "smell_test"),
}


def allowed_actions(space):
    if space not in ACTION_SPACES:
        raise ValueError("Unknown action space")
    return [a for a in ACTION_SPECS if
            not (space == "motor_excluded" and a == "motor_exam")]


def ranking_utility(model, success, action, score):
    return (expected_gain(model, action, score) * success[action] /
            max(1, ACTION_SPECS[action]["cost"]))


def run_episode(model, success, values, policy, budget, space):
    """Keep evidence private until requested; no test-availability peeking."""
    allowed = allowed_actions(space)
    env = EvidenceOracle(values, budget)
    log_odds = 0.0
    actions_taken = []

    def order_one(action):
        nonlocal log_odds
        if action not in allowed or action in env.requested:
            raise ValueError("Invalid or repeated policy action")
        raw_value = env.request(action)
        actions_taken.append(action)
        if raw_value is not None:
            b = bin_value(action, raw_value)
            if b is not None:
                log_odds += math.log(model[action]["PD"][b] /
                                     model[action]["HC"][b])

    # Uniform zero-cost initial request, including if age is missing.
    order_one("age")
    if policy in ORDERS:
        for action in ORDERS[policy]:
            if action not in allowed or action in env.requested:
                continue
            if ACTION_SPECS[action]["cost"] <= env.budget-env.spent:
                order_one(action)
    elif policy in ("static_ig", "adaptive_ig"):
        static_rank = sorted(
            (a for a in allowed if a != "age"),
            key=lambda a: (-ranking_utility(model, success, a, 0.0), a))
        while True:
            affordable = [a for a in allowed if a not in env.requested
                          and ACTION_SPECS[a]["cost"] <= env.budget-env.spent]
            if not affordable:
                break
            if policy == "static_ig":
                # Order never changes based on the patient's responses.
                chosen = next(a for a in static_rank if a in affordable)
            else:
                chosen = min(
                    affordable,
                    key=lambda a: (-ranking_utility(model,success,a,log_odds),a))
            order_one(chosen)
    else:
        raise ValueError("Unknown policy")
    return log_odds, env.spent, env.returned, tuple(actions_taken)


def export_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(a, b, budget, bootstraps, seed, minimum):
    """Patient-paired comparison. Individual scores are never exported."""
    if a.keys() != b.keys():
        raise ValueError("Policy evaluations involve different patients")
    pairs = []
    different = 0
    for pid in sorted(a):
        score_a, label_a, cost_a, returned_a, sequence_a = a[pid]
        score_b, label_b, cost_b, returned_b, sequence_b = b[pid]
        if label_a != label_b:
            raise ValueError("Inconsistent true labels")
        pairs.append((label_a, score_a, score_b))
        if sequence_a != sequence_b:
            different += 1
    # Paired bootstrap delta is B - A, both scored on the SAME patients.
    differences = paired_bootstrap(pairs, bootstraps, seed)
    for item in differences:
        item.update({
            "comparison": "adaptive_ig_minus_" + item["comparison"].split("_minus_")[-1],
            "budget": budget,
            "different_request_sequences_pct": round(100*different/len(pairs),2),
            "n_subjects": len(pairs),
        })
    return differences


def audit(source,engine,manifest,splits,out,folds,budgets,seed,bootstraps,minimum):
    source,manifest,splits,out = [
        Path(x).expanduser().resolve() for x in (source,manifest,splits,out)]
    if not all(path.is_file() for path in (source,manifest,splits)):
        raise FileNotFoundError("Source, manifest or split DB unavailable")
    if out in (source,manifest,splits):
        raise ValueError("Report dir must differ from databases")
    if folds<2 or not budgets or min(budgets)<0 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid folds, budgets, bootstrap or cell size")
    patients = load_training(source,engine,manifest,splits)
    records = [(pid,label,anchor,values) for pid,(label,anchor,values)
               in sorted(patients.items())]
    assignment = assign_folds(records,folds,seed)
    results = defaultdict(dict)
    action_counts = Counter()
    for held in range(folds):
        fit = [(label,anchor,values) for pid,label,anchor,values in records
               if assignment[pid]!=held]
        test = [(pid,label,values) for pid,label,anchor,values in records
                if assignment[pid]==held]
        if not test or {label for label,anchor,values in fit}!={"PD","HC"}:
            raise ValueError("Invalid out-of-fold split")
        model,success = train_model(fit)
        for pid,label,values in test:
            for space in ACTION_SPACES:
                for budget in budgets:
                    for policy in POLICIES:
                        score,cost,n_results,requests=run_episode(
                            model,success,values,policy,budget,space)
                        key=(space,budget,policy)
                        if pid in results[key]:
                            raise ValueError("Patient scored twice")
                        results[key][pid]=(score,label=="PD",cost,n_results,requests)
                        for action in requests:
                            action_counts[(space,budget,policy,action)]+=1

    metrics=[]
    for space in ACTION_SPACES:
        for budget in budgets:
            for policy in POLICIES:
                batch=results[(space,budget,policy)]
                if len(batch)!=len(records):
                    raise ValueError("Missing out-of-fold episodes")
                row=evaluate([(score,label,cost,returned)
                              for score,label,cost,returned,req in batch.values()])
                metrics.append(dict(action_space=space,budget=budget,policy=policy,
                                    evaluated_split="PPMI_train_out_of_fold",**row))
    comparisons=[]
    for space in ACTION_SPACES:
        for budget in budgets:
            baseline = results[(space,budget,"adaptive_ig")]
            for other in ("clinical","motor_first","static_ig"):
                a=results[(space,budget,other)]
                delta=compare(a,baseline,budget,bootstraps,
                              seed + budget*17 + (1 if space=="motor_allowed" else 2),minimum)
                for item in delta:
                    item["comparison"]="adaptive_ig_minus_"+other
                    item["action_space"]=space
                    comparisons.append(item)
    action_rows = [
        dict(action_space=space,budget=budget,policy=policy,action=action,
             n_requests=n)
        for (space,budget,policy,action),n in sorted(action_counts.items())
        if n>=minimum]
    out.mkdir(parents=True,exist_ok=True)
    export_csv(out/"policy_comparison_cv.csv",
        ["action_space","budget","policy","evaluated_split","n","n_PD",
         "n_HC","balanced_accuracy","macro_f1","auroc","mean_cost","mean_results"],
        metrics)
    export_csv(out/"paired_adaptivity.csv",
        ["action_space","budget","comparison","metric","point_difference",
         "bootstrap_lower_95","bootstrap_upper_95","bootstrap_replicates",
         "different_request_sequences_pct","n_subjects"],comparisons)
    export_csv(out/"requested_action_counts.csv",
        ["action_space","budget","policy","action","n_requests"],action_rows)
    summary={
        "status":"PPMI_TRAIN_ONLY_static_vs_dynamic_policy_audit",
        "n_train_patients":len(records),
        "n_PD":sum(label=="PD" for pid,label,anchor,val in records),
        "n_HC":sum(label=="HC" for pid,label,anchor,val in records),
        "budgets":list(budgets),"policies":list(POLICIES),"action_spaces":list(ACTION_SPACES),
        "folds":folds,"seed":seed,"bootstrap_replicates":bootstraps,
        "interpretation":[
            "Static IG and adaptive IG use the same inner-fold-fitted numeric model.",
            "Static IG uses a fixed global expected IG/cost ordering.",
            "Adaptive IG recomputes expected IG/cost after requested clinical results.",
            "Both policies can skip unaffordable actions; no per-patient readiness access.",
            "Paired differences use matched out-of-fold PPMI TRAIN predictions.",
            "Request sequence differences measure adaptation but not clinical utility.",
            "A motor-first fixed baseline is essential because motor dominates diagnosis.",
            "Candidate outcomes, clinical codebook interpretation and motor-state mixtures remain unvalidated.",
            "All data are retrospective; neither date order nor prospective clinical performance is established.",
            "No validation/test/BioFIND observations or individual predictions are used/exported.",
            "Bootstrap intervals are descriptive and unadjusted for multiple testing.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--split-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase16_static_vs_adaptive"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--budgets",type=int,nargs="+",default=[1,2,3,4,5])
    p.add_argument("--bootstraps",type=int,default=300)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--min-cell-count",type=int,default=10)
    a=p.parse_args()
    result=audit(a.source_db,a.source_engine,a.manifest_db,a.split_db,
                 a.report_dir,a.folds,a.budgets,a.seed,a.bootstraps,
                 a.min_cell_count)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Only upload aggregate Phase16 reports; patient DBs remain on cluster.")

if __name__=="__main__":
    main()
