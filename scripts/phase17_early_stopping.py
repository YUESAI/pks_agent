#!/usr/bin/env python3
"""Phase 17 — Train-only cost-aware early-stopping diagnostic policy audit.

Use ONLY frozen PPMI TRAIN patients. Candidate PD/HC labels are NOT clinically
validated. Every model is fitted within the inner-training fold and evaluated
on disjoint held-out patients from the TRAIN partition. Never read val/test or
BioFIND records. No patient-level values, predictions or IDs are exported.

All policies see only observations they request. The optional stop decision
uses uncalibrated Naive Bayes score confidence; stopping thresholds here are
prespecified exploratory sweeps, NOT calibrated clinical risk thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase13_numeric_policy_cv import (
    bin_value, evaluate, load_training, train_model
)
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase15_motor_budget import EvidenceOracle, expected_gain
from scripts.budgeted_env_v0 import ACTION_SPECS

POLICIES = ("clinical", "motor_first", "static_ig", "adaptive_ig")
SPACES = ("motor_allowed", "motor_excluded")
ORDERS = {
    "clinical": ("age", "sleep_item", "autonomic_item", "cognitive_test",
                 "motor_exam", "smell_test"),
    "motor_first": ("age", "motor_exam", "cognitive_test", "sleep_item",
                    "autonomic_item", "smell_test"),
}
# None is the no-stopping reference. Values are MODEL scores, not calibrated risks.
STOP_THRESHOLDS = (None, 0.70, 0.80, 0.90, 0.95, 0.99)


def logistic(score):
    return 1.0 / (1.0 + math.exp(-max(-500., min(500., score))))


def allowed(space):
    if space not in SPACES:
        raise ValueError("Unknown motor condition")
    return tuple(a for a in ACTION_SPECS
                 if space != "motor_excluded" or a != "motor_exam")


def global_rank(model, success, actions):
    # Fixed order fitted on this inner training fold: NEVER patient-dependent.
    return sorted((a for a in actions if a != "age"),
                  key=lambda a: (-expected_gain(model, a, 0.) * success[a] /
                                 max(1, ACTION_SPECS[a]["cost"]), a))


def replay(model, success, values, policy, space, budget, threshold):
    """Return (diagnostic log-odds, spent, returned, requests, stop_reason).

    Even if unavailable, a test costs money. Missingness never updates score.
    No individual-level readiness masks or medication metadata are revealed.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if threshold is not None and not (0.5 < threshold < 1):
        raise ValueError("Threshold must be between 0.5 and 1")
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    actions = allowed(space)
    oracle = EvidenceOracle(values, budget)
    score = 0.0
    paid_values = 0
    stop_reason = "budget_or_catalog_exhausted"

    def request(action):
        nonlocal score, paid_values
        raw = oracle.request(action)
        if raw is not None:
            category = bin_value(action, raw)
            if category is not None:
                score += math.log(model[action]["PD"][category] /
                                  model[action]["HC"][category])
                if ACTION_SPECS[action]["cost"] > 0:
                    paid_values += 1

    def should_stop():
        if threshold is None or paid_values < 1:
            return False
        p = logistic(score)
        return max(p, 1-p) >= threshold

    # Everyone explicitly requests the same zero-cost age background.
    request("age")
    if policy in ORDERS:
        order = ORDERS[policy]
        for action in order:
            if action not in actions or action in oracle.requested:
                continue
            if ACTION_SPECS[action]["cost"] > budget - oracle.spent:
                continue
            request(action)
            if should_stop():
                stop_reason = "score_threshold"
                break
    else:
        fixed = global_rank(model, success, actions)
        while True:
            available = [a for a in actions if a not in oracle.requested
                         and ACTION_SPECS[a]["cost"] <= budget-oracle.spent]
            if not available:
                break
            if policy == "static_ig":
                chosen = next(a for a in fixed if a in available)
            else:
                chosen = min(available, key=lambda a: (
                    -expected_gain(model, a, score) * success[a] /
                    max(1, ACTION_SPECS[a]["cost"]), a))
            request(chosen)
            if should_stop():
                stop_reason = "score_threshold"
                break
    return (score, oracle.spent, oracle.returned,
            len(oracle.requested), stop_reason)


def brier_by_class(rows):
    """Class-balanced Brier, to counter the strong 728:99 prevalence skew."""
    values = defaultdict(list)
    for score, truth, cost, returned, requested, reason in rows:
        p = logistic(score)
        values[truth].append((p - float(truth)) ** 2)
    if not values[True] or not values[False]:
        raise ValueError("Need both labels for Brier score")
    return round((sum(values[True])/len(values[True]) +
                  sum(values[False])/len(values[False]))/2, 5)


def export_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(source, engine, manifest, splits, report_dir, folds, budgets, seed):
    source, manifest, splits, report_dir = (
        Path(p).expanduser().resolve()
        for p in (source, manifest, splits, report_dir))
    if not all(p.is_file() for p in (source, manifest, splits)):
        raise FileNotFoundError("Missing source DB, manifest or locked split DB")
    if report_dir in (source, manifest, splits) or folds < 2 or not budgets or min(budgets) < 0:
        raise ValueError("Invalid output, folds or budgets")
    data = load_training(source, engine, manifest, splits)
    population = [(pid,label,anchor,values) for pid,(label,anchor,values)
                  in sorted(data.items())]
    assignment = assign_folds(population, folds, seed)
    results = defaultdict(list)
    for fold in range(folds):
        fit = [(label,anchor,values) for pid,label,anchor,values in population
               if assignment[pid] != fold]
        held = [(pid,label,values) for pid,label,anchor,values in population
                if assignment[pid] == fold]
        if not held or {label for label,_,_ in fit} != {"PD","HC"}:
            raise ValueError("Empty held-out fold or missing label in fit")
        model, success = train_model(fit)
        for pid,label,values in held:
            for space in SPACES:
                for budget in budgets:
                    for policy in POLICIES:
                        for threshold in STOP_THRESHOLDS:
                            result = replay(model,success,values,policy,space,
                                            budget,threshold)
                            results[(space,budget,policy,threshold)].append(
                                (result[0],label=="PD",result[1],result[2],
                                 result[3],result[4]))
    rows = []
    for space in SPACES:
        for budget in budgets:
            for policy in POLICIES:
                for threshold in STOP_THRESHOLDS:
                    group = results[(space,budget,policy,threshold)]
                    if len(group) != len(population):
                        raise ValueError("Lost patient during OOF replay")
                    classic = evaluate([(s,l,c,r) for s,l,c,r,q,reason in group])
                    stopped = sum(reason=="score_threshold" for s,l,c,r,q,reason in group)
                    row = {
                        "action_space":space, "budget":budget, "policy":policy,
                        "threshold": "no_stop" if threshold is None else str(threshold),
                        "evaluated_population":"PPMI_train_out_of_fold",
                        "mean_actions":round(sum(q for s,l,c,r,q,reason in group)/len(group),3),
                        "stopped_by_threshold_pct":round(100*stopped/len(group),2),
                        "class_balanced_brier":brier_by_class(group),
                        **classic,
                    }
                    rows.append(row)
    report_dir.mkdir(parents=True,exist_ok=True)
    export_csv(report_dir/"stopping_policy_cv.csv",
               ["action_space","budget","policy","threshold","evaluated_population",
                "n","n_PD","n_HC","balanced_accuracy","macro_f1","auroc",
                "mean_cost","mean_actions","mean_results",
                "stopped_by_threshold_pct","class_balanced_brier"],
               rows)
    counts=Counter(label for pid,label,anchor,val in population)
    summary={
        "research_status":"TRAIN_only_exploratory_stopping_policy_sweep",
        "n_train":len(population),"PD":counts["PD"],"HC":counts["HC"],
        "folds":folds,"budgets":list(budgets),"stop_thresholds":
            ["no_stop" if x is None else x for x in STOP_THRESHOLDS],
        "policies":list(POLICIES),"action_spaces":list(SPACES),
        "seed":seed,"output_rows":len(rows),
        "limitations":[
            "Model posterior scores are NOT calibrated clinical probabilities.",
            "A threshold triggers only after a paid test returned valid numeric evidence.",
            "Candidate labels are retrospective and may have motor-diagnosis circularity.",
            "ON/OFF motor source conditions are still mixed when motor is available.",
            "All policies use the same within-fold model and patient evidence oracle.",
            "No PPMI val/test or BioFIND patient records are read.",
            "Thresholds are exploratory, not selected or approved for clinical use.",
            "Static information gain may be sufficient; stopping tests cost savings, not proof of RSI.",
            "Reports contain aggregates only; local DBs must not leave the cluster.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (report_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),default="duckdb")
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--split-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,default=Path("outputs/phase17_stopping"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--budgets",type=int,nargs="+",default=[2,3,5])
    p.add_argument("--seed",type=int,default=20261018)
    args=p.parse_args()
    result=audit(args.source_db,args.source_engine,args.manifest_db,
                 args.split_db,args.report_dir,args.folds,args.budgets,args.seed)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Share only the aggregated reports, not the private research databases.")


if __name__=="__main__":
    main()
