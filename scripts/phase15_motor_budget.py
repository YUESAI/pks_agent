#!/usr/bin/env python3
"""Phase 15: PPMI TRAIN-only budget curves and motor-state sensitivity.

Compare adaptive vs fixed/random under budgets 1..5, with motor included or
excluded, on (A) all TRAIN and (B) motor-UNSPECIFIED-ready TRAIN subjects.
The selected subset is a selection-biased sensitivity analysis, NOT a new
external or held-out cohort. Labels only fit inner CV folds and score results.
No val/test/BioFIND measurements, IDs, scores or trajectories are exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase13_numeric_policy_cv import (
    bin_value, evaluate, load_training, train_model
)
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase14_feature_attribution import paired_bootstrap
from scripts.budgeted_env_v0 import ACTION_SPECS

POLICIES = ("clinical", "smell_first", "random", "adaptive")
ORDERS = {
    "clinical": ("age", "sleep_item", "autonomic_item", "cognitive_test",
                 "motor_exam", "smell_test"),
    "smell_first": ("age", "smell_test", "motor_exam", "cognitive_test",
                    "sleep_item", "autonomic_item"),
}
VIEWS = ("motor_allowed", "motor_excluded")
POPULATIONS = ("all_train", "motor_unspecified_ready")


def load_unspecified_train_ids(manifest_path, train_ids):
    """Read only PPMI TRAIN readiness status, never labels or raw scores."""
    db = sqlite3.connect("file:" + str(manifest_path.resolve()) + "?mode=ro",
                         uri=True)
    ready = set()
    try:
        ordered = sorted(train_ids)
        for start in range(0, len(ordered), 200):
            ids = ordered[start:start+200]
            q = ",".join("?" for _ in ids)
            for pid, in db.execute(
                "SELECT patient_id FROM episode_actions "
                "WHERE cohort='PPMI' AND action_key='updrs3_total@UNSPECIFIED' "
                "AND status='ready' AND patient_id IN ("+q+")", ids
            ).fetchall():
                ready.add(pid)
    finally:
        db.close()
    return ready


def sigmoid(score):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, score))))


def entropy(p):
    return (-p*math.log(p)-(1-p)*math.log(1-p)) if 0 < p < 1 else 0.0


def expected_gain(model, action, score):
    """Information gain from observed clinical value, not from missingness."""
    prior = sigmoid(score)
    before = entropy(prior)
    posterior_entropy = 0.0
    for p_pd,p_hc in zip(model[action]["PD"],model[action]["HC"]):
        mass = prior*p_pd + (1-prior)*p_hc
        if mass > 0:
            posterior_entropy += mass*entropy(prior*p_pd/mass)
    return max(0.0, before - posterior_entropy)


class EvidenceOracle:
    """The policy obtains values exclusively via request()."""
    def __init__(self, private_values, budget):
        self._private_values = private_values
        self.budget = budget
        self.spent = 0
        self.requested = set()
        self.returned = 0

    def request(self, action):
        if action in self.requested:
            raise ValueError("Duplicate test request")
        cost = ACTION_SPECS[action]["cost"]
        if self.spent + cost > self.budget:
            raise ValueError("Budget exceeded")
        self.requested.add(action)
        self.spent += cost
        value = self._private_values.get(action)
        if bin_value(action, value) is None:
            return None
        self.returned += 1
        return value


def random_order(seed, patient_id, view):
    options = [a for a in ACTION_SPECS if a != "age" and
               (view != "motor_excluded" or a != "motor_exam")]
    key = hashlib.sha256((str(seed)+"|"+view+"|"+patient_id).encode()).digest()
    random.Random(int.from_bytes(key[:8], "big")).shuffle(options)
    return ("age",) + tuple(options)


def play(model, success, values, policy, pid, budget, seed, view):
    allowed = [a for a in ACTION_SPECS
               if view != "motor_excluded" or a != "motor_exam"]
    env = EvidenceOracle(values, budget)
    score = 0.0

    def request(action):
        nonlocal score
        raw = env.request(action)
        if raw is not None:
            category = bin_value(action, raw)
            score += math.log(model[action]["PD"][category] /
                              model[action]["HC"][category])

    # Zero-cost baseline background offered identically through a request.
    request("age")
    if policy == "adaptive":
        while True:
            possibilities = [a for a in allowed if a not in env.requested
                             and ACTION_SPECS[a]["cost"] <= budget-env.spent]
            if not possibilities:
                break
            possibilities.sort(key=lambda a: (
                -expected_gain(model,a,score)*success[a]/
                    max(1,ACTION_SPECS[a]["cost"]), a))
            request(possibilities[0])
    else:
        order = (random_order(seed,pid,view) if policy=="random"
                 else ORDERS[policy])
        for a in order:
            if a not in allowed or a in env.requested:
                continue
            if ACTION_SPECS[a]["cost"] <= budget-env.spent:
                request(a)
    return score, env.spent, env.returned


def write_csv(path, names, records):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def analyse(source, engine, manifest, splits, output, folds, budgets, seed,
            bootstraps=300, minimum=10):
    source, manifest, splits, output = [
        Path(path).expanduser().resolve()
        for path in (source,manifest,splits,output)
    ]
    if not all(x.is_file() for x in (source,manifest,splits)):
        raise FileNotFoundError("Missing source, manifest or frozen split")
    if output in (source,manifest,splits):
        raise ValueError("Output directory equals input database")
    if folds<2 or not budgets or min(budgets)<0 or bootstraps<100 or minimum<10:
        raise ValueError("Invalid folds/budgets/bootstraps/minimum")

    all_data = load_training(source,engine,manifest,splits)
    ids = set(all_data)
    unspecified = load_unspecified_train_ids(manifest,ids)
    if not unspecified.issubset(ids):
        raise ValueError("Motor-ready subjects outside frozen PPMI TRAIN")
    subsets = {
        "all_train": sorted(ids),
        "motor_unspecified_ready": sorted(unspecified),
    }
    metrics_rows = []
    bootstrap_rows = []
    population_rows = []

    for population, pids in subsets.items():
        rows = [(pid,*all_data[pid]) for pid in pids]
        labels = Counter(label for _,label,_,_ in rows)
        population_rows.extend([
            {"population":population,"label":lab,"patients":n}
            for lab,n in sorted(labels.items())
        ])
        if min(labels["PD"],labels["HC"])<folds:
            raise ValueError("Not enough PD/HC for stratified CV: "+population)
        assignments = assign_folds(rows,folds,seed)
        # Same folds and fitted models across every budget/policy/view.
        predictions = defaultdict(list)
        for held in range(folds):
            fit_rows = [(label,anchor,values)
                        for pid,label,anchor,values in rows
                        if assignments[pid]!=held]
            fold_eval = [(pid,label,values)
                         for pid,label,anchor,values in rows
                         if assignments[pid]==held]
            if {x[0] for x in fit_rows}!={"PD","HC"} or not fold_eval:
                raise ValueError("Invalid inner fold")
            model, success = train_model(fit_rows)
            for pid,label,values in fold_eval:
                for view in VIEWS:
                    for budget in budgets:
                        for policy in POLICIES:
                            score,cost,n_returned = play(
                                model, success, values, policy, pid,
                                budget, seed, view)
                            predictions[(view,budget,policy)].append(
                                (pid,score,label=="PD",cost,n_returned))

        for view in VIEWS:
            for budget in budgets:
                for policy in POLICIES:
                    data = predictions[(view,budget,policy)]
                    if len(data)!=len(rows):
                        raise ValueError("Missing out-of-fold patient score")
                    result = evaluate([
                        (score,truth,cost,n_returned)
                        for pid,score,truth,cost,n_returned in data
                    ])
                    metrics_rows.append({
                        "population":population,"action_space":view,
                        "budget":budget,"policy":policy,
                        "evaluated_split":"PPMI_train_out_of_fold",
                        **result,
                    })
                if budget not in (2,3,5):
                    continue
                clinical = {pid:(score,truth) for pid,score,truth,cost,n in
                            predictions[(view,budget,"clinical")]}
                adaptive = {pid:(score,truth) for pid,score,truth,cost,n in
                            predictions[(view,budget,"adaptive")]}
                if clinical.keys()!=adaptive.keys():
                    raise ValueError("Unpaired patient scores")
                pairs = []
                for pid in sorted(clinical):
                    a,lab = clinical[pid]
                    b,lab2 = adaptive[pid]
                    if lab!=lab2:
                        raise ValueError("Inconsistent patient outcome")
                    pairs.append((lab,a,b))
                for item in paired_bootstrap(
                        pairs,bootstraps,seed+budget*11+
                        (1 if view=="motor_allowed" else 2)):
                    bootstrap_rows.append({
                        "population":population,"action_space":view,
                        "budget":budget,**item,
                    })

    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"budget_policy_cv.csv",
        ["population","action_space","budget","policy","evaluated_split",
         "n","n_PD","n_HC","balanced_accuracy","macro_f1","auroc",
         "mean_cost","mean_results"],metrics_rows)
    write_csv(output/"paired_policy_differences.csv",
        ["population","action_space","budget","comparison","metric",
         "point_difference","bootstrap_lower_95","bootstrap_upper_95",
         "bootstrap_replicates"],bootstrap_rows)
    write_csv(output/"population_counts.csv",
        ["population","label","patients"],population_rows)
    report = {
        "status":"PPMI_TRAIN_ONLY_exploratory_motor_sensitivity",
        "train_total":len(all_data),"motor_unspecified_ready":len(unspecified),
        "budgets":list(budgets),"folds":folds,"seed":seed,
        "bootstraps":bootstraps,
        "warnings":[
            "Shared motor-condition subset is selection-biased, NOT matched cohort.",
            "UNSPECIFIED is a missing metadata state, NOT untreated OFF condition.",
            "Masked motor means that the policy cannot request it and never sees the value.",
            "PD/HC candidate labels and numeric coding are not clinically validated.",
            "Predictive motor severity is expected in PD vs HC and may reflect circularity.",
            "Paired bootstrap intervals are descriptive and not multiplicity-adjusted.",
            "No PPMI val/test or BioFIND measurements loaded.",
            "No patient-level data in these reports; local databases are restricted.",
        ],
        "created_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db",type=Path,required=True)
    parser.add_argument("--source-engine",choices=("duckdb","sqlite"),
                        default="duckdb")
    parser.add_argument("--manifest-db",type=Path,required=True)
    parser.add_argument("--split-db",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,
                        default=Path("outputs/phase15_motor_budget"))
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--budgets",type=int,nargs="+",default=[1,2,3,4,5])
    parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--bootstraps",type=int,default=300)
    parser.add_argument("--min-cell-count",type=int,default=10)
    args=parser.parse_args()
    result=analyse(args.source_db,args.source_engine,args.manifest_db,
                   args.split_db,args.report_dir,args.folds,args.budgets,
                   args.seed,args.bootstraps,args.min_cell_count)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print("Upload only aggregate Phase15 reports; NEVER any patient database.")


if __name__=="__main__":
    main()
