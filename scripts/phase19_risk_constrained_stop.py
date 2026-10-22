#!/usr/bin/env python3
"""Phase 19 — Inner-fold stopping-threshold selection, PPMI TRAIN only.

Within each outer CV fit fold: first reserve a disjoint threshold-selection
subset, then split the remaining subjects into model proper-fit and isotonic
calibration subsets. Choose a stopping threshold only on the selection
subjects under a predefined maximum class-balanced accuracy degradation
(upper 95% paired bootstrap bound), then evaluate on outer held-out TRAIN
subjects. No PPMI validation/test or BioFIND patient rows are read.

Exploratory, retrospective candidate PD/HC outcomes only. This is NOT
clinical calibration or a prospective test-ordering reconstruction.
No patient-level values/IDs/predictions exported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase13_numeric_policy_cv import load_training, train_model
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase18_nested_calibration import (
    calibration_split, trace_static, fit_calibrator, sigmoid
)
from scripts.phase10_leakage_splits import auc

SPACES = ("motor_allowed", "motor_excluded")
CANDIDATES = (0.70, 0.80, 0.90, 0.95, 0.99)
POLICIES = ("no_stop", "fixed_0.90", "inner_selected")


def select(trajectory, policy, calibrator, threshold):
    """Stopping can only occur after a valid paid result.

    Calibration direction must agree with uncalibrated score direction.
    Otherwise the model could stop 'confidently' on an opposite prediction.
    """
    if policy == "no_stop":
        score, cost, n = trajectory[-1]
        return score, cost, n, False, False
    if not 0.5 < threshold < 1:
        raise ValueError("Stopping threshold must lie between 0.5 and 1")
    disagreements = False
    for score, cost, n in trajectory:
        if n < 1:
            continue
        calibrated = calibrator.predict(score)
        if max(calibrated, 1-calibrated) < threshold:
            continue
        agrees = ((score >= 0) == (calibrated >= 0.5))
        if not agrees:
            disagreements = True
            continue
        return score, cost, n, True, disagreements
    score, cost, n = trajectory[-1]
    return score, cost, n, False, disagreements


def score_metrics(rows):
    """Rows: (is_PD, score, examination_cost, stopped, disagreement)."""
    n = len(rows)
    pd = sum(r[0] for r in rows)
    hc = n-pd
    if not n or not pd or not hc:
        raise ValueError("Metrics require both PD and HC patients")
    tp = sum(truth and score >= 0 for truth, score, cost, stop, disagree in rows)
    tn = sum((not truth) and score < 0
             for truth, score, cost, stop, disagree in rows)
    tpr = tp/pd
    tnr = tn/hc
    probability = [(score, truth) for truth, score, cost, stop, disagree in rows]
    curve = auc(probability)
    return {
        "balanced_accuracy": (tpr+tnr)/2,
        "sensitivity_PD": tpr,
        "specificity_HC": tnr,
        "false_negative_rate_PD": 1-tpr,
        "false_positive_rate_HC": 1-tnr,
        "auroc": curve,
        "mean_cost": sum(r[2] for r in rows)/n,
        "mean_cost_PD": sum(r[2] for r in rows if r[0])/pd,
        "mean_cost_HC": sum(r[2] for r in rows if not r[0])/hc,
        "stopped_pct": 100*sum(r[3] for r in rows)/n,
        "direction_disagreement_pct": 100*sum(r[4] for r in rows)/n,
    }


def strata_bootstrap(base, candidate, iterations, seed, metrics):
    """Return paired bootstrap upper bound on BA loss / all metric deltas.

    Both input lists are ordered by identical patient IDs. Resample separately
    inside PD and HC, retaining patient pairs for each bootstrap replicate.
    """
    if len(base) != len(candidate):
        raise ValueError("Unequal paired evaluation lengths")
    positives = [i for i, x in enumerate(base) if x[0]]
    negatives = [i for i, x in enumerate(base) if not x[0]]
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("Need >=2 PD and HC for paired uncertainty")
    if any(x[0] != y[0] for x, y in zip(base, candidate)):
        raise ValueError("Unequal paired patient labels")
    rng = random.Random(seed)
    samples = {m: [] for m in metrics}
    for _ in range(iterations):
        positions = [positives[rng.randrange(len(positives))] for _ in positives]
        positions.extend(negatives[rng.randrange(len(negatives))] for _ in negatives)
        first = score_metrics([base[i] for i in positions])
        second = score_metrics([candidate[i] for i in positions])
        for metric in metrics:
            samples[metric].append(second[metric]-first[metric])
    point_a = score_metrics(base)
    point_b = score_metrics(candidate)
    output = {}
    for metric, differences in samples.items():
        differences.sort()
        lo = differences[max(0, int(.025*(len(differences)-1)))]
        hi = differences[min(len(differences)-1,
                             math.ceil(.975*(len(differences)-1)))]
        output[metric] = (point_b[metric]-point_a[metric], lo, hi)
    return output


def evaluate_trajectory(traces, threshold):
    """Each trace: (patient ID, label, trajectory). Patient IDs stay in RAM."""
    output = {}
    for pid, label, trajectory, calibrator in traces:
        score, cost, n, stopped, disagrees = select(
            trajectory, "stop", calibrator, threshold)
        output[pid] = (label=="PD", score, cost, stopped, disagrees)
    return output


def no_stop_rows(traces):
    return {
        pid: (label=="PD", *select(traj, "no_stop", calibrator, .90)[:2],
              False, False)
        for pid, label, traj, calibrator in traces
    }


def choose_threshold(traces, risk_tolerance, bootstraps, seed):
    """Maximize savings with a conservative upper bound on BA loss.

    Threshold search happens only on disjoint selection subjects. If no
    candidate satisfies bound, do not stop. Conservative tie breaking
    prefers a HIGHER threshold.
    """
    baseline = no_stop_rows(traces)
    ids = sorted(baseline)
    baseline_rows = [baseline[pid] for pid in ids]
    baseline_cost = score_metrics(baseline_rows)["mean_cost"]
    options = []
    for threshold in CANDIDATES:
        variant = evaluate_trajectory(traces, threshold)
        current = [variant[pid] for pid in ids]
        uncertainty = strata_bootstrap(
            baseline_rows, current, bootstraps,
            seed + round(threshold*1000), ("balanced_accuracy",))
        # Loss = BA(no_stop)-BA(stop), upper bound = -lower delta CI.
        delta, lower, upper = uncertainty["balanced_accuracy"]
        upper_loss = -lower
        cost = score_metrics(current)["mean_cost"]
        if upper_loss <= risk_tolerance and cost < baseline_cost - 1e-9:
            options.append((cost, -threshold, threshold, upper_loss))
    if not options:
        return None, baseline_cost, None
    cost, _, threshold, upper_loss = min(options)
    return threshold, cost, upper_loss


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit(source, engine, manifest, split, output, folds=5, budget=5,
          seed=20261018, tolerance=.02, bootstraps=300):
    source, manifest, split, output = [
        Path(p).expanduser().resolve() for p in
        (source, manifest, split, output)
    ]
    if not all(p.is_file() for p in (source, manifest, split)):
        raise FileNotFoundError("Missing source, manifest or split database")
    if output in (source, manifest, split):
        raise ValueError("Output path overlaps protected database")
    if (folds < 2 or budget < 1 or bootstraps < 100
            or tolerance < 0 or tolerance >= 0.5):
        raise ValueError("Invalid folds, budget, tolerance or bootstrap count")
    training = load_training(source, engine, manifest, split)
    records = [(pid, label, anchored, values)
               for pid, (label, anchored, values) in sorted(training.items())]
    fold_map = assign_folds(records, folds, seed)
    predictions = {space: defaultdict(dict) for space in SPACES}
    fold_choices = []
    for fold in range(folds):
        outer_fit = [r for r in records if fold_map[r[0]] != fold]
        outer_eval = [r for r in records if fold_map[r[0]] == fold]
        if not outer_eval:
            raise ValueError("Empty outer evaluation fold")
        # Selection never overlaps the model fitting or calibration partition.
        remaining, selection = calibration_split(
            outer_fit, seed + 127*fold, fraction=.25)
        proper, calibration = calibration_split(
            remaining, seed + 283*fold, fraction=1/3)
        model, success = train_model(
            [(label, anchor, values) for pid, label, anchor, values in proper])
        for space in SPACES:
            calibrator = fit_calibrator(model, success, calibration, space, budget)
            select_traces = [
                (pid, label, trace_static(model,success,values,space,budget),
                 calibrator)
                for pid,label,anchor,values in selection
            ]
            threshold, selected_cost, upper_loss = choose_threshold(
                select_traces, tolerance, bootstraps, seed+fold*1000+
                (1 if space=="motor_allowed" else 2))
            fold_choices.append({
                "outer_fold":fold, "action_space":space,
                "model_fit_n":len(proper),"calibration_n":len(calibration),
                "selection_n":len(selection),"outer_eval_n":len(outer_eval),
                "selection_HC":sum(r[1]=="HC" for r in selection),
                "chosen_threshold":"no_stop" if threshold is None else threshold,
                "selection_mean_cost":round(selected_cost,3),
                "selection_upper_95_BA_loss":
                    "" if upper_loss is None else round(upper_loss,4),
            })
            for pid,label,anchor,values in outer_eval:
                traj = trace_static(model,success,values,space,budget)
                for policy in POLICIES:
                    if policy=="no_stop":
                        score,cost,n,stopped,disagree = select(
                            traj,"no_stop",calibrator,.90)
                    else:
                        level = (.90 if policy=="fixed_0.90" else threshold)
                        if level is None:
                            score,cost,n,stopped,disagree = select(
                                traj,"no_stop",calibrator,.90)
                        else:
                            score,cost,n,stopped,disagree = select(
                                traj,"stop",calibrator,level)
                    if pid in predictions[space][policy]:
                        raise ValueError("Outer subject evaluated twice")
                    predictions[space][policy][pid] = (
                        label=="PD",score,cost,stopped,disagree)
    summary_rows=[]
    comparison_rows=[]
    headline_metrics = (
        "balanced_accuracy","sensitivity_PD","specificity_HC",
        "false_positive_rate_HC","false_negative_rate_PD",
        "auroc","mean_cost",
    )
    for space in SPACES:
        control = predictions[space]["no_stop"]
        if len(control)!=len(records):
            raise ValueError("Missing outer predictions")
        for policy in POLICIES:
            mapping=predictions[space][policy]
            ids=sorted(mapping)
            values=[mapping[pid] for pid in ids]
            report=score_metrics(values)
            summary_rows.append({
                "action_space":space,"policy":policy,
                "evaluation":"PPMI_TRAIN_outer_OOF",
                "n":len(ids),"n_PD":sum(v[0] for v in values),
                "n_HC":sum(not v[0] for v in values),
                **{k:round(v,5) for k,v in report.items()},
            })
            if policy=="no_stop":
                continue
            control_values=[control[pid] for pid in ids]
            diffs=strata_bootstrap(
                control_values,values,bootstraps,
                seed + (7 if space=="motor_allowed" else 17)
                + (1 if policy=="fixed_0.90" else 2),
                headline_metrics)
            for metric,(point,low,high) in diffs.items():
                comparison_rows.append({
                    "action_space":space,
                    "comparison":policy+"_minus_no_stop",
                    "metric":metric,"difference":round(point,5),
                    "lower_95":round(low,5),"upper_95":round(high,5),
                    "bootstrap_replicates":bootstraps,
                })
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"selected_stopping_metrics.csv",
              ["action_space","policy","evaluation","n","n_PD","n_HC",
               "balanced_accuracy","sensitivity_PD","specificity_HC",
               "false_negative_rate_PD","false_positive_rate_HC",
               "auroc","mean_cost","mean_cost_PD","mean_cost_HC",
               "stopped_pct","direction_disagreement_pct"],summary_rows)
    write_csv(output/"selected_vs_no_stop_paired.csv",
              ["action_space","comparison","metric","difference",
               "lower_95","upper_95","bootstrap_replicates"],comparison_rows)
    write_csv(output/"inner_selection_by_fold.csv",
              ["outer_fold","action_space","model_fit_n","calibration_n",
               "selection_n","outer_eval_n","selection_HC",
               "chosen_threshold","selection_mean_cost",
               "selection_upper_95_BA_loss"],fold_choices)
    info={
        "status":"exploratory_PPMI_TRAIN_nested_threshold_selection",
        "n_train":len(records),
        "outer_folds":folds,"budget":budget,
        "candidate_thresholds":list(CANDIDATES),
        "noninferiority_tolerance_balanced_accuracy":tolerance,
        "paired_bootstraps":bootstraps,"seed":seed,
        "data_partitions":"outer CV: proper model fit, disjoint isotonic calibration,"
                          " disjoint threshold selection, held-out TRAIN evaluation",
        "cautions":[
            "Outcomes are retrospective candidate labels, not clinically validated.",
            "The outer folds are all within already examined PPMI TRAIN.",
            "No PPMI validation/test or BioFIND patient observations are accessed.",
            "Threshold-selection guarantee is an exploratory bootstrap bound,"
            " NOT a medical or prospective safety guarantee.",
            "Selection groups contain few HC patients, limiting risk precision.",
            "No model probability is calibrated to real-world population prevalence.",
            "Confidence direction must agree with the Naive Bayes class sign.",
            "Repeated same-month/ON-OFF motor conditions remain confounded.",
            "Only aggregated statistics and chosen thresholds are exported.",
        ],
        "generated_utc":datetime.now(timezone.utc).isoformat(),
    }
    (output/"summary.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-db",type=Path,required=True)
    p.add_argument("--source-engine",choices=("duckdb","sqlite"),
                   default="duckdb")
    p.add_argument("--manifest-db",type=Path,required=True)
    p.add_argument("--split-db",type=Path,required=True)
    p.add_argument("--report-dir",type=Path,
                   default=Path("outputs/phase19_threshold_selection"))
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--budget",type=int,default=5)
    p.add_argument("--seed",type=int,default=20261018)
    p.add_argument("--max-balanced-accuracy-loss",type=float,default=.02)
    p.add_argument("--bootstraps",type=int,default=300)
    a=p.parse_args()
    result=audit(
        a.source_db,a.source_engine,a.manifest_db,a.split_db,
        a.report_dir,a.folds,a.budget,a.seed,
        a.max_balanced_accuracy_loss,a.bootstraps)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print("Upload ONLY aggregate reports. Never upload private databases.")


if __name__=="__main__":
    main()
