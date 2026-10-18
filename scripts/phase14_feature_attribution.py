#!/usr/bin/env python3
"""Phase 14: numerical-feature attribution and paired policy uncertainty.

Only read the FROZEN PPMI TRAIN patients via Phase13 loader. Predict outer
fold patients using models fitted on other TRAIN folds. No val/test/BioFIND
patient measurements are accessed. No patient-level results are exported.

These are retrospective candidate labels, not clinically validated outcomes.
An all-observed-values feature ablation is NOT a budgeted Doctor policy.
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
    bin_value, evaluate, load_training, policy_episode, train_model,
)
from scripts.phase11_protocol_preflight import assign_folds
from scripts.phase10_leakage_splits import auc

# Prespecified ablations, no subset selected using held-out fold performance.
SUBSETS = {
    "all_numeric": ("age", "sleep_item", "cognitive_test", "autonomic_item",
                    "motor_exam", "smell_test"),
    "no_motor": ("age", "sleep_item", "cognitive_test", "autonomic_item",
                 "smell_test"),
    "nonmotor_no_age": ("sleep_item", "cognitive_test", "autonomic_item",
                        "smell_test"),
    "motor_only": ("motor_exam",),
    "motor_and_age": ("motor_exam", "age"),
    "age_only": ("age",),
    "moca_only": ("cognitive_test",),
    "upsit_only": ("smell_test",),
    "rbd_item_only": ("sleep_item",),
    "scopa_item_only": ("autonomic_item",),
}


def record_score(model, values, allowed):
    """Missing/nonnumeric tests are neutral, never treated as diagnostic evidence."""
    log_odds = 0.0
    n_values = 0
    for action in allowed:
        bucket = bin_value(action, values.get(action))
        if bucket is None:
            continue
        pd = model[action]["PD"][bucket]
        hc = model[action]["HC"][bucket]
        log_odds += math.log(pd / hc)
        n_values += 1
    return log_odds, n_values


def bounded_percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.floor((len(ordered) - 1) * fraction)))
    return ordered[index]


def balanced_accuracy(scored):
    tp = sum(score >= 0 and truth for score, truth in scored)
    tn = sum(score < 0 and not truth for score, truth in scored)
    pd = sum(truth for _, truth in scored)
    hc = len(scored) - pd
    if not pd or not hc:
        raise ValueError("Both PD and HC required")
    return 0.5 * (tp / pd + tn / hc)


def paired_bootstrap(pairs, iterations, seed):
    """Stratified *paired* patient resampling; same patients for both policies.

    pairs: (truth_is_PD, clinical_score, adaptive_score); never written to disk.
    """
    pos = [row for row in pairs if row[0]]
    neg = [row for row in pairs if not row[0]]
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError("Need at least two patients per class")
    rand = random.Random(seed)
    deltas_balanced = []
    deltas_auc = []
    def compare(rows):
        baseline = [(clinical, truth) for truth, clinical, adaptive in rows]
        adapt = [(adaptive, truth) for truth, clinical, adaptive in rows]
        return (balanced_accuracy(adapt) - balanced_accuracy(baseline),
                auc(adapt) - auc(baseline))
    point_ba, point_auc = compare(pairs)
    for i in range(iterations):
        resampled = [pos[rand.randrange(len(pos))] for _ in pos]
        resampled += [neg[rand.randrange(len(neg))] for _ in neg]
        d_ba, d_auc = compare(resampled)
        deltas_balanced.append(d_ba)
        deltas_auc.append(d_auc)
    return [
        dict(comparison="adaptive_minus_clinical", metric="balanced_accuracy",
             point_difference=round(point_ba, 4),
             bootstrap_lower_95=round(bounded_percentile(deltas_balanced, .025), 4),
             bootstrap_upper_95=round(bounded_percentile(deltas_balanced, .975), 4),
             bootstrap_replicates=iterations),
        dict(comparison="adaptive_minus_clinical", metric="auroc",
             point_difference=round(point_auc, 4),
             bootstrap_lower_95=round(bounded_percentile(deltas_auc, .025), 4),
             bootstrap_upper_95=round(bounded_percentile(deltas_auc, .975), 4),
             bootstrap_replicates=iterations),
    ]


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def motor_source_counts(manifest_path, train_labels, minimum):
    """Review label-specific medication-state *selection*, aggregate only.

    This is not Doctor-facing information and must not become a policy input.
    """
    con = sqlite3.connect("file:" + str(Path(manifest_path).resolve()) + "?mode=ro",
                          uri=True)
    counts = Counter()
    try:
        for offset in range(0, len(train_labels), 200):
            ids = list(sorted(train_labels))[offset:offset+200]
            marks = ",".join("?" for _ in ids)
            seen = defaultdict(set)
            rows = con.execute(
                "SELECT patient_id,action_key FROM episode_actions "
                "WHERE cohort='PPMI' AND status='ready' AND patient_id IN (" +
                marks + ")", ids)
            for pid, action in rows:
                if action in ("updrs3_total@UNSPECIFIED",
                              "updrs3_total@OFF", "updrs3_total@ON"):
                    seen[pid].add(action)
            for pid in ids:
                selected = next((name for name in
                                 ("UNSPECIFIED", "OFF", "ON")
                                 if "updrs3_total@" + name in seen[pid]), "ABSENT")
                counts[(train_labels[pid], selected)] += 1
    finally:
        con.close()
    return [
        dict(label=label, selected_motor_condition=condition,
             n_patients=str(n) if n == 0 or n >= minimum else "<" + str(minimum))
        for (label, condition), n in sorted(counts.items())
    ]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(source, engine, manifest, splits, out, folds=5, budget=5,
          seed=20261018, bootstraps=500, minimum=10):
    source, manifest, splits, out = map(
        lambda p: Path(p).expanduser().resolve(),
        (source, manifest, splits, out))
    if not all(p.is_file() for p in (source, manifest, splits)):
        raise FileNotFoundError("Missing source, episode or split DB")
    if out in (source, manifest, splits):
        raise ValueError("Output directory overlaps local database")
    if folds < 2 or budget < 0 or bootstraps < 100 or minimum < 10:
        raise ValueError("Require folds>=2, budget>=0, bootstraps>=100, min_count>=10")
    data = load_training(source, engine, manifest, splits)
    rows = [(pid, label, anchor, values)
            for pid, (label, anchor, values) in sorted(data.items())]
    if len(rows) < 2 * folds:
        raise ValueError("Too few PPMI TRAIN patients")
    folds_by_id = assign_folds(rows, folds, seed)
    ablations = defaultdict(list)
    paired_scores = {}
    bin_totals = Counter()
    numeric_state = Counter()
    for pid, label, anchor, values in rows:
        for action in SUBSETS["all_numeric"]:
            bucket = bin_value(action, values.get(action))
            status = ("unavailable" if action not in values else
                      "outside_numeric_range" if bucket is None else "in_range")
            numeric_state[(label, action, status)] += 1
            if bucket is not None:
                bin_totals[(label, action, bucket)] += 1
    for held in range(folds):
        train = [(label, anchor, values) for pid, label, anchor, values
                 in rows if folds_by_id[pid] != held]
        test = [(pid, label, values) for pid, label, anchor, values
                in rows if folds_by_id[pid] == held]
        if not test or {l for l, _, _ in train} != {"PD", "HC"}:
            raise ValueError("One fold is empty or lacks PD/HC in fitting")
        model, success = train_model(train)
        for pid, label, values in test:
            truth = label == "PD"
            for subset, actions in SUBSETS.items():
                score, n_used = record_score(model, values, actions)
                ablations[subset].append((score, truth, 0, n_used))
            clinical, _, _ = policy_episode(
                values, model, success, "clinical", budget, seed, pid)
            adaptive, _, _ = policy_episode(
                values, model, success, "adaptive", budget, seed, pid)
            paired_scores[pid] = (truth, clinical, adaptive)
    if set(paired_scores) != set(data):
        raise ValueError("Missing train OOF predictions")
    metric_rows = []
    for subset, items in sorted(ablations.items()):
        m = evaluate(items)
        metric_rows.append(dict(
            feature_subset=subset,
            evaluation_mode="all_recorded_values_unbudgeted",
            evaluated_population="PPMI_train_out_of_fold",
            folds=folds,
            n=m["n"], n_PD=m["n_PD"], n_HC=m["n_HC"],
            balanced_accuracy=m["balanced_accuracy"],
            macro_f1=m["macro_f1"], auroc=m["auroc"],
            mean_available_numeric_results=m["mean_results"],
        ))
    uncertainty = paired_bootstrap(
        list(paired_scores.values()), bootstraps, seed + 19)
    hist_rows = [dict(label=l, action=a, binned_category=b,
                      patients=str(n) if n >= minimum else "<" + str(minimum))
                 for (l, a, b), n in sorted(bin_totals.items())]
    qc_rows = [dict(label=l, action=a, status=s,
                    patients=str(n) if n >= minimum else "<" + str(minimum))
               for (l, a, s), n in sorted(numeric_state.items())]
    state_rows = motor_source_counts(
        manifest, {pid: label for pid, label, _, _ in rows}, minimum)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"feature_ablation_cv.csv",
              ["feature_subset", "evaluation_mode", "evaluated_population",
               "folds", "n", "n_PD", "n_HC", "balanced_accuracy",
               "macro_f1", "auroc", "mean_available_numeric_results"],
              metric_rows)
    write_csv(out/"adaptive_vs_clinical_paired_bootstrap.csv",
              ["comparison", "metric", "point_difference",
               "bootstrap_lower_95", "bootstrap_upper_95",
               "bootstrap_replicates"], uncertainty)
    write_csv(out/"numeric_value_qc.csv",
              ["label", "action", "status", "patients"], qc_rows)
    write_csv(out/"numeric_bin_distributions.csv",
              ["label", "action", "binned_category", "patients"], hist_rows)
    write_csv(out/"selected_motor_state_by_label.csv",
              ["label", "selected_motor_condition", "n_patients"], state_rows)
    summary = {
        "status": "train_only_exploratory_feature_attribution",
        "candidate_label_population": "PPMI train only",
        "n_patients": len(rows), "n_PD": sum(x[1] == "PD" for x in rows),
        "n_HC": sum(x[1] == "HC" for x in rows),
        "folds": folds, "seed": seed, "budget_for_paired_policies": budget,
        "bootstraps": bootstraps,
        "frozen_split_sha256": digest_file(splits),
        "frozen_manifest_sha256": digest_file(manifest),
        "safety_notes": [
            "All-recorded-value ablations are not budgeted Doctor strategies.",
            "No PPMI validation/test or BioFIND patient data are accessed.",
            "Same held-out train-fold patients are paired for Adaptive vs Clinical.",
            "Month-only data cannot reconstruct diagnosis-time order.",
            "Candidate labels and numerical test codes lack clinical validation.",
            "The Phase9 motor selection mixes treatment states; state may confound score.",
            "High scores may reflect cohort selection or diagnostic circularity.",
            "No patient IDs, raw numeric scores, per-subject predictions or source dates exported.",
            "Bootstrap intervals describe resampling uncertainty, not causal significance.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
                                    encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--source-engine", choices=("duckdb","sqlite"), default="duckdb")
    parser.add_argument("--manifest-db", type=Path, required=True)
    parser.add_argument("--split-db", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path,
                        default=Path("outputs/phase14_feature_ablation"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--min-cell-count", type=int, default=10)
    args = parser.parse_args()
    result = audit(args.source_db, args.source_engine, args.manifest_db,
                   args.split_db, args.report_dir, args.folds, args.budget,
                   args.seed, args.bootstraps, args.min_cell_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Share ONLY the aggregate reports, never patient-level databases")


if __name__ == "__main__":
    main()
