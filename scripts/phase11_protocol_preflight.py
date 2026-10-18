#!/usr/bin/env python3
"""Phase 11: TRAIN-ONLY cross-validated shortcut assessment and holdout lock.

This file does NOT fit a clinical prediction model. It fits simple
Bernoulli availability baselines exclusively inside the existing PPMI TRAIN
split, comparing raw motor state actions to collapsed motor families.

No patient IDs, scores, individual predictions or source data are exported.
The local source/manifest/split databases must remain on the research cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.phase10_leakage_splits import (
    auc, classify_naive_bayes, open_source, score_nb
)

CANONICAL_NONMOTOR = (
    "age_at_visit", "moca_total", "rbd_vivid_dreams",
    "scopa_aut_item1", "upsit_total",
)
MOTOR_TOTAL_KEYS = (
    "updrs3_total@ON", "updrs3_total@OFF", "updrs3_total@UNSPECIFIED",
)
REPRESENTATIONS = ("all_raw_actions", "collapsed_motor", "nonmotor_only")


def digest_file(path):
    """Content-address local split/manifest DBs, not their patient-level rows."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            batch = handle.read(1024 * 1024)
            if not batch:
                break
            hasher.update(batch)
    return hasher.hexdigest()


def project_actions(ready, representation, catalog):
    """Return presence-only features. Never include medication-state metadata."""
    if representation == "all_raw_actions":
        return frozenset(ready)
    result = {a for a in CANONICAL_NONMOTOR if a in ready and a in catalog}
    if representation == "collapsed_motor":
        if any(a in ready for a in MOTOR_TOTAL_KEYS):
            result.add("motor_any_state")
    return frozenset(result)


def allowed_actions(representation, catalog):
    if representation == "all_raw_actions":
        return sorted(catalog)
    result = [a for a in CANONICAL_NONMOTOR if a in catalog]
    if representation == "collapsed_motor":
        result.append("motor_any_state")
    return sorted(result)


def assign_folds(records, k, seed):
    """Stratified deterministic patient folds, respecting baseline-anchor stratum."""
    groups = defaultdict(list)
    for pid, label, anchor, actions in records:
        groups[(label, bool(anchor))].append(pid)
    assignments = {}
    for (label, anchored), ids in sorted(groups.items()):
        ordered = sorted(ids, key=lambda pid: hashlib.sha256(
            f"{seed}|{label}|{anchored}|{pid}".encode("utf-8")).hexdigest())
        for pos, pid in enumerate(ordered):
            assignments[pid] = pos % k
    return assignments


def safe_cell(n, minimum):
    return str(n) if n == 0 or n >= minimum else f"<{minimum}"


def crossvalidated_predictions(records, catalog, representation, folds, seed):
    """Out-of-fold label scores; returned only in memory, never serialized."""
    if len({label for _, label, _, _ in records}) != 2:
        raise ValueError("Both PD and HC required in training split")
    fold_by_id = assign_folds(records, folds, seed)
    action_space = allowed_actions(representation, catalog)
    observed = []
    for held in range(folds):
        train = [(label, project_actions(ready, representation, catalog))
                 for pid, label, anchor, ready in records if fold_by_id[pid] != held]
        test = [(label, project_actions(ready, representation, catalog))
                for pid, label, anchor, ready in records if fold_by_id[pid] == held]
        model = classify_naive_bayes(train, action_space)
        if model is None:
            raise ValueError("One training fold has only one label")
        for label, observed_actions in test:
            observed.append((score_nb(model, observed_actions), label == "PD"))
    return observed


def evaluate_scores(scored):
    """Pooled train-only out-of-fold metrics, no per-person predictions exported."""
    tp = sum(score >= 0 and truth for score, truth in scored)
    fn = sum(score < 0 and truth for score, truth in scored)
    tn = sum(score < 0 and not truth for score, truth in scored)
    fp = sum(score >= 0 and not truth for score, truth in scored)
    pd_n, hc_n = tp + fn, tn + fp
    if not pd_n or not hc_n:
        raise ValueError("Both classes needed to evaluate metrics")
    bal = 0.5 * (tp / pd_n + tn / hc_n)
    f_pd = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    f_hc = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
    return {
        "n": len(scored), "n_PD": pd_n, "n_HC": hc_n,
        "balanced_accuracy": round(bal, 4),
        "macro_f1": round((f_pd + f_hc) / 2, 4),
        "auroc": round(auc(scored), 4),
    }


def write_csv(path, fields, data):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def audit(source_db, engine, manifest_db, split_db, out, folds=5, seed=20261018, minimum=10):
    """Read all data only to validate integrity; model on PPMI train only."""
    source_db, manifest_db, split_db, out = [
        Path(p).expanduser().resolve() for p in (source_db, manifest_db, split_db, out)
    ]
    if not all(p.is_file() for p in (source_db, manifest_db, split_db)):
        raise FileNotFoundError("Missing source, manifest or split DB")
    if out in (source_db, manifest_db, split_db):
        raise ValueError("Report path overlaps a database path")
    if folds < 2 or minimum < 10:
        raise ValueError("Need >=2 folds and min-cell-count >=10")

    src = open_source(source_db, engine)
    man = sqlite3.connect("file:" + str(manifest_db) + "?mode=ro", uri=True)
    spl = sqlite3.connect("file:" + str(split_db) + "?mode=ro", uri=True)
    try:
        labels = {(c, p): l for c, p, l in src.execute(
            "SELECT cohort,patient_id,candidate_label FROM restricted_labels "
            "WHERE candidate_label IN ('PD','HC')").fetchall()}
        splits = {(c, p): sp for c, p, sp in spl.execute(
            "SELECT cohort,patient_id,split FROM patient_splits").fetchall()}
        episode_anchor = {(c, p): bool(m) for c, p, m in man.execute(
            "SELECT cohort,patient_id,baseline_month FROM episode_candidates").fetchall()}
        if not labels or labels.keys() != splits.keys() or labels.keys() != episode_anchor.keys():
            raise ValueError("Patient populations differ across source, manifest and split")
        if any((c == "PPMI" and sp not in ("train", "val", "test"))
               or (c == "BioFIND" and sp != "external_test")
               for (c, p), sp in splits.items()):
            raise ValueError("Unexpected split assignments")
        catalog = {action for c, action in man.execute(
            "SELECT cohort,action_key FROM action_catalog") if c == "PPMI"}
        expected = set(MOTOR_TOTAL_KEYS) | set(CANONICAL_NONMOTOR)
        missing = expected - catalog
        if missing:
            raise ValueError("Manifest missing expected actions; check Phase 9: " +
                             ", ".join(sorted(missing)))
        masks = defaultdict(set)
        n_actions = Counter()
        for c, pid, act, status in man.execute(
            "SELECT cohort,patient_id,action_key,status FROM episode_actions"):
            key = (c, pid)
            if key not in labels:
                raise ValueError("Orphan action row")
            n_actions[(c, pid, act)] += 1
            if status == "ready":
                masks[key].add(act)
        if any(n > 1 for n in n_actions.values()):
            raise ValueError("Duplicate patient-action records in manifest")
        local_train = [
            (pid, labels[(c, pid)], episode_anchor[(c, pid)], masks[(c, pid)])
            for (c, pid), sp in sorted(splits.items())
            if c == "PPMI" and sp == "train"
        ]
        if len(local_train) < 2 * folds:
            raise ValueError("Not enough training subjects for requested folds")

        split_counts = Counter((c, sp, labels[(c, p)]) for (c, p), sp in splits.items())
        leakage_rows = []
        for representation in REPRESENTATIONS:
            scored = crossvalidated_predictions(local_train, catalog, representation, folds, seed)
            leakage_rows.append({
                "availability_representation": representation,
                "evaluated_split": "PPMI_train_out_of_fold",
                "folds": folds,
                "n_actions": len(allowed_actions(representation, catalog)),
                **evaluate_scores(scored),
            })

        bias_rows = []
        for rep in REPRESENTATIONS:
            space = allowed_actions(rep, catalog)
            for action in space:
                pd_total = sum(l == "PD" for _, l, _, _ in local_train)
                hc_total = len(local_train) - pd_total
                pd_ready = sum(l == "PD" and action in project_actions(ready, rep, catalog)
                               for _, l, _, ready in local_train)
                hc_ready = sum(l == "HC" and action in project_actions(ready, rep, catalog)
                               for _, l, _, ready in local_train)
                bias_rows.append({
                    "representation": rep, "action": action,
                    "train_PD": pd_total, "train_HC": hc_total,
                    "PD_ready": safe_cell(pd_ready, minimum),
                    "HC_ready": safe_cell(hc_ready, minimum),
                    "PD_ready_percent": round(pd_ready * 100 / pd_total, 2)
                        if pd_ready == 0 or pd_ready >= minimum else "",
                    "HC_ready_percent": round(hc_ready * 100 / hc_total, 2)
                        if hc_ready == 0 or hc_ready >= minimum else "",
                })
    finally:
        spl.close()
        man.close()
        src.close()

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "train_only_shortcut_cv.csv",
              ["availability_representation", "evaluated_split", "folds", "n_actions",
               "n", "n_PD", "n_HC", "balanced_accuracy", "macro_f1", "auroc"],
              leakage_rows)
    write_csv(out / "train_action_family_bias.csv",
              ["representation", "action", "train_PD", "train_HC",
               "PD_ready", "HC_ready", "PD_ready_percent", "HC_ready_percent"], bias_rows)
    write_csv(out / "frozen_split_counts.csv",
              ["cohort", "split", "label", "patients"],
              [{"cohort": c, "split": sp, "label": l, "patients": n}
               for (c, sp, l), n in sorted(split_counts.items())])
    summary = {
        "status": "exploratory_protocol_preflight",
        "evaluated_predictions": "Only cross-validated PPMI train; NO val/test/external predictions",
        "frozen_split_sha256": digest_file(split_db),
        "frozen_manifest_sha256": digest_file(manifest_db),
        "source_db_schema": "v2 month precision (not clinically validated)",
        "train_patient_count": len(local_train),
        "cv_folds": folds, "seed": seed,
        "representations": list(REPRESENTATIONS),
        "interpretation": [
            "All-raw actions encode ON/OFF and repeat motor subscales.",
            "Collapsed motor is only an availability grouping; not a harmonized numeric motor score.",
            "Nonmotor-only removes motor availability, not all study-process confounding.",
            "Clinical labels, treatment effects, and true diagnosis decision time remain unvalidated.",
            "Phase10 test metrics have already been inspected; do not repeatedly tune on them.",
            "No patient-level records exported. Keep DB and split artifacts on cluster.",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                                      encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-db", type=Path, required=True)
    ap.add_argument("--source-engine", choices=("duckdb", "sqlite"), default="duckdb")
    ap.add_argument("--manifest-db", type=Path, required=True)
    ap.add_argument("--split-db", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, default=Path("outputs/protocol_preflight"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--min-cell-count", type=int, default=10)
    args = ap.parse_args()
    result = audit(args.source_db, args.source_engine, args.manifest_db,
                   args.split_db, args.report_dir, args.folds, args.seed,
                   args.min_cell_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Upload only the aggregate protocol_preflight report directory.")


if __name__ == "__main__":
    main()
