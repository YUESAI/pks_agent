#!/usr/bin/env python3
"""Phase63: 20 CPU jobs = 5 prespecified representations x 4 regressors.

Each job evaluates exactly the same three clinical prediction tasks:
Prodromal 12m, Prodromal 24m, and PD 6m, with five stable patient-hash
folds per task. All models get REAL measured BL features only, except
the explicitly labeled R01-augmented representation, which also gets
an actual earlier remote R01 visit (4-8 months since BL) when available.

Only previously frozen Prodromal TRAIN subjects and strict PD cohort.
No historical Prodromal VAL/TEST; no raw records/predictions exported.
No LLM or causal intervention in this experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.phase36_masked_site_feasibility import ALL, VISIBLE
from scripts.phase60_cpu_multistep_wm import auc
from scripts.phase62_ppmi_multicohort_audit import (
    load_cohorts, future_site_pairs,
)

SEED = 20261018
REPRESENTATIONS = (
    "visible27_scores",
    "visible27_mask",
    "full33_scores",
    "full33_mask",
    "full33_r01",
)
ESTIMATORS = ("ridge", "elasticnet", "random_forest", "hist_gbm")
TASKS = (("prodromal", 12), ("prodromal", 24), ("PD", 6))
VISIBLE_IDX = tuple(ALL.index(k) for k in VISIBLE)
REFERENCE_ARMS = (
    ("visible27_scores", "ridge"),
    ("full33_mask", "ridge"),
)


def arm_from_id(task_id):
    if not 0 <= task_id < 20:
        raise ValueError("Task index must be 0..19")
    return REPRESENTATIONS[task_id // len(ESTIMATORS)], ESTIMATORS[
        task_id % len(ESTIMATORS)]


def hash_id(pid, task):
    raw = f"phase63|{SEED}|{task[0]}|{task[1]}|{pid}"
    return hashlib.sha256(raw.encode("utf8")).hexdigest()


def fixed_folds(rows, task, n_splits=5):
    """Fold ID is a hash of patient ID, independent of labels and row order."""
    if len(rows) < n_splits * 8:
        raise ValueError("Need at least 40 people")
    ids = [r["pid"] for r in rows]
    if len(set(ids)) != len(rows):
        raise ValueError("Duplicate person in one prediction task")
    ranked = sorted(range(len(rows)), key=lambda i: hash_id(ids[i], task))
    folds = np.zeros(len(rows), dtype=np.int64)
    for j, i in enumerate(ranked):
        folds[i] = j % n_splits
    return folds


def _real_scores(visit, indices):
    return np.array([
        float(visit["score"][j]) if visit["status"][j] == 1 else np.nan
        for j in indices], dtype=np.float64)


def _statuses(visit, indices):
    return np.array([float(visit["status"][j]) for j in indices],
                    dtype=np.float64)


def _valid_r01(person):
    """R01 is strictly before the first 12-month target eligibility window."""
    bl_month = person["baseline_month"]
    valid = [
        v for v in person["visits"][1:]
        if v["event"] == "R01" and 4 <= v["month"] - bl_month <= 8
    ]
    # Repeated R01 events are unavailable, rather than selecting by outcome.
    return valid[0] if len(valid) == 1 else None


def build_inputs(rows, people_by_id, representation, task):
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown information panel")
    features = []
    history_n = 0
    for record in rows:
        bl = record["baseline"]
        if representation.startswith("visible27"):
            indices = VISIBLE_IDX
        else:
            indices = tuple(range(len(ALL)))
        values = _real_scores(bl, indices)
        if representation.endswith("_scores"):
            row = values
        else:
            row = np.concatenate((values, _statuses(bl, indices)))
        if representation == "full33_r01":
            # At 6m, R01 may occur AFTER the chosen target visit.
            # Do not use it at all for 6m forecasting.
            # Adds information for 12/24m only; this panel is NOT BL-only.
            person = people_by_id[record["pid"]]
            remote = _valid_r01(person) if task[1] >= 12 else None
            if remote is not None:
                history_n += 1
                rvalue = _real_scores(remote, VISIBLE_IDX)
                rstatus = _statuses(remote, VISIBLE_IDX)
                gap = float(remote["month"] - person["baseline_month"])
                present = 1.
            else:
                rvalue = np.full(len(VISIBLE_IDX), np.nan)
                rstatus = np.zeros(len(VISIBLE_IDX))
                gap = 0.
                present = 0.
            row = np.concatenate((
                row, rvalue, rstatus, np.array([present, gap])))
        features.append(row)
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != len(rows):
        raise AssertionError("Feature matrix mismatch")
    return X, history_n


def train_fold_preprocess(x_train, x_test, representation):
    """Training-only imputation. No future values used.

    Score-only panels: training-set median, thus missingness is not
    explicitly visible. Mask panels: missing scores set to 0 ONLY as a
    numerical placeholder, accompanied by 0/1/2 observation status.
    101/UR NEVER becomes a measured clinical score.
    """
    x_train = np.array(x_train, dtype=np.float64, copy=True)
    x_test = np.array(x_test, dtype=np.float64, copy=True)
    n = len(VISIBLE_IDX) if representation.startswith("visible27") else len(ALL)
    if representation.endswith("_scores"):
        median = np.zeros(n)
        for j in range(n):
            observed = x_train[:, j]
            actual = observed[np.isfinite(observed)]
            median[j] = float(np.median(actual)) if len(actual) else 0.
        for a in (x_train, x_test):
            vals = a[:, :n]
            vals[:] = np.where(np.isfinite(vals), vals, median)
    else:
        for a in (x_train, x_test):
            a[:] = np.where(np.isfinite(a), a, 0.)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise AssertionError("Nonfinite model inputs")
    return x_train, x_test


def make_estimator(kind, fold):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.))
    if kind == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=.025, l1_ratio=.5, max_iter=6000,
                       selection="cyclic", random_state=SEED+fold))
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=240, min_samples_leaf=6,
            max_features=.65, n_jobs=2, random_state=SEED+fold)
    if kind == "hist_gbm":
        return HistGradientBoostingRegressor(
            max_iter=160, learning_rate=.05, max_leaf_nodes=9,
            min_samples_leaf=15, l2_regularization=10.,
            random_state=SEED+fold)
    raise ValueError("Unsupported estimator")


def predict_oof(rows, people_by_id, task, candidates, folds=5):
    """Fit candidate AND fixed references on same folds inside each job.

    Output arrays exist only in process memory, never in exported CSVs.
    """
    rows = sorted(rows, key=lambda x: x["pid"])
    assignments = fixed_folds(rows, task, folds)
    y = np.array([r["target_burden"] for r in rows], dtype=np.float64)
    needed = tuple(dict.fromkeys(candidates))
    xs = {}
    history = {}
    for name, _ in needed:
        if name not in xs:
            xs[name], history[name] = build_inputs(
                rows, people_by_id, name, task)
    estimates = {config: np.full(len(rows), np.nan, dtype=float)
                 for config in needed}
    for fold in range(folds):
        tr = np.flatnonzero(assignments != fold)
        te = np.flatnonzero(assignments == fold)
        if len(te) == 0 or len(tr) == 0 or set(tr) & set(te):
            raise AssertionError("Invalid patient-disjoint fold")
        # Training and evaluation use exactly the same person sets
        # across all 20 CPU jobs and for all reference comparisons.
        for representation, kind in needed:
            x_train, x_test = train_fold_preprocess(
                xs[representation][tr], xs[representation][te],
                representation)
            model = make_estimator(kind, fold)
            model.fit(x_train, y[tr])
            estimates[(representation, kind)][te] = np.maximum(
                0., np.asarray(model.predict(x_test), dtype=float))
    if any(not np.isfinite(p).all() for p in estimates.values()):
        raise AssertionError("OOF predictions incomplete")
    return rows, estimates, history


def metrics(rows, prediction, task, config, extra_history_n):
    truth = np.array([r["target_burden"] for r in rows], dtype=float)
    outcome = np.array([r["target_positive"] for r in rows], dtype=int)
    risk = 1. - np.exp(-np.clip(prediction, 0, 50))
    n = len(rows)
    k = max(1, int(.20*n+.5))
    ranks = sorted(range(n), key=lambda i: (
        -risk[i], hash_id(rows[i]["pid"], task)))
    captured = sum(outcome[i] for i in ranks[:k])
    pos = int(outcome.sum())
    return {
        "cohort": task[0],
        "horizon_months": task[1],
        "representation": config[0],
        "estimator": config[1],
        "patients": n,
        "positive": pos if pos >= 10 else "<10",
        "quota20": k,
        "captured_positive": captured if captured >= 10 else "<10",
        "recall_at_20": captured / pos if min(captured, pos) >= 10 else "",
        "mae_future_six_sum": float(np.mean(np.abs(prediction-truth))),
        "rmse_future_six_sum": float(np.sqrt(
            np.mean((prediction-truth)**2))),
        "auc_nonzero": auc(risk.tolist(), outcome.tolist()),
        "uncalibrated_brier": float(np.mean((risk-outcome)**2)),
        "n_with_real_R01_history": extra_history_n,
        "per_patient_predictions_exported": 0,
    }


def pair_ci(rows, left_pred, right_pred, task, comparison, bootstrap):
    """Paired subject bootstrap of candidate-minus-reference errors.

    Both are OOF within the same folds, trained without outer person.
    CI conditions on the already fitted models and reuses explored TRAIN.
    """
    target = np.array([r["target_burden"] for r in rows], dtype=float)
    diffs = (np.abs(right_pred-target) - np.abs(left_pred-target))
    seed = int(hashlib.sha256(
        f"phase63|{SEED}|{task}|{comparison}".encode()).hexdigest()[:8], 16)
    generator = np.random.default_rng(seed)
    results = np.zeros(bootstrap, dtype=float)
    n = len(rows)
    for i in range(bootstrap):
        results[i] = diffs[generator.integers(0, n, size=n)].mean()
    low, high = np.quantile(results, [.025, .975])
    return {
        "cohort": task[0], "horizon_months": task[1],
        "contrast": comparison,
        "delta_mae_candidate_minus_reference": float(np.mean(diffs)),
        "paired_bootstrap_ci_low": float(low),
        "paired_bootstrap_ci_high": float(high),
        "bootstrap_replicates": bootstrap,
        "inference_scope": "exploratory_same_cohort_fixed_fitted_models",
    }


def evaluate(patients, arm_id, folds=5, bootstrap=600):
    if folds != 5 or bootstrap < 100:
        raise ValueError("Frozen five-fold protocol; >=100 bootstrap replicates")
    candidate = arm_from_id(arm_id)
    reference = tuple(REFERENCE_ARMS)
    outputs = []
    contrasts = []
    coverage = []
    for task in TASKS:
        cohort, horizon = task
        subset = [p for p in patients if p["cohort"] == cohort]
        records, _ = future_site_pairs(subset, horizon)
        records = sorted(records, key=lambda row: row["pid"])
        n = len(records)
        npos = sum(r["target_positive"] for r in records)
        eligibility = {"cohort": cohort, "horizon_months": horizon,
                       "n_eligible": n if n >= 10 else "<10",
                       "n_positive": npos if npos >= 10 else "<10"}
        if n < 40 or min(npos, n-npos) < 10:
            coverage.append({**eligibility, "status": "insufficient_eligible_labels"})
            continue
        people = {p["pid"]:p for p in subset if p["pid"] in
                  {r["pid"] for r in records}}
        config_list = (candidate,) + reference
        scored, pred, history = predict_oof(
            records, people, task, config_list, folds)
        candidate_values = pred[candidate]
        outputs.append(metrics(
            scored, candidate_values, task, candidate,
            history[candidate[0]]))
        for baseline in reference:
            ref_values = pred[baseline]
            diff_name = (f"{candidate[0]}_{candidate[1]}_minus_"
                         f"{baseline[0]}_{baseline[1]}")
            contrasts.append(pair_ci(
                scored, ref_values, candidate_values,
                task, diff_name, bootstrap))
        coverage.append({
            **eligibility, "status": "complete",
            "r01_history_available": history.get("full33_r01", None)
                if candidate[0] == "full33_r01" else "",
        })
    return candidate, outputs, contrasts, coverage


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest-db", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--arm-id", required=True, type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=600)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    manifest = args.manifest_db.expanduser().resolve()
    out = args.report_dir.expanduser().resolve()
    if not (root / "PPMI").is_dir() or not manifest.is_file():
        raise FileNotFoundError("Original PPMI source and frozen manifest required")
    if out == root or root in out.parents or out == manifest or out == manifest.parent:
        raise ValueError("Unsafe aggregate output location")
    started = time.perf_counter()
    patients, source = load_cohorts(root, manifest)
    config, results, contrasts, capacity = evaluate(
        patients, args.arm_id, args.folds, args.bootstrap)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "arm_metrics.csv", results)
    write_csv(out / "paired_reference_contrasts.csv", contrasts)
    write_csv(out / "task_capacity.csv", capacity)
    report = {
        "status": "phase63_cpu_20arm_formal_exploratory_complete",
        "array_task_id": args.arm_id,
        "representation": config[0], "estimator": config[1],
        "experiment_tasks": [list(x) for x in TASKS],
        "cohort_separation": "independent per-cohort fixed-horizon five-fold OOF",
        "fold_seed": SEED, "folds": args.folds,
        "bootstrap": args.bootstrap,
        "completed_tasks": len(results),
        "elapsed_seconds": round(time.perf_counter()-started, 2),
        "source_motor_table": source["source_motor_table"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "limits": [
            "Only original frozen Prodromal TRAIN, separate strict PD cohort",
            "No new Prodromal val/test, no clinical independent validation",
            "Same target patients, label-blind fixed fold hash across all 20 jobs",
            "Twelve- and twenty-four-month future labels are true measurements",
            "BL-only predictors except explicitly named full33_r01 arm",
            "R01 is forbidden in 6m tasks because it can postdate the target",
            "R01 if used at 12/24m was observed at 4..8m before outcome window",
            "R01 missingness retained; never impute a 101/UR as measured zero",
            "Historical 6 item BL scoring included only in full33 panels",
            "Model hyperparameters fixed before OOF; no outer-label tuning",
            "Subject bootstrap fixes fitted models; post-hoc 20x3 comparisons",
            "Model outcomes not longitudinal causal interventions or Qwen RSI",
            "Only aggregate CSV/JSON; identifiers/features/individual predictions not exported",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({
        "status": "success", "array_task_id": args.arm_id,
        "representation": config[0], "estimator": config[1],
        "finished_tasks": len(results), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
