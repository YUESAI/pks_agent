#!/usr/bin/env python3
"""Phase60: CPU-only coherent observation WM and 1/2-step imagined questions.

Exploratory, patient-disjoint frozen TRAIN V04 *masking simulation*.
The 27 visible V04 scores are real only after PatientEvidenceService.request().
The six hidden V04 scores never enter the planner or a patient tool.
No real R01->V04 intervention and NO Qwen inference in this experiment.

Factorial arms: naive_greedy, naive_rollout, mixture_greedy, mixture_rollout.
A class-conditional latent categorical mixture is a proper joint density:
   p(y, x_1,...,x_27) = p(y) sum_z p(z|y) prod_j p(x_j|y,z).
Hypothetical answers only update a temporary copy of model state.
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

from scripts.phase45_world_model import PatientEvidenceService

SEED = 20261018
ARMS = ("naive_greedy", "naive_rollout", "mixture_greedy", "mixture_rollout")
KS = (1, 2, 3)
DEFAULT_BUDGETS = (4, 8, 12)


def stable_hash(value, seed=SEED):
    return hashlib.sha256(f"phase59|{seed}|{value}".encode()).hexdigest()


def split_train(records, seed=SEED):
    """Identical 65/15/20 percent hash assignment used by Phase59."""
    if len(records) < 200 or len({r["pid"] for r in records}) != len(records):
        raise ValueError("Require >=200 distinct frozen TRAIN V04 people")
    ranked = sorted(records, key=lambda r: stable_hash(r["pid"], seed))
    a, b = int(.65 * len(ranked)), int(.80 * len(ranked))
    fit, dev, outer = ranked[:a], ranked[a:b], ranked[b:]
    assert not ({r["pid"] for r in fit} & {r["pid"] for r in dev + outer})
    assert not ({r["pid"] for r in dev} & {r["pid"] for r in outer})
    if len(outer) < 20:
        raise ValueError("Insufficient untouched TRAIN-internal evaluation people")
    return fit, dev, outer


class CategoricalMixtureWM:
    """Dirichlet-smoothed EM for a class-conditioned latent categorical WM.

    The model is a coherent joint probability distribution over the true
    measured visible-item scores and binary outcome, not a causal simulator.
    """

    def __init__(self, items, components=1, alpha=1., epochs=24, seed=SEED):
        if not items or len(items) != len(set(items)):
            raise ValueError("Items must be nonempty and distinct")
        if components not in KS or alpha <= 0 or epochs < 1:
            raise ValueError("Invalid model configuration")
        self.items = tuple(items)
        self.n_components = components
        self.alpha = float(alpha)
        self.epochs = epochs
        self.seed = seed
        self.fitted = False

    def fit(self, records):
        records = list(records)
        if len(records) < 50 or len({r["pid"] for r in records}) != len(records):
            raise ValueError("Need >=50 independent fit people")
        grouped = [[], []]
        for r in records:
            y = r["label"]
            if type(y) is not int or y not in (0, 1):
                raise ValueError("Bad target")
            x = []
            for item in self.items:
                v = r["features"][item]
                if type(v) is not int or v not in range(5):
                    raise ValueError("Only actually measured 0..4 visible scores")
                x.append(v)
            grouped[y].append((r["pid"], tuple(x)))
        if min(map(len, grouped)) < self.n_components * 2:
            raise ValueError("Too few positives/negatives for mixture")
        k = self.n_components
        a = self.alpha
        class_prior = [(len(g) + a) / (len(records) + 2 * a) for g in grouped]
        self.comp_priors = []
        self.comp_categories = []
        self.comp_classes = []
        for y in (0, 1):
            group = grouped[y]
            n = len(group)
            # Stable balanced deterministic initial assignment, FIT IDs only.
            ordered = sorted(group, key=lambda row: hashlib.sha256(
                f"phase60-init|{self.seed}|{y}|{row[0]}".encode()).hexdigest())
            resp = [[float(i % k == z) for z in range(k)]
                    for i in range(n)]
            # Use the same order through the complete EM update sequence.
            for _ in range(self.epochs):
                counts = [a] * k
                bins = [[[a] * 5 for _ in self.items] for _ in range(k)]
                for (_, scores), probs in zip(ordered, resp):
                    for z in range(k):
                        w = probs[z]
                        counts[z] += w
                        for j, v in enumerate(scores):
                            bins[z][j][v] += w
                weights = [v / sum(counts) for v in counts]
                cats = [
                    [tuple(v / sum(hist) for v in hist) for hist in bins[z]]
                    for z in range(k)
                ]
                new_resp = []
                for _, scores in ordered:
                    logp = [
                        math.log(weights[z]) +
                        sum(math.log(cats[z][j][v])
                            for j, v in enumerate(scores))
                        for z in range(k)
                    ]
                    high = max(logp)
                    unnorm = [math.exp(v - high) for v in logp]
                    total = sum(unnorm)
                    new_resp.append([v / total for v in unnorm])
                resp = new_resp
            # Final M-step uses final responsibilities (not stale estimates).
            counts = [a] * k
            bins = [[[a] * 5 for _ in self.items] for _ in range(k)]
            for (_, scores), probs in zip(ordered, resp):
                for z, w in enumerate(probs):
                    counts[z] += w
                    for j, v in enumerate(scores):
                        bins[z][j][v] += w
            for z in range(k):
                self.comp_priors.append(class_prior[y] * counts[z] / sum(counts))
                self.comp_categories.append(tuple(
                    tuple(v / sum(hist) for v in hist) for hist in bins[z]))
                self.comp_classes.append(y)
        self.n_total = len(self.comp_priors)
        self.fitted = True
        return self

    def initial_state(self):
        if not self.fitted:
            raise RuntimeError("Fit before querying")
        return tuple(self.comp_priors)

    def risk(self, state):
        return sum(p for p, y in zip(state, self.comp_classes) if y == 1)

    def predict(self, state, item_idx):
        if not 0 <= item_idx < len(self.items):
            raise ValueError("Not an allowed visible action")
        probs = [0.] * 5
        for z, w in enumerate(state):
            hist = self.comp_categories[z][item_idx]
            for v in range(5):
                probs[v] += w * hist[v]
        return tuple(probs)

    def imagined_transition(self, state, item_idx, value, likelihood=None):
        """Hypothetical Bayesian update; caller must NEVER release it as fact."""
        if value not in range(5):
            raise ValueError("Invalid hypothetical score")
        p = (self.predict(state, item_idx)[value]
             if likelihood is None else likelihood)
        if p <= 0:
            raise ArithmeticError("Expected positive smoothed probability")
        updated = tuple(
            state[z] * self.comp_categories[z][item_idx][value] / p
            for z in range(self.n_total))
        # Control accumulated float drift over lookahead.
        s = sum(updated)
        return tuple(x / s for x in updated)


def binary_entropy(p):
    p = min(1. - 1e-12, max(1e-12, p))
    return -p * math.log(p) - (1-p) * math.log(1-p)


def expected_after_one(model, state, item_idx):
    probabilities = model.predict(state, item_idx)
    return sum(
        p * binary_entropy(model.risk(
            model.imagined_transition(state, item_idx, v, p)))
        for v, p in enumerate(probabilities)
    )


def choose_action(model, state, available, remaining_budget, lookahead):
    """Genuine branching depth 2: E_x[min_j E_x2 H(Y|x,x2)].

    Inner min is computed separately for each hypothetical x, so
    the second measurement can adapt to the hypothetical answer.
    A depth-2 calculation is NOT used when only one action remains.
    """
    candidates = tuple(sorted(available))
    if not candidates or remaining_budget < 1:
        raise ValueError("No permissible actions")
    if not lookahead or remaining_budget == 1 or len(candidates) == 1:
        return min(candidates, key=lambda j: (
            expected_after_one(model, state, j), j))
    best = (math.inf, None)
    for first in candidates:
        others = tuple(j for j in candidates if j != first)
        pv = model.predict(state, first)
        objective = 0.
        for v, p in enumerate(pv):
            hypothetical = model.imagined_transition(state, first, v, p)
            objective += p * min(
                expected_after_one(model, hypothetical, next_item)
                for next_item in others)
        candidate = (objective, first)
        if candidate < best:
            best = candidate
    return best[1]


def dev_masked_nll(model, dev, budgets=(4, 8, 12), seed=SEED):
    """DEV-only predictive likelihood on deterministically masked items.

    Target chosen by predeclared hash permutation. Hidden SIX labels and
    outer-evaluation people cannot inform model selection.
    """
    loss = []
    for row in dev:
        ordering = sorted(range(len(model.items)), key=lambda j:
                          stable_hash(f"phase60-dev|{row['pid']}|{model.items[j]}", seed))
        for budget in budgets:
            if budget >= len(ordering):
                raise ValueError("DEV masks must leave a target item")
            state = model.initial_state()
            for j in ordering[:budget]:
                value = row["features"][model.items[j]]
                state = model.imagined_transition(state, j, value)
            target = ordering[budget]
            p = model.predict(state, target)[row["features"][model.items[target]]]
            loss.append(-math.log(max(p, 1e-15)))
    return sum(loss) / len(loss)


def select_models(fit, dev, items, epochs, seed=SEED):
    """K selected by DEV item NLL only, with K=1 as safe fallback."""
    models = {}
    dev_scores = {}
    for k in KS:
        model = CategoricalMixtureWM(
            items, components=k, epochs=epochs, seed=seed).fit(fit)
        models[k] = model
        dev_scores[k] = dev_masked_nll(model, dev, seed=seed)
    selected = min(KS, key=lambda k: (dev_scores[k], k))
    return models[1], models[selected], selected, dev_scores


def simulate(row, model, budget, lookahead):
    """Planner reads only the belief, real tool responses and action budget."""
    evidence = PatientEvidenceService(row["features"], model.items, budget)
    state = model.initial_state()
    available = set(range(len(model.items)))
    queried_nll = 0.
    for step in range(budget):
        idx = choose_action(model, state, available, budget - step, lookahead)
        probs = model.predict(state, idx)
        event = evidence.request(model.items[idx])  # Only real observation path
        if event["provenance"] != "observed_V04_masking_simulation":
            raise AssertionError("Not a recorded V04 observation")
        observed = event["recorded_score"]
        queried_nll -= math.log(max(probs[observed], 1e-15))
        state = model.imagined_transition(state, idx, observed, probs[observed])
        available.remove(idx)
    if len(evidence.observed) != budget:
        raise AssertionError("Query budget mismatch")
    return model.risk(state), queried_nll / budget


def auc(scores, labels):
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda r: r[0])
    rank = 1
    positive_ranks = 0.
    while rank <= len(pairs):
        end = rank
        while end < len(pairs) and pairs[end][0] == pairs[rank-1][0]:
            end += 1
        average = (rank + end) / 2.
        positive_ranks += sum(label for _, label in pairs[rank-1:end]) * average
        rank = end + 1
    return (positive_ranks - n1 * (n1 + 1) / 2) / (n1 * n0)


def aggregate(rows, predictions, q, arm):
    """No subject-level predictions or selected-item trajectories are exported."""
    n = len(rows)
    labels = [int(r["label"]) for r in rows]
    risks = [predictions[r["pid"]][0] for r in rows]
    nlls = [predictions[r["pid"]][1] for r in rows]
    npos = sum(labels)
    if not 0 < npos < n:
        raise ValueError("Both classes needed")
    quota = min(n, max(1, int(.20 * n + .5)))
    chosen = sorted(range(n), key=lambda i: (
        -risks[i], stable_hash(rows[i]["pid"])))[:quota]
    captured = sum(labels[i] for i in chosen)
    small = captured < 10 or npos < 10
    return {
        "arm": arm,
        "budget": q,
        "n_patients": n,
        "n_positive": npos if npos >= 10 else "<10",
        "site_quota": quota,
        "captured_positive": captured if captured >= 10 else "<10",
        "recall_at_20pct": "" if small else captured / npos,
        "auroc": auc(risks, labels),
        "brier": sum((p-y)**2 for p, y in zip(risks, labels)) / n,
        "queried_item_nll": sum(nlls) / n,
        "real_observation_actions": n * q,
        "imagined_answers_released_to_tool": 0,
        "qwen_prompts": 0,
    }


def bootstrap_pairs(rows, left, right, budget, reps, seed=SEED):
    """Paired patient bootstrap in RAM; no patient-level export."""
    n = len(rows)
    truth = [int(r["label"]) for r in rows]
    a = [left[r["pid"]][0] for r in rows]
    b = [right[r["pid"]][0] for r in rows]
    rng = random.Random(seed + budget)
    diffs = []
    for _ in range(reps):
        indices = [rng.randrange(n) for __ in range(n)]
        da = sum((a[i] - truth[i])**2 for i in indices) / n
        db = sum((b[i] - truth[i])**2 for i in indices) / n
        diffs.append(db - da)  # negative favors right
    diffs.sort()
    lo = diffs[int(.025 * (len(diffs)-1))]
    hi = diffs[int(.975 * (len(diffs)-1))]
    return {
        "budget": budget,
        "left": "naive_greedy",
        "right": "mixture_rollout",
        "delta_brier_right_minus_left": (
            sum((b[i]-truth[i])**2 - (a[i]-truth[i])**2
                for i in range(n)) / n),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "bootstrap_replicates": reps,
        "ci_scope": "fixed-fitted-model TRAIN-internal paired patients only",
    }


def write_csv(path, values):
    if not values:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def run(records, items, arm="all", budgets=DEFAULT_BUDGETS,
        epochs=24, bootstrap=500, seed=SEED):
    if arm not in ARMS + ("all",):
        raise ValueError("Invalid arm")
    budgets = tuple(sorted(set(budgets)))
    if not budgets or budgets[0] < 1 or budgets[-1] > len(items):
        raise ValueError("Invalid budgets")
    fit, dev, outer = split_train(records, seed)
    naive, mixture, selected, dev_scores = select_models(
        fit, dev, items, epochs, seed)
    arms = ARMS if arm == "all" else (arm,)
    metrics = []
    prediction_sets = {}
    for policy in arms:
        model = naive if policy.startswith("naive_") else mixture
        lookahead = policy.endswith("_rollout")
        for budget in budgets:
            estimates = {
                r["pid"]: simulate(r, model, budget, lookahead) for r in outer
            }
            metrics.append({
                **aggregate(outer, estimates, budget, policy),
                "selected_mixture_components": selected
                    if policy.startswith("mixture_") else 1,
            })
            prediction_sets[(policy, budget)] = estimates
    paired = []
    if arm == "all" and bootstrap:
        for q in budgets:
            paired.append(bootstrap_pairs(
                outer,
                prediction_sets[("naive_greedy", q)],
                prediction_sets[("mixture_rollout", q)],
                q, bootstrap, seed))
    metadata = {
        "protocol": "frozen_TRAIN_same_V04_27_visible_masking_65_15_20",
        "seed": seed,
        "patient_counts": {"fit": len(fit), "dev": len(dev), "eval": len(outer)},
        "eval_positive": sum(r["label"] for r in outer)
            if sum(r["label"] for r in outer) >= 10 else "<10",
        "selected_mixture_components_by_DEV_NLL": selected,
        "dev_masked_item_nll": {str(k): dev_scores[k] for k in KS},
        "budgets": budgets,
        "arms": arms,
        "model_family": "class_conditional_latent_categorical_mixture",
        "planning": "adaptive_two_step_expected_binary_entropy_lookahead",
        "patient_partition_fingerprint": hashlib.sha256(
            "|".join(sorted(r["pid"] for r in outer)).encode()).hexdigest()[:16],
        "disclosure": [
            "Exploratory TRAIN-internal reused cohort; Phase44 TEST already consumed",
            "Retrospective same-V04 masking; NOT an R01->future V04 action",
            "Only a measured 27-visible-item value can become a Patient tool response",
            "Latent imagined scores never enter observed evidence",
            "Hidden six outcomes are never offered as an action",
            "No Qwen, no real Doctor prompting, no claim of integrated RSI in Phase60",
            "DEV chooses number of latent mixture components; EVAL does not",
            "Bootstrap holds trained models fixed and does not correct exploration",
            "Outputs contain aggregate metrics only, with positive cells under 10 suppressed",
        ],
    }
    return metrics, paired, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS + ("all",), default="all")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-db", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 0 or args.epochs < 1:
        raise ValueError("Invalid requested full experiment")
    data = args.data_root.expanduser().resolve()
    db = args.manifest_db.expanduser().resolve()
    report = args.report_dir.expanduser().resolve()
    if not (data / "PPMI").is_dir() or not db.is_file():
        raise FileNotFoundError("Missing frozen TRAIN source or manifest DB")
    if report == data or data in report.parents or db == report or db.parent == report:
        raise ValueError("Unsafe aggregate report directory")
    from scripts.phase45_wm_rsi_agents import load_train
    started = time.perf_counter()
    rows, items, qc, source = load_train(data, db)
    metrics, paired, metadata = run(
        rows, items, arm=args.arm, budgets=args.budgets,
        epochs=args.epochs, bootstrap=args.bootstrap)
    report.mkdir(parents=True, exist_ok=False)
    write_csv(report / "arm_metrics.csv", metrics)
    write_csv(report / "paired_brier_bootstrap.csv", paired)
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (report / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase60_complete",
        "arm": args.arm,
        "aggregate_report_dir": str(report),
        "metrics_rows": len(metrics),
        "paired_comparisons": len(paired),
        "selected_k": metadata["selected_mixture_components_by_DEV_NLL"],
    }, indent=2))


if __name__ == "__main__":
    main()
