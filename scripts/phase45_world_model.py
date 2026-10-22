"""Small probabilistic observation World Model and grounded Doctor/Patient agents.

V0 is a generative model of *same-visit* MDS-UPDRS III item observations,
NOT a learned temporal Parkinson's disease state-transition model. Data
supplied to the patient service must be measured onsite V04 records only.
"""
from __future__ import annotations
import hashlib
import math
from collections import Counter

ITEM_VALUES = tuple(range(5))


def entropy(p):
    p = min(1 - 1e-12, max(1e-12, p))
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    q = math.exp(z)
    return q / (1 + q)


class ObservationWorldModel:
    """Naive Bayes categorical p(y, visible items), Laplace-Dirichlet prior.

    The discrete hidden outcome y is whether the six unobservable Part III
    scores sum to > 0. This model can simulate *possible observed values*
    of each as-yet-unqueried visible item conditional on observed history.
    """

    def __init__(self, items, alpha=1.0, temperature=1.0):
        if not items or len(set(items)) != len(items):
            raise ValueError("Expected distinct visible item names")
        if alpha <= 0 or temperature <= 0:
            raise ValueError("Invalid smoothing/temperature")
        self.items = tuple(items)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.fitted = False

    def fit(self, records):
        records = list(records)
        if len(records) < 12:
            raise ValueError("Too few independent training people")
        class_counts = Counter()
        hist = {k: [[0] * 5 for _ in range(2)] for k in self.items}
        for row in records:
            y = int(row["label"])
            if y not in (0, 1):
                raise ValueError("Label must be 0/1")
            class_counts[y] += 1
            for k in self.items:
                v = row["features"][k]
                if isinstance(v, bool) or v not in ITEM_VALUES:
                    raise ValueError("Invalid recorded visible item")
                hist[k][y][int(v)] += 1
        a = self.alpha
        self.prior = (class_counts[1] + a) / (len(records) + 2 * a)
        self.conditional = {
            k: [[(hist[k][y][v] + a) / (class_counts[y] + 5 * a)
                  for v in ITEM_VALUES] for y in (0, 1)]
            for k in self.items
        }
        self.fitted = True
        return self

    def posterior(self, observations):
        if not self.fitted:
            raise RuntimeError("Fit the world model first")
        logodds = math.log(self.prior / (1 - self.prior))
        for k, v in observations.items():
            if k not in self.conditional or v not in ITEM_VALUES:
                raise ValueError("Unrecognized clinical observation")
            logodds += math.log(self.conditional[k][1][v] /
                                self.conditional[k][0][v])
        return sigmoid(logodds / self.temperature)

    def predict_item(self, observations, name):
        """A distribution over a permissible next observed symptom score."""
        if name in observations or name not in self.conditional:
            raise ValueError("Item already known or inaccessible")
        p = self.posterior(observations)
        return tuple((1 - p) * self.conditional[name][0][v] +
                     p * self.conditional[name][1][v]
                     for v in ITEM_VALUES)

    def information_gain(self, observations, name):
        prior = entropy(self.posterior(observations))
        probs = self.predict_item(observations, name)
        future = sum(probs[v] * entropy(self.posterior({**observations, name: v}))
                     for v in ITEM_VALUES)
        return max(0.0, prior - future)

    def static_order(self):
        return sorted(self.items, key=lambda k: (-self.information_gain({}, k), k))


class PatientEvidenceService:
    """Only measured 27-item V04 values are accessible through request()."""

    def __init__(self, measured_visible, allowed, budget):
        self._measured = dict(measured_visible)
        self.allowed = frozenset(allowed)
        self.budget = budget
        self.observed = {}
        if set(self._measured) != self.allowed:
            raise ValueError("Evidence availability mismatch")
        if budget < 0 or budget > len(self.allowed):
            raise ValueError("Invalid patient question budget")

    def request(self, item):
        if item not in self.allowed or item in self.observed:
            raise PermissionError("Missing, forbidden or repeated measurement")
        if len(self.observed) >= self.budget:
            raise PermissionError("Question budget exhausted")
        value = self._measured[item]
        if value not in ITEM_VALUES:
            raise ValueError("Invalid recorded score")
        self.observed[item] = value
        return {"item": item, "recorded_score": value,
                "provenance": "observed_V04_masking_simulation"}


def query_policy(world, patient, mode, question_budget, pid, seed):
    """Doctor chooses queries without receiving any hidden target scores."""
    if mode not in ("random", "static", "dynamic", "variance"):
        raise ValueError("Unknown Doctor controller")
    if mode == "random":
        fixed = sorted(world.items, key=lambda k: hashlib.sha256(
            f"phase45|{seed}|{pid}|{k}".encode()).hexdigest())
    elif mode == "static":
        fixed = world.static_order()
    else:
        fixed = None
    while len(patient.observed) < question_budget:
        remaining = [k for k in world.items if k not in patient.observed]
        if mode in ("random", "static"):
            next_item = next(k for k in fixed if k in remaining)
        elif mode == "dynamic":
            next_item = max(remaining, key=lambda k:
                            (world.information_gain(patient.observed, k), -world.items.index(k)))
        else:
            next_item = max(remaining, key=lambda k:
                            (sum(v*v*prob for v,prob in enumerate(world.predict_item(patient.observed,k))) -
                             sum(v*prob for v,prob in enumerate(world.predict_item(patient.observed,k)))**2,
                             -world.items.index(k)))
        patient.request(next_item)
    return world.posterior(patient.observed), len(patient.observed)
