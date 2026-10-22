# Phase 43 — Frozen validation protocol (v1)

**Pre-specified before opening the PPMI validation cohort.** This is a one-shot independent, observational, patient-level *validation* of the simple strategy discovered in Phase 41/42 on the frozen PPMI Prodromal TRAIN split. The frozen TEST split is not opened.

## Prediction task and cohort

At real remote **R01** (4–8 months after the frozen BL anchor), select patients for *future-risk stratification* based on the **sum of all 27 valid remotely assessable MDS-UPDRS III items**. Six hands-on items must all be unavailable/101; the remote total must be blank. Include all such frozen **VAL** patients, not just those later observed. Exclude ambiguous repeated R01 or V04 records per the Phase36 conservative parser.

The binary **future** endpoint is whether those six hands-on items have a positive total during the patient's later, complete and internally sum-verified **V04** clinic examination (9–15 months from BL; V04 strictly after R01). A patient lacking an eligible V04 label remains *unknown*, not negative. This is NOT R01 same-visit ground truth or causal referral benefit.

## Locked policy and estimand

* Ranking: descending **27-item observable R01 sum** only; NO fitted regression, MoCA, age, outcome, actual future gap, or future observability feature.
* Ties: ascending SHA256 of literal `Phase41|20261018|{patient_id}`, used only in local RAM; seed fixed `20261018`.
* Primary budget: **20% of all eligible VAL R01 patients**, exact number `floor(0.20*n+0.5)`. Rank all eligible remote patients before checking V04 labels.
* Primary statistic: `sharp_lower_recall - k/n`, where `sharp_lower_recall=a/(p+v)`, `a` is selected known future positives, `p` all known future positives, and `v` the **unselected** unknown V04 labels. `k/n` is expected recall under uniformly random selection of exactly `k` patients, for any fixed completed labeling.
* Factual statement tested: is the primary lower-bound advantage **strictly positive**? This is a deterministic identification-bound comparison, **NOT** a hypothesis test or p-value.
* Secondary *descriptive* budgets: 10%, 30%, 40%, 50%, no selection or adjustment based on their values. Other secondary outcomes are observed-case tied-score AUROC and tie-aware Average Precision, released only when both observed classes have at least 10 patients.
* Small cell counts are suppressed in aggregate exports. No row-level labels, scores, decisions, or patient IDs leave the cluster.

## No after-the-fact tuning

The above cohort criteria, clinical item definitions, primary budget, tie rule, missing-label bounds, and reporting fields are frozen before this validation. Evaluate VAL once. Do NOT optimize the policy after inspecting VAL. Keep frozen TEST unopened until a separately timestamped final-test protocol is explicitly fixed. Validation size may be small, yielding inconclusive evidence.

## Threats to interpretation

1. The endpoint is a **later V04 observation**, not missing six R01 scores.
2. Follow-up missingness may be informative; identification bounds permit all assignments of those missing labels.
3. Even a positive lower-bound advantage does not prove clinical benefit, clinical cost savings, or that an LLM agent is needed.
4. A selected complete-case sample is insufficient for deployment; source definitions, visit modality, treatment state, and cohort transportability remain to be clinically validated.
5. Exact outcome percentages can reveal small counts; reports must be checked for permitted sharing.

Run via `scripts/phase43_locked_validation.py`; do not invoke the future TEST cohort.
