# Parkinson agent research: verified snapshot through Phase 44

**Frozen record:** 2026-10-19 (America/Toronto). **Sources:** the local aggregate report archives `pks_phase23_report.zip` … `pks_phase44_report.zip`, their `summary.json` and the specifically named CSVs below. No patient-level data are committed. **Status:** TEST was consumed in Phase44; it is NOT a fresh held-out set for future iterations.

This note is a retrospective record of analyses already performed, **not** a new experiment. Exact source-archive checksums and the execution playbook are in `docs/phase23_44_reproduction.md` and `docs/phase23_44_report_index.csv`.

## Research question and scope

Real PPMI remote `R01` MDS-UPDRS Part III visits often cannot rate the five rigidity items and postural stability. At the later onsite `V04` visit, those items can be observed. Can the **27 remote-visible item sum at R01** identify people more likely to have a **nonzero six-item sum at FUTURE V04**, while respecting a fixed observation budget? This is **prognostic risk stratification**, not the causal utility of referral or an R01 same-day missing-score label.

The experiment series also tested baseline-to-longitudinal Motor prediction and a simulated **same-visit V04 masking** benchmark. Keep these three task types separate.

## Frozen population, eligibility, leakage rules

- Original cohort: PPMI Prodromal; frozen Phase23 **patient-level** splits: **TRAIN 1,312 / VAL 281 / TEST 281** from 1,874 candidate people. Split seed **20261018**. No cross-split patient sharing.
- Source `MDS-UPDRS_Part_III_06Oct2026.csv`, private data rooted at `/scratch/ylong66/data/pks/data`. Frozen manifest `outputs/longitudinal_manifests/prodromal_motor_v0.sqlite` relative to code checkout `/home/ylong66/workspace/pks_agent`. Raw data and code are NOT in the same parent directory.
- Medication-state filter `UNSPECIFIED` is a source-data label, **not** a clinically proven uniform OFF or ON treatment state.
- BL uses anchored month. R01 remote requires a valid visit at BL+4..8 months; onsite V04 at BL+9..15 months, strictly later than R01. Reject ambiguous repeated exams. For the real-remote task, all 27 visible `NP3*` items must be numeric 0..4, all six hands-on items are unrateable (numeric 101/UR), and R01 full `NP3TOT` is missing.
- A labeled V04 requires all 33 items numeric 0..4 and their sum exactly equal the documented onsite `NP3TOT`; no eligible V04 means **unknown future label**, not a negative.
- At R01, only the visible 27-item score enters the deployed rank policy. No future V04 score, label, visit gap, outcome readiness, age or hidden item is an input. Compare future-V04 proxies separately from contemporaneous V04-masking experiments.
- Do NOT publish raw scores/identifiers/clinical row files, patient-level decisions, private manifests or unscreened ZIPs. Public percentages may permit reconstruction of counts smaller than ten; this file contains only study-level aggregates.

## Results ledger (source CSV indicated)

| Phases | Split / sample | Evidence | Main conclusion |
|---|---|---|---|
| 23, 28, 32 | TRAIN 1,312 | Frozen target counts: 6m 34 / 12m 586 / 24m 377; 12m+24m intersection 332; all 6/12/24 just 25. Phase32 raw-to-manifest discrepancies **0**. | Longitudinal timepoints are sparse, particularly at 6m; source consistency verified, not clinical validity. |
| 27 | TRAIN, 586 12m pairs | Persistence MAE/RMSE 3.0836/4.6570; Motor-only Ridge 3.1169/4.4519; Motor+MoCA 3.1170/4.4506; adding RBD/SCOPA/UPSIT did not yield robust incremental benefit. | Extra baseline modalities do **not** demonstrate stable incremental benefit for this target. |
| 29–30 | TRAIN, 332 BL→12→24 trajectories | 12m persistence MAE/RMSE 3.2470/5.2607; **fixed (BL+12m)/2** 2.9081/4.7986; 2-visit Ridge 3.0694/4.7592. Trained blend selected 50/50 in **all five folds**. | Historical averaging is a powerful baseline. Disease-transition learning is **not established**. |
| 31–35 | TRAIN source | Phase32: 410 rejected patient-month keys, all empty `NP3TOT` (not observed numeric disagreements); 286 six-month R01 blank totals. Phase35: 274/286 R01 blanks have all six hands-on fields coded numeric 101. | Remote modality induces **structural unobservability**, not six true zero scores. |
| 36–37 | TRAIN, **605** complete onsite V04 exams | 33-item sums matched documented full totals. Six hidden items summed to zero for **399/605**. V04 mask target: zero MAE/RMSE 0.7785/1.6993; 27-item Ridge 0.8477/1.3471. | Hidden-total prediction has a severe zero-mode baseline; learned model improves RMSE, not MAE. |
| 38 | TRAIN, 605 V04 simulated masked visits | At 20% idealized reveal budget, visible-total ranking captures **81/206** hidden-positive patients vs **40/206** for one random allocation; residual MAE 0.4066 vs 0.6265. 27-item Ridge ranking has no clear benefit over visible total. | Simulated same-visit budgeted information acquisition works; clinical referral benefits **not** tested. |
| 39–40 | TRAIN, real R01 229, V04 605, paired 199 | Visible 27-item sum mean R01 3.459 vs V04 4.782 (SMD -0.271). Paired cross-visit Spearman **0.599** (500-bootstrap CI 0.490–0.691). Transferred V04 20% cutoff selects only 13.54% of R01; exact top-k fixes capacity. | Real remote/site distribution shift and temporal rank instability require separate evaluation. |
| 41–42 | TRAIN, real R01 229, later V04 observed for 199 | On observed cases: AUROC **0.6806**, AP **0.4933** for future nonzero V04 six-item score. Of all 229 R01 at exact 20% budget, *worst-case future-positive recall* 26.19%, random expected recall 20.09%; 30 outcomes unknown. | Descriptive risk ranking exceeds random-budget expectation under adversarial missing-future-label assignments, not a causal effect. |
| 43 | **Independent VAL**, eligible R01 49; V04 observed 41 | Locked 20%-budget primary: sharp recall lower **25.00%**, random expected **20.41%**, lift **+4.59 pp**. Known-label AUROC **0.6140**, AP **0.4229**. | Precommitted direction-positive independent validation, but low precision. |
| 44 | **Final TEST, now consumed**, eligible R01 37; V04 observed 34 | Same locked 20%-budget primary: sharp recall lower **30.77%**, random expected **18.92%**, lift **+11.85 pp**. Known-label AUROC **0.8696**, AP **0.7259**. | Final direction-positive test on a *small* cohort; no p-value or causal/clinical utility validation. |

**Do not silently mix metrics:** a 20% quota ranks all eligible R01 people, including the unknown-label ones; AUROC and AP necessarily use observed future labels only. Sharp missing-label recall bounds are identification bounds, **not bootstrap confidence intervals**. The numerical lift is `lower_bound_recall - k/n` in percentage points; `k = floor(0.20*n + 0.5)`. This is NOT a statistical significance criterion.

## Final precommitted policy: immutable v1

```text
Inputs: genuine R01, 27 visible valid Part III scores only.
risk_i = SUM(visible_27_item_scores_i)
Rank: descending risk_i.
Tie: ascending SHA256("Phase41|20261018|{patient_id}") in local process only.
Decision capacity: k = floor(0.20 * total_eligible_R01 + 0.5).
Output: exactly k ranked people for simulated risk-based prioritization.
Outcome: LATER qualifying V04 hidden-six sum > 0 (or UNKNOWN).
Primary report: sharp worst-case future-positive Recall minus k/N.
```

Formula: let `a` = selected known future positives, `p` = all known future positives, and `v` = **unselected** R01 people missing a qualifying future V04 label. Then `Recall_lower = a/(p+v)`. The random exact-k allocation reference has `E[Recall] = k/N` for any fixed complete labels.

The lock was documented in `docs/phase43_frozen_validation_protocol.md` before VAL and in `docs/phase44_final_test_protocol.md` before TEST. Phase43/44 must not be turned into a hyperparameter search or repeated final assessment.

## Known limitations and permitted claims

1. Real R01 six hidden scores are **not observed**, and a future V04 visit is not concurrent remote ground truth. This is future risk stratification, not contemporaneous six-item imputation.
2. A policy ranking future abnormalities does not estimate benefits, harms, or costs of ordering a new clinic visit at R01. Phase38's perfect reveal is an **idealized same-V04 simulation**.
3. R01→V04 gap, treatment conditions, enrollment and follow-up selection, incomplete data dictionaries, source-date precision and cohort shift can confound interpretation.
4. External validation in BioFIND/FoxInsight has **not** been established for this exact future-on-site outcome. Data access/licensing may restrict reproduction.
5. No evidence so far shows **Multi-Agent**, **World Model** or **Recursive Self-Improvement (RSI)** outperforming the simple final policy. This is a research **baseline**; it does not establish agent-based state-of-the-art performance.
6. **Final TEST consumed** at report generation time **2026-10-19T02:55:59Z**. Any later changes to models or decision rules require a **new independent cohort or a properly independent evaluation design**, never reuse this TEST as unseen.

## Source traceability

All values above were checked against the user-provided ZIPs and their internal aggregate CSVs/JSON. Canonical report paths include `outputs/phase27_fair_incremental/nested_multimodal_oof.csv`, `outputs/phase30_smoothing/model_metrics.csv`, `outputs/phase32_exclusion_audit/summary.json`, `outputs/phase37_masked_site_benchmark/masked_site_oof_metrics.csv`, `outputs/phase38_budgeted_acquisition/budgeted_acquisition_curve.csv`, `outputs/phase41_future_risk/prospective_rank_discrimination.csv`, `outputs/phase42_missing_followup_bounds/missing_outcome_budget_bounds.csv`, `outputs/phase43_locked_validation/locked_validation_policy.csv`, and `outputs/phase44_locked_final_test/locked_validation_policy.csv`. Archive hashes are in the artifact index.
