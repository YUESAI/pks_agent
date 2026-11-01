# Phase83 final five-fold report and research decision (2026-11-01)

## Input and job completeness

User-provided `phase83_fixed_reports.zip` contains 25 job directories,
including five *previously failed* Qwen jobs. Only the newest COMPLETE
report for each of twenty arms was included. New Qwen jobs:
arm3=336490, arm7=336491, arm11=336492, arm15=336493,
arm19=336494. All 20 arms now completed 3/3 tasks each.
Selected reports contain 225 aggregate method rows, 840 paired
bootstrap comparison rows, 60 coverage rows and 15 DEV/AUDIT
verification events. The earlier FAILED Qwen directories are excluded.

All cross-method values below are computed **only within arms
3,7,11,15,19**, one arm per person-disjoint outer fold, so
repeated base models from other arms are not double counted.
RPS/median-MAE/Brier/net-RPS are patient-count-weighted fold means.
RMSE is the square root of patient-count-weighted fold RMSE-squared.
No patient-level predictions or identifiers are exported.

| outcome | n | method | RPS | median-MAE | pooled RMSE(mean) | Brier | EVAL q | net-RPS@0.00025 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Prod12 |584| motor_q0 | .024495193 | .758562 | 1.487824 | .212211 | 0 | .024495193 |
| Prod12 |584| marginal_fixed | **.024328266** | .758562 | 1.479008 | .210044 | 584 | .024578266 |
| Prod12 |584| conditional_fixed | .024413196 | .760274 | 1.484844 | .210714 | 584 | .024663196 |
| Prod12 |584| conditional_oof | .024485368 | **.755137** | 1.485782 | .212239 | 17 | .024492645 |
| Prod12 |584| conditional_qwen | .024485368 | .755137 | 1.485782 | .212239 | 17 | .024492645 |
| Prod12 |584| verified_policy | .024485368 | .755137 | 1.485782 | .212239 | 17 | .024492645 |
| Prod24 |378| motor_q0 | .027548359 | .875661 | 1.724469 | .209287 | 0 | **.027548359** |
| Prod24 |378| marginal_fixed | .027449879 | **.841270** | 1.721325 | **.205272** | 378 | .027699879 |
| Prod24 |378| conditional_fixed | **.027423640** | .846561 | **1.720892** | .206500 | 378 | .027673640 |
| Prod24 |378| conditional_oof | .027543358 | .873016 | 1.724185 | .209128 | 64 | .027585686 |
| Prod24 |378| conditional_qwen | .027543358 | .873016 | 1.724185 | .209128 | 64 | .027585686 |
| Prod24 |378| verified_policy | .027543358 | .873016 | 1.724185 | .209128 | 64 | .027585686 |
| PD12 |156| motor_q0 | .073336908 | 2.480769 | 3.158771 | .050485 | 0 | .073336908 |
| PD12 |156| marginal_fixed | .073248065 | **2.448718** | **3.153993** | .051413 | 153 | .073493258 |
| PD12 |156| conditional_fixed | **.073239248** | 2.467949 | 3.155553 | .050818 | 153 | .073484440 |
| PD12 |156| conditional_oof | .073249853 | 2.467949 | 3.156565 | **.050434** | 23 | **.073286712** |
| PD12 |156| conditional_qwen | .073249853 | 2.467949 | 3.156565 | .050434 | 23 | .073286712 |
| PD12 |156| verified_policy | .073249853 | 2.467949 | 3.156565 | .050434 | 23 | .073286712 |

Tiny differences of 0.0000025 in Prod12 net-RPS favor OOF over
the q0 baseline under the stipulated RPS-equivalent cost; they
have NO practical or inferential support. For Prod24 and PD12,
q0 has lower net-RPS than all test policies. Cost is experimental
score-equivalent, not patient time, pain, dollars, or real
healthcare unit costs.

## Agent-specific outcomes

Across the five new *complete* Qwen jobs there were 245 Qwen
CPU inferences total: 40 in Prod12 tasks, 143 in Prod24,
62 in PD12, counted across DEV, AUDIT and outer EVAL.
No invalid actions or Qwen-only abstentions were recorded.
Actual q1 releases in outer EVAL: 17/584 Prod12, 64/378
Prod24, and 23/156 PD12, the same counts as conditional
OOF. In all 15 paired task/fold aggregates, Qwen and OOF
had exactly the same output metrics (RPS, median-MAE, mean-RMSE,
Brier, NLL and query counts). Do not assert that all Qwen
*action names* are identical without a privacy-safe per-patient
action agreement aggregate. All 15 DEV/AUDIT Qwen-policy
proposals were REJECTED; every DEV and AUDIT net-RPS
candidate-minus-incumbent difference was exactly zero.
Verifier therefore did not improve forecast accuracy.

Conditional fixed vs marginal fixed RPS was worse for
Prod12 and slightly better for Prod24/PD12 in the aggregate,
but fold signs are mixed; per-fold paired confidence
intervals generally include zero.
In Prod12 conditional_oof was worse RPS than
conditional_fixed in 4/5 folds; in Prod24 3/5 folds.
No robust active-acquisition uplift above fixed query exists.
The value of full-length LLM inference does not appear
justified by these exploratory results.

## Reproducibility issue discovered from repeated jobs

Although base model seeds and outer patient folds nominally
match, independently executed arms 2 and 3 within the same
fold yield slightly different q0 RPS values, up to about
0.000104 for Prod12 (larger than some claimed updates).
Code inspection identified a probable mechanism:
`scripts/phase62_ppmi_multicohort_audit.py` loops over the
unordered set `cohorts.keys()|diagnoses.keys()` while building
the patient registry; iteration order is process dependent
when PYTHONHASHSEED is not set. `build_auxiliary` receives
the resulting patient order, so random seeded minibatch indices
may select different auxiliary examples. This is a plausible
cause but not a proven exclusive explanation.

Before any further controlled experiment, impose stable
patient ID order and auxiliary example order, set hash seed
BEFORE interpreter start, freeze file and split hashes,
and add cross-process same-seed reproducibility checks.
Do NOT quietly mix old and post-fix fold outcomes.

## Cross-fitting limitation

The Phase83 FIT-only value model trains an inner OOF Motor
prediction for each FIT person. Then it fits the inner
conditional evidence likelihood on OOF predictions from
other FIT people, some of whose OOF predictors may themselves
have trained on the current validation subjects. Thus some
inner FIT validation targets can affect emission inputs
*indirectly*. This is not strictly nested sample splitting;
do not claim rigorous independent FIT cross-fitting.
No outer heldout labels were directly used in model fitting
and Qwen did not see clinical values or labels before query.
Repair would require nested cross-fitting of the
conditional emissions or training emissions from inner
train-only models and assessing inner validation separately.

## Scientific decision

Stop automated Phase84/85 hyperparameter and prompt search
on the repeatedly explored original PPMI TRAIN. Package
Phase62–83 results into a coherent research audit:
- positive longitudinal Motor distribution forecasting gains
  vs matched clinical baseline, properly scored by mean RMSE,
  median MAE and full-posterior RPS;
- repeated negative evidence for WM- and LLM-driven active
  test selection compared with fixed clinical baseline;
- non-independent repeated TRAIN fold exploration;
- downstream work: stable deterministic data/model pipeline,
  strictly nested OOF conditional evidence utility,
  one genuine pre-registered prospective/temporally external
  patient holdout and possibly fully observed fixed test panels;
- do not call current model a clinical causal simulator or
  autonomous self-improving RSI.

The original consumed final TEST has NOT been reopened.
No patient-level private data was exported to this document.
