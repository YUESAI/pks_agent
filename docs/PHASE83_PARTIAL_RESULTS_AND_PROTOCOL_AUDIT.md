# Phase83 conditional-evidence experiment: verified interim report (2026-11-01)

Source: owner-uploaded `phase83_reports.zip`. **NOT a complete 20/20
final evaluation**. Archive contains 20 job directories and 107
non-directory files. The 15 non-Qwen jobs (arms 0,1,2 in each of five
outer folds) all completed three tasks each. All 5 real-Qwen arms
are INCOMPLETE in this snapshot, NOT automatically assumed failed:

| Qwen arm | Slurm ID | Last completed task | Progress |
|---|---:|---|---|
| 3 | 336397 | Prodromal24 | 2/3 |
| 7 | 336401 | Prodromal12 | 1/3 |
| 11 | 336405 | none | 0/3 |
| 15 | 336409 | Prodromal12 | 1/3 |
| 19 | 336413 | none | 0/3 |

No Slurm logs included, so whether queued/running/timeout/failed
cannot be inferred. DO NOT resubmit these jobs until checking sacct
and squeue. This ZIP has 159 aggregate method rows, 576 paired
comparison rows, 49 completed task coverage rows, and 4 Qwen
DEV/AUDIT decisions.

## Five-fold robust non-Qwen findings

Use arms **2,6,10,14,18 only** as a single comparable reference
set for every model; these are 5 patient-disjoint outer folds and
contain each method in the same job. Patient-weighted means (MAE and
RPS are averages weighted by each fold's heldout `patients`).
RMSE below is a weighted average of per-fold RMSE, not the exact
pooled patient-level square-root MSE.

| Cohort | N | Policy | median-MAE | mean-RMSE* | RPS | Brier | queries | net-RPS cost0.00025 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Prodromal12 |584| motor_q0 | .758562|1.480207|.024502|.211980|0|.024502 |
| Prodromal12 |584| marginal_fixed | .758562|1.471333|**.024340**|.209860|584|.024590 |
| Prodromal12 |584| conditional_fixed | .761986|1.477071|.024418|.210499|584|.024668 |
| Prodromal12 |584| conditional_oof | .760274|1.480560|.024517|.211938|13|.024523 |
| Prodromal24 |378| motor_q0 | .875661|1.692110|.027542|.209314|0|**.027542** |
| Prodromal24 |378| marginal_fixed | .851852|1.686086|.027454|.205339|378|.027704 |
| Prodromal24 |378| conditional_fixed | .851852|1.690110|**.027413**|.206165|378|.027663 |
| Prodromal24 |378| conditional_oof | .873016|1.692063|.027530|.209016|62|.027571 |
| PD12 |156| motor_q0 | 2.474359|3.140965|.073329|.050486|0|.073329 |
| PD12 |156| marginal_fixed | 2.461539|3.137553|.073239|.051416|153|.073484 |
| PD12 |156| conditional_fixed | **2.455128**|3.138252|**.073231**|.050819|153|.073476 |
| PD12 |156| conditional_oof | 2.461539|3.138425|.073242|**.050435**|23|**.073279** |

Note: for illustrative per-query 0.00025 penalty, q0 net RPS
is best for Prodromal12/24; PD12 conditional OOF has slightly better
net RPS than q0 (difference ~0.00005), still exploratory and not
statistically established. No real healthcare dollar cost is implied.

Per-fold RPS changes for conditional_fixed vs marginal_fixed:
Prodromal12: +0.000168,-0.000060,+0.000178,-0.000125,+0.000196.
Prodromal24: -0.000212,-0.000386,+0.000556,-0.000487,+0.000408.
These are mixed. For conditional_oof vs conditional_fixed:
Prodromal12: +.000020,+.000036,+.000109,+.000062,+.000230
(**5/5 worse**). Prodromal24: 4/5 worse, 1/5 slightly better.
Each task's difference is small; multiple per-fold paired 95%
bootstrap intervals include zero. These are repeatedly examined
original frozen Prodromal TRAIN (not independent confirmation).

Conditional OOF query count: 13/584 (2.2%) Prodromal12;
62/378 (16.4%) Prodromal24; 23/156 (14.7%) PD12.
Across-fold query decisions are unstable. More abstention does not
establish a predictive uplift.

## Incomplete real-Qwen evaluation

At upload time the Qwen jobs had completed only 4 task/fold
combinations (Prodromal12 fold0, fold1, fold3; Prodromal24
fold0). In those four matched pairs `conditional_qwen` equals
`conditional_oof` exactly, with identical per-fold RPS,
median-MAE, RMSE, Brier and released test count. Four audit
decisions all rejected switching. This does NOT establish a
five-fold Qwen conclusion. Qwen calls logged by the 4 partial
tasks total 99 across DEV/AUDIT/EVAL; no LLM-only abstention
or invalid actions logged in these partial tasks.

## Critical protocol issue in crossfit internals

`OOFConditionalValue.fit()` first creates a 3fold `motor_oof`.
For an inner outer-fold k, the conditional emission for
validation subjects is then fitted from `motor_oof[train_idx]`.
Some of those training-prior predictions were produced by models
trained on validation-subject outcomes (since other OOF models
were fitted on all folds except their respective held folds).
Consequently the emission model for a held person could have
**indirectly** depended on that held person's label through
other people's OOF predictions. The manuscript MUST NOT claim
this is strict nested cross-fitting or completely leakage-free.
This does not mean labels were explicitly given to Qwen or that
an outer evaluation fold entered model fitting. It matters for
the internal FIT-only value estimates and may overfit selection.

A future strict repair would refit inner train-cell priors using
models trained **only on the inner train subjects**, then evaluate
inner held subjects on predictions from a separately trained model;
or create an actually nested sub-crossfit inside each inner
train block. Preserve an independently untouched outer heldout.

## Immediate next steps

1. Check `squeue` and `sacct` for Qwen IDs 336397,336401,
   336405,336409,336413. If they are still running, wait for
   existing jobs to finish; do not submit duplicates.
2. When done collect Qwen task output and original Slurm logs;
   re-evaluate all five folds separately without double counting.
3. Before expanding to another phase, inspect the nested OOF
   dependence and decide whether a repaired prospective acquisition
   benchmark with a genuinely hidden availability process and
   an independent patient cohort is feasible. Do not claim
   autonomous RSI or agent clinical benefit.

No patient IDs, raw scores or individual-level predictions are
recorded here. The original consumed TEST remains excluded.
