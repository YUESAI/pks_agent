# Phase82 complete five-fold evidence audit and Phase83 conditional clinical channel

phase82_reports.zip: 20 of 20 full job summaries (336369–336388);
all 60 tasks completed. There are 405 method metric rows, 1020
paired-bootstrap rows, 60 retrospective oracle-headroom rows and
15 DEV/AUDIT verifier decisions. Five Qwen groups made **354 total**
CPU calls. One of 15 candidate switches accepted (Prodromal24 fold3);
Qwen alone abstained zero times when invoked.

Use arm3,7,11,15,19 for same-job matched comparison to avoid
double-counting. Patient-weighted fivefold RPS:

| method | Prodromal12 n584 | Prodromal24 n378 | PD12 n156 |
|---|---:|---:|---:|
| Motor q0 | .024503 | .027539 | .073328 |
| Fixed measured q1 | **.024336** | **.027438** | **.073238** |
| Expected-KL q1 | .024827 | .027739 | .073889 |
| OOF net-value q1 (cost .00025) | .024509 | .027652 | .074015 |
| Value-guarded real Qwen | .024509 | .027572 | .074015 |

Qwen query counts on EVAL: Prodromal12 63/584 (10.8%),
Prodromal24 15/378 (4.0%), PD12 75/156 (48.1%). Hard guard
blocked most proposals; no LLM-only ABSTAIN. Thus this system
saved Qwen calls but did not establish clinical RPS uplift.
Actual queries of OOF and Qwen were identical in the first
two tasks except limited different selected items in 24m fold3.

Using per-query .00025 RPS-equivalent cost, *no query* gives
best net RPS in all three tasks. This is an illustrative utility,
not an actual medical service price. Patient-paired bootstrap
intervals for fixed->OOF RPS include zero within each available fold.

Retrospective outcome-aware clairvoyant single-test oracle ceiling:
| task | q0 RPS | hindsight best RPS | apparent relative headroom |
|---|---:|---:|---:|
| Prodromal12 | .024503 | .022236 | 9.3% |
| Prodromal24 | .027539 | .024698 | 10.3% |
| PD12 | .073328 | .063956 | 12.8% |

The oracle sees the EVAL patient's true Y and selects whichever
already measured candidate minimizes that patient's RPS. It is
highly optimistic due to per-patient hindsight/multiple comparisons,
NOT the achievable gain from prospective testing. Do not train
from the oracle or use it for clinical claims.

## Phase83: test source of acquisition failure

Hypothesis: Phase80–82 assumes marginal P(BL test | future total
burden category) and may double-count the history already reflected
in Motor prior p(Y|Motor histories).

Use a conditional evidence channel approximating P(BL test bucket
| 4-category future Motor burden, 3-band *existing Motor risk*).
Fit risk-band emissions from FIT **OOF** predictions, never use
outer patient outcomes. Shrink conditional estimates strongly toward
risk-band test marginals and temper likelihood ratios at fixed
alpha=.5. This is NOT full conditioning on the Motor history, and
conditioning on a scalar risk can lose essential information.

For OOF query-value estimation, train three distinct Motor models on
other FIT subjects, obtain genuine out-of-fold priors for each FIT
subject, and train emission likelihoods excluding the OOF validation
subset. Use those inner-held subjects' observed BL clinical values
and future outcomes only to estimate test value on the FIT pool.
The outer conditional emission uses the full FIT OOF priors and
labels, not EVAL outcomes.

Four ordinary arms per fold (20 independent jobs):
0 Motor q0 vs Phase80 original marginal FIT-priority query.
1 Add new risk-conditional fixed query; isolate update mechanism.
2 Add new conditional OOF value selection with ABSTAIN and cost threshold.
3 Real Qwen chooses among conditional OOF-positive candidates;
  DEV/AUDIT reviews a single Qwen-versus-OOF policy switch.

Evaluate paired RPS, MAE from median, RMSE from mean, Brier,
net RPS at query cost .00025, query count and Qwen calls.
Assess whether conditional update beats marginal update under
identical fixed test selection BEFORE claiming an Agent uplift.

These are repeatedly explored frozen Prodromal TRAIN and strict
PD; original consumed TEST is not accessed. Same pre-cutoff Motor
measurements, genuinely recorded BL clinical data and patient-level
splits. No Qwen access to test value/outcome, no fake observations,
no treatment recommendations, and no autonomous RSI.

## Launch 20 ordinary independent CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE83_ARM_ID="$arm" slurm/phase83_dusky_cpu.sbatch
done
```

8 CPUs reserved, 48 GB memory, strict 12h maximum, no array, no
GPU, PyTorch single-thread and MKLDNN disabled to avoid Phase77
12-hour GRU backward stalls. Only arm3,7,11,15,19 run local
Qwen3-4B CPU inference. Per-task aggregates saved, then summary.json
on complete job. Code is committed, not yet tested on Dusky.
