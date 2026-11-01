# Phase81 completed results and Phase82 cost-aware agent-value audit

Source phase81_reports.zip received Nov 1, 2026. All 20 jobs complete
(IDs 336040..336060, one skipped ID), each has all three task results.
300 method rows, 1500 paired-bootstrap contrast rows, 15
DEV/AUDIT Qwen-vs-OOF policy comparisons. Across five Qwen jobs,
2449 CPU Qwen inference calls, no invalid actions, none chose
ABSTAIN. Qwen jobs took ~4.2–4.5 hours each, others 14–23s.

Deduplication: the same reference models are repeated in multiple
standalone arms. For strict same-fold matched comparison, use **only
arm 3,7,11,15,19**, and weight MAE/RPS by outer heldout size.
Results:
| Policy | Prodromal12 N584 MAE_median/RPS | Prodromal24 N378 MAE_median/RPS |
|---|---|---|
| Motor q0 | .758562/.024488 | .873016/.027565 |
| Fixed q1 | .758562/**.024323** | **.851852/.027464** |
| WM expected-KL q1 | .765411/.024809 | .857143/.027776 |
| Deterministic random q1 | .768836/.024733 | .873016/.027577 |
| 3fold OOF utility, can abstain | .756849/.024476 | .870370/.027881 |
| Qwen can abstain | **.755137**/.024481 | .875661/.027979 |
| DEV/AUDIT verified policy | .756849/.024476 | .870370/.027881 |

PD12 N156: fixed MAE 2.461538/RPS .073232;
OOF MAE 2.455128/RPS .073376; Qwen MAE 2.461538/RPS .073367.

OOF abstention was 445/584 (76.2%) for Prodromal12,
288/378 (76.2%) for Prodromal24, but fit and observed
query rates were highly variable across folds. Qwen selected
a test for every available case across DEV/AUDIT/EVAL, despite
ABSTAIN in prompts. Verifier accepted only one of fifteen
Qwen policy switches (PD12), no Prodromal acceptance.
The small fixed RPS advantage is exploratory, NOT established
statistical significance. All cohorts are repeatedly developed
original frozen Prodromal TRAIN or exploratory PD; the consumed
original TEST remains untouched.

## Phase82 prespecified falsification criteria

This study tests whether any reliable *net clinical information
value* exists for active BL test selection, not whether a prompt
looks convincing. The existing Phase80 emission model conditions
on future severity but not full historical Motor state; therefore
causal/incremental-conditional gain cannot be claimed.

For every same-fold FIT-trained categorical WM and FIT-only
clinical emission channel, evaluate the following matched q0/q1
policies. 5 outer folds x four standalone CPU arms = 20 jobs:

0: q0, fixed q1, WM expected KL q1, reproducible random q1;
   outcome-aware hindsight oracle headroom for ANALYSIS ONLY.
1: cross-fitted OOF-value acquisition with net-RPS costs
   {0.0001, 0.00025, 0.0005}; comparator controls.
2: same OOF-value cost frontier with stricter 0.0005 primary
   selection and oracle/headroom analysis; no LLM overhead.
3: a **real Qwen** Doctor receives only tests whose
   FIT cross-fitted expected RPS gain exceeds 0.00025,
   or ABSTAIN. Hard value-filtering is enforced OUTSIDE the LLM.
   DEV/AUDIT gates one Qwen-vs-OOF policy switch under
   net RPS, MAE, RMSE and Brier guardrails.

Cost is an illustrative score-equivalent threshold, **not**
a currency estimate or real healthcare cost. Report raw RPS,
net RPS, median MAE, mean RMSE, Brier, query fraction and
the proportion of candidate Qwen actions blocked by FIT
evidence. The oracle chooses the best recorded test using
the evaluation patient's TRUE future outcome, so it is an
unattainable optimistic upper bound and MUST NOT be used
for candidate or deployment policy selection. Interpret as
maximum headroom under retrospective observed-test availability,
not causal value of measuring an unobserved test.

Five CPU Qwen arms (3,7,11,15,19) only, 12-hour cap, 48GB,
8 reserved CPU cores, conservative single-thread GRU backward
with MKLDNN disabled. Other arms are short. Every task's
aggregates are saved before the next task; summary.json only
appears on complete execution.

## Submit ordinary independent jobs (no array or new branch)

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE82_ARM_ID="$arm" slurm/phase82_dusky_cpu.sbatch
done
```

Upload `phase82_value_*/` directories, with method_metrics.csv,
paired_bootstrap.csv, coverage.csv, oracle_headroom.csv, audit.json,
progress.json, summary.json. Do not export PHI. This does not
implement autonomous RSI or prospective ordering.
