# Phase65 results and Phase66 fivefold CPU plan

Phase65 user-uploaded phase65_reports.zip: 20 arms, 60 aggregate metrics
rows and 15 real verifier proposals, all accepted. 12m Prodromal 126
patients (45 positives), 24m Prodromal 80 (24 positives), PD12m 35
(all positive, undefined AUC). Archive is complete at report-file level.

Selected Phase65 observations:
- Prodromal12 P1 plain Qwen MAE .7123/AUC .7228, vs P1 WM-guided
  Qwen 1.1757/.6572, and P1 Qwen+RSI-lite 1.0179/.5936.
- Prodromal24 P0 plain Qwen MAE .8807, versus WM-guided 1.3323.
- Plain Qwen invalid actions 1814 across 15 task reports,
  WM-guided 359 and RSI-lite 363.
- All accepted RSI proposals chose scalar 0.8 (candidate styles
  variance/risk_shift). Indicates shrinkage/calibration, not verified
  neural self-improvement. Confidence intervals are repeated TRAIN
  fixed-model bootstrap, not confirmatory clinical validation.

## Phase66 motivation and honest scope

20 CPU-only ordinary sbatch jobs = five outer subject folds (0..4)
times four policies (0..3). Every person in a task has one and only
one outer evaluation fold per policy; assemble later by cohort and
policy, using patient counts to compute weighted MAE/Brier. AUROC and
recall cannot generally be combined by simple averaging; a future
cluster-local aggregation step is needed for pooled OOF AUROC.

A coherent categorical observation model now computes posterior
p(future 6-item burden class | actually disclosed BL/R01 measurements).
Future burden is divided into 0,1,2..3,4+ strata, with class
conditional categorical feature likelihoods and Laplace shrinkage.
The posterior yields predicted burden and nonzero risk. For each
candidate action, the expected reduction in class entropy or future
burden variance is derived from the model's own predicted conditional
score-bin probabilities. No fake raw 0..4 MoCA/UPSIT scores. Baseline
features and actual pre-target R01 come from existing local PPMI.
Additional unlabeled FIT-fold patients can contribute measurement
bin edges/frequency but never future labels.

This is still an observational finite class WM, NOT a causal
intervention world model, neural longitudinal dynamics network,
nor parameter-updating joint RSI. Qwen3-4B prompts contain only
observed evidence, current allowed item names and optionally WM
score advice. Invalid actions counted; plain Qwen fallback is
label-free and deterministic. "Verified" variant selects
style/temperature on inner TUNE and independently accepts or
rolls back ONCE on AUDIT using a deterministic greedy policy
proxy; it does not test the revised policy using Qwen on DEV.
A clean external test cohort is still required.

## Submit 20 independent CPU jobs, no Slurm array, no smoke job

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE66_ARM_ID="$arm" slurm/phase66_dusky_cpu.sbatch
done
```

Resources per job: 8 CPU, 48GB, 12h. Most arms run offline Qwen,
so CPU runtime is significant. Slurm may queue jobs. Outputs:
fold_metrics.csv, cohort_coverage.csv, verifier_trace.json, summary.json
under phase66_fold_<ARM>_<JOBID>. ZIP aggregate results only.
