# Phase78 distributional Motor WM: 20 independent CPU jobs

Phase77 complete 20/20, new job IDs 335758–335777, each 3
tasks completed. Detailed actual results in
PHASE77_COMPLETE_RESULTS_PHASE78_DISTRIBUTIONAL_PLAN.md.

Five outer person folds x four arms:
0. Existing CT-direct Motor WM plus matched HGB.
1. A hurdle WM with learned p(burden>0) and p(burden=1..24 | positive);
   evaluate both exact distribution mean and categorical median.
2. A 25-way categorical Motor WM with total burden 0..24, evaluate
   distribution mean and median.
3. Fit both hurdle and categorical models plus CT-direct; select one
   prespecified candidate on DEV and verify once on AUDIT, requiring
   MAE improvement without meaningful RMSE or Brier worsening.
Every job computes its own matched Motor-only HGB and CT-direct
reference; some models are intentionally repeated in multiple arms
to preserve in-job patient-level contrasts.

The Motor history used for input is the actual visible27 visits up to
horizon-dependent prediction cutoff. True six-site future target
cannot enter input. All auxiliary adjacent Motor visit supervision
is from FIT persons, not evaluation, DEV, or AUDIT subjects.

This is NOT evidence of a full causal patient WM or a self-modifying RSI
and does not include Qwen. The aim is to isolate distributional
forecasting and supervised decision-functional calibration after the
Phase77 simple OOF median shift improved MAE but worsened RMSE.
For any final paper, use independent new patients not previously
selected, repeatedly tuned or leaked into model choice.

## Launch all 20 independent CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE78_ARM_ID="$arm" slurm/phase78_dusky_cpu.sbatch
done
```

No array. 8 CPU cores requested but PyTorch intraop and BLAS threads
conservatively capped at 1 to avoid the Phase77 GRU backward stall;
48GB and 12h walltime. Each job writes per-task aggregate metrics,
paired bootstrap, audit, coverage and progress; summary only when
all tasks finish. Upload ZIP of phase78_distribution_*/ and omit
patient-level data. The scripts are committed but untested on Dusky.
