# Phase84: switch from fivefold exploration to reproducible WM development

Scientific goals: stop repeated fivefold agent prompt tweaks; develop
a plausible *structural* advance over direct total-burden categorical
forecasting. The positive hypothesis is that 6 real future Motor item
distributions provide supervision and their convolution produces a
coherent posterior for total future 6-site burden (0..24), which may
learn better with limited data and offer more interpretable outputs.

A single **patient-global 80% FIT / 20% DEV** hash split is used
across all horizon tasks; it is deterministic and disjoint by
patient ID, independent of Python set iteration. The DEV 20%
is a development set and WILL BE reused for model selection, so is
not a scientific holdout or final test. Never touch the previously
consumed original final TEST.

Three standalone ordinary CPU jobs:
0. Phase78 25-class categorical baseline.
1. New six-item categorical (5 scores per item), convolved via
   differentiable probability convolution into an exactly
   normalized 25-category burden posterior; optimize true
   future item CE plus total-burden NLL.
2. Same architecture plus actual FIT-person adjacent Motor
   visit next-item auxiliary supervision.

For each identical DEV patient evaluate exact probability
quality (RPS, NLL, Brier), median MAE, mean RMSE,
80% prediction interval coverage/width. The underlying
site-independent convolution is an explicit approximation:
cross-site conditional dependence may be important, so do
not claim a full causal state-transition WM from it.

Source fixes committed in the same branch:
- Phase62 cohort iteration sorted (patient registry, visits,
  patients) instead of hash-random set iteration.
- Phase72 auxiliary visit sequence sorted by patient ID.
- Phase73 outer heldout ordering stable across processes.
- Phase83 conditional emission model for an inner split now
  trains on ONLY its inner-train subjects, eliminating
  prior crossfit indirect dependence through held validation
  labels. The inner-train emission priors are IN-SAMPLE and
  should NOT be described as strict nested OOF calibrations.
- New tests for convolution normalization, 0/24 support,
  differentiable site score likelihood, and split stability.

Note: Phase62 ordering change means old phases' matching fold
model predictions may shift. For apples-to-apples WM model
comparison, rerun arm0 categorical baseline in the SAME new
Phase84 protocol. Do not cross-compare raw MAE from old phase
archives without controlling model protocol and split.

Run tests on cluster login (no Slurm smoke job):
```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python -m unittest discover -s tests -p test_phase84_wm.py -v
```

Submit the **three** candidates in parallel (not a 20-job sweep):
```bash
for arm in 0 1 2; do
  sbatch --export=ALL,PHASE84_ARM_ID="$arm" slurm/phase84_dusky_cpu.sbatch
done
```
8 CPU cores allocated, but verified fast single-thread
PyTorch path with MKLDNN disabled, 48GB, 12h hard limit,
no GPU, no Slurm arrays. Results under
`/scratch/ylong66/pks_agent_outputs/phase84_wm_*/`.
When candidates look promising, separately plan robust
baselines/ablations and *truly independent* confirmation.
