# Phase48 — stronger probabilistic observation WM (controlled upgrade)

**Motivation.** Phase47's 100-patient q=4/8/12/16 TRAIN sweep does
not justify a claim that no WM helps Qwen. Its World Model was a
categorical Naive Bayes estimator over a binary hidden-six outcome,
with conditionally independent observed Motor items. That is a
mechanistic starter, not an advanced clinical World Model.

## Correct interpretation of Phase47 report

The provided Phase47 source ZIP contained `cost_curve_metrics.csv`,
`question_agreement_by_budget.csv` and `summary.json` for 100
TRAIN V04 subjects, 29 hidden-positive. At q=4/8/12/16:

- Qwen-only AUROC: **0.751/0.731/0.726/0.743**.
- Qwen + Naive Bayes WM AUROC: **0.649/0.680/0.710/0.718**.
- Dynamic naive WM AUROC: **0.646/0.687/0.701/0.718**.
- At q=16, all three Qwen+WM / static WM / dynamic WM policies
  selected the same 20 people, but not necessarily in the same order.
- All metrics come from an exploratory, repeatedly visited TRAIN
  partition, not an independent test of these architectures.

## New model and method limitations

`scripts/phase48_masked_transformer_wm.py` implements an **ensemble
of small masked-item Transformers**, trained on patient-held-out
PPMI Prodromal TRAIN complete V04 measured observations. Each item
has an identity embedding; scores 0..4 and unobserved status are
different token values. Random query masks are sampled across
varying budgets; the model predicts the hidden-six-positive outcome
and, conditioned on it, the categorical distribution of each masked
item. An approximate mixture over outcome classes gives
`p(X_j | X_observed)`, while the class head estimates
`p(y | X_observed)`.

A requested measurement never comes from the model: only the
PatientEvidenceService returns recorded PPMI scores. The neural WM
produces distributions and estimated conditional mutual information.
Its posterior and conditional item heads are trained together but
are NOT guaranteed to satisfy exact global joint-distribution
consistency. It is not a causal simulator.

**This is still an observation WM, not an advanced longitudinal or
action-conditioned Parkinson's disease progression WM.** A future
temporal WM must explicitly model visit interval, treatment state,
missingness/modality and distribution of future actual measurements.

## Comparison before exposing neural WM to Qwen

Hold fixed the same hash-disjoint TRAIN fit and 100 heldout patients.
Train three transformer ensemble members, with 20% nested TRAIN dev
early stopping. Compare Naive Bayes and masked Transformer at
unbiased observed subsets of q=4/8/12/16 with:
- heldout masked-item conditional negative log-likelihood (primary
  check for observation modelling),
- hidden-positive Brier and AUROC,
- strict 20% top-k captured-positive recall and hidden burden.

**Only after showing reasonable calibration and item predictive
performance should Doctor agents receive neural WM advice.**
A complex neural model does not automatically improve small-data
performance, and model complexity alone is not a scientific result.

## Run on Dusky CPU first (no downloads or GPU)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit

/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python \
  -m unittest tests.test_phase48_masked_transformer_wm -v

# Optional fast smoke: 1 small ensemble member, 5 epochs, 40 patients
PHASE48_ENSEMBLE=1 PHASE48_EPOCHS=5 PHASE48_MAX_PATIENTS=40 \
  PHASE48_WIDTH=32 PHASE48_DEPTH=1 \
  sbatch slurm/phase48_dusky_cpu_wm.sbatch

# Full TRAIN-only small-model comparison:
sbatch slurm/phase48_dusky_cpu_wm.sbatch
```

The job uses Python and PyTorch from
`/scratch/ylong66/conda/envs/pks_qwen46_cu118` and stores its
aggregate report under
`/scratch/ylong66/pks_agent_outputs/phase48_cpu_<JOBID>/`.
Both login-node downloads and GPU resources are **unnecessary**
for this step. Upload only the aggregate `wm_quality.csv`,
`ensemble_training.csv` and `summary.json`.

All outcomes remain in already explored **TRAIN**. Original
Phase44 TEST was consumed and is not accessed. The work is a
controlled WM capability comparison, not independent confirmatory
evidence and not an implemented LLM-RSI system.
