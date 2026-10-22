# Phase58 — ten methods, one matched Train-only R01-to-V04 study

**Status:** implemented; actual Dusky execution pending. All
data stay on the cluster, and *only aggregate tables* are exported.

## Why this is the correct next experiment

Phase56 had 197 strict genuine remote R01 -> later complete V04
pairs, 57 later hidden-six-positive; the 20% onsite referral
quota chose exactly 39 patients. R01 visible sum captured
22/57, tuned Markov 23/57, Phase54 temporal WM 19/57.
These point differences cannot be tested with separately
uploaded aggregate output files. Also, a fast Phase54 job
does not imply the model is advanced: it was a small random-
initialized, recurrent numerical latent WM trained on 658
patients, not pretrained Dreamer4 or multimodal Qwen.

## Ten predefined matching strategies

1. `visible27_sum`: unweighted sum of actual R01 visible items;
2. `visible27_nonzero`: number of real R01 visible items >0;
3. `visible27_ridge`: fit-only ridge regression on 27 item scores
   predicting later V04 six-item nonzero (L2=20; fixed);
4. `visible27_logistic`: fixed-L2 logistic regression, 27 observed
   items, training cohort labels only;
5. `visible27_naive_bayes`: item-by-label categorical probabilities
   fitted from outer-training *real remote-future pairs*;
6. `markov_tuned`: Phase56 time-gap/Dirichlet Markov selected using
   inner development and fitted to full outer TRAIN;
7. `latent_wm_64`: masked recurrent time-gap WM, width 64,
   patient-level inner epoch selection and full outer fit;
8. `latent_wm_128`: same architecture, width 128, same fold rule;
9. `wm_markov_blend`: 0.5 WM64 risk + 0.5 Markov risk;
10. `visible_wm_blend`: 0.5 WM64 risk + 0.5 fitted
    visible27 logistic probability.

Fixed mixtures are **not optimized on outer-heldout**. Each model
uses identical five patient-disjoint folds from Phase56's
`split_patients` function and the same strict `find_pairs`
and frozen TRAIN manifest. Other than inner epoch selection for WM
and prior/gap selection for Markov, hyperparameters are fixed in
advance. Never access original VAL or consumed Phase44 TEST.

**The LLM Doctor/Qwen and RSI are not used in Phase58.** This
benchmark tests whether a stronger risk policy actually helps
the limited onsite quota before adding costly agent planning.
The two WM sizes are *architecture capacity ablations*, not
2026 foundation-world-model pretrained checkpoints.

## Primary objective and statistical controls

- Exactly 20% of the strict matched future-V04 cohort are selected:
  `k=floor(0.2*n+0.5)`.
- Same deterministic Phase56 hash tie-break for all rankings.
- Primary descriptive criterion: future six-positive recall@k.
- Report AUROC and differences from `visible27_sum`.
- Paired 500-resample **patient** bootstrap recomputes all
  rankings and quota on identical resampled patient indices
  (not separately sampled runs).
- Include WM vs Markov, blend vs Markov, WM128 vs WM64.
- Report selected patient-list overlap (Jaccard; small cell
  counts suppressed), and per-fold training-time seconds,
  chosen epoch counts and WM parameter counts.
- No model can use future V04 item scores in an inference feature.
  Neural models may receive the retrospectively known future
  target visit **month**, so this is not a prospective timing model.
- Bootstrap keeps fitted out-of-fold predictions fixed;
  it does not quantify variability from retraining.
- No patient-level data or selected patient IDs leave RAM.

## Dusky CPU commands; no internet, GPU or download

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase58_paired_metrics.py \
  scripts/phase58_ten_strategy_sweep.py
"$PY" -m unittest tests.test_phase58_ten_strategy_sweep -v

# Fast test of all ten policies / models (5-fold, 2 max epochs):
PHASE58_EPOCHS=2 PHASE58_BOOTSTRAP=100 \
  sbatch slurm/phase58_dusky_ten_strategy.sbatch

# Full 10-policy study, 5-fold, up to 25 epochs, 500 bootstrap draws:
sbatch slurm/phase58_dusky_ten_strategy.sbatch

# Do not start the full job until smoke job exits successfully.
```

Each Slurm job first runs the synthetic regression tests; the
job's exit status and logs must be checked. Results:
`/scratch/ylong66/pks_agent_outputs/phase58_ten_<JOBID>/`.

Outputs:
- `ten_strategy_metrics.csv` — 10 rows of exact-k capture, recall,
  AUROC, baseline random expectation;
- `paired_bootstrap.csv` — model-vs-visible27 and prespecified
  other paired 95% intervals;
- `topk_overlap.csv` — selected top-k overlap and Jaccard;
- `fold_training_compute.csv` — real model sizes, selected epochs,
  fit/heldout counts and elapsed seconds;
- `summary.json` — study flags, runtime and test restrictions.

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -dt phase58_ten_*
zip -r phase58_ten_report.zip phase58_ten_<ACTUAL_JOB_ID>/
```

### Interpretation gate

These 197 repeatedly explored TRAIN people, the same policy
and budget as previous stages, are *not new confirmatory test*.
Ten strategies introduce substantial multiplicity; their
exploratory best AUROC/Recall cannot be treated as a publication
claim. For final paper: freeze a clinically motivated primary
method and compare it on **genuinely new independent subjects**
with an uncertainty analysis.

If a fixed blend/WM cannot beat visible sum and learned Markov
under equal quota, prioritize more informative representations
and long-horizon prediction, not GPU scaling as a substitute
for signal.
