# Phase54 — masked irregular-time longitudinal Motor World Model (implementation)

**Status:** code, synthetic regression tests and offline CPU Slurm job committed.
Cluster PPMI performance is *not yet known*. This is a new temporal
observation WM, **not** a complete Dreamer 4 reproduction and **not**
a medication-intervention simulator.

## Data definition

- Read only frozen PPMI Prodromal TRAIN patient identities from the
  Phase23 SQLite manifest and original local Motor source CSV.
- Same Phase53 BL anchor, R01 4–8 month window, V04 9–15 month window,
  UNSPECIFIED medication state, duplicate month conflicts and strict
  complete-33 total verification.
- One patient has >=2 valid baseline-anchored encounters; 33 actual
  Motor item ratings, an item validity status, visit month and event.
- Status 0 = ordinary missing, 1 = actual score 0..4,
  2 = unable to rate (101/UR). Placeholder zero for status 0/2
  is never accepted as an actual measurement; loss ignores these.
- Patient IDs and row-level arrays remain *in RAM on the cluster*;
  reports export only aggregate counts and evaluation metrics.

Phase53 observed 2,702 usable complete/partial episodes across
1,312 frozen TRAIN people; 658 had baseline + >=1 visit,
426 >=3, 272 >=4, 98 >=5. **Phase54 counts may be different**
because the individual-level data construction is its own strict
eligibility procedure. Report discrepancies rather than forcing
them to match. Phase53's 197 strict R01-to-V04 matches vs earlier
locked 199 must also be independently reconciled in any future
cohort comparison.

## World Model

`scripts/phase54_temporal_wm.py` has:
- item identity embedding;
- measured score (0..4), ordinary-missing and 101/UR token embeddings;
- time-interval-conditioned recurrent latent transition;
- observed-visit GRU belief update;
- 33 categorical next-visit heads over score 0..4.

The *next visit* is predicted **before** that visit's actual
item scores/statuses are incorporated into the belief.
A regression test mutates future scores and verifies that their
own predictions are invariant. Subsequent predictions may
legitimately depend on those newly measured scores.

**Important prospective caveat:** this initial method uses the
actual heldout next visit's calendar month as a specified time
gap, but no actual next-visit value, mask or event. Therefore it
is a **retrospective time-conditioned next-visit prediction**,
not a prospective forecast of when the visit occurs.

## Evaluation

`scripts/phase54_evaluate.py` performs patient-disjoint 5-fold
OUTER TRAIN CV. An inner 15% subset selects number of epochs
(2, 5, 10, 25 by default); the final model restarts from scratch
and refits on **all** outer TRAIN patients for the selected count.
No early stopping from an outer-heldout patient.

Comparison uses the *identical measured next-visit score targets*
for all methods:
1. neural temporal masked WM;
2. last-known-item persistence with fixed confidence .85
   (falls back to fit-TRAIN item priors for unseen items);
3. fit-TRAIN smoothed categorical item marginals.

Primary exploratory endpoint: **negative log-likelihood averaged
first across actually scored future items per patient, then equally
over heldout TRAIN patients**. Pairwise patient-bootstrap 95% CI
compares neural WM minus persistence. A fixed .85 persistence
assumption is an initial baseline; tuning/strengthening it needs
inner TRAIN-only selection before any comparative claim.

This is an *observed-future-items* endpoint: selection/missingness
can bias interpretation; masks are not treated as targets or evidence
of actual zero scores. Adjacent visit pairs are never randomly
split across patients. Original Phase44 Test has been consumed
and is not used.

## Run using installed offline Conda (Dusky CPU)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python

"$PY" -m py_compile scripts/phase54_masked_sequences.py \
  scripts/phase54_temporal_wm.py scripts/phase54_evaluate.py
"$PY" -m unittest tests.test_phase54_masked_temporal_wm -v

# Small neural capacity, short run: tests run automatically in job too
PHASE54_WIDTH=16 PHASE54_EPOCHS=2 \
  sbatch slurm/phase54_dusky_cpu_masked_wm.sbatch

# Full exploratory TRAIN-only run:
sbatch slurm/phase54_dusky_cpu_masked_wm.sbatch
```

Output:
`/scratch/ylong66/pks_agent_outputs/phase54_masked_<ACTUAL_JOBID>/`

Expected aggregate outputs:
`patient_oof_item_nll.csv`,
`paired_nll_contrast.csv`,
`fold_training_sizes.csv`, `summary.json`.

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -d phase54_masked_*
zip -r phase54_aggregate_report.zip phase54_masked_<ACTUAL_JOBID>/
```

No network, new packages, Hugging Face weights, or GPU needed.
The code has synthetic tests; actual cluster success requires a
real Slurm run and cannot be guaranteed before it executes.

## Beyond Phase54

The next research-quality upgrade would include patient-level
distributional calibration, proper scoring decomposition by
mask/provenance/horizon, stronger learned persistence baselines,
multi-step autonomous rollouts, more PPMI modalities and a
verifier-gated connection to offline Qwen Doctor. Treat this
prototype as an **initial genuine temporal prediction test**,
not evidence that advanced clinical WM outperforms simple rules.
