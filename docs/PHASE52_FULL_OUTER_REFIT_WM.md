# Phase52 — fix low effective training size in longitudinal World Model CV

**Problem (Phase50):** 332 frozen PPMI Prodromal TRAIN BL/12m/24m
trajectories, 3-fold outer CV; each outer train has only 221–222
people, and reserving 20% inner development left just 177–178 people
in the weights used to evaluate heldout patients. Phase50's latent
WM MAE 2.91681 did not improve meaningfully over fixed BL/12m
50/50 smoothing MAE 2.90813 (paired MAE difference +0.00868;
95% bootstrap [-0.07738,+0.11848]).

## Why simply requesting more folds is not sufficient

| Outer folds | Approx outer TRAIN n | Approx inner fitting n | FINAL refit n |
|---|---:|---:|---:|
| Old Phase50 3 | 221–222 | 177–178 | **177–178 (no refit)** |
| Phase52 3 | 221–222 | 177–178 | 221–222 |
| **Phase52 5 (primary)** | **265–266** | 212–213 | **265–266** |
| Phase52 10 (sensitivity) | 298–299 | 238–239 | 298–299 |

All rows above are derived from the known 332 patient trajectories
and may differ by one person depending on exact fold assignments.
**More folds do NOT create new independent patients.**

## Method — nested epoch selection then full outer refit

For every outer fold, each ensemble member:

1. Fits a provisional model on 80% of the OUTER TRAIN patients.
2. Chooses best number of epochs based exclusively on the other
   20% of OUTER TRAIN patients (the inner-development split).
3. Discards that checkpoint, initializes a FRESH model, and trains
   for the fixed selected number of epochs on **all** outer-training
   patients. There is no further early stopping using outer-heldout.
4. Predicts the untouched outer-heldout patients only, combining
   three independently seeded model members.

Three patient-disjoint evaluation alternatives are supported:
`--folds 3` / `--folds 5` / `--folds 10`.
The exact choice of 5-fold primary is made before new results are
seen. Compare to historical last value, BL, fixed 50/50 average,
and linear extrapolation. Paired patient-bootstrap MAE contrast
vs fixed 50/50 is computed over OOF predictions.

New aggregate outputs:
- `fold_training_sizes.csv`: inner-fit/dev sizes and TRUE final
  refit sizes, plus selected epoch and inner dev MAE;
- `oof_metrics.csv`: MAE/RMSE/Bias for all baselines and latent WM;
- `fold_aggregate_mae.csv`: fold-level MAE for main contrast;
- `paired_smoothing_contrast.csv`: paired OOF bootstrap interval;
- `summary.json`: split provenance and limits.

## Run on offline Dusky CPU

No new Conda packages, network, GPU or model downloads.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase52_full_outer_refit_wm.py
"$PY" -m unittest tests.test_phase52_full_outer_refit_wm -v

# Optional brief functionality smoke:
PHASE52_FOLDS=5 PHASE52_MEMBERS=1 PHASE52_EPOCHS=3 \
  PHASE52_WIDTH=16 sbatch slurm/phase52_dusky_cpu_refit_wm.sbatch

# Primary full 5-fold, 3-member benchmark:
PHASE52_FOLDS=5 sbatch slurm/phase52_dusky_cpu_refit_wm.sbatch

# Ten-fold sensitivity run, only after verifying 5-fold:
PHASE52_FOLDS=10 sbatch slurm/phase52_dusky_cpu_refit_wm.sbatch
```

Aggregate output:
`/scratch/ylong66/pks_agent_outputs/phase52_<FOLDS>fold_<actualJobID>/`.

## Interpret correctly

Moving to 5 folds AND retraining on full outer TRAIN are two changes,
so comparing Phase50 old 3-fold metrics with Phase52 5-fold is
**not a controlled ablation**. The Phase52 optional 3-fold with full
refit provides a partial controlled contrast; exact train-size
ablation would use identical folds/hyperparameters/epochs.

The goal is a fairer estimate of how much a tiny longitudinal WM
can learn from a 332-patient cohort, NOT proof that it implements
Dreamer 4, gains from RSI, or captures intervention effects.

Only the original frozen TRAIN registry and targets are queried.
Original Phase44 TEST is consumed and remains unopened; VAL is
not used. All individual records, identifiers, scores, trajectories,
model predictions and checkpoint weights stay on the cluster.
