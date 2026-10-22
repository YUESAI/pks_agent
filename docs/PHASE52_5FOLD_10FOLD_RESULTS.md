# Phase52 results: 5-fold versus 10-fold full-outer refit (2026-10-20)

**Evidence:** user-provided aggregate archives `phase52_5fold.zip` and
`phase52_10fold.zip`, jobs `330379` and `330381` respectively.
Each contains `oof_metrics.csv`, `paired_smoothing_contrast.csv`,
`fold_training_sizes.csv`, `fold_aggregate_mae.csv`, `summary.json`.

## Design and actually fitted patient counts

332 independent PPMI Prodromal TRAIN BL/12m/24m trajectories, three
ensemble members per outer fold. Epochs were selected using a disjoint
inner-development subset, then each member was **reinitialized and
refit on the full outer-training patients**, with no outer-heldout
labels used during fitting or epoch selection.

| Run | outer folds | inner fit for epoch selection | inner dev | FINAL fitted per member | outer heldout |
|---|---:|---:|---:|---:|---:|
| Phase50 original | 3 | 177–178 | 44 | **177–178 (no full refit)** | 110–111 |
| Phase52 job 330379 | 5 | 212–213 | 53 | **265–266** | 66–67 |
| Phase52 job 330381 | 10 | 238–239 | 60 | **298–299** | 33–34 |

The Phase52 per-model training-volume fix succeeded. Selected inner
epochs in 5-fold were 12–27, and in 10-fold 12–36, both within the
60 epoch maximum. The patient count remains 332; more folds do not
create independent training trajectories.

## Heldout OOF scores (lower is better)

| Model | MAE | RMSE | Signed bias |
|---|---:|---:|---:|
| fixed 50/50 BL + observed 12m | **2.90813** | **4.79858** | -0.84488 |
| Phase50 old 3-fold WM | 2.91681 | 4.79599 | -0.89289 |
| Phase52 5-fold latent WM | 2.94740 | 4.87617 | -1.02732 |
| Phase52 10-fold latent WM | 2.93544 | 4.87214 | -1.00831 |

Paired OOF contrast `WM absolute error - fixed_50_50 absolute error`:

| Run | delta MAE | patient bootstrap 95% interval | resamples |
|---|---:|---|---:|
| Phase52 5-fold | **+0.03927** | [-0.06518, +0.14351] | 500 |
| Phase52 10-fold | **+0.02731** | [-0.06481, +0.12404] | 500 |

Both intervals include zero. The 10-fold WM's MAE is 0.01196
below the 5-fold WM, but is 0.02731 above the fixed 50/50
historical reference. Do **not** claim superiority, infer
significance, or interpret this as a causal training-size ablation.

## Fold heterogeneity

**5-fold**: WM beats fixed 50/50 in 2/5 folds, loses in 3/5.
Largest fold-level disadvantage +0.21031 (fold 4);
largest advantage -0.10267 (fold 2), calculated as WM minus baseline
fold MAE.

**10-fold**: WM beats fixed 50/50 in 6/10 folds, loses in 4/10.
Largest disadvantage +0.37215 (fold 1) and largest advantage
-0.18926 (fold 9). These small folds are not independent
hypothesis tests, and fold numbering has no clinical meaning.

The exact same 332 original Train patients were included in both
OOF evaluations, but prediction models differ due to fold
allocation and fitting volume. The per-person paired difference
between the **two WM CV schemes** cannot be reconstructed from
aggregate files, and their delta-MAE confidence intervals are
for comparisons to smoothing, not for direct 5-versus-10
statistical inference.

## Decision / research pivot

**Stop treating additional CV folds as a path to a stronger WM.**
The bottleneck is representation and data: Phase50/52 train from only
three scalar Motor totals (BL and 12m observed; 24m supervision)
per patient, with no many-visit multimodal longitudinal sequences or
recorded time-aligned intervention transitions.

The modern WM + LLM Doctor + verifier-guided RSI hypothesis is not
tested by these experiments. The next priority is Phase51's strict
TRAIN-only multi-visit availability audit, followed by construction
of real irregular-time multi-measurement trajectories. Then develop
action-conditioned observation/belief updates and calibrated
temporal dynamics before exposing the learned WM to Doctor.

If later testing uses a truly independent newly acquired external
cohort, predefine the endpoint and policy; the original Phase44
TEST was consumed. Phase52 metrics are exploratory results on
repeatedly inspected TRAIN data and should not be reported as
independent confirmatory performance.

**Privacy:** the cited ZIPs contain only aggregate statistics;
no patient-level data are copied into the repository.
