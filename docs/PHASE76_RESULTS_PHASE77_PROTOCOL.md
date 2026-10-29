# Phase77: Phase76 result assessment + cross-fitted clinical residual WM

Phase76 submitted report (phase76_gated.zip) contains 20 directories but
only eight full reports: arms 4..7 and 12..15, corresponding to outer
folds 1 and 3. Remaining 12 directories are empty; there is no Slurm
state in the uploaded archive. Completed reports' method metrics have
repeated matched baselines; deduplicate by (cohort,horizon,fold,model)
before computing patient-weighted results.

Exact two-fold patient-weighted MAE:
| method | Prodromal12 N219 | Prodromal24 N155 | PD12 N67 |
|--|--:|--:|--:|
| Motor HGB | .94772 | 1.17577 | 2.53540 |
| Motor CT-WM | .76386 | .98775 | 2.59550 |
| Early multimodal CT | .80865 | .99675 | 2.58577 |
| Gated multimodal CT | .77523 | .97368 | 2.56361 |
| Verified candidate | .70256 | .98350 | 2.59499 |

The learned gates stayed near their sigmoid(-3) initialization
(mean weight ~0.048), thus Phase76 does NOT demonstrate that a gate
learned to infer reliability. Other improvements may stem from
clinical dropout or differently trained models, and only TWO folds
are completed, so there is no confirmed five-fold result.

Verifier: 6 decisions on 2 folds, 3 accepted / 3 rejected.
Prodromal12 gated L1 revision accepted on both folds and produced
strong MAE gain, but Brier can deteriorate; no formal multi-testing
correction and development data repeatedly explored.

## Phase77 hypothesis and methodology

Rather than concatenating uncertain clinical modalities into the
trained Motor encoder, freeze a strong CT Motor backbone and learn
only the *OOF residual severity* associated with nonmotor tests.
Train correction labels from FIT-only 3fold inner OOF Motor
predictions, not in-sample motor predictions, to prevent the correction
from learning motor training optimism.

Four outer strategies (20 standalone jobs, five folds):
0. Motor-only CT WM plus motor-only and multimodal HGB baselines.
1. Early-fusion CT as a fair reference.
2. Cross-fitted residual model and intercept-only calibration control.
3. Same residual proposal with DEV gamma selection from
   [.25,.5,1.0] and independent AUDIT accept/reject or rollback.

Every job exports repeated, same-fold Motor WM and HGB references
plus selected strategy, along with patient-paired bootstrap contrasts.
This is a controlled statistical model revision, NOT autonomous
recursive RSI or a causal intervention model. The correction changes
burden only, not the binary risk prediction; Brier veto therefore
is deliberately noninformative and cannot demonstrate risk safety.
Compare candidate to bias_only to test nonmotor-specific uplift.

Patient-level disjoint partitions are frozen. All optional values
come from true recorded baseline clinical test rows. No image data,
no raw PPMI export, no Qwen CPU load. Current cohort still frozen
original Prodromal TRAIN plus separately defined PD; no fresh final
independent TEST. The project's hard per-job walltime limit is 12h;
aggregate report files are saved after each task.

## Submit 20 independent non-array CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE77_ARM_ID="$arm" slurm/phase77_dusky_cpu.sbatch
done
```

8 CPU, 48 GB, 12-hour cap each. No Slurm array, no GPU and no
separate smoke job. Per task aggregate results include
method_metrics.csv, paired_bootstrap.csv, coverage.csv, audit.json,
progress.json and summary.json (latter only on completed execution).
If a directory is empty, check Slurm queue and sacct rather than
assigning a model score.
