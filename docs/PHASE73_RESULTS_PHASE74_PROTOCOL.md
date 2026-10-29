# Phase73 actual 5-fold results + Phase74 matched WM mechanism tests

Report zip phase73_matched.zip received Oct 29 2026.
Five complete Qwen+verified jobs: arm3/7/11/15/19.
All five jobs included within-job **matched** Ridge, single CT-WM,
ensemble, and Qwen-selected single/ensemble/verified outcomes,
so the cross-validation comparisons are available on all 5 folds.
Total 90 model rows, 60 paired-bootstrap contrasts, 15 verifier
decisions. 5/15 candidate switches accepted (Prodromal12=1,
Prodromal24=4, PD12=0). All Qwen requests were valid (2449 total);
reported run time 1.18-3.13 hours per job.

Patient-weighted fivefold MAE:
| method | Prodromal12 N584 | Prodromal24 N378 | PD12 N156 |
|---|---:|---:|---:|
| Ridge, fixed evidence | .99204 | 1.14974 | 2.68581 |
| Single CT WM, fixed | 1.04172 | 1.17294 | 2.65808 |
| Ensemble CT WM, fixed | 1.04333 | 1.16669 | 2.67543 |
| Single CT WM, Qwen evidence | 1.04351 | 1.17528 | 2.66470 |
| Ensemble CT WM, Qwen evidence | 1.04548 | 1.16800 | 2.67784 |
| Verified CT WM, Qwen evidence | 1.03575 | 1.16433 | 2.66470 |

On identical Qwen evidence, selected checkpoint improvement over
the single model was 0.00777 MAE at Prodromal12 and 0.01096 at
Prodromal24; still worse than the fixed-evidence Ridge reference.
Qwen evidence choice itself made ensemble MAE slightly worse than
fixed selection. These repeated TRAIN analyses remain exploratory.

Phase73 reports show 24-month subset has 356 of 378 evaluated
patients with 2+ historical visits, 226 with 3+ histories. Current
WM cross-entropy objective is NOT aligned to primary six-rating SUM
MAE even though original seven clinical outcomes are available.

Phase74 next hypothesis: shared fully recorded visible27 longitudinal
features, nonmotor baseline values and missingness, same person splits
and identical evidence for 4 methods:
0 hand-engineered longitudinal slopes/last observation Ridge;
1 same features histogram gradient boosting;
2 CT direct six-item SUM prediction plus six-item categorical,
  binary outcome, and actual FIT-only next-Motor auxiliary task;
3 same neural model with different MAE-targeted loss, allowed
  revision only when DEV improves and independent AUDIT also
  improves without Brier deterioration.

20 non-array CPU jobs = five outer folds x four models. Each job
also computes internal matched Ridge and HGB comparisons. Qwen is
intentionally paused: first establish if temporal neural WM improves
against strong longitudinal tabular baselines. Only FIT patient
true historical visits contribute to self-supervised auxiliary
training. Never use heldout or future target scores as model inputs.

**Important**: All per-task results are saved before proceeding
to the next task. If a Slurm task is killed at 12h, completed task
reports remain along with progress.json. An incomplete job has no
final summary.json; do not treat partial reports as full success.
Exact resumption of neural optimizer state is not yet implemented,
so 'partial result persistence' is NOT training checkpoint resume.

## Submit independent jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE74_ARM_ID="$arm" slurm/phase74_dusky_cpu.sbatch
done
```

Each independent CPU job: 8 cores, 48GB, 12 hours maximum.
No array jobs, no GPUs, no separate smoke-test jobs.
Export phase74_direct_* complete aggregate report folders only:
method_metrics.csv, paired_bootstrap.csv, coverage.csv, audit.json,
progress.json, summary.json if fully completed.
