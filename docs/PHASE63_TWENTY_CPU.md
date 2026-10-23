# Phase63: exactly 20 formal CPU array jobs (no separate smoke run)

Branch: feat/phase62-ppmi-cohort-expansion, reused IN PLACE.
Same frozen PPMI data safety boundary as Phase62.

20 deterministic jobs = five feature panels x four models. Each job
evaluates all three tasks using exactly the same five patient-disjoint
hash folds: Prodromal 12m (584 phase62 participants); Prodromal 24m
(378); PD 6m (295). Counts are references from completed Phase62,
not guaranteed Phase63 results.

Panel index: 0 visible27_scores, 1 visible27_mask,
2 full33_scores, 3 full33_mask, 4 full33_r01.
Estimator index: 0 Ridge, 1 ElasticNet, 2 RandomForest, 3 HistogramGBM.
ID=4*panel_index+estimator_index, 0..19.
The full33_r01 representation includes actual R01 observations ONLY
for 12/24-month prediction. For PD 6m it uses BL full33_mask only
because a 4..8m R01 may occur AFTER the 6m target. Do not interpret
those PD6 models as having additional longitudinal inputs.

Each job ALSO fits two in-job references (visible27_scores Ridge and
full33_mask Ridge) on the SAME OUTER FOLDS for paired comparison.
Preprocessing computed within training fold. Missing score entries are
not real zeros; when numerical placeholder 0 is used for mask panels,
the status 0/1/2 accompanies it. Only BL or pre-target observed
R01 input; no future scores, visit data or label-derived covariates.
No per-person data/transcripts exported. Strict eligible follow-ups
and previously frozen Prodromal TRAIN only. PD is separate cohort.

The experiment isolates additional information gain versus model
family gains. These are exploratory post-hoc 20x3 experiments on
previously explored patients, not a new external test. Paired
bootstrap conditions on fitted models; no family-wise correction.
At fixed 20% quota, small captured-positive cells are suppressed.
Primary outcome: next-visit six-item score sum MAE. Secondary:
RMSE, AUROC, uncalibrated binary Brier and recall.

## Submit all twenty simultaneously (Slurm caps concurrency at 20)

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
sbatch --array=0-19%20 slurm/phase63_dusky_twenty_cpu.sbatch
```

Single array submission schedules 20 distinct CPU jobs. Allocation
depends on scheduler capacity; not necessarily starts simultaneously.
Each task requests 4 CPU and 16G RAM (up to 80 CPU cores + 320G if
all 20 start together). No GPU. The 8-hour limit is a cap not a
runtime prediction.

```bash
squeue -u "$USER"
sacct -j <ARRAY_ID> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS
ls /scratch/ylong66/pks_agent_outputs/phase63_arm_*
tail -n 50 slurm-phase63-<ARRAY_ID>_0.log
cd /scratch/ylong66/pks_agent_outputs
zip -r phase63_twenty_reports.zip phase63_arm_*/
```

Files per task: arm_metrics.csv, paired_reference_contrasts.csv,
task_capacity.csv and summary.json. Export only aggregate report
directory, never PPMI raw tables, frozen SQLite manifest, model
matrices or per-person predictions.

Scientific next gate: bring together all 20 task reports, use
prespecified MAE and information-matched full33_mask vs
full33_scores / visible27_mask comparisons, then pursue a true
longitudinal WM and independent patients if warranted.
