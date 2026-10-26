# Phase69 actual results and Phase70 horizon-fused WM experiment

Uploaded report: phase69_temporal.zip, 20 complete config folders, 60
fold metric lines, and all verifier reports. No Slurm exit codes.

| Cohort/task | N | Greedy MAE | Plain Qwen | Qwen+WM | Qwen+verified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prodromal 12m | 584 | 1.051917 | 1.048838 | 1.048828 | 1.047184 |
| Prodromal 24m | 378 | 1.210427 | **1.129181** | 1.194448 | 1.189423 |
| PD 12m | 156 | **2.573894** | 2.637672 | 2.590271 | 2.693924 |

15 Qwen-run verifier candidate audit records: four accepted (all PD12),
11 rejected. Plain Qwen invalid requests: 265/1168 Prodromal12;
191/756 Prodromal24. Guided WM: 1/1168 and 0/756.

Major Phase69 limitation: the FIT-only time-gap transition head trained
on genuine consecutive motor encounters, but final risk estimates and
question utilities did NOT use it. This prevents attributing any
decision gain to temporal planning.

Phase70 directly uses the transition head in a prespecified fixed
0.35 mixture with the ordinary masked WM horizon-specific prediction.
For six missing onsite items it calculates the future-score expectation
and model-implied probability that at least one score is nonzero.
It uses task horizon 12/24; when R01 evidence exists, an approximate
six-month offset is used. That offset is an approximation, not a
record-specific visit time; no fabricated value enters the PatientTool.
When temporal training transitions are absent, the original
outcome head is used. Temporal outputs are *observational* and their
six-item independence approximation may be miscalibrated.

20 independent ordinary sbatch jobs: five outer patient folds and
four Qwen strategies. Within a fold the training people, trained WM
and Qwen budget are the same:
 0 Plain Qwen, static WM risk (NO WM action hints);
 1 Qwen + static masked WM hints;
 2 Qwen + temporally fused WM hints;
 3 Qwen + temporally fused WM + true-Qwen DEV/AUDIT policy revision.
Each job uses two actual data requests per evaluated person. These
are internal exploratory studies of originally frozen TRAIN.
No array, no extra smoke job, no independent test claim.
All four strategies perform offline Qwen on CPU and may approach the
12-hour base-partition walltime limit.

## Run

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE70_ARM_ID="$arm" slurm/phase70_dusky_cpu.sbatch
done
```

All jobs request 8 CPU cores, 48GB RAM and 12h. Results:
fold_metrics.csv, cohort_coverage.csv, verifier_trace.json, summary.json.
If any job fails, report its Slurm stderr/log; don't treat missing
reports as zero-valued outcomes.
