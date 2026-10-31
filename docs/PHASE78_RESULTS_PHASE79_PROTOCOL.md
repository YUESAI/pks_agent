# Phase78 completed results and Phase79 dual-functional ordinal Motor WM

Source ZIP: phase78_reports.zip uploaded 2026-10-31. All 20 ordinary
CPU jobs complete, each with three clinical outcomes; reports:
285 method metric rows, 390 comparison rows, 15 verifier decisions.
Repeated baselines across arms are deduplicated by
(cohort,horizon,outer_fold,method) and fold MAE weighted by patient count.

| Model (mean unless median noted) | Prodromal12 N584 MAE | Prodromal24 N378 MAE | PD12 N156 MAE |
|---|---:|---:|---:|
| Motor HGB | 0.99633 | 1.14285 | 2.48944 |
| Motor CT direct | 0.87304 | 0.98166 | 2.58728 |
| Hurdle mean | 0.99245 | 1.14224 | 2.59252 |
| Hurdle median | 0.77055 | **0.84392** | 2.55769 |
| Categorical mean | 0.99759 | 1.12375 | 2.62103 |
| Categorical median | **0.75856** | 0.87566 | **2.48077** |
| Verified decision | 0.86874 | 0.96164 | 2.50091 |

Mean-vs-median tradeoff is real: Prodromal12 CT-direct RMSE
1.46014 vs categorical median 1.59760; Prodromal24 CT-direct
RMSE 1.72611 vs hurdle median 1.84489. The underlying *distribution*
mean may have better RMSE than median, so scoring must be applied to
the correct decision functional. The original Phase78 combined
mean-severity, median-severity and risk into a single (MAE, RMSE,
Brier) tuple, which penalizes a good median for its expected higher
squared error.

Verifier accepted 5/15 proposals: Prodromal12 0/5,
Prodromal24 1/5, PD12 4/5. Prodromal MAE-improving
median proposals were commonly rejected because their RMSE as a
*median* deteriorated beyond +0.05, sometimes also Brier. This is
a decision-theoretic mismatch, not proof the posterior is well
calibrated or useful.

## Phase79 corrected evaluation design

Five patient-disjoint outer folds, four ordinary CPU arms (20 jobs):
0. Motor HGB and old CT direct references.
1. Baseline Phase78 categorical and hurdle posterior (0..24).
2. Add a new ordinal probability model using strictly proper
   categorical CE + ranked probability score (RPS), with a small
   auxiliary squared-error term for the distribution mean.
3. On DEV, choose temperature from {0.75,1,1.25,1.5,2}
   to minimize RPS; compare exactly once on independent AUDIT
   against the unchanged ordinal distribution. Reject if AUDIT
   RPS is not improved, Brier rises > .005, mean-RMSE rises
   > .05 or median-MAE rises > .03. Report rollback.

Each distribution yields:
* MAE of conditional *median*;
* RMSE of conditional *mean*;
* Brier for p(Y>0), coherent with P(Y=0);
* RPS normalized by 24, log score, 80% interval coverage/width;
* AUROC when calculable.

Paired bootstrap compares corresponding losses for each patient;
no extra output of patient identifiers, scores or predictions.
All original future targets and histories follow established
strict PPMI rules; auxiliary next-visit labels come exclusively
from FIT person histories, not DEV/AUDIT/evaluation people.

These remain repeatedly explored frozen Prodromal TRAIN results,
not independent clinical confirmation. This experiment tests
probabilistic forecasts, not a genuine agent RSI or Qwen doctor.
No GPU.

## Submit 20 independent ordinary CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE79_ARM_ID="$arm" slurm/phase79_dusky_cpu.sbatch
done
```

Each job: partition base, account def-hup-ab_cpu, 8
allocated cores but conservative PyTorch/BLAS 1 thread,
48 GB RAM, strict maximum 12h. No array, no smoke job.
Complete outputs: method_metrics.csv, paired_bootstrap.csv,
coverage.csv, audit.json, progress.json, summary.json.
