# Phase62: Expand strict PPMI cohorts before further RSI tuning

Goals:
1. Read complete local PPMI cohort, baseline diagnosis and MDS-UPDRS Part III data;
2. Count data eligible across **PD**, **HC**, frozen **Prodromal TRAIN**;
3. Keep 101/UR and missing masks, separate medication states, quarantine duplicates;
4. For 6, 12 and 24 month retrospective visits, run patient-disjoint 5-fold
   CPU future site hands-on six-item burden regression and secondary binary ranking.

## Correct scientific framing

No prospective causal clinical intervention or real question-request actions.
BL actually measured 33-item motor data/masks are input. Future six
hands-on item values are *labels only*, selected using horizon windows
6±2, 12±3, 24±4 months. Only verified complete 33-item future
measurements with internally consistent total can supply labels.
Future event, date, score, diagnosis and availability do not enter model
features. Study window and state are chosen prior to model fitting.
Never interpret 'UNSPECIFIED' as medication OFF.

**Data fence:** frozen Prodromal original VAL/TEST patients are not read
by the new patient registry or any exported cohort label/motor analysis.
PD and HC enter newly formed development tasks only, not an independent
external confirmation cohort. Cross-cohort diagnosis is never used as
a predictor or 'Doctor tool'. Raw patients stay on the cluster.
No full dataset download required beyond locally present files.

All methods are CPU-only:
- train-fold target mean
- BL remotely visible 27-item sum + ridge regression
- BL 33 item scores plus item states (missing/UR masked) + ridge
- same masked features + regularized random forest

Model search is not performed on outer folds. 5-fold OOF applies
separately to each cohort/horizon; no person crosses folds. Strict
minimum 40 people and 10 from each binary class; unavailable tasks
are marked SKIPPED, not fabricated. Paired bootstrap of masked ridge
minus visible-sum ridge MAE is conditional on trained models.

The secondary nonzero risk is an **uncalibrated** monotone transform
of predicted six-score burden; primary outcome is future burden
MAE/RMSE, not AUROC.

## No-smoke production CPU jobs

From your working checkout:
```bash
cd /home/ylong66/workspace/pks_agent
git fetch origin feat/phase62-ppmi-cohort-expansion
git switch --track origin/feat/phase62-ppmi-cohort-expansion

sbatch slurm/phase62_dusky_cohort_audit.sbatch
for h in 6 12 24; do
  sbatch --export=ALL,PHASE62_HORIZON="$h" slurm/phase62_dusky_forecast.sbatch
done
```

No separate smoke test, GPU request, network download, or Conda install.
Four independent CPU jobs; Slurm schedules them based on allocations.
Each produces only aggregate CSV/JSON in
/scratch/ylong66/pks_agent_outputs/phase62_*_<JOBID>/.
Check with `squeue -u "$USER"` and `sacct -j <JOBID>`.
Use actual job IDs when creating ZIPs.

**Warning:** Original Phase44 TEST was consumed. This is exploratory
method development only. Missing follow-up outcomes may be informative.
The new all-cohort data-capacity analysis and model evaluation should
not be described as clinical external validation or true joint RSI.
