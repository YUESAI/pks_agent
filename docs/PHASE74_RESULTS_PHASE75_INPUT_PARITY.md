# Phase74 uploaded report and Phase75 corrected-input experiments

Phase74 uploaded ZIP: 20 unique directories but only 8 contain result
files. Completed arm IDs: 4,5,6,7,12,13,14,15. Other 12 are empty.
Only folds 1 and 3 are evaluated; do NOT call this complete 5fold CV.
No job logs are included, so unknown whether other 12 are still
running, failed, or were killed. Completed runs reported 4-34 seconds,
not 12-hour run times.

Deduplicated identical within-job references across arms, patient-weighted
across folds 1 and 3:
- Prodromal 12m n=219: Ridge MAE 1.08868, HGB .97020,
  Direct CT .77931, verified CT .77931.
- Prodromal 24m n=155: Ridge 1.37957, HGB 1.14063,
  Direct CT .98290, verified CT .98290.
- PD 12m n=67: Ridge 2.98704, HGB 2.45730,
  Direct CT 2.57705, verified CT 2.53814.
- Six verifier candidates on 2 folds x 3 tasks: one PD12 accepted;
  five rejected. There is no evidence of a Prodromal RSI uplift.

**Critical correctness defect in Phase74**: Ridge/HGB engineered()
received baseline clinical optional values, but DirectWM.fit/predict
fed records with their `released={}` dictionary to
phase72_ctlatent_wm.sequence_features(), so the neural model never
actually received those nonmotor baseline values. This is a
different-information comparison; Phase74's favorable neural signal
should not be interpreted as a multimodal superiority claim.
Also Phase74 exported only CT versus Ridge paired intervals, not
the more relevant CT versus HGB.

## Phase75 repair and design

- Fix neural tensor representation: copy actual observed baseline
  optional clinical values into `released` BEFORE creating tensors.
  Missing modality remains absent, not imputed as a measured score.
- FIVE outer patient folds x FOUR arms: 0 same-patient matched
  Ridge/HGB; 1 Motor-only CT against Motor-only HGB; 2 Motor+nonmotor
  CT against same-panel HGB; 3 MAE loss revision on full panel with
  DEV proposal, independent AUDIT once-only veto/accept.
- Each job computes its own matched tabular baselines; 20 ordinary
  CPU submissions, no array and no Qwen/GPU.
- All models have same pre-target visible27 histories. Longitudinal
  clinical scores remain true measurements. Extra future endpoints
  are never used as predictors. Shared 12m/24m/PD12m tasks.
- Main inference: patient-count-weighted MAE and Brier on FIVE folds.
  Each relevant CT-HGB or multimodal-motor comparison is exported as
  a paired 500-bootstrap MAE delta with confidence interval.
- Task-level results are saved after each completed task, retaining
  partial outputs on 12h walltime; no exact optimizer-state resume.
- Results remain repeated frozen Prodromal TRAIN exploratory analyses,
  not confirmatory validation. A neural WM is not automatically a
  causal patient world model.

## Submit all twenty independent CPU jobs

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
  sbatch --export=ALL,PHASE75_ARM_ID="$arm" slurm/phase75_dusky_cpu.sbatch
done
```

8 CPU/48GB/12h per independent job. Each writes progress.json,
method_metrics.csv, paired_bootstrap.csv, coverage.csv, audit.json
and final summary.json if complete. ZIP aggregate folders only.
