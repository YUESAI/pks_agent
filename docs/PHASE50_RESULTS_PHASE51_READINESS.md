# Phase50 results and Phase51 next step (2026-10-20)

## Observed Phase50 report (Dusky job 330365)

The user-provided archive `phase50_report.zip` contained
`oof_metrics.csv`, `fold_training.csv`,
`paired_smoothing_contrast.csv`, `summary.json`.

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| last_value | 3.24699 | 5.26067 | -0.50000 |
| baseline_value | 3.26807 | 5.27982 | -1.18976 |
| **fixed_50_50_smoothing** | **2.90813** | 4.79858 | -0.84488 |
| linear_extrapolation | 4.66077 | 7.34075 | +0.56794 |
| **latent_imagination_wm** | 2.91681 | **4.79599** | -0.89289 |

All 332 sequences were TRAIN data; three outer patient-held-out
folds, three ensemble members. Actual per-model inner fitting
sizes were 177–178 people, with 44 inner development records per
fold. The latent WM MAE difference vs fixed 50/50 is **+0.00868**,
paired bootstrap 95% interval **[-0.07738,+0.11848]**,
300 bootstrap replications. No demonstrated MAE gain, and no
credible generalization claim beyond repeatedly explored TRAIN.

The model uses a neural latent transition and imagined forward
state, but there is *no evidence yet that the learned state dynamics
outperform strong historical smoothing*. The 2026 Dreamer/JEPA
references motivate future architecture, not this model's superiority.

## Immediate bottleneck: only three total-score visits

The Phase50 source consisted of ~332 complete-case BL/12m/24m
Prodromal TRAIN trajectories, of which each model was fit on fewer
than 180 patients per outer fold. Deep temporal WM training on
these coarse (1 score/visit) sequences is intrinsically data-limited.

Before increasing WM capacity, count ALL eligible followup visits in
the original PPMI Motor CSV, without restricting to canonical three
visit windows or to complete-case sequences.

## Phase51 strict observation capacity audit

Code: `scripts/phase51_longitudinal_visit_readiness.py`.

- frozen manifest `SELECT ... WHERE split='train'` exclusively;
- BL exam month exactly matches frozen baseline anchor;
- exclude invalid followup event codes and pre-BL followups;
- exclude ambiguous score/event values per person/month/drug state;
- count unique per-person Motor exams and adjacent visit intervals;
- keep `UNSPECIFIED`, `ON`, `OFF` distinct; the primary
  sequence counts UNSPECIFIED ONLY, **not untreated patients**;
- output suppressed (<10) aggregate tables only, never patient rows;
- never open frozen validation or consumed Phase44 TEST subjects.

Reports include `primary_visit_distribution.csv`,
`primary_baseline_anchored_distribution.csv`,
`medication_state_visit_distribution.csv`,
`primary_followup_gap_bins.csv`, `data_quality_counts.csv`
and `summary.json`.

### Run (Dusky CPU, no internet, no new dependencies)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m unittest tests.test_phase51_longitudinal_visit_readiness -v
sbatch slurm/phase51_dusky_cpu_readiness.sbatch
```

Output: `/scratch/ylong66/pks_agent_outputs/phase51_readiness_<jobid>/`.

If enough patients have 4+ nonambiguously observed visits, next
develop an **irregular-time multi-step WM** with real heldout
next-visit predictions and forecast calibration; if coverage remains
low, broaden to additional PPMI cohorts / multimodal outcomes
with provenance and true missingness masks before scaling model.

Never claim simulation supports treatment counterfactuals. Neither
the frozen original TEST nor new independent patient populations
are used for this audit.
