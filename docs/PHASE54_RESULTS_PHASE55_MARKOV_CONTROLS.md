# Phase54 report and Phase55 conditional transition controls

## Verified Phase54 result — user-provided zip `phase54_masked.zip`

Job **330531**, timestamp 2026-10-21T22:07:48Z, frozen Prodromal
PPMI **TRAIN only**. The archive contains aggregate report files
`patient_oof_item_nll.csv`, `paired_nll_contrast.csv`,
`fold_training_sizes.csv` and `summary.json`, with no raw
patient-level information.

| Model | Patient-average observed next-visit item NLL (lower better) |
|---|---:|
| Recurrent masked time-conditioned Motor WM | **0.413298** |
| Training-patient item marginal probabilities | 0.448684 |
| Fixed 0.85 prob item persistence | 0.669542 |

**658 patients**, five patient-disjoint outer folds, 131–132 heldout
patients per fold, **526–527 full outer-fit patients** per fold,
inner fitting 447–448 and inner development 79 per fold.
Chosen neural epochs 25, 10, 10, 5, 25.

WM - fixed .85 persistence difference in patient mean NLL:
**-0.256243**, 400-resample patient bootstrap 95% interval
**[-0.269142,-0.242533]**.

Relative to fit-only item marginals, mean NLL decrease:
**0.035386**, or about **7.89%** relative. The original
uploaded ZIP does NOT contain the patient-paired confidence interval
for this **stronger marginal reference**, nor any class-by-class
reliability / calibration report. The fixed 0.85 persistence assumption
is arbitrary and may be badly overconfident.

These results support **promising longitudinal future-observation
modelling on exploratory frozen TRAIN**, NOT proof of state-of-the-art
clinical WM, prospective forecasting, or intervention planning.
Target visit time is supplied retrospectively, and the original
Phase44 TEST was already consumed.

## Phase55: more credible comparison against item Markov transitions

Code `scripts/phase55_markov_baseline.py` fits an item-wise
Dirichlet-smoothed conditional categorical distribution:

`p(score_next | last_actual_score, item, time_gap_bin)`

The `last_actual_score` is updated only when that item is
truly measured 0..4; `101/UR` and missing never enter the last
observation state. `time_gap_bin` is either omitted or
0–8, 9–15, >=16 months. Four Dirichlet shrinkage strengths
are considered: 2, 10, 50, 200. Both the use of
time-gap bins and shrinkage are chosen **inside each outer
fold on inner development patients only**, and refitted
on the entire outer training cohort.

Updated `scripts/phase54_evaluate.py` re-runs the *same*
5-fold neural experiment with new comparisons:
- recurrent temporal neural WM,
- fixed 0.85 item persistence,
- TRAIN-fit marginal frequencies,
- **inner-tuned item-wise Markov conditional transitions**.

The report now contains `fold_markov_selections.csv` and
paired patient-bootstrap WM contrasts against *all three*
baselines, not just the weak fixed-persistence baseline.
This re-run avoids trying to infer individual paired intervals
from earlier aggregate means (which is not possible).

**Important:** The modified training source is the same
Phase54 model/algorithm. However, any model refit is subject
to environmental and pseudorandom determinism; compare with
the original Phase54 aggregate to check reproducibility.

## Launch on Dusky (no network, downloads, or new Conda packages)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase55_markov_baseline.py scripts/phase54_evaluate.py
"$PY" -m unittest tests.test_phase55_markov_baseline tests.test_phase54_masked_temporal_wm -v

# 5-fold full run, with tuned Markov baseline / all paired CIs
sbatch slurm/phase55_dusky_markov_controls.sbatch
```

After success, substitute actual ID:

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -dt phase55_markov_*
zip -r phase55_report.zip phase55_markov_<actualJobID>/
```

What to read: `patient_oof_item_nll.csv` for all four
methods, `paired_nll_contrast.csv` for patient-paired intervals,
`fold_markov_selections.csv` for fold-level selected shrinkage
and gap-conditioning.

This controlled comparison is TRAIN-only and repeatedly exploratory,
not an independent confirmatory result; only a new external cohort
could support future locked confirmation.
