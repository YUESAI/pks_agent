# Phase56 Results and Phase57 Simple Clinical Decision Baseline

## Actual Phase56 report from the user

Archive `phase56_full.zip`, Dusky job **330755**, generated
2026-10-22T00:16:23Z. It contains exactly four aggregate files:
`real_r01_to_v04_metrics.csv`, `fold_inner_selection.csv`,
`paired_nll_contrast.csv`, `summary.json`.

This is a *retrospective R01 -> future V04* evaluation, NOT
same-visit imputation. All patients from frozen Prodromal TRAIN;
the final Phase44 TEST was consumed and not queried.

| Metric | Value |
|---|---:|
| Longitudinal TRAIN people with >=2 visits | 658 |
| Strict real R01 -> later verified V04 paired people | **197** |
| Hidden six >=1 positive at later V04 | **57** |
| Selected for onsite verification (20% exact budget) | **39** |
| Frozen TRAIN-only patient folds | 5 |

| Controller | Future six-item patient mean NLL | AUROC | Positive capture (39 selected) |
|---|---:|---:|---:|
| Temporal masked latent WM | 0.315892 | **0.679261** | **19 / 57 (33.33%)** |
| Inner-tuned item Markov | **0.314551** | 0.650940 | **23 / 57 (40.35%)** |
| Fit-only V04 site item marginal | 0.339238 | 0.436842 | 10 / 57 (17.54%) |

Patient-paired bootstrap of future-site six-score categorical NLL:
- WM minus tuned Markov: **+0.001341**, 95% interval
  **[-0.024964, +0.025792]**, 500 resamples.
- WM minus onsite V04 marginal: **-0.023346**, 95% interval
  **[-0.045918, -0.001786]**, 500 resamples.

**Interpretation:** WM predicts future onsite scores better than the
uninformed marginal reference in this TRAIN cohort, but is not
demonstrably better than tuned Markov. AUROC and 20%-quota
capture point in different directions; no paired intervals for
these decision metrics exist in this report.

### Training detail

Outer heldout R01/V04 paired people by fold: 44,42,39,27,45.
Total outer fitting patients 526 or 527, with 447/448 inner fit
and 79 inner development. Selected neural epochs:
25,25,10,25,2. Inner-tuned Markov selects
gap modes [gap,pooled,gap,pooled,gap] and alpha
[2,2,10,2,2]. Future V04 MONTH was supplied as
a **retrospective forecast horizon**; its actual scores,
status mask and recorded target values were not model inputs.

### Key limitations and next controls

The six-item at-least-one-positive risk in Phase56 is constructed as
`1 - product_h(P(hidden_item_h=0))`. This product factorization
implicitly assumes conditional independence of the six items,
and its calibration is not established.

The strict 197 people are not necessarily identical to the
previous 199-person frozen R01->V04 cohort; do not relabel or
update previous reports without a local eligibility reconciliation.

In particular, the previously frozen **real R01 visible-score sum**
policy was missing from the Phase56 clinical selection table.
Before asserting any benefit of WM-guided Qwen Doctor, compare it
at the same 20%-quota, same strict eligible people, same later V04
outcome and deterministic tie-break.

## Phase57: same-cohort real R01 visible 27-score baseline

New implementation `scripts/phase57_visible27_same_cohort_baseline.py`
calls the **identical Phase56 `find_pairs` selection**. It uses
only the 27 actually measured remote R01 scores, ranks patients
by their sum, and measures AUROC plus captured future six-positive
people under the exact same 20% quota. The risk score is a ranking
quantity, not a calibrated probability, so categorical NLL is
NOT computed for it. It also provides the exact-k random expected
capture and patient bootstrap bounds for standalone metrics.

Only frozen TRAIN is used, never the consumed TEST or VAL. No
individual patient identifiers, item scores, outcomes or rankings
are written to report files.

### Run on offline Dusky CPU

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase57_visible27_same_cohort_baseline.py
"$PY" -m unittest tests.test_phase57_visible27_baseline -v
sbatch slurm/phase57_dusky_cpu_visible27.sbatch
```

Once done, use the actual job ID:

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -dt phase57_visible27_*
zip -r phase57_report.zip phase57_visible27_<ACTUAL_JOBID>/
```

This is an *additional TRAIN diagnostic*; the results cannot be
used as independent confirmation. A direct paired bootstrap of
WM vs visible-sum clinical capture requires the same patient-wise
OOF predictions to be compared **inside a new controlled run**,
not computed from separate aggregate CSV files.
