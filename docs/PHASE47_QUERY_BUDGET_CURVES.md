# Phase47 — One-pass Qwen Doctor query-budget curves

**Purpose:** Phase46 showed at 100 frozen TRAIN V04 patients, q=4:

| Controller | hidden-positive recall @ 20% | AUROC | Brier |
|---|---:|---:|---:|
| Qwen Doctor without WM guidance | 12/29 (41.38%) | 0.74381 | 0.18256 |
| Qwen Doctor with WM guidance | 10/29 (34.48%) | 0.64862 | 0.20225 |
| Static WM | 10/29 (34.48%) | 0.64570 | 0.20766 |
| Dynamic WM | 10/29 (34.48%) | 0.64570 | 0.20766 |
| Visible sum of ALL 27 items | 12/29 (41.38%) | 0.69136 | N/A |

Behavior audit: WM-guided LLM has 95.75% item-set overlap with
dynamic WM and exactly the same selected top-k people; static
and dynamic WM have 100% sequence agreement at q=4.
The WM guidance may cause Qwen to imitate a simplistic planner.
These are TRAIN-only exploratory results, NOT proof the LLM is superior.

## Design

Run a SINGLE grounded Doctor/Patient trajectory through 16 queries
for each patient and each LLM mode. Store risk posterior in process
RAM after q=4,8,12,16 queries. This uses only 16 model generations
per LLM mode/person, not the sum of 4+8+12+16.
At each checkpoint, recompute strict 20% top-k selection, AUROC,
hidden-positive recall, Brier and burden capture. The same TRAIN-only
fit/heldout split and seed are fixed across checkpoints.

Include static WM, adaptive dynamic WM, all-27 sum, and full-27 WM
as comparators. Compare q-matched methods primarily; 27-item
controls are a higher-information reference, not equal cost.
An aggregate behavioral comparison of LLM-only vs LLM+WM and WM
controller question sequences is exported per budget, with no
individual clinical data or question transcripts.

## Run with existing offline Qwen Conda and model

No new dependencies, internet access or model weights required.

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase46_qwen_doctor.py scripts/phase47_question_budget_curve.py
/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python -m unittest \
  tests.test_phase46_qwen_doctor tests.test_phase47_question_budget_curve -v

# Dusky L40S, preferred if available
sbatch slurm/phase47_dusky_qwen_budget.sbatch

# Or Nibi H100 MIG, not both unless replicating for system checks
sbatch slurm/phase47_nibi_qwen_budget.sbatch
```

Result directories:
`/scratch/ylong66/pks_agent_outputs/phase47_dusky_<JOBID>/` or
`/scratch/ylong66/pks_agent_outputs/phase47_nibi_<JOBID>/`.
Reports include `cost_curve_metrics.csv`,
`question_agreement_by_budget.csv`, and `summary.json`.

```bash
cd /scratch/ylong66/pks_agent_outputs
ls -d phase47_*
zip -r phase47_report.zip phase47_dusky_<ACTUAL_JOB_ID>/
```

**Important:** Phase47 is a V04 same-visit masking simulation
inside previously explored TRAIN patients. It cannot establish
actual clinical R01 examination value, new external generalization,
or benefits of temporal WM/LLM-driven recursive self-improvement.
The original Phase44 final TEST was already consumed and stays
unopened by this script.
