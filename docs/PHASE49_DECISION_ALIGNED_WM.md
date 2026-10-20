# Phase49 — decision-aligned masked observation World Model

## Grounded observation from user's Phase48 aggregate report

**Run:** `phase48_cpu_330330`, dated 2026-10-20. The complete
three-member masked Transformer ensemble used 453 TRAIN V04 patients
(363 inner fit, 90 internal dev); 100 TRAIN-heldout patients (29
hidden-six positive) were scored at 4/8/12/16 visible-item budgets.

| q | Naive item NLL | Transformer item NLL | NB AUROC | Transformer AUROC | NB Recall@20% | Transformer Recall@20% |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.387643 | 0.375646 | 0.691598 | 0.673142 | 12/29 | 12/29 |
| 8 | 0.372952 | 0.358808 | 0.745993 | 0.717824 | 12/29 | 11/29 |
| 12 | 0.362463 | 0.349092 | 0.673142 | 0.677513 | 12/29 | 11/29 |
| 16 | 0.356810 | 0.344624 | 0.691598 | 0.685284 | 12/29 | 11/29 |

Thus Transformer item prediction improved by approximately 3.5%
relative NLL on average across these 4 checkpoints, but endpoint
ranking and acquisition results did not consistently improve.
**Do not conclude that stronger WM cannot help Doctor Agent.**
Phase48 was already repeatedly explored TRAIN and does not provide
independent confirmation.

## Phase49: a narrower, falsifiable change

Add a patient-pair ranking term to the Phase48 model's existing
binary-outcome and masked-item categorical objectives:

`L = L_binary + 0.5 * L_masked_item + λ * L_pairwise_rank`

where `L_pairwise_rank` is the mean
`softplus(-(logit_positive-logit_negative))` for each positive/
negative pair in a TRAIN fitting batch. Candidate
`λ in {0, 0.3, 1.0}` is chosen **only on inner development
patients**, via a pre-specified score averaging q=8 and q=12:

`Brier + 0.25*(1-AUROC) + 0.15*MaskedItemNLL`

The outer 3-fold, patient-disjoint TRAIN evaluation includes
Naive Bayes, λ=0 masked Transformer and selected decision-aligned
Transformer, with the same masks for each q. Reports contain:
`oof_decision_quality.csv`, `inner_candidate_verifier.csv`,
`fold_selected_objectives.csv`, `summary.json`.

A positive result would show a **decision-aware representation
gain**, not yet that a multi-agent RSI or time-evolving WM works.
If λ=0 is repeatedly selected, interpret that as failure of this
particular ranking objective to add value on these data.

## Run (Dusky CPU, no model download and no GPU)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
PY=/scratch/ylong66/conda/envs/pks_qwen46_cu118/bin/python
"$PY" -m py_compile scripts/phase49_decision_aligned_wm.py
"$PY" -m unittest tests.test_phase49_decision_aligned_wm -v

# Reduced workload / smoke:
PHASE49_EPOCHS=3 PHASE49_WIDTH=32 PHASE49_DEPTH=1 \
    sbatch slurm/phase49_dusky_cpu_wm.sbatch

# Full exploratory nested TRAIN-CV:
sbatch slurm/phase49_dusky_cpu_wm.sbatch
```

All inputs remain local on the cluster; reports under
`/scratch/ylong66/pks_agent_outputs/phase49_cpu_<JOBID>/`.
Use the REAL job ID to create the aggregate ZIP.

**Important:** the original Phase44 final TEST was consumed.
No claim of fresh heldout generalization can come from
retesting any architecture on that original TEST split.

This is a lightweight, controlled research experiment, **not**
a state-of-the-art temporal/action-conditioned World Model.
