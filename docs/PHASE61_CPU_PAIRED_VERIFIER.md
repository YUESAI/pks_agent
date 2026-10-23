# Phase59 and Phase60 completed; Phase61 CPU paired verifier

## Phase59 real local Qwen CPU (uploaded results)

All four arms ran 60 shared people, 8 requested V04 recorded items,
17 positives, 12 referral slots (frozen TRAIN development reuse).
AUROC/Brier/recall: plain .7770/.1654/.4706; observation
.8393/.1666/.3529; dual .8536/.1535/.4118;
RSI .8564/.1446/.4706. RSI accepted one out of three
allowlisted edits, with two subsequent verifier rejections.
No per-patient predictions were exported, so pairwise CI cannot
be computed from Phase59 reports. Four-arm aggregate winner is
NOT a confirmatory or clinical claim.

## Phase60 (uploaded result)

121 people, 39 positives, at 4/8/12 budgets, 4 model/planner arms
all had exactly identical Brier, AUROC and recall.
DEV item NLL K=1 .4510133, K=2 .4536716, K=3 .4559294;
K=1 was selected, effectively reducing the mixture to naive model.
Two-step lookahead did not produce observable advantage on current
measured same-V04 masking simulation. No model-refit CI in Phase60.
Don't infer that "all rollouts never help"; not established.

## Phase61 experiment purpose

Isolate longitudinal prior, question policy, temperature calibration,
and a predeclared one-shot internal verification gate using CPU and
cohort-paired inference. Six arms: random_fixed, observation_fixed,
temporal_fixed, observation_calibrated, temporal_verified,
temporal_shuffled. The negative-control temporal_shuffled permutes
baseline temporal priors without using outcomes.

A 65/15/20 FIT/DEV/EVAL TRAIN split follows prior Phases;
DEV is additionally split 60/40 as TUNE/AUDIT (by label for
stratification). Only TUNE selects from a frozen grid of
weights/temperatures/planner objectives. AUDIT is used ONCE to
accept/rollback the preselected candidate; EVAL is read only for
report. Longitudinal neural WM training excludes all DEV and EVAL IDs.
All observed patient responses come from real V04 recorded entries.

IMPORTANT: This is a CPU World Model + deterministic planner study,
NOT a real 4B Qwen Doctor rerun and NOT joint WM weight RSI.
Original Phase44 TEST is consumed; repeated TRAIN reuse remains
exploratory. Paired bootstrap is conditional on fitted models and
does not correct past exploration. Aggregate only; suppress small
captured positive cells (<10).

## Run full CPU formal job; no separate smoke Slurm task

```bash
cd /home/ylong66/workspace/pks_agent
git fetch origin feat/phase61-cpu-paired-verifier
git switch --track origin/feat/phase61-cpu-paired-verifier
sbatch slurm/phase61_dusky_cpu_paired.sbatch
```

The script requests 4 CPU, 24 GB, 12 hours; no GPU.
Outputs under /scratch/ylong66/pks_agent_outputs/
phase61_cpu_paired_<JOBID>/:
arm_metrics.csv, paired_bootstrap.csv, verifier_trace.json, summary.json.
Track with squeue -u "$USER" and
sacct -j <JOBID> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS.
