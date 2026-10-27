# Phase71 CPU jobs: matched evidence longitudinal WM mechanism gate

Based on [PHASE70_RESULTS_PHASE71_DECISION.md](PHASE70_RESULTS_PHASE71_DECISION.md).
This batch deliberately avoids Qwen CPU inference. The Phase70 temporal
prediction had worse main-task MAE than static and Plain Qwen:
Prodromal 12m 1.0841 vs static 1.0511; Prodromal24m 1.3043
vs static 1.1999. Running more LLM jobs before isolating that
temporal prediction defect would be costly and hard to interpret.

20 stand-alone jobs: 5 patient evaluation folds times four methods.
0 static masked neural WM
1 temporal auxiliary only, fusion disabled
2 temporal auxiliary plus fixed 0.35 fusion
3 temporal auxiliary plus train-side TUNE select and once-only AUDIT gate
  for fusion coefficient drawn from [0,.1,.2,.35,.5].

All arms use two **ACTUALLY MEASURED** baseline or eligible R01
items selected by FIT-only feature coverage, with lexical tie-break.
No model can invent a tool result. R01 time offset is still estimated
as six months when an R01 score is provided, which remains a known
limitation rather than an exact per-patient timestamp.
Every job fits a matched static baseline and does patient-level
paired bootstrap in memory. Aggregates are exported only. Main outcome:
future six-item score MAE; Brier of any-nonzero event is guardrail.

The current internal frozen Prodromal TRAIN is repeatedly explored;
Phase71 is a *mechanistic* gate, not an independent-confirmation
experiment. It does not implement clinical action RSI, Qwen, or a
causal disease-state model.

## Launch all 20 separately with no array

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
for arm in $(seq 0 19); do
    sbatch --export=ALL,PHASE71_ARM_ID="$arm" slurm/phase71_dusky_cpu.sbatch
done
```

Per-job 8 CPU, 48 GB RAM, 12h; no GPU/LLM model load.
Each outputs fold_metrics.csv, coverage.csv, audit.json, summary.json
inside /scratch/ylong66/pks_agent_outputs/phase71_gate_<ARM>_<JOBID>.
Collect with `zip -r phase71_reports.zip phase71_gate_*/`.
Do not include source patient records, trajectories, or PPMI SQLite.

Code is committed, but job execution on Dusky has not been verified.
