# Phase64: Real multimodal PPMI / Qwen / WM / verified calibration

Branch: feat/phase62-ppmi-cohort-expansion (NO new branch).

This exploratory Phase64 expands beyond Motor scores to actually downloaded
baseline MDS-UPDRS I/II, MoCA, UPSIT, RBD and SCOPA-AUT measurement rows,
where columns are present. Retains BL Motor and pre-target R01 observations.
Missing tables or values remain unavailable; no invented score.

20 separate array tasks = 5 feature panels x 4 decision strategies:

Panel 0: BL Motor remotely visible 27 only.
Panel 1: BL remotely visible 27 plus baseline multimodal clinical values.
Panel 2: Panel 1 + recorded 4..8m R01 Motor items.
Panel 3: BL full33 Motor plus multimodal baseline.
Panel 4: Panel 3 + measured R01 Motor.

Policy 0: deterministic WM action heuristic.
Policy 1: real local Qwen3-4B Doctor, no WM advice.
Policy 2: real local Qwen3-4B Doctor with WM action-value advice.
Policy 3: Qwen + WM plus deterministic two-tier verified scalar risk calibration.

**Important limitation:** Policy 3 is constrained verifier-gated
calibration and is NOT autonomous LLM-proposed / neural-weight Joint RSI.
The WM is masked Ridge future-burden prediction plus an empirical
candidate-value heuristic, not a generative clinical disease simulator.
These jobs are an end-to-end multimodal engineering pilot, NOT an adequate
confirmatory top-conference Joint RSI main-result by themselves.

Outcomes from ACTUAL later verified 6 hidden Motor items at 12/24 months.
Four tasks: Prodromal12/24, PD12/24 (if enough patients).
Prodromal original frozen TRAIN ONLY; PD new strict cohort, no external
validation. Strict data split by hashed patient:
folds 2,3,4 fit; fold0 subdivided tune / verifier audit;
fold1 final TRAIN-internal evaluation. Data after visit horizon
never becomes observation or prompt. Tool only returns ACTUAL recorded
BL/pre-target R01 values, status of 101/UR is never scored zero.
All 20 arms share the same split and budgets. Missing tests cannot be
requested; available feature panels differ in real coverage.

## Submit 20 CPU tasks

```bash
cd /home/ylong66/workspace/pks_agent
git switch feat/phase62-ppmi-cohort-expansion
git pull --ff-only origin feat/phase62-ppmi-cohort-expansion
sbatch --array=0-19%5 slurm/phase64_dusky_multimodal_cpu.sbatch
```

20 jobs are queued, at most 5 concurrently by default. If you have
appropriate CPU quota and want all 20 *concurrently*, change to
`sbatch --array=0-19%20 slurm/phase64_dusky_multimodal_cpu.sbatch`.
15 of these 20 tasks load a genuine 4B Qwen model in float32.
CPU inference may exceed the 24h per-task allocation; check actual
logs. Each task asks four actual-observation questions per evaluated
person at each horizon. If Qwen CPU tasks time out, that is a resource
limit, not evidence of clinical model failure. No separate smoke jobs.

```bash
squeue -u "$USER"
sacct -j <ARRAYID> --format=JobID,State,Elapsed,MaxRSS
tail -n 50 slurm-phase64-<ARRAYID>_0.log
cd /scratch/ylong66/pks_agent_outputs
zip -r phase64_multimodal_reports.zip phase64_multimodal_*/
```

Reports per task: metrics.csv, cohort_sizes.csv, verifier_trace.json,
summary.json. Aggregate only, no patient-level outputs, text prompts
or PPMI raw records.

Next milestone for a defensible method paper:
fit a real autoregressive masked longitudinal WM across VISITS,
demonstrate predictive improvements over simple regularized baselines,
implement actual allowlisted WM checkpoint/policy revisions with
independent audit and rollback, and compare joint vs parameter/policy
only under matched budgets and a genuinely new test cohort.
