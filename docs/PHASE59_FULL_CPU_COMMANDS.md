# Phase59 FULL CPU execution — four independent real Qwen3-4B arms

**Created 2026-10-22.** This document supersedes GPU-only submission
commands if L40S allocations are scarce. No need to run a separate
smoke Slurm experiment: each production job directly loads frozen
TRAIN data and performs full evaluation; any import/runtime failure
causes the job to exit nonzero.

## CPU changes

- Existing `TransformersDoctor` in
  `scripts/phase46_qwen_doctor.py` now accepts
  `device="cpu"` or `device="cuda"`. GPU default is retained.
- Explicit CPU inference uses **Qwen3-4B-Instruct-2507,
  torch.float32, device_map="cpu"**. The model and tokenizer are
  read exclusively from the already-downloaded Scratch checkpoint.
  This is a real local LLM, **not** a fake/mocked Doctor.
- `scripts/phase59_integrated_pilot.py` passes
  `--doctor-device cpu`. The lightweight temporal WM stays on CPU.
- `slurm/phase59_dusky_cpu_four_arm.sbatch` uses Dusky
  `def-hup-ab_cpu` / `base`, requests 12 CPU cores, 64GB RAM
  and 12h time limit per arm. **No `--gres` or `--gpus` flags.**
- No pre-experiment py_compile, unittest suite, synthetic smoke job,
  CUDA preflight, pip install, Conda install, internet or model download
  is required in the production CPU job.

**Do not confuse small dataset with cheap LLM inference.**
WM training has tens of thousands of parameters and ~658 histories;
the **four-billion parameter Qwen** must be evaluated repeatedly.
Each 60-person × 8-question arm generates about 480 separate LLM
responses (plus a few RSI proposal rounds in the RSI arm).
CPU token generation may be far slower than an L40S. Expect potential
memory pressure and clock-time overruns. Actual runtimes must come
from `sacct`, not a promised estimate.

## Four commands (submit concurrently to Slurm)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit

sbatch --export=ALL,PHASE59_ARM=plain,PHASE59_QUESTIONS=8,PHASE59_MAX_PATIENTS=60 \
  slurm/phase59_dusky_cpu_four_arm.sbatch

sbatch --export=ALL,PHASE59_ARM=observation,PHASE59_QUESTIONS=8,PHASE59_MAX_PATIENTS=60 \
  slurm/phase59_dusky_cpu_four_arm.sbatch

sbatch --export=ALL,PHASE59_ARM=dual,PHASE59_QUESTIONS=8,PHASE59_MAX_PATIENTS=60 \
  slurm/phase59_dusky_cpu_four_arm.sbatch

sbatch --export=ALL,PHASE59_ARM=rsi,PHASE59_QUESTIONS=8,PHASE59_MAX_PATIENTS=60 \
  slurm/phase59_dusky_cpu_four_arm.sbatch
```

Each job is independent and selects exactly the same
frozen TRAIN site fitting/development/evaluation patients.
Slurm will run them concurrently only if sufficient CPU/RAM
allocations are available; otherwise they queue.
Do not submit each job multiple times unless intentionally
repeating a predeclared seed.

## Monitoring and reports

```bash
squeue -u "$USER"
sacct -j <actual_ID> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS
ls -1t slurm-phase59-cpu-*.log | head -n 8
find /scratch/ylong66/pks_agent_outputs -maxdepth 1 \
 -type d -name 'phase59_cpu_*'
```

Successful aggregate output dirs:
- `phase59_cpu_plain_<JOB_ID>/`
- `phase59_cpu_observation_<JOB_ID>/`
- `phase59_cpu_dual_<JOB_ID>/`
- `phase59_cpu_rsi_<JOB_ID>/`

```bash
cd /scratch/ylong66/pks_agent_outputs
zip -r phase59_cpu_four_reports.zip \
 phase59_cpu_plain_<ID1>/ \
 phase59_cpu_observation_<ID2>/ \
 phase59_cpu_dual_<ID3>/ \
 phase59_cpu_rsi_<ID4>/
```

Only aggregate CSV/JSON are exported. Positive/count cells under 10
are conservatively suppressed. Do not upload private raw PPMI,
manifest, checkpoints or row-level trajectories.

## Research boundaries

The model chooses 8 simulated questions about true recorded V04
27 remotely observable items and predicts **same-V04** six
hands-on score nonzero. This is onsite masking SIMULATION,
not the separate real R01->future V04 trial or a prospective
clinical action. The temporal prior is trained on earlier
measured Motor history excluding development and evaluation
IDs. It is not an actual action-conditioned longitudinal Dreamer4
simulator.

The RSI condition only proposes allowlisted prior/temperature/
strategy updates with development Brier accept/reject and rollback;
it does not rewrite neural weights or code. The previous Phase44
TEST is consumed, and all four reports are repeatedly explored
TRAIN outcomes without an independent confirmatory comparison.
