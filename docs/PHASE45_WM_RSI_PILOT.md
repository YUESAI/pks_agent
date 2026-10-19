# Phase45: CPU-first WM × Grounded Agents × RSI-lite pilot

Status: **implemented experimental prototype**, not validated on real PPMI by the authoring environment.
Phase23–44 results and the consumed final TEST remain frozen.

## What is genuinely implemented

An observation World Model estimates categorical likelihoods of each
same-visit V04 0–4 Motor item conditional on a binary hidden-six
outcome, using smoothed class-conditional likelihoods and Bayes updates.
It chooses queries via conditional expected entropy reduction.
The Doctor policy selects which of the 27 visible scores to request;
a deterministic Patient Evidence Service reveals only measured V04
values, checking allowed fields and action budget. The hidden six
labels cannot be requested. A verifier-guided RSI-lite outer loop
proposes changes to observation model smoothing/temperature and
Doctor query planner, keeping candidates only when inner
development recall minus 0.2 × Brier improves.

Three outer patient folds stay separate from inner RSI proposal
selection. Data is limited to Phase23 frozen TRAIN patients with a
complete, internally sum-verified V04 MDS-UPDRS Part III exam.
Every experiment outputs ONLY aggregate CSV / JSON reports.

**Not yet implemented:** disease-progressing temporal WM, real
LLM/VLM inference, autonomous source-code rewriting RSI, multi-agent
natural-language dialogue, clinical action economics or prospective
doctor/patient decisions. Phase45 is proof of mechanism, not a claim
that any of these are already effective.

## Experiment matrix

| Name | Visible items per patient | Method |
|---|---:|---|
| random_topk | 0 | deterministic random onsite capacity allocation |
| visible_sum_27 | 27 | clinical score rank; strong full-information baseline |
| wm_full_27 | 27 | WM posterior with every recorded visible score |
| random_questions | q | random grounded Doctor questions + fixed WM |
| static_questions | q | static information-gain ordering + fixed WM |
| dynamic_questions | q | adaptive conditional information-gain + fixed WM |
| rsi_policy | q | RSI-lite evolves planner, WM fixed |
| rsi_wm | q | RSI-lite evolves WM smoothing/temperature, planner fixed |
| rsi_joint | q | RSI-lite jointly searches planner and WM |

Default q = 8; onsite selection budget = 20% of all frozen TRAIN
onsite V04 patients. Main outcome is recall of hidden-six-positive
patients; also report burden capture, precision, AUROC, probability
calibration (Brier) where applicable, proposal traces and pairwise
conditional bootstrap intervals. Questions=27 baselines cost MORE
information actions than q=8 methods; do not call cross-budget
comparisons matched-cost performance improvements.

## Local run (login only for smoke tests)

```bash
cd /home/ylong66/workspace/pks_agent
git pull --ff-only origin feat/parkinson-data-audit
python -m py_compile scripts/phase45_world_model.py scripts/phase45_wm_rsi_agents.py
python -m unittest tests.test_phase45_wm_rsi_agents -v
```

## Actual TRAIN run on Dusky CPU (recommended)

```bash
cd /home/ylong66/workspace/pks_agent
sbatch slurm/phase45_dusky_cpu.sbatch
squeue -u "$USER"
```

Every job writes aggregates to
`/scratch/ylong66/pks_agent_outputs/phase45_dusky_<jobid>/`.
Check `slurm-phase45-dusky-<jobid>.log` and the corresponding `.err`.
The Slurm file uses the exact CPU allocation provided for Dusky
and loads Python from `/scratch/ylong66/conda/envs/ms127/bin/python`.

Direct CPU run without Slurm, for initial smoke checking:

```bash
mkdir -p /scratch/ylong66/pks_agent_outputs/phase45_smoke
python scripts/phase45_wm_rsi_agents.py \
  --data-root /scratch/ylong66/data/pks/data \
  --manifest-db ./outputs/longitudinal_manifests/prodromal_motor_v0.sqlite \
  --report-dir /scratch/ylong66/pks_agent_outputs/phase45_smoke \
  --folds 3 --questions 8 --rounds 2 --bootstraps 100
```

## Nibi backup (GPU reserved but UNUSED by V0)

```bash
sbatch slurm/phase45_nibi_gpu_optional.sbatch
```

Do not choose this path merely to accelerate Phase45 V0; it is
CPU-only and reserves an H100 MIG GPU without utilizing it.
On a CPU partition of Nibi, change resource flags according to
local allocation policy rather than request a GPU unnecessarily.

## No-internet compute nodes & optional model downloads

**Phase45 needs no HF LLM/VLM and downloads nothing.** If a later
iteration uses real LLM or VLM inference, on a NETWORKED LOGIN NODE:

```bash
cd /home/ylong66/workspace/pks_agent
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/ylong66/.cache
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" /scratch/ylong66/models
python -m pip install -U huggingface_hub
python scripts/prefetch_optional_agent_models.py --kind text
# Optional, only if actual image input/VLM experiment becomes justified:
python scripts/prefetch_optional_agent_models.py --kind vlm
```

Verified public model IDs (as of 2026-10):
`Qwen/Qwen3-0.6B` (text) and
`Qwen/Qwen2.5-VL-3B-Instruct` (vision-language). Model files go to
`/scratch/ylong66/models/Qwen--Qwen3-0.6B/` and
`/scratch/ylong66/models/Qwen--Qwen2.5-VL-3B-Instruct/`.
On a compute node, set:

```bash
export HF_HOME=/scratch/ylong66/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/scratch/ylong66/.cache
```

Load the *local model directory* with `local_files_only=True`.
Do not attempt Hugging Face network requests on compute nodes.
For a reproducible paper, use immutable model revision SHA rather than
`main`, and record all licenses, model revisions and exact dependencies.
VLM models are not needed for the currently tabular PPMI Part III data.

## Reporting

After the job, upload `model_metrics.csv`,
`paired_policy_contrasts.csv`, `evolution_trace.csv`,
`selected_configs.csv` and `summary.json` only, as an aggregate ZIP.
Never export individual score vectors, patient IDs, simulated agent
transcripts containing record values or the SQLite manifest.

Do not choose hyperparameters based on Phase43 Validation or Phase44
consumed TEST. Successful TRAIN-only pilot results require a new
independent confirmation cohort before any final new research claim.

## Recommended first science sweep: three Dusky CPU jobs

The CPU array evaluates **q=4, q=8, q=12** remote item requests with
identical recorded TRAIN cohort and fixed seed. These are exploratory
TRAIN experiments, not independent hypothesis tests. It creates one
scratch aggregate directory per task:

```bash
cd /home/ylong66/workspace/pks_agent
sbatch slurm/phase45_dusky_cpu_array.sbatch
squeue -u "$USER"
```

Outputs:
`/scratch/ylong66/pks_agent_outputs/phase45_sweep_<array_job_id>/q4/`,
`q8/`, `q12/` (only aggregate CSV and summary JSON). Compare
`model_metrics.csv` and `evolution_trace.csv` across question budgets.
Check Slurm stdout/stderr in `slurm-phase45-array-<jobid>_<task>.log`.
